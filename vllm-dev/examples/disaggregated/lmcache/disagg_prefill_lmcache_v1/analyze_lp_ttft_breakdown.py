import os
import re
import csv
import math
import sqlite3
import argparse
import subprocess
import bisect
import calendar
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# This script lives in the directory it reads logs from, so anchor on itself
# rather than an absolute path tied to one machine.
BASE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description='Build TTFT breakdown plots and CSV from Dynamo/vLLM logs + nsys.'
)
parser.add_argument('--log-name', default=None,
                    help='Log tag without the trailing -logs, resolved under '
                         'dynamo-log/. Example: no-local-lookup-scale2')
parser.add_argument('--log-dir', default=None,
                    help='Absolute log directory. Overrides --log-name.')
parser.add_argument('--out-subdir', default='ttft_breakdown_static')
parser.add_argument('--sqlite-dir', default='/tmp')
parser.add_argument('--no-export', action='store_true',
                    help='Reuse existing sqlite exports instead of running nsys export.')
args = parser.parse_args()

if args.log_dir:
    LOG = Path(args.log_dir)
elif args.log_name:
    LOG = BASE / f'dynamo-log/{args.log_name}-logs'
else:
    parser.error('one of --log-dir or --log-name is required')

# Output files live inside the run's own log directory, so they need no run
# prefix. The sqlite exports are shared under --sqlite-dir, so those do keep a
# per-run name to avoid one run clobbering another's export.
RUN = re.sub(r'-logs$', '', LOG.name).replace('-', '_')
OUT = LOG / args.out_subdir
OUT.mkdir(parents=True, exist_ok=True)

def worker_num(path: Path) -> int:
    m = re.search(r'prefill-(\d+)', path.name)
    return int(m.group(1)) if m else -1

NSYS = {
    worker_num(rep): {
        'sqlite': Path(args.sqlite_dir) / f'{RUN}_prefill_{worker_num(rep)}.sqlite',
        'rep': rep,
    }
    for rep in sorted(LOG.glob('prefill-*.nsys-rep'), key=worker_num)
}

for wid in list(NSYS):
    paths = NSYS[wid]
    rep = paths['rep']
    db = paths['sqlite']
    if not rep.exists():
        print(f'WARNING: nsys report missing for worker {wid}: {rep}; requests on this worker will have coverage=False')
        del NSYS[wid]
        continue
    if args.no_export and db.exists():
        continue
    if db.exists() and db.stat().st_mtime_ns >= rep.stat().st_mtime_ns:
        continue
    print(f'exporting {rep} -> {db}')
    subprocess.run([
        'nsys', 'export', '--type=sqlite', '--force-overwrite=true',
        '--quiet=true', f'--output={db}', str(rep)
    ], check=True)
if not NSYS:
    raise FileNotFoundError(f'no prefill nsys report found under {LOG}')

ANSI = re.compile(r'\x1b\[[0-9;]*m')
ISO_RE = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)')
BRACKET_TS_RE = re.compile(r'\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]')
RID_RE = re.compile(r'(?P<rid>\d{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})')
REQ_LINE_RE = re.compile(r'^\[req\] idx=(\d+) ts=(\d+)ms in=(\d+) out=(\d+) hashes=(\d+)')
OK_LINE_RE = re.compile(r'^\[ok \] idx=(\d+) out=(\d+)/(\d+) ttft=([0-9.]+)ms tpot=([0-9.]+)ms e2e=([0-9.]+)ms chunks=(\d+)')
LM_RE = re.compile(r'Reqid: (?P<full>\d{8}-[0-9a-f-]{27,40}-[0-9a-f]{8}), Total tokens (?P<total>\d+), Inference Engine computed tokens: (?P<computed>\d+), LMCache hit tokens: (?P<hit>\d+), need to load: (?P<need>\d+)')
P2D_RE = re.compile(r'NIXL p2d transfer done: request_id=(?P<full>\d{8}-[0-9a-f-]{27,48}) xfer_ms=(?P<xfer>[0-9.]+) post_ms=(?P<post>[0-9.]+) bytes=(?P<bytes>\d+) descs=(?P<descs>\d+) handles=(?P<handles>\d+)')
WORKER_ID_RE = re.compile(r'prefill_worker_id=(\d+)')
DECODE_ID_RE = re.compile(r'decode_worker_id=(\d+)')
INPUT_TOKENS_RE = re.compile(r'input_tokens=(\d+)')
TTFT_RE = re.compile(r'ttft_ms="?([0-9.]+)"?')
INSTANCE_RE = re.compile(r'instance_id=(\d+)')
NVTX_REQ_RE = re.compile(r'\.req_(?P<rid>[0-9a-f-]{8,48})')
SELECT_PREFILL_RE = re.compile(r'Selected worker: worker_type=prefill, worker_id=(\d+)')

def strip(line: str) -> str:
    return ANSI.sub('', line)

def iso_to_ns(text: str) -> int | None:
    m = ISO_RE.search(text)
    if not m:
        return None
    s = m.group(1)
    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
    return calendar.timegm(dt.utctimetuple()) * 1_000_000_000 + dt.microsecond * 1000

def bracket_to_ns(text: str) -> int | None:
    m = BRACKET_TS_RE.search(text)
    if not m:
        return None
    dt = datetime.strptime(f'{m.group(1)} {m.group(2)}.{m.group(3)}', '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
    return calendar.timegm(dt.utctimetuple()) * 1_000_000_000 + dt.microsecond * 1000

def base_req_id(full: str) -> str:
    return re.sub(r'-[0-9a-f]{8}$', '', full)

def req_idx(rid: str) -> int:
    return int(rid.split('-', 1)[0])

# Benchmark metadata.
bench = {}
time_scale = None
with (LOG / 'benchmark.log').open(errors='replace') as f:
    for raw in f:
        line = strip(raw).strip()
        m = REQ_LINE_RE.match(line)
        if m:
            idx, ts, inp, out, hashes = m.groups()
            idx = int(idx)
            bench.setdefault(idx, {})
            bench[idx].update({
                'idx': idx,
                'trace_ts_ms': int(ts),
                'arrival_scaled_ms': None,
                'input_tokens': int(inp),
                'output_tokens_expected': int(out),
                'hash_count': int(hashes),
            })
            continue
        m = OK_LINE_RE.match(line)
        if m:
            idx, out_a, out_b, ttft, tpot, e2e, chunks = m.groups()
            idx = int(idx)
            bench.setdefault(idx, {})
            bench[idx].update({
                'output_tokens_actual': int(out_a),
                'ttft_ms': float(ttft),
                'tpot_ms': float(tpot),
                'e2e_ms': float(e2e),
                'chunks': int(chunks),
            })
            continue
        if line.startswith('time_scale='):
            try:
                time_scale = float(line.split('=', 1)[1])
            except ValueError:
                pass
if time_scale:
    for row in bench.values():
        if row.get('trace_ts_ms') is not None:
            row['arrival_scaled_ms'] = row['trace_ts_ms'] / time_scale

# Frontend arrival/completion and frontend-selected workers.
front = {}
rid_by_idx = {}
with (LOG / 'frontend.log').open(errors='replace') as f:
    for raw in f:
        line = strip(raw)
        ridm = RID_RE.search(line)
        if not ridm:
            continue
        rid = ridm.group('rid')
        idx = req_idx(rid)
        if idx not in bench:
            continue
        rid_by_idx.setdefault(idx, rid)
        rec = front.setdefault(rid, {})
        ts_ns = iso_to_ns(line)
        if ts_ns is None:
            continue
        if 'dynamo_llm::http::service::metrics' in line and 'request received' in line:
            rec['frontend_received_ns'] = min(rec.get('frontend_received_ns', ts_ns), ts_ns)
        if 'dynamo_llm::http::service::metrics' in line and 'request completed' in line:
            rec['frontend_completed_ns'] = ts_ns
            if (m := TTFT_RE.search(line)):
                rec['frontend_ttft_ms'] = float(m.group(1))
            if (m := WORKER_ID_RE.search(line)):
                rec['prefill_worker_id'] = m.group(1)
            if (m := DECODE_ID_RE.search(line)):
                rec['decode_worker_id'] = m.group(1)
            if (m := INPUT_TOKENS_RE.search(line)):
                rec['frontend_input_tokens'] = int(m.group(1))
        if (m := SELECT_PREFILL_RE.search(line)):
            rec.setdefault('selected_prefill_worker_ids', []).append(m.group(1))

# Prefill logs: request ingress/completion, worker-id mapping, LMCache hit info.
prefill = {}
worker_id_to_file = {}
for path in sorted(LOG.glob('prefill-*.log'), key=worker_num):
    wid = worker_num(path)
    with path.open(errors='replace') as f:
        for raw in f:
            line = strip(raw)
            ridm = RID_RE.search(line)
            if ridm:
                rid = ridm.group('rid')
                idx = req_idx(rid)
                if idx in bench:
                    rid_by_idx.setdefault(idx, rid)
                    rec = prefill.setdefault(rid, {})
                    if (im := INSTANCE_RE.search(line)):
                        worker_id_to_file.setdefault(im.group(1), wid)
                    ts_ns = iso_to_ns(line)
                    if ts_ns is not None and 'component="prefill"' in line and 'endpoint="generate"' in line:
                        if 'request received' in line:
                            rec['prefill_received_ns'] = min(rec.get('prefill_received_ns', ts_ns), ts_ns)
                            rec['prefill_file'] = wid
                        elif 'request completed' in line:
                            rec['prefill_completed_ns'] = max(rec.get('prefill_completed_ns', ts_ns), ts_ns)
                            rec['prefill_file'] = wid
            if 'LMCache INFO:' in line and 'Reqid:' in line and 'LMCache hit tokens:' in line:
                m = LM_RE.search(line)
                ts_ns = bracket_to_ns(line)
                if m and ts_ns is not None:
                    rid = base_req_id(m.group('full'))
                    idx = req_idx(rid)
                    if idx not in bench:
                        continue
                    rid_by_idx.setdefault(idx, rid)
                    rec = prefill.setdefault(rid, {})
                    # Keep earliest scheduler/LMCache line as runtime start.
                    rec['lmcache_start_ns'] = min(rec.get('lmcache_start_ns', ts_ns), ts_ns)
                    rec['prefill_file'] = wid
                    rec['lm_total_tokens'] = int(m.group('total'))
                    rec['lm_engine_computed_tokens'] = int(m.group('computed'))
                    rec['lm_hit_tokens'] = int(m.group('hit'))
                    rec['lm_need_load_tokens'] = int(m.group('need'))

for rid, rec in front.items():
    if 'prefill_worker_id' in rec and rec['prefill_worker_id'] in worker_id_to_file:
        rec['prefill_file'] = worker_id_to_file[rec['prefill_worker_id']]

# Decode logs: request-level P->D NIXL transfer telemetry. For TP workers the
# critical-path transfer time is max(rank xfer+post), not sum across ranks.
p2d = {}
for path in sorted(LOG.glob('decode-*.log')):
    with path.open(errors='replace') as f:
        for raw in f:
            line = strip(raw)
            m = P2D_RE.search(line)
            if not m:
                continue
            rid = base_req_id(m.group('full'))
            idx = req_idx(rid)
            if idx not in bench:
                continue
            rec = p2d.setdefault(rid, {'rank_ms': [], 'xfer_ms': [], 'post_ms': [], 'bytes': 0})
            xfer_ms = float(m.group('xfer'))
            post_ms = float(m.group('post'))
            rec['rank_ms'].append(xfer_ms + post_ms)
            rec['xfer_ms'].append(xfer_ms)
            rec['post_ms'].append(post_ms)
            rec['bytes'] += int(m.group('bytes'))

# Interval helpers.
def merge_intervals(rows):
    # Sorting here, not at the call sites: callers pass concatenations of two
    # individually-sorted lists (compute + loads, when deriving exposed store),
    # and the sweep below only ever compares against merged[-1]. Unsorted input
    # makes it swallow every interval that starts before the running maximum,
    # silently.
    clean = []
    for s, e in rows:
        if e is None or e <= s:
            continue
        clean.append((int(s), int(e)))
    clean.sort()
    merged = []
    for s, e in clean:
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]

def subtract_intervals(a_intervals, b_intervals):
    out = []
    j = 0
    nb = len(b_intervals)
    for a, b in a_intervals:
        cur = a
        while j < nb and b_intervals[j][1] <= a:
            j += 1
        k = j
        while k < nb and b_intervals[k][0] < b:
            c, d = b_intervals[k]
            if c > cur:
                out.append((cur, min(c, b)))
            if d > cur:
                cur = d
            if cur >= b:
                break
            k += 1
        if cur < b:
            out.append((cur, b))
    return out

def intersect_intervals(a_intervals, b_intervals):
    """A and B, computed as A - (A - B).

    Both inputs must be sorted and non-overlapping, which is what
    merge_intervals and subtract_intervals both return.
    """
    return subtract_intervals(a_intervals, subtract_intervals(a_intervals, b_intervals))


class IntervalIndex:
    def __init__(self, intervals):
        self.intervals = intervals
        self.starts = [a for a, _ in intervals]
        self.ends = [b for _, b in intervals]
        self.prefix = [0]
        total = 0
        for a, b in intervals:
            total += b - a
            self.prefix.append(total)
    def overlap(self, a, b):
        if b <= a or not self.intervals:
            return 0
        i = bisect.bisect_right(self.ends, a)
        j = bisect.bisect_left(self.starts, b)
        if i >= j:
            return 0
        total = self.prefix[j] - self.prefix[i]
        if self.starts[i] < a:
            total -= a - self.starts[i]
        if self.ends[j - 1] > b:
            total -= self.ends[j - 1] - b
        return max(0, total)

def nsys_base_ns_from_report_mtime(sqlite_path: Path, rep_path: Path) -> int:
    con = sqlite3.connect(sqlite_path)
    stop = con.execute('select stopTime from ANALYSIS_DETAILS limit 1').fetchone()[0]
    con.close()
    return os.stat(rep_path).st_mtime_ns - int(stop)

def nsys_capture_range_ns(sqlite_path: Path) -> tuple[int, int] | None:
    """The profiler's own collection window, in the sqlite timebase.

    Needed to tell "the GPU was idle here" apart from "nsys was not recording
    here". Inferring the window from the first and last profiled activity
    cannot do that: it would mark the idle gap before the first kernel -- which
    is genuine queue/startup time -- as uncaptured. Returns None when the table
    or columns are absent, and the caller then reports nothing rather than
    guessing.
    """
    con = sqlite3.connect(sqlite_path)
    try:
        row = con.execute('select startTime, stopTime from ANALYSIS_DETAILS limit 1').fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        con.close()
    if not row or row[0] is None or row[1] is None:
        return None
    return int(row[0]), int(row[1])


def table_has_rows(con: sqlite3.Connection, name: str) -> bool:
    """Whether `name` exists in this export and holds at least one row.

    Existence alone is not the question: an exporter that always emits the
    table would then look like graph-mode capture on a node-mode profile.
    """
    if con.execute(
        "select 1 from sqlite_master where type='table' and name=? limit 1",
        (name,),
    ).fetchone() is None:
        return False
    return con.execute(f'select 1 from {name} limit 1').fetchone() is not None


def worker_last_completed_ns(wid: int) -> int | None:
    vals = [rec.get('prefill_completed_ns') for rec in prefill.values()
            if rec.get('prefill_file') == wid and rec.get('prefill_completed_ns')]
    return max(vals) if vals else None

intervals = {}
for wid, paths in NSYS.items():
    db = paths['sqlite']
    rep = paths['rep']
    con = sqlite3.connect(db)
    # Profiles must be captured with `--cuda-graph-trace=node`, which traces
    # every graph node into CUPTI_ACTIVITY_KIND_KERNEL and emits no
    # GRAPH_TRACE table. Under `=graph` the nodes inside a graph are not
    # traced individually and the kernel table alone undercounts compute, so
    # that mode is rejected rather than silently analyzed: an undercounted
    # compute union inflates every exposed-load number, which is the one
    # quantity this script exists to measure.
    if table_has_rows(con, 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE'):
        con.close()
        raise SystemExit(
            f'worker={wid}: {rep} was captured with --cuda-graph-trace=graph '
            f'(CUPTI_ACTIVITY_KIND_GRAPH_TRACE is populated). This script only '
            f'supports --cuda-graph-trace=node, where each graph node lands in '
            f'CUPTI_ACTIVITY_KIND_KERNEL. Under graph mode the kernels inside a '
            f'graph are never traced individually, so the compute union would be '
            f'undercounted and exposed KV load correspondingly overstated. '
            f'Re-profile with --cuda-graph-trace=node.'
        )
    compute_intervals = merge_intervals(
        con.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
    load_rows = con.execute("""
        select n.start,n.end,coalesce(s.value,n.text,'') as nm
        from NVTX_EVENTS n left join StringIds s on n.textId=s.id
        where n.end is not null and (
            coalesce(s.value,n.text,'') like 'LMCache.retrieve_layer.%'
            or coalesce(s.value,n.text,'') like 'LMCache.batched_to_gpu.copy_chunks.%'
            or coalesce(s.value,n.text,'') like 'LMCache.mooncake.batch_get_into.%'
        )
        order by n.start
    """).fetchall()
    store_rows = con.execute("""
        select n.start,n.end from NVTX_EVENTS n left join StringIds s on n.textId=s.id
        where n.end is not null and (
            coalesce(s.value,n.text,'') like 'LMCacheConnectorV1Impl.save_kv_layer%'
            or coalesce(s.value,n.text,'') like 'LMCacheConnectorV1Impl.wait_for_save%'
            or coalesce(s.value,n.text,'') like 'LMCacheEngine.store_layer%'
            or coalesce(s.value,n.text,'') like 'LMCache.mooncake.batch_put%'
            or coalesce(s.value,n.text,'') like 'LMCache.mooncake.put%'
            or coalesce(s.value,n.text,'') like 'LMCache.store_layer.%'
        )
        order by n.start
    """)
    loads = merge_intervals([(s, e) for s, e, _ in load_rows])
    # Per-request load intervals. LMCache tags each load range with the owning
    # request (".req_<uuid>" suffix), so a range can be charged to exactly one
    # request instead of to every request whose TTFT window contains it.
    loads_by_rid_raw = {}
    tagged = untagged = 0
    for s, e, nm in load_rows:
        m = NVTX_REQ_RE.search(nm)
        if m is None:
            untagged += 1
            continue
        tagged += 1
        loads_by_rid_raw.setdefault(base_req_id(m.group('rid')), []).append((s, e))
    loads_by_rid = {
        rid: IntervalIndex(merge_intervals(sorted(rows)))
        for rid, rows in loads_by_rid_raw.items()
    }
    # Earliest NVTX moment per request, used to anchor the sqlite timeline to
    # the wall clock (see base_ns below). The log line this pairs with is the
    # earliest LMCache line for that request, so both sides must be the min.
    lmcache_nvtx_first_ns = {
        rid: min(s for s, _ in rows) for rid, rows in loads_by_rid_raw.items()
    }
    stores = merge_intervals(store_rows)
    max_kernel_end = con.execute('select max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0] or 0
    max_load_end = con.execute("""
        select max(n.end) from NVTX_EVENTS n left join StringIds s on n.textId=s.id
        where n.end is not null and (
            coalesce(s.value,n.text,'') like 'LMCache.retrieve_layer.%'
            or coalesce(s.value,n.text,'') like 'LMCache.batched_to_gpu.copy_chunks.%'
            or coalesce(s.value,n.text,'') like 'LMCache.mooncake.batch_get_into.%'
        )
    """).fetchone()[0] or 0
    max_store_end = con.execute("""
        select max(n.end) from NVTX_EVENTS n left join StringIds s on n.textId=s.id
        where n.end is not null and (
            coalesce(s.value,n.text,'') like 'LMCacheConnectorV1Impl.save_kv_layer%'
            or coalesce(s.value,n.text,'') like 'LMCacheConnectorV1Impl.wait_for_save%'
            or coalesce(s.value,n.text,'') like 'LMCacheEngine.store_layer%'
            or coalesce(s.value,n.text,'') like 'LMCache.mooncake.batch_put%'
            or coalesce(s.value,n.text,'') like 'LMCache.mooncake.put%'
            or coalesce(s.value,n.text,'') like 'LMCache.store_layer.%'
        )
    """).fetchone()[0] or 0
    capture_range = nsys_capture_range_ns(db)
    con.close()
    max_activity_end = max(int(max_kernel_end), int(max_load_end), int(max_store_end))
    last_completed_ns = worker_last_completed_ns(wid)
    # Preferred anchor: pair each request's first LMCache NVTX range with the
    # LMCache log line for the same request id. Both are emitted by this
    # worker process, so their difference is exactly the sqlite->wall offset,
    # with no assumption about which event happened last.
    #
    # The old anchor (last "request completed" minus max nsys activity) is
    # wrong whenever the GPU keeps working after this worker's final request
    # -- background stores, or traffic that outlives the benchmark window. On
    # the small-logs run that misaligned prefill-0 by ~58 s and dropped its
    # in-window kernel time from 20.2 s to 8.6 s, which showed up as a bogus
    # 80% "in-prefill idle" bar while prefill-1 (correctly aligned) showed 1%.
    offsets = []
    for rid, first_ns in lmcache_nvtx_first_ns.items():
        log_ns = prefill.get(rid, {}).get('lmcache_start_ns')
        if log_ns is not None:
            offsets.append(int(log_ns) - int(first_ns))
    if offsets:
        offsets.sort()
        base_ns = offsets[len(offsets) // 2]
        base_source = f'lmcache_nvtx_to_log_median_over_{len(offsets)}_requests'
        spread_ms = (offsets[-1] - offsets[0]) / 1e6
    elif last_completed_ns and max_activity_end:
        base_ns = int(last_completed_ns - max_activity_end)
        base_source = 'last_prefill_completed_minus_max_nsys_activity'
        spread_ms = float('nan')
    else:
        base_ns = nsys_base_ns_from_report_mtime(db, rep)
        base_source = 'report_mtime_minus_analysis_stopTime_fallback'
        spread_ms = float('nan')
    exposed = subtract_intervals(loads, compute_intervals)
    exposed_store = subtract_intervals(stores, merge_intervals(compute_intervals + loads))
    # A request's own load is exposed when no kernel was running at that moment.
    # Subtracting the worker-global compute set (not just this request's own
    # kernels) is deliberate: the GPU is busy for TTFT purposes regardless of
    # which request the running kernel belongs to.
    exposed_loads_by_rid = {
        rid: IntervalIndex(subtract_intervals(idx.intervals, compute_intervals))
        for rid, idx in loads_by_rid.items()
    }
    intervals[wid] = {
        'base_ns': base_ns,
        'capture_range_ns': capture_range,
        'compute': IntervalIndex(compute_intervals),
        'loads': IntervalIndex(loads),
        'stores': IntervalIndex(stores),
        'exposed_loads': IntervalIndex(exposed),
        'exposed_stores': IntervalIndex(exposed_store),
        'loads_by_rid': loads_by_rid,
        'exposed_loads_by_rid': exposed_loads_by_rid,
        'load_ranges_tagged': tagged,
    }
    fallback_base = nsys_base_ns_from_report_mtime(db, rep)
    compute_busy = sum(b - a for a, b in compute_intervals)
    # An empty compute union is not a valid analysis: every load range would
    # count as exposed and the breakdown would read 0% compute. Say so instead
    # of emitting a plot that looks like a finding.
    if not compute_intervals:
        raise SystemExit(
            f'worker={wid}: CUPTI_ACTIVITY_KIND_KERNEL is empty in {db}. Without '
            f'kernels there is no compute union, so all KV load would be reported '
            f'as exposed. Check that the profile actually captured GPU activity.'
        )
    if offsets and spread_ms > 1000.0:
        print(
            f'WARNING: worker={wid} the NVTX-to-log offset varies by '
            f'{spread_ms:.0f} ms across {len(offsets)} requests. The two clocks '
            f'should differ by a constant, so a large spread means the pairing is '
            f'unreliable and every interval bucket may be shifted.'
        )
    print(
        f'worker={wid} base_ns={base_ns} source={base_source} '
        f'offset_spread_ms={spread_ms:.1f} '
        f'fallback_mtime_base_ns={fallback_base} delta_vs_fallback_ms={(base_ns-fallback_base)/1e6:.3f} '
        f'last_completed_ns={last_completed_ns} max_activity_end_ns={max_activity_end} '
        f'legacy_base_delta_ms='
        f'{((last_completed_ns - max_activity_end - base_ns)/1e6) if last_completed_ns else float("nan"):.1f} '
        f'compute={len(compute_intervals)} loads={len(loads)} stores={len(stores)} '
        f'compute_busy_ms={compute_busy/1e6:.1f} '
        f'exposed_load={len(exposed)} exposed_store={len(exposed_store)} '
        f'load_ranges_tagged={tagged} load_ranges_untagged={untagged} '
        f'rids_with_loads={len(loads_by_rid)}'
    )

rows = []
clamped_before_arrival = 0
clamped_after_ttft = 0
for idx in sorted(bench):
    b = bench[idx]
    if 'ttft_ms' not in b:
        continue
    rid = rid_by_idx.get(idx)
    fr = front.get(rid, {}) if rid else {}
    pr = prefill.get(rid, {}) if rid else {}
    front_recv = fr.get('frontend_received_ns')
    ttft_ms = float(b['ttft_ms'])
    if front_recv is None:
        # Cannot place on absolute wall clock; keep row with no nsys coverage.
        rows.append({**b, 'request_id': rid or '', 'coverage': False, 'reason': 'missing_frontend_received'})
        continue
    ttft_end = front_recv + int(ttft_ms * 1_000_000)
    run_start = pr.get('lmcache_start_ns') or pr.get('prefill_received_ns') or front_recv
    # These clamps keep the breakdown well-formed, but if the log families ever
    # land on different clocks the second one fires for every request, silently
    # zeroing queue wait and pushing that time into the exposed-load and other
    # buckets. Count them so the summary below reports it instead of hiding it.
    if run_start < front_recv:
        run_start = front_recv
        clamped_before_arrival += 1
    if run_start > ttft_end:
        run_start = front_recv
        clamped_after_ttft += 1
    queue_ms = max(0.0, (run_start - front_recv) / 1e6)

    wid = fr.get('prefill_file')
    if wid is None:
        wid = pr.get('prefill_file')
    compute_ms = exposed_ms = load_total_ms = load_overlap_ms = 0.0
    store_total_ms = store_exposed_ms = store_overlap_ms = 0.0
    load_attribution = 'none'
    coverage = wid in intervals
    if coverage:
        ns_base = intervals[wid]['base_ns']
        a = int(run_start - ns_base)
        c = int(ttft_end - ns_base)
        if c > a:
            compute_ms = intervals[wid]['compute'].overlap(a, c) / 1e6
            # Charge only this request's own load ranges. Falling back to the
            # worker-global set would re-count every concurrent request's load
            # (measured 10x over-count on an 8-way concurrent run), which makes
            # zero-hit requests appear to spend TTFT loading KV they never read.
            own_loads = intervals[wid]['loads_by_rid'].get(rid)
            own_exposed = intervals[wid]['exposed_loads_by_rid'].get(rid)
            if own_loads is not None and own_exposed is not None:
                load_total_ms = own_loads.overlap(a, c) / 1e6
                exposed_ms = own_exposed.overlap(a, c) / 1e6
                load_attribution = 'per_request'
            elif intervals[wid]['load_ranges_tagged'] > 0:
                # Ranges are tagged but this request owns none: it loaded
                # nothing. Zero is the correct answer, not the worker-global sum.
                load_total_ms = 0.0
                exposed_ms = 0.0
                load_attribution = 'per_request_no_loads'
            else:
                # Profile predates req-id tagging; fall back to the old
                # worker-global attribution and mark the row so the
                # over-counting stays visible downstream.
                load_total_ms = intervals[wid]['loads'].overlap(a, c) / 1e6
                exposed_ms = intervals[wid]['exposed_loads'].overlap(a, c) / 1e6
                load_attribution = 'worker_global_untagged'
            load_overlap_ms = max(0.0, load_total_ms - exposed_ms)
            store_total_ms = intervals[wid]['stores'].overlap(a, c) / 1e6
            store_exposed_ms = intervals[wid]['exposed_stores'].overlap(a, c) / 1e6
            store_overlap_ms = max(0.0, store_total_ms - store_exposed_ms)
    runtime_ms = max(0.0, ttft_ms - queue_ms)
    if compute_ms > runtime_ms:
        compute_ms = runtime_ms
    if exposed_ms > max(0.0, runtime_ms - compute_ms):
        exposed_ms = max(0.0, runtime_ms - compute_ms)
    if store_exposed_ms > max(0.0, runtime_ms - compute_ms - exposed_ms):
        store_exposed_ms = max(0.0, runtime_ms - compute_ms - exposed_ms)
    p2d_rec = p2d.get(rid or '', {})
    # Critical path across TP ranks is the slowest rank, not the sum.
    p2d_kv_transfer_raw_ms = max(p2d_rec.get('rank_ms', []) or [0.0])
    # The per-request column is clamped to whatever TTFT has left, so the
    # stacked per-request bars cannot exceed 100%. The aggregate view needs the
    # unclamped value: it places the transfer as a real interval on the wall
    # clock and intersects it with the served span itself, and a value already
    # trimmed by this request's leftover budget would be trimmed twice.
    p2d_kv_transfer_ms = min(
        p2d_kv_transfer_raw_ms,
        max(0.0, ttft_ms - queue_ms - compute_ms - exposed_ms - store_exposed_ms),
    )
    other_ms = max(0.0, ttft_ms - queue_ms - compute_ms - exposed_ms - store_exposed_ms - p2d_kv_transfer_ms)

    total = pr.get('lm_total_tokens') or b.get('input_tokens') or 0
    hit = pr.get('lm_hit_tokens') or 0
    need = pr.get('lm_need_load_tokens') or 0
    hit_rate = (hit / total) if total else 0.0
    # Where each prompt token's KV came from. LMCache reports `hit` as a match
    # counted from token 0, so it *includes* the part vLLM already had resident
    # in HBM (`computed`); the two are nested, not disjoint. Subtracting gives
    # the disjoint split, which is what makes a high hit rate and a small load
    # coexist: most of `hit` is often already-resident HBM, needing no transfer.
    #
    # The DRAM part is derived as hit-minus-computed rather than read from
    # `need to load` because LMCache decrements `need` by one token in the
    # full-prompt-hit case (it must recompute the last token), which would stop
    # the three buckets from summing to total. `lm_need_load_tokens` is kept as
    # its own column for anyone who wants the literal transfer size.
    computed = pr.get('lm_engine_computed_tokens', 0) or 0
    hbm_tokens = min(computed, total)
    dram_tokens = max(0, min(hit, total) - hbm_tokens)
    miss_tokens = max(0, total - hbm_tokens - dram_tokens)
    rows.append({
        'idx': idx,
        'request_id': rid or '',
        'trace_ts_ms': b.get('trace_ts_ms'),
        'arrival_scaled_ms': b.get('arrival_scaled_ms'),
        'input_tokens': b.get('input_tokens'),
        'output_tokens': b.get('output_tokens_actual'),
        'hash_count': b.get('hash_count'),
        'ttft_ms': ttft_ms,
        'tpot_ms': b.get('tpot_ms'),
        'e2e_ms': b.get('e2e_ms'),
        'frontend_ttft_ms': fr.get('frontend_ttft_ms'),
        'prefill_worker_id': fr.get('prefill_worker_id', ''),
        'prefill_file': wid if wid is not None else '',
        'decode_worker_id': fr.get('decode_worker_id', ''),
        'frontend_received_ns': front_recv,
        'run_start_ns': run_start,
        'prefill_received_ns': pr.get('prefill_received_ns'),
        'prefill_completed_ns': pr.get('prefill_completed_ns'),
        'lm_total_tokens': total,
        'lm_engine_computed_tokens': pr.get('lm_engine_computed_tokens', 0),
        'lm_hit_tokens': hit,
        'lm_need_load_tokens': need,
        'kv_hit_rate': hit_rate,
        'hbm_hit_tokens': hbm_tokens,
        'dram_hit_tokens': dram_tokens,
        'no_hit_tokens': miss_tokens,
        'hbm_hit_frac': hbm_tokens / total if total else 0.0,
        'dram_hit_frac': dram_tokens / total if total else 0.0,
        'no_hit_frac': miss_tokens / total if total else 0.0,
        'queue_wait_ms': queue_ms,
        'runtime_ms': runtime_ms,
        'gpu_compute_ms': compute_ms,
        'kv_load_exposed_ms': exposed_ms,
        'kv_load_total_ms': load_total_ms,
        'kv_load_overlap_compute_ms': load_overlap_ms,
        'kv_load_attribution': load_attribution,
        'kv_store_total_ms': store_total_ms,
        'kv_store_overlap_compute_or_load_ms': store_overlap_ms,
        'kv_store_exposed_ms': store_exposed_ms,
        'p2d_kv_transfer_ms': p2d_kv_transfer_ms,
        'p2d_kv_transfer_raw_ms': p2d_kv_transfer_raw_ms,
        'p2d_kv_transfer_bytes': p2d_rec.get('bytes', 0),
        'other_ms': other_ms,
        'queue_pct': 100 * queue_ms / ttft_ms if ttft_ms else 0,
        'gpu_compute_pct': 100 * compute_ms / ttft_ms if ttft_ms else 0,
        'kv_load_exposed_pct': 100 * exposed_ms / ttft_ms if ttft_ms else 0,
        'kv_store_exposed_pct': 100 * store_exposed_ms / ttft_ms if ttft_ms else 0,
        'p2d_kv_transfer_pct': 100 * p2d_kv_transfer_ms / ttft_ms if ttft_ms else 0,
        'other_pct': 100 * other_ms / ttft_ms if ttft_ms else 0,
        'coverage': bool(coverage),
        'reason': '' if coverage else 'missing_prefill_worker_or_nsys',
    })

if not rows:
    raise SystemExit(
        f'No requests parsed from {LOG / "benchmark.log"}. This script reads the '
        f'"[req] idx=... ts=...ms in=... out=... hashes=..." lines emitted by '
        f'mooncake_trace_replay.py / benchmark_serving_multi_turn.py; a run logged '
        f'by a different harness will not match. Check the file exists and '
        f'contains those lines.'
    )

df = pd.DataFrame(rows).sort_values('idx')
# Keep the raw continuous overlap metrics in the CSV so any threshold-based
# classification can be done downstream.
df['kv_load_exposed_frac_of_load'] = df.apply(
    lambda r: (r.kv_load_exposed_ms / r.kv_load_total_ms)
    if r.kv_load_total_ms and r.kv_load_total_ms > 0 else 0.0,
    axis=1,
)
df['kv_load_overlap_ratio'] = (1.0 - df['kv_load_exposed_frac_of_load']).clip(lower=0.0, upper=1.0)

# Sanity checks on the attribution. Both of these caught a real bug where load
# ranges were charged worker-globally: attributed load summed to ~10x the
# wall-clock span, and requests with zero LMCache hit tokens still carried
# seconds of "KV load".
print(f"load attribution modes: {df.kv_load_attribution.value_counts().to_dict()}")
if clamped_after_ttft > len(df) * 0.5:
    print(
        f'WARNING: run_start fell after TTFT end for {clamped_after_ttft}/{len(df)} '
        f'requests, so queue wait was forced to 0 for them. That means the LMCache '
        f'bracket timestamps and the dynamo ISO timestamps are on different clocks. '
        f'dynamo_script.sh exports TZ=UTC0 to keep them aligned; logs captured '
        f'before that, or with TZ overridden, will show this.'
    )
elif clamped_after_ttft:
    print(f'run_start clamped after ttft_end for {clamped_after_ttft}/{len(df)} requests')
for wid, grp in df[df.coverage].groupby('prefill_file'):
    if wid not in intervals:
        continue
    union_ms = sum(b - a for a, b in intervals[wid]['loads'].intervals) / 1e6
    attributed_ms = float(grp.kv_load_total_ms.sum())
    ratio = attributed_ms / union_ms if union_ms else 0.0
    flag = '' if ratio <= 1.05 else '  <-- OVER-COUNTED, attribution is not per-request'
    print(
        f'worker={wid} attributed_load={attributed_ms:.0f}ms '
        f'nvtx_union_load={union_ms:.0f}ms ratio={ratio:.2f}{flag}'
    )
zero_hit = df[(df.lm_hit_tokens == 0) & (df.kv_load_total_ms > 0)]
if len(zero_hit) > 0:
    print(
        f'WARNING: {len(zero_hit)} requests have lm_hit_tokens=0 but nonzero '
        f'kv_load_total_ms (mean {zero_hit.kv_load_total_ms.mean():.0f}ms). '
        f'A request that hit nothing has nothing to load; this indicates '
        f'cross-request contamination in load attribution.'
    )
csv_path = OUT / 'ttft_breakdown.csv'
df.to_csv(csv_path, index=False)
print(f'wrote {csv_path} rows={len(df)}')


# Plots.
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 160,
})
colors = {
    'queue_pct': '#9CA3AF',
    'gpu_compute_pct': '#2563EB',
    'kv_load_exposed_pct': '#DC2626',
    'kv_store_exposed_pct': '#7C3AED',
    'p2d_kv_transfer_pct': '#06B6D4',
    'other_pct': '#F59E0B',
    'in_prefill_idle_pct': '#78350F',
}
labels = [
    ('queue_pct', 'Queue wait'),
    ('gpu_compute_pct', 'GPU compute'),
    ('kv_load_exposed_pct', 'Exposed KV load'),
    ('kv_store_exposed_pct', 'Exposed KV store'),
    ('p2d_kv_transfer_pct', 'P2D KV transfer'),
    ('other_pct', 'Other'),
]

def make_plots(sort_name, sort_cols, ascending=True, x_desc='rank',
               tick_col=None, tick_fmt=None, tick_desc=None):
    """One stacked bar per request, paged, sorted by `sort_cols`.

    `tick_col`/`tick_fmt`/`tick_desc` say what the x tick labels read: the
    column to sample, how to render one value, and the axis caption. They are
    passed per call rather than switched on `sort_name` inside, so adding a
    sort cannot silently inherit another sort's labels.
    """
    sdf = df.sort_values(sort_cols, ascending=ascending, kind='mergesort').reset_index(drop=True)
    per_page = 300
    pdf_path = OUT / f'ttft_breakdown_by_{sort_name}.pdf'
    with PdfPages(pdf_path) as pdf:
        pages = math.ceil(len(sdf) / per_page)
        for page in range(pages):
            sub = sdf.iloc[page*per_page:(page+1)*per_page].copy()
            x = list(range(len(sub)))
            fig, ax = plt.subplots(figsize=(16, 5.6))
            bottom = [0.0] * len(sub)
            for col, lab in labels:
                vals = sub[col].fillna(0).clip(lower=0).tolist()
                ax.bar(x, vals, bottom=bottom, width=0.92, color=colors[col], label=lab, linewidth=0)
                bottom = [a+b for a,b in zip(bottom, vals)]
            ax.set_ylim(0, 100)
            ax.set_ylabel('Share of TTFT (%)')
            ax.set_xlabel(f'Requests sorted by {x_desc} (page {page+1}/{pages})')
            start_rank = page*per_page + 1
            end_rank = page*per_page + len(sub)
            title = (
                f'{RUN} TTFT breakdown by {sort_name}: ranks {start_rank}-{end_rank} '
                f'| mean TTFT={sub.ttft_ms.mean():.1f} ms, mean queue={sub.queue_pct.mean():.1f}%, '
                f'mean exposed KV={sub.kv_load_exposed_pct.mean():.1f}%'
            )
            ax.set_title(title)
            if len(sub) > 0:
                tick_step = 50
                ticks = list(range(0, len(sub), tick_step))
                if ticks[-1] != len(sub)-1:
                    ticks.append(len(sub)-1)
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [tick_fmt(sub.iloc[t][tick_col]) for t in ticks],
                    rotation=45, ha='right')
                ax.set_xlabel(tick_desc)
            ax.grid(axis='y', alpha=0.25, linewidth=0.6)
            ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False)
            fig.tight_layout(rect=(0, 0.03, 1, 1))
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'wrote {pdf_path} pages={pages}')



# (ms column, pct column, legend label, key into `colors`). The pct name is
# spelled out rather than derived from the ms name so a bucket whose two names
# differ (queue_stall_ms / queue_pct) cannot silently miss its color or column.
AGG_BUCKETS = [
    ('queue_stall_ms', 'queue_pct', 'Queue wait', 'queue_pct'),
    ('gpu_compute_ms', 'gpu_compute_pct', 'GPU compute (prefill)', 'gpu_compute_pct'),
    ('kv_load_exposed_ms', 'kv_load_exposed_pct', 'Exposed KV load', 'kv_load_exposed_pct'),
    ('kv_store_exposed_ms', 'kv_store_exposed_pct', 'Exposed KV store', 'kv_store_exposed_pct'),
    ('p2d_kv_transfer_ms', 'p2d_kv_transfer_pct', 'P2D KV transfer', 'p2d_kv_transfer_pct'),
    ('post_prefill_other_ms', 'post_prefill_other_pct',
     'Post-prefill (handoff + 1st decode)', 'other_pct'),
    ('in_prefill_idle_ms', 'in_prefill_idle_pct', 'In-prefill idle', 'in_prefill_idle_pct'),
]


def aggregate_ttft_share():
    """Totals on each worker's own wall clock, not per-request means.

    The per-request columns cannot be summed: with N requests in flight the
    same kernel lands inside N TTFT windows and is charged to all of them, so
    summed compute grows with concurrency while load -- which *is* attributed
    per request -- does not. That asymmetry is what pushes exposed_load/ttft
    far below the cost model's T_load/T_comp.

    Here the denominator is the union of the request windows [arrival, first
    token) on one worker, so concurrent requests collapse into one span and
    idle gaps between requests are excluded. Every bucket is a union measured
    on that same timeline.

    Bucket order is also the priority order when two things overlap, and it is
    chosen so each bucket answers "what was the prefill GPU blocked on":
    prefill compute wins over everything (the GPU was busy), then exposed load,
    then exposed store, and only then P2D. P2D runs on the decode GPU and can
    overlap prefill work on another request, so charging it ahead of prefill
    compute would hide compute that really was running.

    P2D placement is a reconstruction, not a measurement. The decode log
    (`nixl/worker.py`) reports only a duration per request, no start or end, so
    the transfer is placed as the interval ending at that request's first token
    -- the last thing that must finish before decode can emit -- and clipped to
    the served span. `p2d_reconstructed` in the CSV marks this.
    """
    covered = df[df.coverage & df.prefill_file.isin(list(intervals))]
    if len(covered) == 0:
        print('no requests with nsys coverage; skipping aggregate TTFT share')
        return
    rows_ = []
    for wid, grp in covered.groupby('prefill_file'):
        iv = intervals[wid]
        base = iv['base_ns']
        windows, queue_spans, p2d_spans, admitted_spans = [], [], [], []
        missing_completed = 0
        for r in grp.itertuples():
            recv = int(r.frontend_received_ns) - base
            start = int(r.run_start_ns) - base
            end = recv + int(round(float(r.ttft_ms) * 1e6))
            if end <= recv:
                continue
            windows.append((recv, end))
            if start > recv:
                queue_spans.append((recv, min(start, end)))
            # [admitted, prefill completed): the request is actually being
            # served, so idle time here is a real prefill-side stall. The start
            # must be run_start, not arrival -- using arrival would fold the
            # queue wait into "in prefill" and, unioned over dozens of
            # requests, would cover the whole span and swallow every other
            # bucket. Missing the completion log means we cannot tell where
            # serving ended, so assume through the first token.
            done = getattr(r, 'prefill_completed_ns', None)
            adm_start = min(max(start, recv), end)
            if done is None or pd.isna(done):
                missing_completed += 1
                admitted_spans.append((adm_start, end))
            else:
                adm_end = min(max(int(done) - base, adm_start), end)
                if adm_end > adm_start:
                    admitted_spans.append((adm_start, adm_end))
            # Anchored at the end of the TTFT window and extended backwards:
            # the P->D KV transfer is the last step before the first token, so
            # its end is the one edge that is known rather than guessed.
            p2d_ns = int(round(float(getattr(r, 'p2d_kv_transfer_raw_ms', 0.0) or 0.0) * 1e6))
            if p2d_ns > 0:
                p2d_spans.append((max(recv, end - p2d_ns), end))
        served = merge_intervals(windows)
        if not served:
            continue
        served_ns = sum(b - a for a, b in served)
        compute_ns = sum(iv['compute'].overlap(a, b) for a, b in served)
        load_total_ns = sum(iv['loads'].overlap(a, b) for a, b in served)
        store_total_ns = sum(iv['stores'].overlap(a, b) for a, b in served)
        exp_load_ns = sum(iv['exposed_loads'].overlap(a, b) for a, b in served)
        exp_store_ns = sum(iv['exposed_stores'].overlap(a, b) for a, b in served)
        # Everything nsys accounts for on the prefill GPU. Its complement
        # inside the span is time the prefill GPU was idle, which is where the
        # P2D transfer and the first decode token live.
        accounted = merge_intervals(
            list(iv['compute'].intervals) + list(iv['loads'].intervals)
            + list(iv['stores'].intervals))
        idle_region = subtract_intervals(served, accounted)
        idle_ns = sum(b - a for a, b in idle_region)
        idle_idx = IntervalIndex(idle_region)
        queue_merged = merge_intervals(queue_spans)
        queue_ns = sum(idle_idx.overlap(a, b) for a, b in queue_merged)
        # `other` is only meaningful once it is split by *where* in a request's
        # life the prefill GPU went idle, so each piece names a different
        # suspect. Two landmarks bound the phases: run_start (the request was
        # admitted) and prefill_completed (the prefill worker logged "request
        # completed"). Idle after the latter cannot be a prefill stall at all --
        # it is the P->D handoff and the first decode step.
        #
        # in-prefill is computed as an explicit intersection rather than as
        # "idle minus the others": a leftover subtraction would absorb every
        # future accounting mistake into this one bucket and still look
        # self-consistent, which is exactly how a span-wide `pre_done` union
        # once made in-prefill idle appear to be the whole run.
        admitted_merged = merge_intervals(admitted_spans)
        in_prefill_region = subtract_intervals(
            intersect_intervals(idle_region, admitted_merged), queue_merged)
        in_prefill_idle_ns = sum(b - a for a, b in in_prefill_region)
        # Idle while no request was admitted-and-unfinished, and not queueing:
        # the handoff window plus scheduler gaps between requests.
        post_prefill_region = subtract_intervals(
            subtract_intervals(idle_region, admitted_merged), queue_merged)
        post_prefill_ns = sum(b - a for a, b in post_prefill_region)
        # P2D is reconstructed backwards from the first token, so it lands in
        # the post-prefill region by construction. Charge it there.
        p2d_region = subtract_intervals(
            intersect_intervals(merge_intervals(p2d_spans), post_prefill_region),
            queue_merged)
        p2d_ns_total = sum(b - a for a, b in p2d_region)
        # compute + exposed_load + exposed_store + idle must reconstruct the
        # span; a mismatch means the three activity sets stopped being disjoint
        # (e.g. an exposed set rebuilt without subtracting the right base).
        drift_ns = served_ns - (compute_ns + exp_load_ns + exp_store_ns + idle_ns)
        if abs(drift_ns) > 1_000_000:
            raise SystemExit(
                f'worker={wid}: aggregate buckets miss the served span by '
                f'{drift_ns/1e6:.3f} ms. compute/exposed-load/exposed-store are '
                f'supposed to be pairwise disjoint unions covering exactly '
                f'served - idle; they are not.'
            )
        # Now that all three are built by explicit intersection, the split must
        # add up on its own. Checking it (instead of deriving one term by
        # subtraction) is what turns an attribution mistake into a crash rather
        # than a plausible-looking bar.
        split_drift_ns = idle_ns - (queue_ns + in_prefill_idle_ns + post_prefill_ns)
        if abs(split_drift_ns) > 1_000_000:
            raise SystemExit(
                f'worker={wid}: the idle split does not add up. idle='
                f'{idle_ns/1e6:.1f} ms but queue {queue_ns/1e6:.1f} + in-prefill '
                f'{in_prefill_idle_ns/1e6:.1f} + post-prefill '
                f'{post_prefill_ns/1e6:.1f} = '
                f'{(queue_ns+in_prefill_idle_ns+post_prefill_ns)/1e6:.1f} '
                f'(drift {split_drift_ns/1e6:.1f} ms). The three regions are '
                f'supposed to partition idle exactly.'
            )
        if p2d_ns_total > post_prefill_ns + 1_000_000:
            raise SystemExit(
                f'worker={wid}: P2D ({p2d_ns_total/1e6:.1f} ms) exceeds the '
                f'post-prefill region ({post_prefill_ns/1e6:.1f} ms) it is carved '
                f'from.'
            )
        # How much of the span the profile could even see, taken from nsys's own
        # collection window. If base_ns is off, or nsys attached late, the span
        # runs past the capture and that time silently piles up in `other`
        # instead of looking like an error. Unknown window -> report 0 rather
        # than inventing one from activity bounds, which would count a genuine
        # pre-first-kernel idle gap as uncaptured.
        cap_range = iv.get('capture_range_ns')
        if cap_range is None:
            outside_ns = 0
        else:
            cap = IntervalIndex([cap_range])
            outside_ns = served_ns - sum(cap.overlap(a, b) for a, b in served)
        rows_.append({
            'worker': wid,
            'requests': len(grp),
            'served_span_ms': served_ns / 1e6,
            'queue_stall_ms': queue_ns / 1e6,
            'gpu_compute_ms': compute_ns / 1e6,
            'kv_load_exposed_ms': exp_load_ns / 1e6,
            'kv_store_exposed_ms': exp_store_ns / 1e6,
            'p2d_kv_transfer_ms': p2d_ns_total / 1e6,
            'post_prefill_other_ms': (post_prefill_ns - p2d_ns_total) / 1e6,
            'in_prefill_idle_ms': in_prefill_idle_ns / 1e6,
            'post_prefill_total_ms': post_prefill_ns / 1e6,
            'requests_missing_completed': missing_completed,
            # Span diagnostics. served_span is a union, so it can only be
            # smaller than the sum of the windows; how much smaller says how
            # much the requests actually overlapped. mean_concurrency near 1
            # means they were served nearly one at a time, which makes a span
            # proportional to the request count expected rather than a bug.
            'window_sum_ms': sum(b - a for a, b in windows) / 1e6,
            'wall_first_to_last_ms': (max(b for _, b in served)
                                      - min(a for a, _ in served)) / 1e6,
            'span_segments': len(served),
            'mean_concurrency': (sum(b - a for a, b in windows) / served_ns
                                 if served_ns else 0.0),
            'admitted_busy_ms': sum(b - a for a, b in admitted_merged) / 1e6,
            'kv_load_total_ms': load_total_ns / 1e6,
            'kv_store_total_ms': store_total_ns / 1e6,
            'p2d_reported_ms': float(grp.p2d_kv_transfer_raw_ms.sum()),
            'requests_with_p2d': int((grp.p2d_kv_transfer_raw_ms > 0).sum()),
            'prefill_idle_ms': idle_ns / 1e6,
            'ttft_sum_ms': float(grp.ttft_ms.sum()),
            'outside_nsys_ms': outside_ns / 1e6,
        })
    if not rows_:
        print('no served span could be built; skipping aggregate TTFT share')
        return
    total = {'worker': 'TOTAL'}
    for k in rows_[0]:
        if k != 'worker':
            total[k] = sum(r[k] for r in rows_)
    # Ratios must be recomputed from the summed totals, not summed themselves:
    # adding two workers' mean concurrency would report ~2 for a run where each
    # served requests strictly one at a time.
    total['mean_concurrency'] = (total['window_sum_ms'] / total['served_span_ms']
                                 if total['served_span_ms'] else 0.0)
    agg = rows_ + [total]
    for r in agg:
        den = r['served_span_ms'] or 1.0
        for ms_col, pct_col, _, _ in AGG_BUCKETS:
            r[pct_col] = 100.0 * r[ms_col] / den
        r['kv_load_total_pct'] = 100.0 * r['kv_load_total_ms'] / den
        r['kv_store_total_pct'] = 100.0 * r['kv_store_total_ms'] / den
        # The two numbers the cost model talks about. Both are wall-clock
        # totals on one timeline, so this is comparable to c*prefix / T_comp
        # in a way that exposed_load/ttft is not.
        comp = r['gpu_compute_ms'] or 1.0
        r['kv_load_total_over_compute'] = r['kv_load_total_ms'] / comp
        r['kv_load_exposed_over_compute'] = r['kv_load_exposed_ms'] / comp
        r['kv_load_overlap_ratio'] = (
            1.0 - r['kv_load_exposed_ms'] / r['kv_load_total_ms']
            if r['kv_load_total_ms'] > 0 else 0.0)
        r['ttft_sum_over_span'] = r['ttft_sum_ms'] / den
        # How much of the reported P2D duration actually landed in the span.
        # A low value means the reconstruction was clipped: the transfers were
        # placed inside prefill-busy or queue time, so the bucket understates
        # them. It is the honest way to read the P2D bar.
        r['p2d_placed_frac'] = (
            r['p2d_kv_transfer_ms'] / r['p2d_reported_ms']
            if r['p2d_reported_ms'] > 0 else 0.0)
        # Set after the TOTAL row is summed, so it stays a flag rather than
        # being added up into a meaningless count.
        r['p2d_reconstructed'] = True

    print()
    print('=== aggregate TTFT share (wall clock, union of [arrival, first token) windows) ===')
    print(f'requests: {len(covered)}/{len(df)} with nsys coverage '
          f'({100*len(covered)/len(df):.1f}%); uncovered requests contribute nothing')
    hdr = (f'{"worker":>7} {"reqs":>5} {"span_ms":>11} {"queue":>8} {"compute":>8} '
           f'{"expLoad":>8} {"expStore":>9} {"p2d":>7} {"postPF":>8} {"inPFidle":>9} '
           f'{"load_tot":>9} {"store_tot":>10} {"outside":>9}')
    print(hdr)
    for r in agg:
        print(f'{str(r["worker"]):>7} {r["requests"]:>5.0f} {r["served_span_ms"]:>11.1f} '
              f'{r["queue_pct"]:>7.1f}% {r["gpu_compute_pct"]:>7.1f}% '
              f'{r["kv_load_exposed_pct"]:>7.1f}% {r["kv_store_exposed_pct"]:>8.1f}% '
              f'{r["p2d_kv_transfer_pct"]:>6.1f}% {r["post_prefill_other_pct"]:>7.1f}% '
              f'{r["in_prefill_idle_pct"]:>8.1f}% '
              f'{r["kv_load_total_pct"]:>8.1f}% '
              f'{r["kv_store_total_pct"]:>9.1f}% {r["outside_nsys_ms"]:>8.1f}ms')
    t = agg[-1]
    print(
        f'TOTAL ms: span={t["served_span_ms"]:.1f} queue={t["queue_stall_ms"]:.1f} '
        f'compute={t["gpu_compute_ms"]:.1f} exposed_load={t["kv_load_exposed_ms"]:.1f} '
        f'exposed_store={t["kv_store_exposed_ms"]:.1f} '
        f'p2d={t["p2d_kv_transfer_ms"]:.1f} post_prefill_other='
        f'{t["post_prefill_other_ms"]:.1f} in_prefill_idle={t["in_prefill_idle_ms"]:.1f} '
        f'| load_total={t["kv_load_total_ms"]:.1f} store_total={t["kv_store_total_ms"]:.1f}'
    )
    # What used to be one opaque `other` bar, split by where in the request's
    # life the prefill GPU was idle. Post-prefill time is not a prefill stall
    # at all: prefill had already reported completion, so it is the P->D
    # handoff, decode-side scheduling and the first decode step.
    print(
        f'idle split: prefill_idle={t["prefill_idle_ms"]:.1f} ms = queue '
        f'{t["queue_stall_ms"]:.1f} + post-prefill {t["post_prefill_total_ms"]:.1f} '
        f'(of which P2D {t["p2d_kv_transfer_ms"]:.1f}) + in-prefill '
        f'{t["in_prefill_idle_ms"]:.1f}'
    )
    # Span sanity, per worker. A span that scales with request count is the
    # signature of requests being served one at a time; a span far below the
    # window sum means they genuinely overlapped. Printing both makes the
    # difference checkable instead of a matter of trust.
    print('span check (served_span is a UNION of [arrival, first token) windows):')
    for r in agg:
        label = f'prefill-{r["worker"]}' if r['worker'] != 'TOTAL' else 'TOTAL'
        per_req = r['served_span_ms'] / r['requests'] if r['requests'] else 0.0
        print(
            f'  {label:>10}: n={r["requests"]:.0f} span={r["served_span_ms"]:.1f}ms '
            f'sum(window)={r["window_sum_ms"]:.1f}ms '
            f'mean_concurrency={r["mean_concurrency"]:.2f} '
            f'segments={r["span_segments"]:.0f} '
            f'wall_first_to_last={r["wall_first_to_last_ms"]:.1f}ms '
            f'span/req={per_req:.1f}ms'
        )
    if t['mean_concurrency'] < 1.2:
        print(
            f'NOTE: mean concurrency is {t["mean_concurrency"]:.2f}, i.e. the TTFT '
            f'windows barely overlap, so the span grows roughly linearly with the '
            f'request count and a worker with 2x the requests will show ~2x the '
            f'span. That is the union behaving correctly, not double counting -- '
            f'the per-request double-counting factor below is the number to watch.'
        )
    # A span much larger than the wall clock it covers would be impossible;
    # a span much smaller than the window sum means heavy overlap.
    for r in rows_:
        if r['served_span_ms'] > r['wall_first_to_last_ms'] + 1.0:
            raise SystemExit(
                f'worker={r["worker"]}: served span ({r["served_span_ms"]:.1f} ms) '
                f'exceeds the wall time from first arrival to last first-token '
                f'({r["wall_first_to_last_ms"]:.1f} ms), which is impossible for a '
                f'union of intervals inside that range.'
            )
    print(
        f'TOTAL vs cost model: load_total/compute={t["kv_load_total_over_compute"]:.3f} '
        f'exposed_load/compute={t["kv_load_exposed_over_compute"]:.3f} '
        f'load_overlap_ratio={t["kv_load_overlap_ratio"]:.3f}'
    )
    # P2D has no timestamps in the decode log, only a duration, so say plainly
    # how much of it the reconstruction could place inside the span.
    print(
        f'P2D (reconstructed, decode log has duration only): reported='
        f'{t["p2d_reported_ms"]:.1f} ms over {t["requests_with_p2d"]:.0f} requests, '
        f'placed={t["p2d_kv_transfer_ms"]:.1f} ms ({100*t["p2d_placed_frac"]:.0f}% of '
        f'reported) in prefill-idle time; prefill_idle={t["prefill_idle_ms"]:.1f} ms'
    )
    if t['p2d_reported_ms'] > 0 and t['p2d_placed_frac'] < 0.5:
        print(
            f'NOTE: only {100*t["p2d_placed_frac"]:.0f}% of the reported P2D time fell in '
            f'prefill-idle time, so the P2D bar understates it. The rest overlapped '
            f'prefill compute or queue wait on this worker, which is expected when '
            f'the prefill GPU stays busy on other requests during the transfer.'
        )
    # The reason this view exists, stated as a number: how many times over the
    # per-request windows cover the same wall clock.
    print(
        f'per-request double counting: sum(ttft)={t["ttft_sum_ms"]:.1f} ms is '
        f'{t["ttft_sum_over_span"]:.2f}x the {t["served_span_ms"]:.1f} ms wall-clock '
        f'span, so summing the per-request columns overstates concurrent work by '
        f'about that factor'
    )
    if t['outside_nsys_ms'] > 0.02 * t['served_span_ms']:
        print(
            f'WARNING: {t["outside_nsys_ms"]:.1f} ms of the served span '
            f'({100*t["outside_nsys_ms"]/t["served_span_ms"]:.1f}%) lies outside the '
            f'nsys capture range. That time can only land in `other`; check base_ns '
            f'and whether the profile covers the whole run.'
        )

    csv_path = OUT / 'ttft_share_aggregate.csv'
    cols = ['worker', 'requests', 'served_span_ms', 'queue_stall_ms', 'gpu_compute_ms',
            'kv_load_exposed_ms', 'kv_store_exposed_ms', 'p2d_kv_transfer_ms',
            'post_prefill_other_ms', 'in_prefill_idle_ms',
            'kv_load_total_ms', 'kv_store_total_ms',
            'queue_pct', 'gpu_compute_pct', 'kv_load_exposed_pct',
            'kv_store_exposed_pct', 'p2d_kv_transfer_pct', 'post_prefill_other_pct',
            'in_prefill_idle_pct',
            'kv_load_total_pct', 'kv_store_total_pct', 'kv_load_total_over_compute',
            'kv_load_exposed_over_compute', 'kv_load_overlap_ratio',
            'p2d_reported_ms', 'p2d_placed_frac', 'requests_with_p2d',
            'p2d_reconstructed', 'prefill_idle_ms', 'post_prefill_total_ms',
            'requests_missing_completed', 'window_sum_ms', 'wall_first_to_last_ms',
            'span_segments', 'mean_concurrency', 'admitted_busy_ms',
            'ttft_sum_ms', 'ttft_sum_over_span', 'outside_nsys_ms']
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in agg:
            w.writerow(r)
    print(f'wrote {csv_path} rows={len(agg)}')

    pdf_path = OUT / 'ttft_share_aggregate.pdf'
    fig, ax = plt.subplots(figsize=(max(5.0, 1.7 * len(agg) + 2.5), 5.5))
    x = list(range(len(agg)))
    bottom = [0.0] * len(agg)
    for ms_col, pct_col, lab, ckey in AGG_BUCKETS:
        vals = [r[pct_col] for r in agg]
        ax.bar(x, vals, bottom=bottom, width=0.6, color=colors[ckey], label=lab,
               linewidth=0)
        bottom = [a + b for a, b in zip(bottom, vals)]
    for i, r in enumerate(agg):
        ax.text(i, 101, f'{r["served_span_ms"]/1e3:.1f}s\nn={r["requests"]:.0f}',
                ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share of wall-clock served span (%)')
    ax.set_xlabel('Prefill worker (labels above bars are span and request count)')
    p2d_note = (f'; P2D reconstructed, {100*t["p2d_placed_frac"]:.0f}% of reported placed'
                if t['p2d_reported_ms'] > 0 else '; no P2D reported')
    ax.set_title(
        f'{RUN} aggregate TTFT share on wall clock\n'
        f'load_total/compute={t["kv_load_total_over_compute"]:.2f}, '
        f'exposed_load/compute={t["kv_load_exposed_over_compute"]:.2f}, '
        f'overlap={100*t["kv_load_overlap_ratio"]:.0f}%{p2d_note}',
        pad=22)
    ax.set_xticks(x)
    ax.set_xticklabels([f'prefill-{r["worker"]}' if r['worker'] != 'TOTAL' else 'TOTAL'
                        for r in agg])
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {pdf_path}')


make_plots('hit_rate', ['kv_hit_rate', 'idx'], True, 'KV hit rate',
           tick_col='kv_hit_rate', tick_fmt=lambda v: f'{100*float(v):.0f}%',
           tick_desc='KV hit rate ascending (tick labels are hit rate)')
# Sorted by DRAM-hit share, the only bucket that costs a transfer. The hit_rate
# plot cannot show this: its x axis lumps HBM-resident tokens in with loaded
# ones, so a request that hit 90% and loaded nothing sits next to one that hit
# 90% and loaded all of it. Ties break on HBM share so the two populations at
# dram=0 (fully resident vs. fully cold) do not interleave.
make_plots('dram_hit_ratio', ['dram_hit_frac', 'hbm_hit_frac', 'idx'], True,
           'DRAM hit ratio',
           tick_col='dram_hit_frac', tick_fmt=lambda v: f'{100*float(v):.0f}%',
           tick_desc='DRAM hit ratio ascending (tick labels are DRAM hit ratio)')
aggregate_ttft_share()
print('DONE output_dir=', OUT)
