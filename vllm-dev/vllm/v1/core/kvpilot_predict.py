#!/usr/bin/env python3
"""Shared dram-safe predictor: cost model, tool-call labelling, p50 lookup.

Imported by both the offline validator (validate_premises.py) and the online
decode-side predictor in dynamo's vLLM handler. Keeping one copy is the point:
if the two drift, the measured accuracy stops describing what runs in serving.
"""

import json
import os
import re
import shlex

# Qwen3-30B-A3B @ bf16: prefill cost-model coefficients and KV bytes per token.
A_FLOPS = 786_432
B_FLOPS = 5_460_983_808
KV_BYTES = 98_304

# Aggregate 8-GPU dense tensor-core peaks (prefill is dense GEMM, so the
# "with sparsity" datasheet figures are unreachable) and DRAM read bandwidth.
MACHINES = {
    "h20": {"bf16_flops": 1.184e15, "bw_dram": 5.04e11},
    "h200": {"bf16_flops": 7.912e15, "bw_dram": 5.04e11},
    "a100": {"bf16_flops": 2.496e15, "bw_dram": 2.52e11},
}

RX_INLINE_CALL = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
RX_INLINE_CMD = re.compile(r"<parameter=command>\s*(.*?)\s*</parameter>", re.S)

# Wrappers precede the real command inside one subcommand and are stepped over.
WRAPPER = {"timeout", "sudo", "nohup", "env", "xargs", "stdbuf", "nice"}
ARG_EATERS = {"timeout", "source", "."}

# Subcommands whose output is a line or two at most, so they move the response
# length by nothing. Dropping them collapses `cd+X` and `X` onto one label,
# which is the point: the label exists to predict response length, and these
# contribute none of it. A command built only of these keeps them all, since
# something has to name the turn.
SILENT = {"cd", "chmod", "chown", "mkdir", "rmdir", "rm", "touch", "mv", "cp",
          "export", "ln", "kill", "pkill"}

# str_replace_editor's own subcommands. The trace holds a handful of turns where
# the model filled `command` with a shell line instead, which would otherwise
# become its own label carrying the whole command text.
EDITOR_COMMANDS = {"view", "str_replace", "create", "insert", "undo_edit"}

# Interpreters whose version suffix carries no signal for response length:
# python, python3, python3.9 and python3.11 all behave the same here.
VERSIONED = ("python", "pytest", "pip", "gcc", "g++", "clang", "node", "ruby",
             "perl", "php", "java", "javac", "gfortran")

# An executed script reports its interpreter. Anything else executed by path is
# a one-off binary (a just-compiled a.out, a generated helper); those names are
# unique per trace and would each become their own label.
SCRIPT_EXT = {".py": "python", ".sh": "sh", ".bash": "sh", ".pl": "perl",
              ".rb": "ruby", ".js": "node", ".ts": "node"}

# Shell keywords are grammar, not programs. `for f in ...; do python $f; done`
# runs python; labelling it `for` describes the loop and hides the payload.
SHELL_KEYWORDS = {"for", "while", "until", "if", "then", "else", "elif", "fi",
                  "do", "done", "case", "esac", "in", "function", "select",
                  "time", "exec", "eval", "source", "."}


def derive_constants(machine=None, compute_scale=1.0, dram_scale=1.0):
    """us/token constants; `machine` defaults to $KVPILOT_MACHINE.

    TP is absent on purpose: it divides compute and KV bytes by the same factor,
    so the verdict is scale-invariant.
    """
    spec = MACHINES[machine or os.environ.get("KVPILOT_MACHINE", "h20")]
    f = spec["bf16_flops"] * compute_scale
    bw = spec["bw_dram"] * dram_scale
    return {
        "a_us": A_FLOPS / f * 1e6,
        "b_us": B_FLOPS / f * 1e6,
        "c_dram_us": KV_BYTES / bw * 1e6,
    }


def dram_safe_times(prefix, append, k):
    """`(t_load_us, t_comp_us)` for this turn, the two sides of the verdict.

    append == 0 means the session ended: nothing will reload this KV, so there
    is no load to hide and both sides are zero.
    """
    if append <= 0:
        return 0.0, 0.0
    t_comp = k["a_us"] * append * (prefix + append / 2.0) + k["b_us"] * append
    return k["c_dram_us"] * prefix, t_comp


def is_dram_safe(prefix, append, k):
    """Whether loading `prefix` from DRAM hides behind this turn's compute.

    append == 0 means the session ended: nothing will reload this KV, so there
    is no load to hide. Feeding 0 into the formula would give T_comp = 0 and
    wrongly demand HBM.
    """
    if append <= 0:
        return True
    t_load, t_comp = dram_safe_times(prefix, append, k)
    return t_load < t_comp


def _bash_label(cmd):
    """Label a bash call by the programs it runs, silent subcommands removed.

    Split on `&&` only: `|` feeds the next stage so only its output returns,
    every `;` in this trace sits inside `python -c "..."`, and `||` means
    "otherwise" so the sides are exclusive. Heredocs are stripped first, being
    the only source of newlines.

    Each subcommand reports the program it runs — for `for f in *.py; do python
    $f; done` that is python, taken from the body after `do`/`then` because the
    loop header holds a variable name and a word list. Version suffixes are
    folded away (python3.11 -> python) since they carry no signal for response
    length, but `python3.11-config` is a different program and keeps its name.
    A script executed by path reports its interpreter; any other executed file
    is a one-off binary whose name is unique to this trace, so it becomes
    `exec`.
    """
    heads = [h for h in (_segment_head(s) for s in _split_segments(cmd)) if h]
    if not heads:
        return "?"
    loud = [h for h in heads if h not in SILENT]
    if loud:
        return "+".join(loud) if len(loud) > 1 else loud[0]
    # Every subcommand was silent. Keep them, minus `cd`, which is pure
    # navigation and never the point of the turn; dedupe so `rm+rm` is `rm`.
    seen, keep = set(), []
    for h in heads:
        if h != "cd" and h not in seen:
            seen.add(h)
            keep.append(h)
    keep = keep or ["cd"]
    return "+".join(keep) if len(keep) > 1 else keep[0]


def _split_segments(cmd):
    cmd = re.sub(r"<<-?\s*['\"]?(\w+)['\"]?.*?^\1", " ", cmd, flags=re.S | re.M)
    cmd = re.sub(r"<<-?\s*['\"]?(\w+)['\"]?.*", " ", cmd, flags=re.S)
    return cmd.split("&&")


def _normalize(name):
    """Fold a version suffix onto its base program, but only a bare one."""
    for base in VERSIONED:
        if name == base:
            return base
        if name.startswith(base):
            tail = name[len(base):]
            if re.fullmatch(r"-?\d+(\.\d+)*", tail):
                return base
    return name


def _segment_head(seg):
    """The program one subcommand runs, or None if it runs none."""
    # A test expression runs no program: `[ -f x ]` and `test -f x` are
    # conditions whose operands would otherwise be read as the command.
    if re.match(r"^\s*(\[\[?|test)\s", seg) or re.match(r"^\s*\[\[?\s*$", seg):
        return None
    body = re.split(r"(?:^|;|\s)(?:do|then)(?:\s|;|$)", seg, maxsplit=1)
    if len(body) > 1 and body[1].strip():
        seg = body[1]
    seg = re.split(r"[;\n]", seg)[0] if re.match(r"^\s*\S", seg) else seg

    # posix mode keeps a quoted value as one word, so
    # `COMP_WORDS="python reproducer.py"` stays attached to its `NAME=` prefix
    # instead of spilling `reproducer.py` out as an apparent program name.
    lex = shlex.shlex(seg, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        words = list(lex)
    except ValueError:
        words = seg.split()

    skip_arg = False
    for tok in words:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            continue
        raw = tok
        tok = tok.strip("\\").lstrip("(").lstrip("{").lstrip("[")
        if not tok or tok.startswith("-") or tok in (">", "<", ">>", "2>"):
            continue
        name = tok.split("/")[-1].strip("\"'")
        if skip_arg:
            skip_arg = False
            continue
        if not re.match(r"^[A-Za-z_.][\w.+-]*$", name):
            continue
        if name in WRAPPER or name in SHELL_KEYWORDS:
            # `source x` / `timeout N` consume their argument; bare keywords
            # like `if` or `while` are followed by the real command.
            skip_arg = name in ARG_EATERS
            continue
        if raw.startswith("./") or "/" in raw.strip("\"'"):
            # An interpreter given by absolute path is still that interpreter,
            # so normalization comes before the script/binary split.
            norm = _normalize(name)
            if norm in VERSIONED:
                return norm
            for ext, interp in SCRIPT_EXT.items():
                if name.lower().endswith(ext):
                    return interp
            return "exec"
        return _normalize(name)
    return None


def _call_label(name, args):
    if name == "str_replace_editor":
        cmd = args.get("command") if isinstance(args, dict) else None
        # Only the tool's own subcommands become labels. A few turns fill this
        # with a shell line, which would otherwise carry the whole command text
        # into the label and make a category of one.
        if cmd not in EDITOR_COMMANDS:
            cmd = "?"
        return f"str_replace_editor:{cmd}"
    if name == "bash":
        cmd = args.get("command") if isinstance(args, dict) else None
        return f"bash:{_bash_label(cmd) if isinstance(cmd, str) else '?'}"
    return name


def turn_labels(msg):
    """Call types in one assistant turn. prepare_swe_trace.py inlines every
    call as `<function=NAME>` text, so no tool_calls array remains."""
    out = []
    for match in RX_INLINE_CALL.finditer(msg.get("content") or ""):
        args = {}
        cmd = RX_INLINE_CMD.search(match.group(2))
        if cmd:
            args["command"] = cmd.group(1)
        out.append(_call_label(match.group(1), args))
    return out


_RESP_LEN = None

# The table is trace-specific and lives with the dataset that produced it: label
# namespaces do not carry across traces (SWE-smith emits `bash:python` and
# `str_replace_editor:view`, TraceLab emits `Read` and `Bash:cd` — measured
# overlap between the two is exactly zero). There is deliberately no default: a
# table from the wrong trace matches nothing, every lookup silently falls back
# to `_default`, and the verdict quietly degrades to a constant while the logs
# look healthy. Failing at startup is the only way that error stays visible.
RESP_LEN_ENV = "KVPILOT_TOOL_RESP_LEN_PATH"


def _lookup_resp_len(label: str) -> int:
    """Estimated tool response length for `label`.

    A label's own measured value is used however few samples back it: one real
    observation of that tool still describes it better than the global figure
    does. Only a label absent from the table falls back.

    The table is read once and kept: this runs on every decode turn, and a
    re-read would cost ~1000x a dict lookup.
    """
    global _RESP_LEN
    if _RESP_LEN is None:
        path = os.environ.get(RESP_LEN_ENV)
        if not path:
            raise RuntimeError(
                f"{RESP_LEN_ENV} is unset. Point it at the tool-response-length "
                f"table produced by the validator for the trace being replayed "
                f"(e.g. datasets/agent-traces/<trace>/tool_resp_len.json). There "
                f"is no default: a table from a different trace shares no labels "
                f"with this one, so every lookup would fall back to the global "
                f"figure and the dram-safe verdict would become a constant."
            )
        with open(path) as f:
            _RESP_LEN = json.load(f)
    entry = _RESP_LEN.get("labels", {}).get(label)
    return int(entry["resp_len"] if entry is not None
               else _RESP_LEN.get("_default", 0))


def predict_dram_safe(prefix: int, decode_len: int, text: str, k: dict) -> bool:
    """Decide dram-safe from what decode knows when it finishes a turn.

    No tool call means the session ended: nothing reloads this KV, so there is
    no load to hide and the blocks are free to sit in DRAM.
    """
    return explain_dram_safe(prefix, decode_len, text, k)[0]


def explain_dram_safe(prefix: int, decode_len: int, text: str, k: dict):
    """`predict_dram_safe` plus the label, append, and the two times compared.

    The caller logs these; deriving them a second time would let the logged
    label drift from the one the verdict actually used.

    A turn's whole call set is one label, joined by `+`: every response lands in
    the same next append, so `Read+Bash:grep` is its own category rather than a
    variant of `Read` (10.5% of TraceLab rounds make several calls, and their
    appends are correspondingly larger). The validators key their tables the
    same way, so this join is what makes a lookup hit.

    Returns `(dram_safe, label, append, t_load_us, t_comp_us)`.
    """
    labels = turn_labels({"content": text})
    label = "+".join(labels) if labels else None
    append = decode_len + _lookup_resp_len(label) if label is not None else 0
    t_load, t_comp = dram_safe_times(prefix, append, k)
    return is_dram_safe(prefix, append, k), label, append, t_load, t_comp


# Measured verdicts, keyed by the trace line encoded in the request id. This is
# the oracle arm of the experiment: same feedback link and re-tiering, but the
# verdict comes from what a prior run actually measured rather than from the
# cost model, which separates "the prediction is wrong" from "the mechanism does
# not pay off".
MEASURED_ENV = "KVPILOT_MEASURED_DRAM_SAFE_PATH"

_MEASURED = None


def measured_enabled() -> bool:
    """Whether a measured dram-safe table is configured."""
    return bool(os.environ.get(MEASURED_ENV))


def _measured_table() -> dict:
    """The measured table, read once and kept (this runs per decode turn)."""
    global _MEASURED
    if _MEASURED is None:
        path = os.environ[MEASURED_ENV]
        with open(path) as f:
            raw = json.load(f)
        # Accept the validator's {"requests": {...}} envelope or a bare mapping,
        # and normalize both to line -> bool so the caller sees one shape.
        entries = raw.get("requests", raw) if isinstance(raw, dict) else {}
        table = {}
        for key, value in entries.items():
            if key.startswith("_"):
                continue
            safe = value.get("dram_safe") if isinstance(value, dict) else value
            if safe is not None:
                table[str(int(key))] = bool(safe)
        _MEASURED = table
    return _MEASURED


def measured_dram_safe(request_id: str):
    """The measured verdict for `request_id`, or None when it is not in the table.

    Keyed by trace line rather than the full id: the id's remaining bits are
    random per attempt, so the line is the only part that identifies the turn
    across runs. Returning None rather than a default keeps a miss visible to
    the caller, which falls back to the prediction and logs it.
    """
    head = request_id.split("-", 1)[0]
    if not head.isdigit():
        return None
    return _measured_table().get(str(int(head)))



def write_resp_len_table(stats, path, percentile=None):
    """Emit the label -> response-length table the online predictor reads.

    Shared by both validators so the file they produce and the file serving
    reads can never disagree on shape. `stats` maps a label to a dict holding
    `pred` and `n`; reusing the validator's own table-1 numbers is the point, so
    what was measured is what serving predicts. Every label is written however
    few samples it has: its own measurement beats the global figure, which only
    exists to cover labels never seen.

    `pred` is whichever percentile the validator was configured with
    (`--predict-percentile`), not necessarily the median, so the key is
    `resp_len` rather than `p50` and the percentile is recorded in the file. A
    table whose provenance is unreadable invites comparing runs built on
    different estimators.
    """
    import statistics

    all_preds = [q["pred"] for q in stats.values() for _ in range(q["n"])]
    table = {
        "_percentile": percentile,
        "_default": int(statistics.median(all_preds)) if all_preds else 0,
        "labels": {
            label: {"resp_len": int(q["pred"]), "n": q["n"]}
            for label, q in sorted(stats.items(), key=lambda kv: -kv[1]["n"])
        },
    }
    path = os.path.abspath(path)
    with open(path, "w") as fh:
        json.dump(table, fh, indent=2, ensure_ascii=False)
    pct = "unspecified" if percentile is None else f"p{percentile:g}"
    print(f"-> {path}: labels={len(stats):,} default={table['_default']} "
          f"percentile={pct}")
    return table
