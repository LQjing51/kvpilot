# SPDX-License-Identifier: Apache-2.0
"""KVPilot decode-side signal: the next turn's safe tier and prefill times.

One signal per finished turn (format in kvpilot_placement), from:

  KVPILOT_ONLINE_PREDICT=0 (default): the signal table at
    KVPILOT_OFFLINE_TABLE (utils/placement_ab/make_req_table.py,
    signal_<trace>.json), keyed by the request's trace line: next_tier, the
    measured request-safe tier of the session's next turn, next_time, its
    prefill e2e with the prefix in HBM / DRAM / SSD, and next_suffix, its
    suffix in tokens (a one-point suffix distribution).
  KVPILOT_ONLINE_PREDICT=1: predicted from this turn's tool calls. The call
    type's tool-response length distribution
    (KVPILOT_TOOL_RESP_DISTRIBUTION_PATH, one dataset's labels) gives the next
    turn's suffix distribution, the time model (kvpilot_time_model,
    coefficients KVPILOT_TIME_COEFFS_PATH from
    paper-figure/Figure3-request-safe/fit_times2.py) its expected prefill
    time per tier, and the tier follows the table's rule (`thresh`).

Request ids carry the trace line in their first, 8-digit group (the replay);
the NIXL warm-up requests of dynamo_script.sh carry ffffffff and have no next
turn.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections import Counter

from vllm.v1.core.kvpilot_time_model import TimeModel

# Block size == LMCache chunk size (the launch scripts): a prompt shorter than
# this leaves no full block, so the prefill side tracks nothing for it.
BLOCK = 512
WARMUP_HEAD = "ffffffff"

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


def trace_line(request_id: str) -> int | None:
    """The request's trace line; None for a warm-up request."""
    head = request_id.split("-", 1)[0]
    if head == WARMUP_HEAD:
        return None
    if len(head) == 8 and head.isdigit():
        return int(head)
    raise ValueError(f"KVPilot: request id {request_id!r} has no trace line")


def from_env():
    """The signal source the environment selects."""
    if os.environ.get("KVPILOT_ONLINE_PREDICT", "0") == "1":
        return OnlinePredictor(
            os.environ["KVPILOT_TOOL_RESP_DISTRIBUTION_PATH"],
            os.environ["KVPILOT_TIME_COEFFS_PATH"],
        )
    return OfflineTable(os.environ["KVPILOT_OFFLINE_TABLE"])


class OfflineTable:
    """Measured signals, looked up by trace line."""

    def __init__(self, path: str) -> None:
        with open(path) as f:
            self._rows = {int(line): row for line, row in json.load(f).items()}
        self.path = path

    def signal(self, request_id: str, prompt_len: int, decode_len: int, text) -> dict:
        line = trace_line(request_id)
        if line is None:
            return {"tier": None, "time": None, "suffix": None}
        row = self._rows[line]
        # The next turn's measured suffix, where the table has it
        # (add_next_suffix.py): one point of a distribution.
        nxt = row.get("next_suffix")
        return {
            "tier": row["next_tier"],
            "time": row["next_time"],
            "suffix": None if nxt is None else [[1.0, nxt]],
        }


class OnlinePredictor:
    """Signals predicted from the turn's tool calls and the time model."""

    def __init__(self, dist_path: str, coeffs_path: str) -> None:
        with open(dist_path) as f:
            raw = json.load(f)
        self._model = TimeModel.load(coeffs_path, BLOCK)
        self._dist = {}
        mass: Counter = Counter()
        for label, hist in raw.items():
            mids = [(lo + hi) / 2.0 for lo, hi, _ in hist["buckets"]]
            probs = [p for _, _, p in hist["buckets"]]
            self._dist[label] = (mids, probs)
            for m, p in zip(mids, probs):
                mass[m] += p * hist["n"]
        total = sum(mass.values())
        # Unseen labels (call combinations the trace never made) use the mix.
        self._global = (sorted(mass), [mass[m] / total for m in sorted(mass)])
        # A distribution from another trace shares no labels with this one and
        # sends every turn to the mix, flattening the signal: count it.
        self.label_misses = 0
        self.labels_seen = 0

    def signal(self, request_id: str, prompt_len: int, decode_len: int, text) -> dict:
        if trace_line(request_id) is None:
            return {"tier": None, "time": None, "suffix": None}
        labels = turn_labels({"content": text()})
        if not labels:
            # No tool call: the session ends with this turn.
            return {"tier": None, "time": None, "suffix": None}
        label = "+".join(labels)
        self.labels_seen += 1
        hist = self._dist.get(label)
        if hist is None:
            self.label_misses += 1
            hist = self._global
        mids, probs = hist
        p = BLOCK * (prompt_len // BLOCK)
        base = prompt_len % BLOCK + decode_len
        acc = [0.0, 0.0, 0.0]
        weight = sum(probs)
        for resp, prob in zip(mids, probs):
            for i, t in enumerate(self._model.times(p, base + resp)):
                acc[i] += prob * t
        th, td, ts = (a / weight for a in acc)
        thresh = self._model.thresh
        tier = (
            "SSD" if (ts - th) / th < thresh
            else "DRAM" if (td - th) / th < thresh
            else "HBM"
        )
        suffix = [[prob / weight, base + resp] for resp, prob in zip(mids, probs)]
        return {"tier": tier, "time": [th, td, ts], "suffix": suffix}

