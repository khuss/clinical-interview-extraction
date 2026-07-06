"""Force-re-extract all 50 patients and (over)write data/cache/*.json — but
RESUMABLE, so hitting a rate limit partway through doesn't cost you the patients
that already succeeded, and it FAILS FAST instead of grinding for an hour.

Why not the notebook's old cache-first check (skip if data/cache/{pid}.json
exists)? Because that file exists for every patient already — it's the STALE
pre-fix cache. "File exists" can't tell stale-old-cache apart from
already-redone-this-pass. So instead this script keeps its own manifest
(`{cache_dir}/_reextraction_manifest.json`) mapping each patient_id to a
fingerprint of (current SYSTEM_PROMPT + schema + THAT PATIENT'S transcript text).
Re-running the script:
  - skips a patient only if their fingerprint still matches (no wasted quota
    re-doing successes)
  - retries anything that failed, was never reached, OR whose fingerprint changed
    since it was marked done — either because you edited the prompt (changes
    everyone's fingerprint) or because just that one patient's transcript was
    edited/extended (e.g. their interview continues on a later day, or the
    dataset gets a correction for one patient) — either way it's re-extracted,
    and nobody else is disturbed for a single-patient change.
The manifest is PERSISTENT — it is NOT deleted after a clean pass. It's the only
record of "this patient's cache is trustworthy as of this exact fingerprint," so
deleting it once a run succeeds would defeat its entire purpose: the next
invocation (a different terminal session, or the notebook, days later) would have
no way to tell an up-to-date cache file apart from a stale one, and would silently
re-call the API for all 50 patients even though nothing changed. (An earlier
version of this script did delete it on a clean pass — that was a bug, not a
feature; if you saw a full re-extraction with no prompt/transcript changes, this
was why.) Any manifest that isn't in this exact per-patient-fingerprint shape (old
formats, hand edits, corruption) is treated as fully stale rather than partially
trusted — the only cost of distrusting a manifest is redoing some API calls, never
a correctness problem.

RATE LIMITS — handled so a low-tier account can still finish in one unattended
command instead of needing manual restarts:
  1. `extract_one`'s default `num_retries=8` means a SINGLE failing call could
     internally retry 8x with exponential backoff before even raising — potentially
     minutes, per patient, invisible to you. This script passes a small
     `num_retries` (2) instead: fail fast, and let the OUTER loop (below) be the
     real retry mechanism, since it's actually visible and interruptible.
  2. A rate-limit error specifically (429 / "rate_limit_error" / "overloaded") is
     NOT treated as a failure of that patient — it retries the SAME patient after
     a `RATE_LIMIT_COOLDOWN_SECONDS` (65s) wait, long enough to clear a per-minute
     window, up to `MAX_RATE_LIMIT_RETRIES_PER_PATIENT` (6) times, and does not
     count toward the stop-early counter. Moving on to the next patient wouldn't
     help anyway — it would hit the exact same account-wide limit immediately.
  3. Only a GENUINE (non-rate-limit) error counts toward `MAX_CONSECUTIVE_FAILURES`
     and stops the whole run early — that's for real, unexpected breakage (a bad
     response the model can't self-correct, a network error), not pacing.
  4. `--throttle-seconds` lets you tune the pace between calls to your account's
     actual requests-per-minute cap without editing code (e.g. a 5 RPM tier needs
     at least 12-13s between calls, plus margin — the 5s default assumes a less
     restrictive tier and will itself trigger rate limits on a tighter one).

It's always safe to interrupt this manually too (Ctrl-C): each success is written
to disk and added to the manifest immediately, so an interrupt only loses the one
call in flight, never previously-completed patients.

Run from the repo root:

    python src/run_extraction.py
    python src/run_extraction.py --throttle-seconds 13          # tune to your account's rate limit
    python src/run_extraction.py --model gemini/gemini-3.1-flash-lite --cache-dir data/cache_gemini

The --model/--cache-dir pair is what makes a full 50-patient run with a different
provider possible (not just the golden 10, which the notebook's Gemini cells cover) —
output lands in its own folder so it never collides with the default Claude cache.
Gemini's free-tier limits vary a lot per account/project — check your own quota at
aistudio.google.com before picking a model, rather than assuming a published default.
gemini-3.1-flash-lite is a reasonable starting point (a lightweight model with a
comparatively high daily cap on most accounts), but swap the --model string for
whatever your account actually has headroom on.

If it stops itself with "N consecutive failures", that's very likely a sustained/
account-level limit (rate window or quota), not a one-off — worth checking your
provider's console before just re-running immediately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract import DEFAULT_MODEL, SYSTEM_PROMPT, _SCHEMA_JSON, extract_one  # noqa: E402

CACHE_DIR = Path("data/cache")  # default; run()/main() can override per-call
INTERVIEWS_PATH = Path("data/interviews.json")
THROTTLE_SECONDS = 5.0  # between successful calls -- override with --throttle-seconds if your
                        # account tier's requests-per-minute cap is tighter than this allows
RATE_LIMIT_COOLDOWN_SECONDS = 65.0  # long enough to clear a per-MINUTE rate limit window
MAX_RATE_LIMIT_RETRIES_PER_PATIENT = 6  # retries of the SAME patient on a rate-limit error
                                         # specifically, before giving up on just that one
MAX_CONSECUTIVE_FAILURES = 3  # stop the whole run — but only for GENUINE (non-rate-limit)
                              # errors; a sustained rate limit is expected to clear on its own
                              # given enough waiting, so it doesn't count toward this
CALL_NUM_RETRIES = 2  # litellm's OWN internal retry count per call — kept small on purpose (see docstring)


_PROMPT_FP = hashlib.sha256((SYSTEM_PROMPT + _SCHEMA_JSON).encode()).hexdigest()[:16]  # prompt+schema, computed once


def _work_fingerprint(transcript: str) -> str:
    """One patient's unit of work: their transcript text under the CURRENT prompt +
    schema. Changes if EITHER changes — the prompt gets edited (affects everyone), or
    just this one patient's transcript gets edited/extended (e.g. their interview
    continues on a later day, or the dataset gets a correction). Either way, "done"
    under the OLD fingerprint must not be trusted as "done" under the new one — that's
    the whole point of storing this per-patient instead of one global flag."""
    return hashlib.sha256(f"{_PROMPT_FP}|{transcript}".encode()).hexdigest()[:16]


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    """Returns {patient_id: work_fingerprint} for patients safe to skip. Any shape
    that isn't this exact dict (old list-of-ids format, old single-fingerprint format,
    corrupt file) is treated as fully stale — always safe, since the only cost of
    distrusting a manifest is redoing some API calls, never a correctness problem."""
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text())
        done = data["done"]
        if not isinstance(done, dict):
            raise (TypeError if not isinstance(done, list) else ValueError)("old format")
        return done
    except Exception:
        print("  (manifest is missing, in an old format, or unreadable — treating as stale, "
              "re-extracting everything)", flush=True)
        return {}


def _save_manifest(manifest_path: Path, done: dict[str, str]) -> None:
    manifest_path.write_text(json.dumps({"done": done}, indent=1, sort_keys=True))


def rebuild_manifest(cache_dir: str | Path, interviews: list[dict] | None = None) -> dict:
    """Reconstruct the manifest from cache files already on disk, with ZERO API calls.

    For anyone upgrading from the old version of this script (the one that deleted
    the manifest after a clean pass, see the module docstring): your data/cache/*.json
    files are still perfectly valid, but with no manifest there's nothing recording
    that fact, so the next run would silently re-extract all 50 patients for no
    reason. This rebuilds the manifest from what's already cached instead of paying
    for that. Only trust this if the cached files really were produced under the
    CURRENT prompt/schema — if you're not sure, just let the next run re-extract
    for real instead of rebuilding a manifest that would tell it not to."""
    cache_dir = Path(cache_dir)
    if interviews is None:
        interviews = json.loads(INTERVIEWS_PATH.read_text())
    done: dict[str, str] = {}
    for item in interviews:
        pid = item["patient_id"]
        if (cache_dir / f"{pid}.json").exists():
            done[pid] = _work_fingerprint(item["interview_transcript"])
    _save_manifest(cache_dir / "_reextraction_manifest.json", done)
    return done


def _looks_like_rate_limit(e: Exception) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return "ratelimit" in name or "429" in msg or "rate limit" in msg or "overloaded" in msg


def run(
    interviews: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: str | Path = CACHE_DIR,
    throttle_seconds: float = THROTTLE_SECONDS,
) -> dict:
    """Re-extract every patient not yet in the manifest. Importable (e.g. from a
    notebook) as well as runnable as a script — both paths share this one loop.
    Stops itself early after MAX_CONSECUTIVE_FAILURES *genuine* errors in a row —
    a sustained rate limit is handled separately (see below) and never counts
    toward that, since giving up on it doesn't fix anything a longer wait wouldn't.

    `model` / `cache_dir` let this run against a different provider (e.g. Gemini)
    without touching the default Claude cache — each (model, cache_dir) pair gets
    its own manifest, so a Gemini run and a Claude run never collide or skip each
    other's patients. `throttle_seconds` should be tuned to your account's actual
    requests-per-minute cap (e.g. a 5 RPM tier needs >=12s between calls, plus
    margin) — the default is a reasonable starting point, not a guarantee."""
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except ImportError:
        pass

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "_reextraction_manifest.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if interviews is None:
        interviews = json.loads(INTERVIEWS_PATH.read_text())
    done = _load_manifest(manifest_path)
    print(f"model={model}  cache_dir={cache_dir}  n={len(interviews)}  throttle={throttle_seconds}s  "
          f"already re-extracted this pass={len(done)}", flush=True)

    ok, failed = 0, []
    consecutive_failures = 0
    stopped_early = False
    t0 = time.time()
    try:
        for i, item in enumerate(interviews, 1):
            pid = item["patient_id"]
            transcript = item["interview_transcript"]
            expected_fp = _work_fingerprint(transcript)
            if done.get(pid) == expected_fp:
                print(f"[{i}/{len(interviews)}] {pid} SKIP (already re-extracted this pass, "
                      "same prompt + same transcript)", flush=True)
                continue

            rate_limit_attempts = 0
            while True:  # one patient; rate-limit hits retry HERE, genuine errors fall through below
                try:
                    rec = extract_one(transcript, pid, model=model, num_retries=CALL_NUM_RETRIES)
                    (cache_dir / f"{pid}.json").write_text(rec.model_dump_json(indent=1))
                    done[pid] = expected_fp
                    _save_manifest(manifest_path, done)  # persist immediately — a crash loses nothing done
                    ok += 1
                    consecutive_failures = 0
                    print(f"[{i}/{len(interviews)}] {pid} OK  ({time.time()-t0:.0f}s elapsed)", flush=True)
                    time.sleep(throttle_seconds)
                    break  # done with this patient, move to the next
                except Exception as e:  # surfaced, not swallowed — see extract.py's docstring
                    if _looks_like_rate_limit(e) and rate_limit_attempts < MAX_RATE_LIMIT_RETRIES_PER_PATIENT:
                        rate_limit_attempts += 1
                        print(f"[{i}/{len(interviews)}] {pid} rate-limited "
                              f"(retry {rate_limit_attempts}/{MAX_RATE_LIMIT_RETRIES_PER_PATIENT} for this "
                              f"patient) -> waiting {RATE_LIMIT_COOLDOWN_SECONDS:.0f}s to clear the window "
                              "(does not count as a failure)", flush=True)
                        print(f"    {type(e).__name__}: {str(e)[:300]}", flush=True)
                        time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                        continue  # retry the SAME patient, not the next one
                    # either a genuine (non-rate-limit) error, or this one patient exhausted its
                    # rate-limit retries -- give up on it for this pass and move on
                    failed.append(pid)
                    consecutive_failures += 1
                    print(f"[{i}/{len(interviews)}] {pid} FAILED "
                          f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES} in a row): "
                          f"{type(e).__name__}: {e}", flush=True)
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"\nStopping early — {consecutive_failures} genuine failures in a row, "
                              "not just rate limits. Re-run this cell/script later to resume "
                              "(already-done patients are skipped).", flush=True)
                        stopped_early = True
                    time.sleep(throttle_seconds)
                    break  # move to next patient (or the outer loop notices stopped_early and exits)
            if stopped_early:
                break
    except KeyboardInterrupt:
        print(f"\nInterrupted by user after {ok} new successes this pass. "
              "Nothing lost — re-run to resume.", flush=True)
        stopped_early = True

    result = {"ok": ok, "failed": failed, "stopped_early": stopped_early, "total_time_s": time.time() - t0}
    print(f"\nDONE for this pass. ok={ok} failed={len(failed)} failed_ids={failed} "
          f"stopped_early={stopped_early} total_time={result['total_time_s']:.0f}s", flush=True)
    print("Re-run this (script or cell) any time — the manifest persists, so already-done "
          "patients (matching fingerprint) are skipped at zero cost, whether or not this "
          "pass finished clean.", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"litellm model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help=f"output dir (default: {CACHE_DIR})")
    parser.add_argument("--throttle-seconds", type=float, default=THROTTLE_SECONDS,
                         help=f"seconds between successful calls (default: {THROTTLE_SECONDS}) — raise this "
                              "if your account's requests-per-minute cap is tight, e.g. 13+ for a 5 RPM tier")
    parser.add_argument("--rebuild-manifest", action="store_true",
                         help="reconstruct the manifest from cache files already on disk (zero API calls) "
                              "instead of running extraction — see rebuild_manifest()'s docstring")
    args = parser.parse_args()

    if args.rebuild_manifest:
        done = rebuild_manifest(args.cache_dir)
        print(f"Rebuilt manifest for {len(done)} cached patient(s) in {args.cache_dir} — "
              "no API calls made.")
        return

    result = run(model=args.model, cache_dir=args.cache_dir, throttle_seconds=args.throttle_seconds)
    if result["failed"] or result["stopped_early"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
