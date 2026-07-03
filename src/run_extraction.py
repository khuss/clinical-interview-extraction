"""Force-re-extract all 50 patients and (over)write data/cache/*.json — but
RESUMABLE, so hitting a rate limit partway through doesn't cost you the patients
that already succeeded, and it FAILS FAST instead of grinding for an hour.

Why not the notebook's old cache-first check (skip if data/cache/{pid}.json
exists)? Because that file exists for every patient already — it's the STALE
pre-fix cache. "File exists" can't tell stale-old-cache apart from
already-redone-this-pass. So instead this script keeps its own manifest
(`data/cache/_reextraction_manifest.json`) of which patient_ids have been
successfully re-extracted with the CURRENT prompt. Re-running the script:
  - skips anything already in the manifest (no wasted quota re-doing successes)
  - retries anything that failed or was never reached
The manifest is deleted automatically once a full clean pass finishes (0 failures),
so it doesn't linger and cause false skips after a future prompt change.

Two things that used to make a sustained rate limit feel like an infinite hang,
both fixed here:
  1. `extract_one`'s default `num_retries=8` means a SINGLE failing call could
     internally retry 8x with exponential backoff before even raising — potentially
     minutes, per patient, invisible to you. This script passes a small
     `num_retries` (2) instead: fail fast, and let the OUTER resume loop (below)
     be the real retry mechanism, since it's actually visible and interruptible.
  2. If the account is sustainedly rate-limited (not just one blip), every
     subsequent patient will ALSO fail the same way — grinding through all
     remaining patients one by one just burns time for no progress. This script
     now stops itself after `MAX_CONSECUTIVE_FAILURES` in a row, rather than
     making you sit there and interrupt it.

It's always safe to interrupt this manually too (Kernel -> Interrupt, or Ctrl-C):
each success is written to disk and added to the manifest immediately, so an
interrupt only loses the one call in flight, never previously-completed patients.

Run from the repo root:

    python src/run_extraction.py
    python src/run_extraction.py --model gemini/gemini-1.5-flash --cache-dir data/cache_gemini

The --model/--cache-dir pair is what makes a full 50-patient run with a different
provider possible (not just the golden 10, which the notebook's Gemini cells cover) —
output lands in its own folder so it never collides with the default Claude cache.

If it stops itself with "N consecutive failures", that's very likely a sustained/
account-level limit (rate window or quota), not a one-off — worth checking the
Anthropic console before just re-running immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract import DEFAULT_MODEL, extract_one  # noqa: E402

CACHE_DIR = Path("data/cache")  # default; run()/main() can override per-call
INTERVIEWS_PATH = Path("data/interviews.json")
THROTTLE_SECONDS = 5.0  # between successful calls
RATE_LIMIT_COOLDOWN_SECONDS = 15.0  # extra pause after a failure before trying the next patient
MAX_CONSECUTIVE_FAILURES = 3  # stop the whole run rather than grinding through the rest
CALL_NUM_RETRIES = 2  # litellm's OWN internal retry count per call — kept small on purpose (see docstring)


def _load_manifest(manifest_path: Path) -> set[str]:
    if manifest_path.exists():
        return set(json.loads(manifest_path.read_text()))
    return set()


def _save_manifest(manifest_path: Path, done: set[str]) -> None:
    manifest_path.write_text(json.dumps(sorted(done), indent=1))


def _looks_like_rate_limit(e: Exception) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return "ratelimit" in name or "429" in msg or "rate limit" in msg or "overloaded" in msg


def run(
    interviews: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: str | Path = CACHE_DIR,
) -> dict:
    """Re-extract every patient not yet in the manifest. Importable (e.g. from a
    notebook) as well as runnable as a script — both paths share this one loop.
    Stops itself early after MAX_CONSECUTIVE_FAILURES rather than hanging.

    `model` / `cache_dir` let this run against a different provider (e.g. Gemini)
    without touching the default Claude cache — each (model, cache_dir) pair gets
    its own manifest, so a Gemini run and a Claude run never collide or skip each
    other's patients."""
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
    print(f"model={model}  cache_dir={cache_dir}  n={len(interviews)}  throttle={THROTTLE_SECONDS}s  "
          f"already re-extracted this pass={len(done)}", flush=True)

    ok, failed = 0, []
    consecutive_failures = 0
    stopped_early = False
    t0 = time.time()
    try:
        for i, item in enumerate(interviews, 1):
            pid = item["patient_id"]
            if pid in done:
                print(f"[{i}/{len(interviews)}] {pid} SKIP (already re-extracted this pass)", flush=True)
                continue
            try:
                rec = extract_one(item["interview_transcript"], pid, model=model, num_retries=CALL_NUM_RETRIES)
                (cache_dir / f"{pid}.json").write_text(rec.model_dump_json(indent=1))
                done.add(pid)
                _save_manifest(manifest_path, done)  # persist immediately — a crash loses nothing already done
                ok += 1
                consecutive_failures = 0
                print(f"[{i}/{len(interviews)}] {pid} OK  ({time.time()-t0:.0f}s elapsed)", flush=True)
                time.sleep(THROTTLE_SECONDS)
            except Exception as e:  # surfaced, not swallowed — see extract.py's docstring
                failed.append(pid)
                consecutive_failures += 1
                print(f"[{i}/{len(interviews)}] {pid} FAILED "
                      f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES} in a row): "
                      f"{type(e).__name__}: {e}", flush=True)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nStopping early — {consecutive_failures} failures in a row looks "
                          "like a sustained limit, not a blip. Re-run this cell/script later "
                          "to resume (already-done patients are skipped).", flush=True)
                    stopped_early = True
                    break
                cooldown = RATE_LIMIT_COOLDOWN_SECONDS if _looks_like_rate_limit(e) else THROTTLE_SECONDS
                print(f"  -> pausing {cooldown:.0f}s before the next attempt "
                      "(interrupt any time — already-done patients are safe)", flush=True)
                time.sleep(cooldown)
    except KeyboardInterrupt:
        print(f"\nInterrupted by user after {ok} new successes this pass. "
              "Nothing lost — re-run to resume.", flush=True)
        stopped_early = True

    result = {"ok": ok, "failed": failed, "stopped_early": stopped_early, "total_time_s": time.time() - t0}
    print(f"\nDONE for this pass. ok={ok} failed={len(failed)} failed_ids={failed} "
          f"stopped_early={stopped_early} total_time={result['total_time_s']:.0f}s", flush=True)
    if failed or stopped_early:
        print("Re-run this (script or cell) to retry only the failed/remaining "
              "patients — already-done ones are skipped via the manifest.", flush=True)
    else:
        manifest_path.unlink(missing_ok=True)  # clean pass finished; manifest has served its purpose
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"litellm model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help=f"output dir (default: {CACHE_DIR})")
    args = parser.parse_args()

    result = run(model=args.model, cache_dir=args.cache_dir)
    if result["failed"] or result["stopped_early"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
