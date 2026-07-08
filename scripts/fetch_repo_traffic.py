"""Log GitHub traffic (clones/views/referrers/paths) and push the log upstream.

GitHub retains repository traffic for only 14 days, so this script is meant to
run daily: it fetches the four traffic endpoints, merges the per-day numbers
into a durable JSON store, appends the raw responses to a snapshot trail, then
commits ONLY the traffic/ directory and pushes -- giving every observation a
GitHub-side commit timestamp (the point of the exercise is provenance).

Auth: the token is read from Windows Credential Manager via `git credential
fill` (the same credentials git push uses); GITHUB_TOKEN overrides. The
traffic API requires push access to the repository.

Run:  python scripts/fetch_repo_traffic.py [--no-push] [--repo owner/name]
Out:  traffic/traffic_log.json   (merged per-day maxima + latest snapshots)
      traffic/snapshots.jsonl    (append-only raw API responses per fetch)

Stdlib only, so it runs under any Python (Task Scheduler friendly).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
TRAFFIC_DIR = REPO_DIR / "traffic"
LOG_PATH = TRAFFIC_DIR / "traffic_log.json"
SNAP_PATH = TRAFFIC_DIR / "snapshots.jsonl"
API_VERSION = "2026-03-10"


def run_git(*args, check=True):
    proc = subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True,
                          text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def detect_repo_slug():
    url = run_git("remote", "get-url", "origin").stdout.strip()
    tail = url.split("github.com", 1)[-1].lstrip(":/")
    if tail.endswith(".git"):
        tail = tail[:-4]
    if tail.count("/") != 1:
        raise RuntimeError(f"could not parse owner/repo from remote '{url}'")
    return tail


def get_token():
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok.strip()
    proc = subprocess.run(["git", "credential", "fill"], cwd=REPO_DIR,
                          input="protocol=https\nhost=github.com\n\n",
                          capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "no GitHub token: set GITHUB_TOKEN or store credentials for "
        "https://github.com (git credential fill returned none)"
    )


def api_get(path, token, repo_slug):
    url = f"https://api.github.com/repos/{repo_slug}/{path}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "state-output-decoupling-traffic-logger",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        hint = ""
        if exc.code in (401, 403):
            hint = " (traffic endpoints need PUSH access; check the token)"
        raise RuntimeError(f"GET {path} -> HTTP {exc.code}{hint}: {detail}")


def merge_daily(store, series, key):
    """Fold [{timestamp, count, uniques}, ...] into store[key] as per-day
    maxima -- the 14-day window slides and today's numbers grow, so max per
    date preserves the fullest observation of each day."""
    days = store.setdefault(key, {})
    for row in series or []:
        day = row["timestamp"][:10]
        prev = days.get(day, {"count": 0, "uniques": 0})
        days[day] = {
            "count": max(prev["count"], row.get("count", 0)),
            "uniques": max(prev["uniques"], row.get("uniques", 0)),
        }
    store[key] = dict(sorted(days.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=None, help="owner/name (default: origin remote)")
    ap.add_argument("--no-push", action="store_true",
                    help="fetch and write files, but skip commit+push")
    args = ap.parse_args(argv)

    repo_slug = args.repo or detect_repo_slug()
    token = get_token()
    fetched_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    clones = api_get("traffic/clones?per=day", token, repo_slug)
    views = api_get("traffic/views?per=day", token, repo_slug)
    referrers = api_get("traffic/popular/referrers", token, repo_slug)
    paths = api_get("traffic/popular/paths", token, repo_slug)

    TRAFFIC_DIR.mkdir(exist_ok=True)
    store = {}
    if LOG_PATH.exists():
        store = json.load(open(LOG_PATH, encoding="utf-8"))
    store["repo"] = repo_slug
    store["updated_at"] = fetched_at
    merge_daily(store, clones.get("clones"), "clones_by_day")
    merge_daily(store, views.get("views"), "views_by_day")
    store["last_window"] = {
        "fetched_at": fetched_at,
        "clones_total": clones.get("count"),
        "clones_uniques": clones.get("uniques"),
        "views_total": views.get("count"),
        "views_uniques": views.get("uniques"),
    }
    store["latest_referrers"] = referrers
    store["latest_paths"] = paths
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=1)
        fh.write("\n")

    with open(SNAP_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "fetched_at": fetched_at, "repo": repo_slug, "clones": clones,
            "views": views, "referrers": referrers, "paths": paths,
        }) + "\n")

    day_rows = store["clones_by_day"]
    last_day = max(day_rows) if day_rows else "n/a"
    print(f"[traffic] {repo_slug} @ {fetched_at}")
    print(f"[traffic] 14d window: {clones.get('count')} clones / "
          f"{clones.get('uniques')} unique cloners; "
          f"{views.get('count')} views / {views.get('uniques')} unique visitors")
    if day_rows:
        d = day_rows[last_day]
        print(f"[traffic] {last_day}: {d['count']} clones, {d['uniques']} unique")
    print(f"[traffic] log spans {len(day_rows)} days -> {LOG_PATH.relative_to(REPO_DIR)}")

    if args.no_push:
        print("[traffic] --no-push: files written, nothing committed.")
        return 0

    # Commit ONLY the traffic directory; the working tree may hold unrelated
    # work that must never ride along on an automated data commit.
    run_git("add", "--", "traffic")
    staged = run_git("status", "--porcelain", "--", "traffic").stdout.strip()
    if not staged:
        print("[traffic] no changes under traffic/; nothing to commit.")
        return 0
    msg = f"traffic: log through {last_day} ({clones.get('count')} clones / {clones.get('uniques')} unique in window)"
    run_git("commit", "-m", msg, "--", "traffic")
    print(f"[traffic] committed: {msg}")
    push = run_git("push", "origin", "HEAD", check=False)
    if push.returncode == 0:
        print("[traffic] pushed to origin.")
    else:
        print(f"[traffic] PUSH FAILED (commit kept locally; next run retries): "
              f"{push.stderr.strip()[:300]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
