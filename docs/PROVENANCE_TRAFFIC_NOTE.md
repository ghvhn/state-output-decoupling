# Provenance / Traffic Note

This file is an evidence log, not a forum post and not an accusation.

## Screenshot Evidence

Source screenshot:

`C:\Users\GAVINP~1\AppData\Local\Temp\codex-clipboard-74ad091b-9012-447e-850a-41cae3da117d.png`

Observed GitHub traffic panel:

- panel title: `Git clones`
- window: `last 14 days`
- total clones: `1,568`
- unique cloners: `448`
- earliest visible date on chart: `06/23`
- tooltip shown on chart: `06/23`, `Unique 0`
- visible pattern: low/zero activity at the start of the window, irregular activity after
  `06/25`, then a sharp increase around `07/05` and `07/06`

Second source screenshot:

`C:\Users\GAVINP~1\AppData\Local\Temp\codex-clipboard-f5a18be9-9a58-4ffe-8027-73becd61c411.png`

Observed GitHub traffic panels:

- referring sites:
  - `github.com`: `77` views, `3` unique visitors
  - `linkedin.com`: `5` views, `3` unique visitors
- popular content:
  - `Overview`: `135` views, `8` unique visitors
  - `/graphs/traffic`: `28` views, `1` unique visitor
  - `/pulse`: `21` views, `1` unique visitor
  - `/tree/main`: `18` views, `5` unique visitors
  - `/commits/main`: `15` views, `1` unique visitor
  - `/blob/main/PUBLIC_ANNOUNCEM...`: `11` views, `4` unique visitors
  - `/tree/main/scripts`: `7` views, `3` unique visitors
  - `/tree/main/invariants`: `6` views, `4` unique visitors
  - `/branches`: `6` views, `2` unique visitors
  - `/blob/main/invariants/agentic_eng...`: `5` views, `4` unique visitors

## Careful Interpretation

This is evidence that the repository received unusually high clone traffic for a new,
lightly publicized independent research repo.

The combination of the two screenshots strengthens the automated-access inference:

- `1,568` clones and `448` unique cloners in the same traffic window
- only `135` overview views and `8` unique visitors visible in the popular-content table
- only two visible referrers, totaling `82` views and `6` unique visitors

That ratio is difficult to explain as ordinary human browsing followed by normal manual
cloning. It is consistent with automated cloning, mirroring, indexing, scanning, or other
scripted retrieval.

It is still not, by itself, evidence that any specific person, lab, company, model provider,
paper, dataset builder, crawler, or automated system copied, read, trained on, evaluated on,
or otherwise substantively used the work.

Possible explanations include:

- normal GitHub crawler or bot behavior
- traffic from link previews, indexers, mirrors, or security scanners
- private sharing by a reader
- GitHub traffic counting quirks
- repeated clones by automation that does not imply substantive reading or ingestion

The scientifically relevant point is provenance: the repo had a nontrivial public traffic
footprint before a polished public writeup existed, and the clone/view ratio is strong
evidence for automated retrieval rather than normal visible readership. If this is
mentioned in a community post, it should be secondary to the experimental claims and
phrased as a documented automated-access pattern, not as proof of any particular downstream
use.

## Automated Logging (from 2026-07-08)

Screenshots are no longer the only record. `scripts/fetch_repo_traffic.py`
pulls the four GitHub traffic endpoints (clones, views, referrers, paths;
API version 2026-03-10), merges per-day maxima into `traffic/traffic_log.json`,
appends the raw responses to `traffic/snapshots.jsonl`, and commits ONLY the
`traffic/` directory before pushing — so every observation carries a
GitHub-hosted commit timestamp. A Windows scheduled task
(`state-output-traffic-log`, daily 10:00) runs it before the 14-day retention
window can drop a day.

First automated observation (2026-07-08, commit c832498): 1,568 clones /
448 unique cloners against 377 views / 11 unique visitors in the same
window. The clone/view asymmetry documented above persists.

Bounds unchanged: pushed logs are self-reported observations made credible by
hosting and timestamps, not cryptographic proof, and they still do not
identify any actor or downstream use.

## Forum-Safe Wording Prompt

Write this in your own words if used publicly:

1. State the observed numbers.
2. State that GitHub traffic attribution is limited.
3. State that the clone/view ratio is consistent with automated cloning or retrieval.
4. State that this does not identify the actor or prove downstream use.
5. Explain why you are preserving the record anyway: provenance, chronology, and auditability.

Do not use this note as a substitute for a human-written paragraph.
