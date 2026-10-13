# Audio Market Radar

Where AI labs are shipping audio models, where commercially licensed speech data is thin, and what changed this week.

**Live:** https://jakem-r.github.io/audio-market-radar/

Built by Jake Makking-Robinson. Real data only: every number comes from the Hugging Face Hub API.

## Views

- **Gap map:** language × task grid. Demand is watched-lab audio models from the last 12 months (weighted by downloads); supply is commercially licensed speech datasets (weighted by size). Top 10 gaps, each with a note on the dataset worth building.
- **Lab briefs:** one card per watched lab: release pace, task and language mix, most-used models, and the gaps their releases point to.
- **This week:** new lab models and datasets in the last 7 days, plus a diff against the snapshot from a week ago.

Method and limitations are on the page.

## How it runs

`.github/workflows/refresh.yml` runs `fetch.py` daily at 06:00 UTC. It pulls the Hugging Face models and datasets APIs, writes `site/data.json` and `site/snapshots/YYYY-MM-DD.json`, commits them and redeploys GitHub Pages. The site is a single static page that reads `data.json`.

```bash
pip install -r requirements.txt
python fetch.py
python -m http.server -d site
```
