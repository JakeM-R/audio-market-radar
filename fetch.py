"""Audio Market Radar: pull Hugging Face data, compute gaps, write data.json + a dated snapshot.

Runs daily from .github/workflows/refresh.yml. Real data only.
"""
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import pycountry

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(ROOT, "site")
API = "https://huggingface.co/api"

# Watched organisations. Every handle verified on 13 Oct 2026 to have at least one
# audio model on Hugging Face (deepseek-ai, ByteDance-Seed and xiaomi had none).
WATCHLIST = {
    # Big labs
    "openai": "OpenAI", "google": "Google", "facebook": "Meta", "nvidia": "NVIDIA",
    "microsoft": "Microsoft", "Qwen": "Qwen (Alibaba)", "mistralai": "Mistral",
    "ByteDance": "ByteDance", "stepfun-ai": "StepFun", "FunAudioLLM": "FunAudioLLM (Alibaba)",
    # Voice-first startups
    "kyutai": "Kyutai", "sesame": "Sesame", "hexgrad": "hexgrad (Kokoro)", "coqui": "Coqui",
    "suno": "Suno", "myshell-ai": "MyShell", "ResembleAI": "Resemble AI",
    "canopylabs": "Canopy Labs", "Zyphra": "Zyphra", "fishaudio": "Fish Audio",
    "nari-labs": "Nari Labs", "bosonai": "Boson AI", "OuteAI": "OuteAI",
    "neuphonic": "Neuphonic", "Marvis-AI": "Marvis", "sarvamai": "Sarvam AI",
    "pyannote": "pyannote",
    # Academic / open-research speech groups
    "speechbrain": "SpeechBrain", "espnet": "ESPnet", "amphion": "Amphion (CUHK-SZ)",
    "ai4bharat": "AI4Bharat (IIT Madras)", "parler-tts": "Parler-TTS",
}

AUDIO_TAGS = [
    "automatic-speech-recognition", "text-to-speech", "audio-to-audio",
    "audio-classification", "voice-activity-detection", "audio-text-to-text", "text-to-audio",
]

# Grid columns: what the model needs -> which dataset task supplies it.
GRID_TASKS = {
    "asr": {"label": "Speech recognition & understanding",
            "model_tags": {"automatic-speech-recognition", "audio-text-to-text"},
            "dataset_task": "automatic-speech-recognition"},
    "tts": {"label": "Speech synthesis",
            "model_tags": {"text-to-speech"},
            "dataset_task": "text-to-speech"},
}

COMMERCIAL = re.compile(r"^(cc0.*|cc-by-[0-9.]+|cc-by-sa-[0-9.]+|apache-2\.0|mit|odc-by|pddl|bsd.*|unlicense)$")
SIZE_WEIGHT = {
    "n<1K": 1, "1K<n<10K": 2, "10K<n<100K": 3, "100K<n<1M": 4, "1M<n<10M": 5,
    "10M<n<100M": 6, "100M<n<1B": 7,
}
SIZE_IMPLAUSIBLE = 1  # self-reported sizes above 1B rows are almost always wrong tags; weight as unknown
# Bare 3-letter model tags that are languages (most bare tags are 2-letter ISO 639-1).
MODEL_3LETTER = {"yue", "fil", "ceb", "haw", "nan", "wuu", "hak", "cmn"}
ALIASES = {"cmn": "zh"}


def get_paged(path, params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    out = []
    while url:
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "audio-market-radar"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    out.extend(json.load(r))
                    link = r.headers.get("Link") or ""
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                wait = 2 ** attempt
                if getattr(e, "code", None) == 429:
                    wait = int(e.headers.get("Retry-After") or 60)
                print(f"retry in {wait}s {url}: {e}", file=sys.stderr)
                time.sleep(wait)
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


def lang_code(code):
    code = ALIASES.get(code.lower(), code.lower())
    if len(code) == 2:
        lang = pycountry.languages.get(alpha_2=code)
    elif len(code) == 3:
        lang = pycountry.languages.get(alpha_3=code)
    else:
        return None
    if not lang:
        return None
    return getattr(lang, "alpha_2", None) or lang.alpha_3


def lang_name(code):
    if code == "yue":
        return "Cantonese"
    lang = pycountry.languages.get(alpha_2=code) if len(code) == 2 else pycountry.languages.get(alpha_3=code)
    name = re.sub(r"\s*\(.*\)$", "", lang.name) if lang else code
    return {"Panjabi": "Punjabi", "Modern Greek": "Greek", "Pushto": "Pashto", "Norwegian Bokmål": "Norwegian"}.get(name, name)


def model_langs(tags):
    langs = set()
    for t in tags:
        if t.startswith("language:"):
            t = t[9:]
        if (len(t) == 2 and t.isalpha()) or t in MODEL_3LETTER:
            c = lang_code(t)
            if c:
                langs.add(c)
    return sorted(langs)


def fetch_models():
    models = {}
    for org in WATCHLIST:
        for tag in AUDIO_TAGS:
            for m in get_paged("models", {"author": org, "pipeline_tag": tag, "limit": 1000}):
                models[m["id"]] = {
                    "id": m["id"], "org": org, "task": m.get("pipeline_tag"),
                    "createdAt": m.get("createdAt"), "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0), "langs": model_langs(m.get("tags", [])),
                    "license": next((t[8:] for t in m.get("tags", []) if t.startswith("license:")), None),
                    "library": m.get("library_name"),
                }
        print(f"models: {org} done ({len(models)} total)", file=sys.stderr)
    return list(models.values())


def fetch_datasets():
    datasets = {}
    for key, cfg in GRID_TASKS.items():
        rows = get_paged("datasets", {"filter": f"task_categories:{cfg['dataset_task']}", "limit": 1000})
        for d in rows:
            tags = d.get("tags", [])
            rec = datasets.setdefault(d["id"], {
                "id": d["id"], "author": d.get("author"), "createdAt": d.get("createdAt"),
                "lastModified": d.get("lastModified"), "downloads": d.get("downloads", 0),
                "likes": d.get("likes", 0),
                "license": next((t[8:] for t in tags if t.startswith("license:")), None),
                "size": next((t[16:] for t in tags if t.startswith("size_categories:")), None),
                "langs": sorted({c for t in tags if t.startswith("language:") for c in [lang_code(t[9:])] if c}),
                "tasks": [],
            })
            rec["tasks"].append(key)
        print(f"datasets: {cfg['dataset_task']} -> {len(rows)}", file=sys.stderr)
    for d in datasets.values():
        d["commercial"] = bool(d["license"] and COMMERCIAL.match(d["license"]))
        # Count toward supply only if someone has used it: filters empty placeholder repos.
        d["used"] = (d["downloads"] or 0) >= 10 or (d["likes"] or 0) >= 1
    return list(datasets.values())


def size_weight(d):
    return SIZE_WEIGHT.get(d["size"], SIZE_IMPLAUSIBLE)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def model_weight(m):
    return 1 + math.log10(1 + (m["downloads"] or 0))


def compute(models, datasets, now):
    year_ago = now - timedelta(days=365)
    recent = [m for m in models if parse_ts(m["createdAt"]) and parse_ts(m["createdAt"]) >= year_ago]

    # Demand: watched-lab models from the last 12 months, weighted by downloads.
    # A model listing many languages spreads its weight (sqrt) so 100-language
    # checkpoints don't swamp targeted releases.
    demand = defaultdict(float)
    demand_models = defaultdict(list)
    for m in recent:
        for key, cfg in GRID_TASKS.items():
            if m["task"] in cfg["model_tags"] and m["langs"]:
                w = model_weight(m) / math.sqrt(len(m["langs"]))
                for lang in m["langs"]:
                    demand[(lang, key)] += w
                    demand_models[(lang, key)].append(m)

    # Supply: commercially usable datasets, weighted by size category.
    supply = defaultdict(float)
    supply_sets = defaultdict(list)
    for d in datasets:
        if not (d["commercial"] and d["used"]):
            continue
        for key in d["tasks"]:
            for lang in d["langs"]:
                supply[(lang, key)] += size_weight(d)
                supply_sets[(lang, key)].append(d)

    all_supply = defaultdict(int)
    for d in datasets:
        for key in d["tasks"]:
            for lang in d["langs"]:
                all_supply[(lang, key)] += 1

    cells = []
    for (lang, key), dem in demand.items():
        ms = demand_models[(lang, key)]
        labs = Counter(m["org"] for m in ms)
        sets = sorted(supply_sets[(lang, key)], key=lambda d: (-size_weight(d), -d["downloads"]))
        cells.append({
            "lang": lang, "langName": lang_name(lang), "task": key,
            "demand": round(dem, 2), "supply": round(supply[(lang, key)], 2),
            "gap": round(dem / (1 + supply[(lang, key)]) * 100, 1),
            "nModels": len(ms), "nLabs": len(labs),
            "labs": [WATCHLIST[o] for o, _ in labs.most_common(5)],
            "topModels": [{"id": m["id"], "downloads": m["downloads"]}
                          for m in sorted(ms, key=lambda m: -m["downloads"])[:3]],
            "nCommercial": len(sets), "nAnyLicence": all_supply[(lang, key)],
            "largestCommercial": sets[0]["size"] if sets else None,
            "topDatasets": [{"id": d["id"], "size": d["size"], "license": d["license"]} for d in sets[:3]],
        })
    cells.sort(key=lambda c: -c["gap"])

    # Only rank cells with real multi-lab demand, so one hobby-scale release can't top the list.
    ranked = [c for c in cells if c["nLabs"] >= 2]
    build = {
        "asr": "{n}: conversational and accented speech with verbatim transcripts, commercially licensed",
        "tts": "{n}: studio-quality multi-speaker read and expressive speech with aligned text, commercially licensed",
    }
    # Cells driven by exactly the same releases (one multilingual model tagging 30 languages)
    # are one signal, not several: fold them into the highest-scoring cell.
    top, seen = [], {}
    for c in ranked:
        sig = (c["task"], frozenset(m["id"] for m in demand_models[(c["lang"], c["task"])]))
        if sig in seen:
            seen[sig]["alsoLangs"].append(c["langName"])
            continue
        if len(top) == 10:
            continue
        entry = {**c, "alsoLangs": [], "build": build[c["task"]].format(n=c["langName"])}
        seen[sig] = entry
        top.append(entry)
    for e in top:
        e["title"] = e["langName"]
        if len(e["alsoLangs"]) >= 5:
            first = ", ".join([e["langName"]] + e["alsoLangs"][:2])
            e["title"] = f'{e["langName"]} and {len(e["alsoLangs"])} other long-tail languages'
            kind = "read and expressive speech" if e["task"] == "tts" else "conversational speech with verbatim transcripts"
            e["build"] = (f"Long-tail starter packs: a few hundred hours of {kind} per language, "
                          f"commercially licensed, starting with {first}")

    # Lab briefs.
    briefs = []
    by_org = defaultdict(list)
    for m in models:
        by_org[m["org"]].append(m)
    for org, name in WATCHLIST.items():
        ms = by_org.get(org, [])
        rec = [m for m in ms if parse_ts(m["createdAt"]) >= year_ago]
        latest = max((m["createdAt"] for m in ms), default=None)
        langs = Counter(l for m in rec for l in m["langs"])
        org_gaps = [c for c in ranked if org in {m["org"] for m in demand_models[(c["lang"], c["task"])]}][:3]
        briefs.append({
            "org": org, "name": name, "total": len(ms), "last12m": len(rec),
            "perMonth": round(len(rec) / 12, 1), "latest": latest,
            "tasks": Counter(m["task"] for m in rec).most_common(),
            "tasksAllTime": Counter(m["task"] for m in ms).most_common(),
            "langs": [[lang_name(l), n] for l, n in langs.most_common(8)],
            "nLangs": len(langs),
            "downloads30d": sum(m["downloads"] for m in ms),
            "top": [{"id": m["id"], "task": m["task"], "downloads": m["downloads"], "createdAt": m["createdAt"]}
                    for m in sorted(ms, key=lambda m: -m["downloads"])[:4]],
            "recentReleases": [{"id": m["id"], "task": m["task"], "createdAt": m["createdAt"]}
                               for m in sorted(rec, key=lambda m: m["createdAt"], reverse=True)[:4]],
            "gaps": [{"langName": g["langName"], "task": g["task"], "gap": g["gap"]} for g in org_gaps],
        })
    briefs.sort(key=lambda b: (-b["last12m"], -b["downloads30d"]))

    return cells, top, briefs


def this_week(models, datasets, now, prev):
    week_ago = now - timedelta(days=7)
    new_models = [m for m in models if parse_ts(m["createdAt"]) >= week_ago]
    new_sets = [d for d in datasets if d["createdAt"] and parse_ts(d["createdAt"]) >= week_ago]
    updated_sets = [d for d in datasets if d["lastModified"] and parse_ts(d["lastModified"]) >= week_ago
                    and d not in new_sets and d["downloads"] >= 1000]
    grouped = defaultdict(list)
    for m in sorted(new_models, key=lambda m: m["createdAt"], reverse=True):
        grouped[WATCHLIST[m["org"]]].append({"id": m["id"], "task": m["task"], "langs": m["langs"][:6],
                                             "createdAt": m["createdAt"], "downloads": m["downloads"]})
    out = {
        "since": week_ago.date().isoformat(),
        "labModels": grouped,
        "newDatasets": [{"id": d["id"], "license": d["license"], "commercial": d["commercial"],
                         "size": d["size"], "langs": d["langs"][:6], "tasks": d["tasks"], "createdAt": d["createdAt"]}
                        for d in sorted(new_sets, key=lambda d: (not d["commercial"], -d["downloads"]))[:25]],
        "nNewDatasets": len(new_sets),
        "nNewCommercial": sum(d["commercial"] for d in new_sets),
        "updatedDatasets": [{"id": d["id"], "license": d["license"], "downloads": d["downloads"],
                             "lastModified": d["lastModified"]}
                            for d in sorted(updated_sets, key=lambda d: -d["downloads"])[:10]],
        "diff": None,
    }
    if prev:
        prev_models = set(prev["models"])
        prev_dl = prev["labDownloads"]
        out["diff"] = {
            "baseline": prev["date"],
            "newModels": sorted(m["id"] for m in models if m["id"] not in prev_models),
            "newDatasets": len({d["id"] for d in datasets} - set(prev["datasets"])),
            "downloadMoves": [],
        }
    return out


def load_baseline(today):
    snap_dir = os.path.join(SITE, "snapshots")
    if not os.path.isdir(snap_dir):
        return None
    files = sorted(f for f in os.listdir(snap_dir) if f.endswith(".json") and f[:-5] < today)
    if not files:
        return None
    # Prefer the snapshot closest to 7 days ago; fall back to the oldest available.
    target = (datetime.fromisoformat(today) - timedelta(days=7)).date().isoformat()
    older = [f for f in files if f[:-5] <= target]
    pick = older[-1] if older else files[0]
    with open(os.path.join(snap_dir, pick)) as f:
        return json.load(f)


def main():
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    cache = os.environ.get("RADAR_CACHE")  # dev only: reuse a raw pull while tuning scores
    if cache and os.path.exists(cache):
        with open(cache) as f:
            models, datasets = json.load(f)
    else:
        models, datasets = fetch_models(), fetch_datasets()
        if cache:
            with open(cache, "w") as f:
                json.dump([models, datasets], f)
    for d in datasets:
        d["commercial"] = bool(d["license"] and COMMERCIAL.match(d["license"]))
        d["used"] = (d["downloads"] or 0) >= 10 or (d["likes"] or 0) >= 1
    cells, top, briefs = compute(models, datasets, now)
    prev = load_baseline(today)
    week = this_week(models, datasets, now, prev)

    lab_dl = Counter()
    for m in models:
        lab_dl[m["org"]] += m["downloads"] or 0
    if prev:
        week["diff"]["downloadMoves"] = sorted(
            ({"org": WATCHLIST[o], "now": lab_dl[o], "before": prev["labDownloads"].get(o, 0)} for o in WATCHLIST),
            key=lambda x: -(x["now"] - x["before"]))[:8]

    langs_in_grid = Counter()
    for c in cells:
        langs_in_grid[c["lang"]] += c["demand"]
    grid_langs = [l for l, _ in langs_in_grid.most_common(30)]

    data = {
        "generatedAt": now.isoformat(timespec="seconds"),
        "tasks": {k: v["label"] for k, v in GRID_TASKS.items()},
        "watchlist": WATCHLIST,
        "stats": {
            "labModels": len(models),
            "labModels12m": sum(1 for m in models if parse_ts(m["createdAt"]) >= now - timedelta(days=365)),
            "datasets": len(datasets),
            "commercialDatasets": sum(d["commercial"] for d in datasets),
            "datasetsNoLicence": sum(1 for d in datasets if not d["license"]),
            "datasetsNoLang": sum(1 for d in datasets if not d["langs"]),
        },
        "gridLangs": [{"code": l, "name": lang_name(l)} for l in grid_langs],
        "cells": [c for c in cells if c["lang"] in set(grid_langs)],
        "topGaps": top,
        "briefs": briefs,
        "week": week,
    }
    os.makedirs(os.path.join(SITE, "snapshots"), exist_ok=True)
    with open(os.path.join(SITE, "data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    snapshot = {
        "date": today,
        "models": sorted(m["id"] for m in models),
        "datasets": sorted(d["id"] for d in datasets),
        "labDownloads": dict(lab_dl),
        "topGaps": [{"lang": c["lang"], "task": c["task"], "gap": c["gap"]} for c in top],
    }
    with open(os.path.join(SITE, "snapshots", f"{today}.json"), "w") as f:
        json.dump(snapshot, f, separators=(",", ":"))
    print(json.dumps(data["stats"]), file=sys.stderr)
    for c in top:
        print(f'{c["gap"]:>7} {c["langName"]:<14} {c["task"]} models={c["nModels"]} labs={c["nLabs"]} '
              f'commercial={c["nCommercial"]} any={c["nAnyLicence"]}', file=sys.stderr)


if __name__ == "__main__":
    main()
