"""시험 세트용 Nexar 영상 받기. 개발용 30개(runs_in/nexar)와 증류용(runs_in/distill/manifest.csv)은 제외.
사고/정상 반반, 각 그룹 안에서 조명(light_conditions) 비율은 원 데이터 비율을 따른다(층화 추출).
결과: runs_in/testset/manifest.csv (영상은 HF 캐시), runs_in/testset/labels.csv (eval_nexar 형식)
사용: python tools/data/get_nexar_testset.py [개수=100]
"""
import csv, random, sys
from collections import defaultdict
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "nexar-ai/nexar_collision_prediction"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs_in" / "testset"
OUT.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

used = {p.stem.split("_", 1)[1] for p in (ROOT / "runs_in" / "nexar").glob("*.mp4")}
dm = ROOT / "runs_in" / "distill" / "manifest.csv"
if dm.exists():
    used |= {Path(r["file_name"]).stem for r in csv.DictReader(open(dm, encoding="utf-8"))}

random.seed(77)
pick = []
for split in ("positive", "negative"):
    meta = hf_hub_download(REPO, f"train/{split}/metadata.csv", repo_type="dataset")
    rows = [r for r in csv.DictReader(open(meta, encoding="utf-8")) if Path(r["file_name"]).stem not in used]
    by = defaultdict(list)
    for r in rows:
        r["split"] = split
        by[r.get("light_conditions") or "Unknown"].append(r)
    want = n // 2
    total = sum(len(v) for v in by.values())
    for light, group in sorted(by.items()):
        k = round(want * len(group) / total)
        pick += random.sample(group, min(k, len(group)))
print("picked", len(pick), flush=True)

man_rows, lab_rows = [], []
for i, r in enumerate(pick):
    try:
        p = hf_hub_download(REPO, f"train/{r['split']}/{r['file_name']}", repo_type="dataset")
    except Exception as e:
        print("fail", r["file_name"], e, flush=True)
        continue
    name = f"{r['split'][:3]}_{r['file_name']}"
    man_rows.append({"file": name, "path": p})
    lab_rows.append({"file": name, "label": 1 if r["split"] == "positive" else 0,
                     "time_of_event": r.get("time_of_event", ""), "time_of_alert": r.get("time_of_alert", ""),
                     "light": r.get("light_conditions"), "weather": r.get("weather"), "scene": r.get("scene")})
    print(i + 1, "/", len(pick), name, flush=True)

with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file", "path"]); w.writeheader(); w.writerows(man_rows)
with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(lab_rows[0])); w.writeheader(); w.writerows(lab_rows)
print("DONE", len(man_rows))
