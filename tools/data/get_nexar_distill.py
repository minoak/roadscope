"""증류 학습용 Nexar 영상 받기. 평가용 30개(runs_in/nexar)는 제외한다.
영상은 HF 캐시에만 두고 runs_in/distill/manifest.csv에 경로를 적는다.
사용: python tools/data/get_nexar_distill.py [개수=270]
"""
import csv, random, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "nexar-ai/nexar_collision_prediction"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs_in" / "distill"
OUT.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 270

held = {p.stem.split("_", 1)[1] for p in (ROOT / "runs_in" / "nexar").glob("*.mp4")}
rows = []
for split in ("negative", "positive"):
    meta = hf_hub_download(REPO, f"train/{split}/metadata.csv", repo_type="dataset")
    for r in csv.DictReader(open(meta, encoding="utf-8")):
        if Path(r["file_name"]).stem in held:
            continue
        r["split"] = split
        rows.append(r)
random.seed(2026)
pick = random.sample(rows, n)
man = OUT / "manifest.csv"
done = set()
if man.exists():
    done = {r["file_name"] for r in csv.DictReader(open(man, encoding="utf-8"))}
f = open(man, "a", newline="", encoding="utf-8")
w = csv.DictWriter(f, fieldnames=["file_name", "split", "light", "scene", "path"])
if not done:
    w.writeheader()
for i, r in enumerate(pick):
    if r["file_name"] in done:
        continue
    try:
        p = hf_hub_download(REPO, f"train/{r['split']}/{r['file_name']}", repo_type="dataset")
    except Exception as e:
        print("fail", r["file_name"], e, flush=True)
        continue
    w.writerow({"file_name": r["file_name"], "split": r["split"], "light": r.get("light_conditions"),
                "scene": r.get("scene"), "path": p})
    f.flush()
    print(i + 1, "/", n, r["file_name"], flush=True)
f.close()
print("DONE")
