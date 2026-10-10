"""Nexar 충돌 예측 데이터셋에서 테스트용 영상 몇 개만 받는다.

사전 조건: Hugging Face 에서 약관 동의 + 이 PC 에서 `hf auth login` 완료.
결과: runs_in/nexar/<id>.mp4, runs_in/nexar/labels.csv
"""
import csv, shutil, random
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "nexar-ai/nexar_collision_prediction"
OUT = Path(__file__).resolve().parents[2] / "runs_in" / "nexar"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(7)


def meta(split):
    p = hf_hub_download(REPO, f"train/{split}/metadata.csv", repo_type="dataset")
    return list(csv.DictReader(open(p, encoding="utf-8")))


pos, neg = meta("positive"), meta("negative")
pick = []
pick += random.sample([r for r in pos if r["light_conditions"] == "Dark"], 3)
pick += random.sample([r for r in pos if r["light_conditions"] == "Normal"], 4)
pick += random.sample([r for r in neg if r["light_conditions"] == "Dark"], 1)
pick += random.sample([r for r in neg if r["light_conditions"] == "Normal"], 2)

rows = []
for r in pick:
    split = "positive" if r["time_of_event"] else "negative"
    src = hf_hub_download(REPO, f"train/{split}/{r['file_name']}", repo_type="dataset")
    dst = OUT / f"{split[:3]}_{r['file_name']}"
    shutil.copy(src, dst)
    rows.append({"file": dst.name, "label": 1 if split == "positive" else 0,
                 "time_of_event": r["time_of_event"], "time_of_alert": r["time_of_alert"],
                 "light": r["light_conditions"], "weather": r["weather"], "scene": r["scene"]})
    print("got", dst.name, r["light_conditions"], r["time_of_event"] or "-")

with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("done", len(rows), "->", OUT)
