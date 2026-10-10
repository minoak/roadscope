"""t41: 저장된 자동차 모드 결과 (frames.jsonl + meta.json) 에 LDW 방식 VB 를 오프라인으로 적용한다.

VB = 차선 영점 신뢰도가 high 가 아닌 구간 (지금 규칙: y_ref 차로 폭이 geom_car.LANE_ZERO_W_RANGE_M 밖) 의 LDW 를
전부 측정 불가로 둔다. 엔진 --ldw-variant VB (engine._ldw_car 의 VB 분기) 와 같은 레코드를 만든다 — 영점 자체는
바꾸지 않으므로 저장된 meta.json 의 car_calib[].lane_baseline 만으로 결정된다. LDW 외 경고 (FCW·BSW·HEAD) 는
그대로 두고 bin·top_warning·top_reason 만 다시 계산한다. 모델·영상·GPU 를 쓰지 않는다.

    python -B tools/evaluate/rejudge_ldw_vb.py <입력 결과 폴더> <출력 폴더> [영상 ...]
      예) runs_out/nexar_t40 → runs_out/ldw_variants_t41/VB  (V0 + VB = VB)
          runs_out/ldw_variants_t41/runs/VA → runs_out/ldw_variants_t41/VAB  (VA + VB = VAB)
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from copy import deepcopy
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
from roadscope import rules as mvp   # noqa: E402  (V5State 만 쓴다)


def low_conf_reason(lb):
    """VB 로 꺼지는 구간이면 LDW 사유, 아니면 None. 영점 실패 구간은 V0 에서 이미 측정 불가 ('차선 영점 실패')."""
    if not lb or not lb.get('ldw_measurable') or lb.get('conf') == 'high':
        return None
    return f'[측정 불가] 차선 영점 신뢰 낮음 (차로 폭 {lb["lane_w_m"]:.2f}m)'


def refresh_summary(rec):
    """엔진 to_record 와 같은 규칙 (mvp.V5State: 최고 레벨, 동률이면 FCW·LDW·BSW·HEAD 순 첫 번째)."""
    w = rec['warnings']
    state = mvp.V5State(**{k.lower(): mvp.WarningResult(k, w[k]['level'], w[k]['reason'])
                           for k in ('FCW', 'LDW', 'BSW', 'HEAD')})
    top = state.top_warning
    rec.update(bin=state.max_severity, bin_label=state.bin,
               top_warning=top.name if top else None, top_reason=top.reason if top else '')


def apply_vb(rec, lane_baselines):
    """프레임 레코드 하나 → VB 레코드 (새 dict). lane_baselines[구간 번호] = meta car_calib[].lane_baseline."""
    out = deepcopy(rec)
    reason = low_conf_reason(lane_baselines[rec.get('calib_seg') or 0])
    if reason is None:
        return out
    out['warnings']['LDW'] = {'level': 0, 'reason': reason, 'offset_ratio': None, 'ref': None}
    out['lane_measurable'] = False
    refresh_summary(out)
    return out


def run(src, dst, names=None):
    src, dst = Path(src), Path(dst)
    names = names or sorted(p.name for p in src.iterdir() if (p / 'meta.json').exists())
    for n in names:
        meta = json.loads((src / n / 'meta.json').read_text(encoding='utf-8'))
        if meta.get('vehicle') != 'car':
            raise ValueError(f'{n}: 자동차 모드 결과만 된다')
        lbs = {c['segment']: c.get('lane_baseline') or {} for c in meta['car_calib']}
        recs = [json.loads(s) for s in (src / n / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        rows = [apply_vb(r, lbs) for r in recs]
        for a, b in zip(recs, rows):   # LDW 외 경고는 그대로여야 한다
            assert all(a['warnings'][k] == b['warnings'][k] for k in ('FCW', 'BSW', 'HEAD')), n
        (dst / n).mkdir(parents=True, exist_ok=True)
        with open(dst / n / 'frames.jsonl', 'w', encoding='utf-8', newline='\n') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        shutil.copyfile(src / n / 'meta.json', dst / n / 'meta.json')   # 영점은 그대로 (meta 바이트 같음)
        print(n, sum(a != b for a, b in zip(recs, rows)), '/', len(rows), '프레임 바뀜', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
