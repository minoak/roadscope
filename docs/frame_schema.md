# 계약: 프레임 결과 스키마 (엔진 → 시각화)

엔진(Claude 담당)은 영상 1개당 `runs_out/<영상이름>/frames.jsonl` 을 만든다. 한 줄이 한 프레임.
시각화(Codex 담당)는 원본 영상 + frames.jsonl 만 보고 그린다. 엔진 내부 객체(_state 등)에 의존하지 않는다.
스키마를 바꿔야 하면 직접 바꾸지 말고 보고의 "관리자 판단 필요"로 올린다.

## 파일 구성
- `runs_out/<영상이름>/meta.json`: {"source": 영상 경로, "fps_src": 원본 fps, "fps_eval": 평가 fps, "size": [W, H], "n_frames": 평가한 프레임 수, "models": {"lane": ..., "det": ..., "depth": ...}}
- `runs_out/<영상이름>/frames.jsonl`: 아래 레코드

## 프레임 레코드
```json
{
  "frame_idx": 120,              // 원본 영상 기준 프레임 번호
  "t_sec": 4.0,
  "bin": 2,                      // 0 안전, 1 주의, 2 위험, 3 치명
  "bin_label": "위험",
  "top_warning": "FCW",          // 없으면 null
  "top_reason": "정면 6.1m 접근",
  "warnings": {
    "FCW":  {"level": 2, "reason": "...", "track_id": 3, "distance_m": 6.1},
    "LDW":  {"level": 1, "reason": "...", "offset_ratio": 0.26},
    "BSW":  {"level": 0, "reason": "", "track_id": null, "side": null},
    "HEAD": {"level": 0, "reason": "", "track_id": null}
  },
  "lane_measurable": true,
  "lanes": {"left": [x1, y1, x2, y2], "right": [x1, y1, x2, y2]},   // 없으면 null
  "vp": [x, y, conf],                                                // 없으면 null
  "corridor": {"critical": [[x, y], ...], "danger": [[x, y], ...], "caution": [[x, y], ...]},  // 없으면 null
  "objects": [
    {"track_id": 3, "cls": "car", "conf": 0.91, "bbox": [x0, y0, x1, y1],
     "lateral_m": -3.2,   // 옆 간격 m (VP 기준, 왼쪽 음수, 소수 1자리). 계산 불가면 null. vp=null 이면 기본 VP(화면 중앙) 기준
     "distance_m": 6.1, "zone_sev": 2, "direction": "front", "warning": "FCW",   // warning 없으면 null
     "source": "yolo",    // yolo | yolo_lowconf (저확신 큰·가까운 박스, cls "unknown_near") | mask (Moveable 마스크 덩어리, cls "mask_object") | ghost (트래커가 놓친 track 의 예측 박스)
     "ghost": false}      // source 가 ghost 면 true. yolo 외 객체는 거리 기반 근접 판정에만 쓰인다
  ],
  "d_front_m": 6.1, "d_left_m": null, "d_right_m": 14.2
}
```
- 좌표는 원본 영상 픽셀 기준. direction 은 "front" | "left" | "right".
- 거리 단위 m, 소수 1자리. 측정 불가는 null.

## 자동차 모드 (`--vehicle car`, t20 추가)
오토바이 모드 레코드는 위와 같고 아래 필드가 없다. 자동차 모드는 위 필드를 모두 그대로 쓰고 아래를 더한다.
기본 평가 fps 는 자동차 10 (오토바이 5). 수식과 기준값은 `geom_car.py`, 결정은 DECISIONS 2026-10-09.

### 기존 필드의 자동차 모드 계산 방법 (의미는 같음)
- `vp`: 보정 단계에서 구간별로 고정한 [소실점 x, 수평선 y, 신뢰도]. 신뢰도 0.9 high / 0.7 medium / 0.4 low. 보정 실패면 null
- `corridor`: 내 차로 |X| ≤ 2.5 m 영역. 경계선은 소실점을 지나고, 구역 윗변은 critical 3 m, danger 5 m, caution 10 m 거리의 행
- `objects[].distance_m`: 바닥 위치 거리와 차폭 거리 중 가까운 쪽 (아래 `dist_geo`). Depth Pro 는 쓰지 않는다
- `objects[].lateral_m`: 옆 거리 X = H·(박스 바닥 중심 x − 소실점 x) / (박스 바닥 y − 수평선 y)
- `objects[].direction`: |X| ≤ 2.5 m 면 "front", 아니면 X 부호로 "left" / "right"

### 추가 필드
```json
{
  "calib_seg": 0,                 // meta.json car_calib 의 구간 번호
  "depth_guard": {                // 박스 없는 근접 물체 보험 (Depth Pro, 0.4초 간격). Depth Pro 를 안 돌린 프레임은 null.
                                  // 보정 신뢰도 low/failed 구간은 보험을 끄고 Depth Pro 도 안 돌린다 (전부 null)
    "status": "clear",            // clear | alarm | no_scale (도로가 안 보여 축척 없음) | no_geom (보정 실패)
    "distance_m": null,           // alarm 일 때 물체 거리 (도로 픽셀로 축척을 맞춘 값)
    "frac": 0.0, "scale": 15.1, "scale_src": "frame",
    "held_from_t": null           // 직전 Depth Pro 프레임의 alarm 을 이어받았으면 그 시각
  },
  "warnings": {
    "FCW": {"level": 2, "reason": "정면 4.1m", "track_id": 4, "distance_m": 4.1,
            "cause": "distance",  // distance | area_no_distance (거리 없는 정면 객체가 화면 25% 초과) | depth_guard | ghost_hold. level 0 이면 null
            "ttc_s": 2.4},        // FCW 대상 객체의 TTC (기록만), 두 방법이 모두 있을 때 큰 값. 아니면 null
    "LDW": {"level": 0, "reason": "", "offset_ratio": 0.08,
            "ref": {              // t40: LDW 를 잰 위치 (구간의 차선 영점 행 y_ref). 측정 불가 프레임은 null
              "y": 538,           // y_ref (px)
              "x_left": 431.5, "x_right": 790.2,   // 그 행에서 실제로 보인 좌우 차선 위치 (차선 픽셀 평균 x, 연장 안 함)
              "ego_x": 602.0,     // 내 차 기준 열 = 소실점 x − b/k (b = 차로 안 평소 위치)
              "ego_w_px": 168.2,  // 내 차 폭 1.86 m 를 y_ref 에서 px 로
              "lane_w_m": 3.45,   // 이 프레임 차로 폭 (m)
              "gap_left_m": 0.59, "gap_right_m": 1.01}}   // 차체 옆면 (기준 열 ± 0.93 m) 에서 각 차선까지 (m). 음수 = 차체가 선 위. 기록만
  },
  "objects": [
    {"in_lane": true,             // |X| ≤ 2.5 m. X 를 못 구하면 null
     "dist_geo": {                // 유령 (ghost) 은 예측 박스의 기하. 기하로 못 구하면 null (외삽 거리 사용)
       "ground_m": 12.3,          // 바닥 위치 거리. 바닥이 보닛·화면 끝에 가렸거나 수평선 보정 실패면 null
       "width_m": 11.8,           // 차폭 거리 (car 1.86, bus/truck 2.5 m 등). 화면 좌우 끝에 잘린 박스는 null
       "width_implied_m": 1.95,   // 바닥 거리로 역산한 차폭. 1.6~2.5 밖이면 차폭 거리를 경보에 안 쓴다
       "used": "width",           // distance_m 에 쓴 값 (셋 중 가까운 쪽): ground | width | hood_bound (바닥이 가린 박스의 보닛 경계까지 거리 = 상한) | null
       "gated": ["width:차폭 범위 밖"]},
     "ttc": {"scale_s": 2.1, "ground_s": 2.6},   // 박스 폭 / 바닥 위치 TTC (초, 0.5초 창, 기록만). 계산 못 하면 null
     "interior": false}           // t27·t29: true 면 내 차 영역 (보닛·대시보드·A필러·반사 등) 으로 판정에서 뺀 검출 박스.
                                  // 기록만: track_id·distance_m·lateral_m·dist_geo·ttc 는 null, warning null, zone_sev 0
  ]
}
```
- 자동차 모드 차선, 주행 가능 영역은 YOLOPv2 (t29). lane_focus 는 오토바이 모드만 쓴다 (자동차 모드에는 Moveable 마스크 보조 박스 `source: "mask"` 가 없다)
- 내 차 영역 (사전 점검, t29): Mask2Former Ego Vehicle, Car Mount (보정 구간 5프레임 과반) ∪ 보닛 띠 ∪ 구간 내내 정지한 테두리 영역 ∪ 반복 검출 박스.
  박스 면적의 50% (되살린 박스 30%) 이상이 그 안이면 판정에서 빼고 `interior: true` 로 남긴다. 사전 점검이 실패하면 지우지 않는다
- TTC 는 판정에 쓰지 않고 기록만 한다 (DECISIONS 2026-10-09: 위험도는 그 순간의 장면으로 평가). `--no-ttc` 는 계산, 기록을 끈다
- 박스 없는 근접 물체 보험: alarm 이면 FCW 위험(2) (`cause` = depth_guard, `track_id` null)
- 보닛 추정 실패 (`hood_src` = default): 기본 비율 띠로 박스를 지우지 않고, 박스 바닥이 가렸는지는 화면 아래 끝으로 판단한다 (t24)
- LDW 는 그 프레임에서 보이는 것만으로 정한다 (두 모드 모두. t10 의 0.8초 이어받기는 t22 에서 뺐다: `reason` 에 "(직전 판정 유지)" 가 더 나오지 않는다).
  자동차 모드 (t40): 구간의 차선 영점 행 y_ref (meta `car_calib[].lane_baseline`) 에서만 잰다. 0.95H 고정 행은 쓰지 않는다.
  y_ref 에서 내 차 기준 열 좌우로 가장 가까운 검증된 선 두 개가 그 행에서 실제로 보여야 하고 (y_ref ± 2행 차선 픽셀),
  안 보이면 연장하지 않고 측정 불가 (`lane_measurable` false, `reason` "[측정 불가] 기준 행에서 차선 안 보임" /
  "기준 행 차선 한쪽 없음" / "차선 없음" / "차선 수렴 안 됨" / "차선 폭 이상"). 영점 실패 구간은 모든 프레임이
  "[측정 불가] 차선 영점 실패". `offset_ratio` = |기준 열 − 차로 중심| / 차로 폭 (y_ref 에서). 차선 위 주행 (L1) 기준은
  t32 와 같은 30% 초과. 15~30% 는 비율만 기록
  t44 (기본값): 차선 불신 구간 (meta `car_calib[].lane_trust.trusted` false, 영점 실패 포함) 은 모든 프레임이
  "[측정 불가] 차선 신뢰 낮음 (차로 폭 X.Xm | 검산 차로 폭 X.Xm | 영점 실패 | 검산 불가)": 경고 없음. `--no-lane-trust` 면 위 t40 문구 그대로

### meta.json (자동차 모드)
- `car_calib`: 구간별 보정값 리스트. `horizon_y`, `horizon_ratio`, `vp_x`, `horizon_src` (foe | lane | vehicle_regression | default), `horizon_conf` (high | medium | low | failed), `hood_row`, `hood_ratio`, `hood_src` (flow | flow_motion_edge (정지점이 없어 움직이는 점의 아래 끝) | flow_none_visible | mask2former (t29) | interior_flow, interior_motion_edge (구간 전체 광류, t27) | manual | default), `cam_h_m`, `cam_h_src` (vehicle | lane | vehicle+lane | vehicle(lane_disagree) | lane(vehicle_disagree) | default), `f_px`, `window_sec` (보정에 쓴 구간), `start_frame`, `end_frame`, `detail` (방법별 원값과 교차 확인)
- `car_calib[].vp_x_src`: foe | lane | center | center(vp_x_out_of_range) (소실점 x 가 화면 폭 20~80% 밖이라 가운데로, 신뢰도 low)
- `car_calib[].interior`: 내 차 영역 사전 점검. `status` (ok | no_motion = 내 차 움직임이 부족해 정지 영역, 반복 검출 없음),
  `frac` (화면 중 내 차 영역 비율), `m2f` {status, n_frames, frac}, `hood_frac`, `edge_static_frac`,
  `static_boxes` (반복 검출 박스 후보: bbox, cls, n, static_frac, moving_cells, 탈락이면 `rejected`), `png` (같은 폴더의
  `interior_seg<구간>.png`, 흰색 = 내 차 영역)
- `car_calib[].lane_baseline` (t40 차선 영점. t32 의 차로 안 평소 위치를 y_ref 에서 다시 잰 것, 키 이름은 유지):
  사전 점검에서 구간의 YOLOPv2 차선 마스크 (보닛 띠, 내 차 영역의 차선 픽셀 제거) 와 고정 소실점으로 구한다. LDW 만 쓴다
  (옆 거리 X, BSW, FCW, 차선 고르기는 그대로).
  - `ldw_measurable` (false = 영점 실패 → 이 구간 LDW 전부 측정 불가), `reason` (ok | no_horizon | insufficient_lane_pairs |
    no_visible_row | insufficient_valid_lane_pairs)
  - `y_ref`, `y_ref_ratio` (기준 행 px, H 비율): 좌우 실제 선이 다 잡힌 프레임 (`pair_frames`) 중 50% 이상에서 두 선이 함께 보이는
    가장 아래 행 (`y_visible`) − 0.02H. 수평선 + 0.08H 보다 위는 후보가 아니다. `vis_frac` (y_ref 에서 두 선이 함께 보인 비율)
  - `lane_w_px`, `lane_w_m` (y_ref 차로 폭 중앙값, m = px × 카메라 높이 / (y_ref − 수평선)), `m_per_px` (그 비율 k),
    `ego_w_px` (내 차 폭 1.86 m 를 y_ref 에서 px 로 = 1.86 / k)
  - `baseline_m` (차로 안 평소 위치 b, m, 오른쪽 +, ±0.5 상한 후. 실패면 null), `raw_median_m`, `capped`,
    `ldw_shift_px` (LDW 기준 열 이동 = −b/k px)
  - `evidence_frames` (y_ref 에서 두 선이 보여 폭, 중심을 잰 프레임), `examined_frames` (본 프레임 = 보정 구간 + 그 뒤 평가 프레임),
    `required_frames` (= max(30, ceil(본 프레임 × 10%)). 쌍 프레임, 근거가 이보다 적으면 실패)
  - `conf` (high | low | failed): 차로 폭이 2.75~3.75 m 밖이면 low, `conf_reasons` 에 이유. `suspect` (low 일 때 의심 쪽):
    cam_h = 카메라 높이를 못 쟀거나 (default) 차량, 차선 높이가 어긋남 (disagree), lane = 높이 근거는 일관 → 차선 쌍, 검출 쪽.
    `implied_cam_h_m` (차로 폭이 3.5 m 가 되는 카메라 높이, 참고값), `check_independent` (false = 카메라 높이를 차선 비율로 구해
    폭 검산이 순환)
- `car_calib[].lane_trust` (t44 차선 신뢰, DECISIONS 2026-10-10. `--no-lane-trust` 면 키 없음):
  - `trusted` (true/false), `reason` (ok | lane_width_out_of_range | lane_width_out_of_range_check |
    lane_zero_failed:<lane_baseline.reason> | no_lane_free_horizon (차선 없이 구한 수평선이 없어 검산 불가) |
    y_ref_near_lane_free_horizon)
  - 판정에 쓴 y_ref 차로 폭 두 개가 모두 `range_m` [2.0, 4.0] 안이어야 신뢰:
    `lane_w_m` = 차선을 쓴 보정값으로 잰 폭 (= `lane_baseline.lane_w_m`, t40 값. 밖이면 lane_width_out_of_range),
    `lane_w_m_check` = 순환 검산 폭 (밖이면 lane_width_out_of_range_check). `check` 의 수평선, 높이로 잰다:
    수평선은 늘 차선 없는 것 (FOE → 차량 박스 회귀), 높이는 차량 박스 track 3개 이상이면 그 값 (`cam_h_independent` true),
    아니면 차선을 쓴 보정값의 높이 그대로 (false: 높이 순환은 남는다. 기본값 1.4 m와 track 1~2 개 값은 심판으로 쓰지 않는다).
    `check`: horizon_y, horizon_ok, horizon_src, cam_h_m, cam_h_src, cam_h_independent, vehicle_tracks.
    `lane_w_px`, `y_ref` (영점 값 그대로)
  - `exclude_lanes` (t48): true = 폭이 범위 밖 (lane_width_out_of_range | lane_width_out_of_range_check) 이라 차선을 모든 계산에서 뺀다.
    false 인 불신 (영점 실패, 검산 불가) 은 LDW 만 측정 불가이고 보정값은 t40 그대로 (DECISIONS 2026-10-10)
  - `applied`: true 면 (= exclude_lanes) 이 구간 보정값 (위 `car_calib` 의 수평선, 소실점, 카메라 높이, 실내 마스크) 이 차선을 뺀 것
    (FOE, 차량 박스 회귀, 차량 높이, 모두 없으면 기본값) 이고, 근접 보험 축척도 차선 픽셀 (class 3) 없이 도로 (class 2) 만 쓴다. `lane_calib` = 바꾸기 전 차선을 쓴 값 (horizon_y, horizon_ok, horizon_src,
    horizon_conf, vp_x, vp_x_src, cam_h_m, cam_h_src), `calib_changed` = 그 둘이 다른지. 이때 `detail.lane` 에 `excluded`
    "lane_untrusted" 가 붙는다. `lane_baseline` 은 차선을 쓴 보정값으로 잰 영점을 기록만 한다
- `car`: 실행 설정과 시간 (`calib_sec`, `judge_sec`, `depth_every_sec`, `ttc` = "record_only" 또는 false 등).
  `lane_trust_range_m` (t44, 차선 신뢰 범위 [2.0, 4.0]. `--no-lane-trust` 면 키 없음)
- `hood_ratio`: 첫 구간에서 실제로 쓴 보닛 비율, `hood_ratio_src`: 그 출처
