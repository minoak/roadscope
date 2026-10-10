"""t44 차선 신뢰 (영점 차로 폭 2.0~4.0 m, 밖이면 차선을 LDW·수평선·카메라 높이에서 뺀다) 단위 테스트.
모델·영상·GPU 없이 합성 보정값·차선으로 확인한다.

python -m unittest tests.test_lane_trust -v
"""
import unittest
from unittest.mock import patch

import numpy as np

from roadscope import engine
from roadscope import geom_car as gc
from tests.test_ldw_variants import Globals, LEFT, RIGHT, VP_X, HORIZON, W, H, calib, lane_mask

FOE = {'n_ok': 30, 'x': 640.0, 'y': 300.0, 'iqr_x': 10.0, 'iqr_y': 5.0, 'hood_row': 680, 'hood_status': 'flow'}
LANE = {'n_frames': 40, 'x': 650.0, 'y': 310.0, 'iqr_y': 5.0, 'cam_h': 1.5, 'cam_h_ok': True}


def decide(foe, lane, reg_y=None, veh_h=None, n_tracks=1):
    reg = (lambda xv: {'y': reg_y}) if reg_y is not None else (lambda xv: {})
    veh = ((lambda yh, xv: {'cam_h': veh_h, 'cam_h_ok': True, 'n_tracks': n_tracks}) if veh_h is not None
           else (lambda yh, xv: {}))
    return gc.decide_calib(W, H, dict(foe), dict(lane), reg_fn=reg, veh_fn=veh, hood_row=680, hood_src='flow',
                           window=(0.0, 8.0))


def zero(w_px=300.0, y_ref=600, lane_w_m=3.5):
    """lane_zero 성공 결과 (필요한 키만). lane_w_m = 차선을 쓴 보정값으로 잰 폭 (t40 값)."""
    return {'ldw_measurable': True, 'reason': 'ok', 'y_ref': y_ref, 'lane_w_px': w_px, 'lane_w_m': lane_w_m}


def lane_cal(cam_h=1.5):
    """차선을 쓴 보정값 — 높이가 차선 비율에서 나온 경우."""
    c = calib(cam_h=cam_h)
    c.cam_h_src = 'lane'
    return c


def free(cam_h=1.4, n_tracks=None, horizon=HORIZON):
    """차선을 뺀 보정값. n_tracks 가 있으면 높이 = 차량 박스 (그 track 수), 없으면 기본값."""
    c = calib(cam_h=cam_h)
    c.horizon_y = horizon
    c.cam_h_src = 'vehicle' if n_tracks else 'default'
    c.detail = {'vehicle_height': {'n_tracks': n_tracks}} if n_tracks else {}
    return c


class LaneTrustTest(unittest.TestCase):
    def test_strong_vehicle_height_breaks_circularity(self):
        # 차선 높이로 잰 폭은 3.5 m (순환이면 늘 이 근처) 라도, 차량 track 3개 높이 1.0 m 로 다시 재면
        # 수평선 300, y_ref 600 → 300px × 1.0 / 300 = 1.0 m → 불신
        lt = gc.lane_trust(zero(w_px=300.0, lane_w_m=3.5), lane_cal(1.5), free(cam_h=1.0, n_tracks=3))
        self.assertEqual((lt['trusted'], lt['reason'], lt['lane_w_m'], lt['lane_w_m_check']),
                         (False, 'lane_width_out_of_range_check', 3.5, 1.0))
        self.assertEqual((lt['check']['cam_h_m'], lt['check']['cam_h_independent']), (1.0, True))
        lt = gc.lane_trust(zero(w_px=300.0, lane_w_m=3.5), lane_cal(1.5), free(cam_h=3.0, n_tracks=3))
        self.assertEqual((lt['trusted'], lt['reason'], lt['lane_w_m_check']), (True, 'ok', 3.0))

    def test_weak_or_default_height_not_used_as_judge(self):
        # track 1~2 개 차량 높이 (pos_00077 세단 2.82 m) 나 기본값 1.4 m (트럭 pos_00283) 로는 차선을 심판하지 않는다:
        # 높이는 차선을 쓴 보정값 그대로 (1.5 m → 300px = 1.5 m … 여기서는 600px = 3.0 m)
        for ref in (free(cam_h=2.82, n_tracks=1), free(cam_h=1.4)):
            lt = gc.lane_trust(zero(w_px=600.0, lane_w_m=3.0), lane_cal(1.5), ref)
            self.assertEqual((lt['trusted'], lt['lane_w_m_check']), (True, 3.0))
            self.assertEqual((lt['check']['cam_h_m'], lt['check']['cam_h_src'], lt['check']['cam_h_independent']),
                             (1.5, 'lane', False))

    def test_lane_free_horizon_always_used(self):
        # 수평선은 늘 차선 없는 것: 차선 교점 300 이 아니라 회귀 200 → 300px × 1.5 / 400 = 1.125 m → 불신
        lt = gc.lane_trust(zero(w_px=300.0, lane_w_m=3.5), lane_cal(1.5), free(horizon=200))
        self.assertEqual((lt['trusted'], lt['reason'], lt['lane_w_m_check']),
                         (False, 'lane_width_out_of_range_check', 1.125))

    def test_lane_calib_width_checked_first(self):
        # 차선을 쓴 보정값으로 5.59 m (pos_00411 꼴) → 검산 폭이 맞아도 그 보정값을 그대로 쓸 수 없다
        lt = gc.lane_trust(zero(w_px=600.0, lane_w_m=5.59), lane_cal(1.5), free(cam_h=1.5, n_tracks=3))
        self.assertEqual((lt['trusted'], lt['reason'], lt['lane_w_m'], lt['lane_w_m_check']),
                         (False, 'lane_width_out_of_range', 5.59, 3.0))

    def test_range_is_inclusive_2_to_4(self):
        self.assertEqual(gc.LANE_TRUST_W_RANGE_M, (2.0, 4.0))
        for w, ok in ((1.99, False), (2.0, True), (4.0, True), (4.01, False)):
            lt = gc.lane_trust(zero(w_px=300.0, lane_w_m=w), lane_cal(3.0), free())   # 검산 3.0 m
            self.assertEqual(lt['trusted'], ok, w)
            lt = gc.lane_trust(zero(w_px=w * 100, lane_w_m=3.0), lane_cal(3.0), free())   # 1px = 0.01 m
            self.assertEqual(lt['trusted'], ok, w)

    def test_zero_failed_is_untrusted_ldw_only(self):
        # t48: 영점 실패는 LDW 만 끈다 — 보정값에서 차선을 빼지 않는다 (exclude_lanes false)
        lt = gc.lane_trust({'ldw_measurable': False, 'reason': 'insufficient_lane_pairs'}, lane_cal(), free())
        self.assertEqual((lt['trusted'], lt['exclude_lanes'], lt['reason'], lt['lane_w_m'], lt['lane_w_m_check']),
                         (False, False, 'lane_zero_failed:insufficient_lane_pairs', None, None))
        self.assertEqual((lambda t: (t['trusted'], t['exclude_lanes']))(gc.lane_trust(None, lane_cal(), free())),
                         (False, False))

    def test_unverifiable_is_untrusted_ldw_only(self):
        lt = gc.lane_trust(zero(), lane_cal(), gc.default_calib(W, H, 680, 'flow'))
        self.assertEqual((lt['trusted'], lt['exclude_lanes'], lt['reason']), (False, False, 'no_lane_free_horizon'))
        lt = gc.lane_trust(zero(y_ref=int(HORIZON + 0.05 * H)), lane_cal(), free())
        self.assertEqual((lt['trusted'], lt['exclude_lanes'], lt['reason']),
                         (False, False, 'y_ref_near_lane_free_horizon'))

    def test_width_out_of_range_excludes_even_if_unverifiable(self):
        # 영점 폭이 범위 밖이면 검산 가능 여부와 상관없이 차선을 뺀다
        lt = gc.lane_trust(zero(lane_w_m=5.0), lane_cal(), gc.default_calib(W, H, 680, 'flow'))
        self.assertEqual((lt['trusted'], lt['exclude_lanes'], lt['reason'], lt['lane_w_m_check']),
                         (False, True, 'lane_width_out_of_range', None))

    def test_exclude_flag_for_width_reasons(self):
        for zl, ref, reason in ((zero(lane_w_m=5.0), free(), 'lane_width_out_of_range'),
                                (zero(w_px=300.0), free(cam_h=1.0, n_tracks=3), 'lane_width_out_of_range_check')):
            lt = gc.lane_trust(zl, lane_cal(1.5), ref)
            self.assertEqual((lt['reason'], lt['exclude_lanes']), (reason, True))
        self.assertFalse(gc.lane_trust(zero(w_px=600.0), lane_cal(1.5), free())['exclude_lanes'])   # 신뢰


class DecideWithoutLaneTest(unittest.TestCase):
    """decide_calib 에 차선 결과를 빈 dict 로 주면 차선 교점·차선 높이가 빠진다 (engine 의 lane_free)."""

    def test_lane_horizon_replaced_by_regression(self):
        no_foe = {'n_ok': 0, 'hood_row': 680, 'hood_status': 'flow'}
        full, free = decide(no_foe, LANE, reg_y=320.0), decide(no_foe, {}, reg_y=320.0)
        self.assertEqual((full.horizon_src, full.horizon_y, full.vp_x), ('lane', 310.0, 650.0))
        self.assertEqual((free.horizon_src, free.horizon_y, free.vp_x, free.horizon_conf),
                         ('vehicle_regression', 320.0, W / 2, 'low'))

    def test_lane_only_falls_back_to_default(self):
        no_foe = {'n_ok': 0, 'hood_row': 680, 'hood_status': 'flow'}
        full, free = decide(no_foe, LANE), decide(no_foe, {})
        self.assertEqual((full.horizon_src, full.cam_h_src), ('lane', 'lane'))
        self.assertEqual((free.horizon_ok, free.horizon_src, free.horizon_conf, free.cam_h, free.cam_h_src),
                         (False, 'default', 'failed', gc.CAM_H_DEFAULT_M, 'default'))

    def test_lane_height_removed(self):
        full, free = decide(FOE, LANE, veh_h=1.3), decide(FOE, {}, veh_h=1.3)
        self.assertEqual((full.cam_h_src, round(full.cam_h, 3)), ('vehicle+lane', 1.4))
        self.assertEqual((free.cam_h_src, free.cam_h), ('vehicle', 1.3))
        free = decide(FOE, {})
        self.assertEqual((free.cam_h_src, free.cam_h), ('default', gc.CAM_H_DEFAULT_M))

    def test_lane_agreement_no_longer_raises_confidence(self):
        self.assertEqual(decide(FOE, LANE).horizon_conf, 'high')     # FOE 300 ↔ 차선 310: 25px 안
        self.assertEqual(decide(FOE, {}).horizon_conf, 'medium')     # 비교할 것 없음
        self.assertEqual(decide(FOE, {}, reg_y=360.0).horizon_conf, 'low')   # 회귀만 남고 어긋남


class ApplyLaneTrustTest(Globals):
    def make(self, w_px, lane_free, lane_w_m=3.5):
        cal = lane_cal(1.5)
        cal.lane_baseline = zero(w_px=w_px, lane_w_m=lane_w_m)
        cal.lane_free = lane_free
        calls = []
        return cal, calls, (lambda c: calls.append(c))

    def test_trusted_keeps_lane_calib(self):
        cal, calls, redo = self.make(600.0, free())   # 검산 600px × 1.5 / 300 = 3.0 m
        out = engine._apply_lane_trust(cal, redo)
        self.assertIs(out, cal)
        self.assertEqual((out.lane_trust['trusted'], out.lane_trust['applied']), (True, False))
        self.assertIsNone(out.lane_free)
        self.assertEqual(calls, [])

    def test_width_out_of_range_switches_to_lane_free(self):
        lf = free(cam_h=1.12, n_tracks=3, horizon=280)
        cal, calls, redo = self.make(600.0, lf, lane_w_m=5.59)
        out = engine._apply_lane_trust(cal, redo)
        self.assertIs(out, lf)
        self.assertEqual(calls, [lf])                       # 실내 마스크를 바뀐 수평선으로 다시
        self.assertIs(out.lane_baseline, cal.lane_baseline)
        lt = out.lane_trust
        self.assertEqual((lt['trusted'], lt['reason'], lt['applied'], lt['calib_changed']),
                         (False, 'lane_width_out_of_range', True, True))
        self.assertEqual(lt['lane_calib'], gc.calib_geom(cal))
        self.assertTrue(gc.lanes_excluded(out))

    def test_zero_failed_keeps_lane_calib(self):
        # t48 (pos_00845): 영점 실패는 LDW 만 — 수평선·신뢰도·높이·실내 마스크는 차선을 쓴 t40 보정값 그대로
        cal = lane_cal(1.5)
        cal.lane_baseline = {'ldw_measurable': False, 'reason': 'insufficient_valid_lane_pairs'}
        cal.lane_free = free(horizon=350)
        calls = []
        out = engine._apply_lane_trust(cal, calls.append)
        self.assertIs(out, cal)
        self.assertEqual(calls, [])
        self.assertEqual((out.lane_trust['trusted'], out.lane_trust['applied']), (False, False))
        self.assertFalse(gc.lanes_excluded(out))
        self.assertEqual((out.horizon_y, out.horizon_conf, out.cam_h), (HORIZON, 'high', 1.5))

    def test_unverifiable_keeps_lane_calib(self):
        cal, calls, redo = self.make(600.0, gc.default_calib(W, H, 680, 'flow'))
        out = engine._apply_lane_trust(cal, redo)
        self.assertIs(out, cal)
        self.assertEqual((out.lane_trust['reason'], out.lane_trust['applied'], calls), ('no_lane_free_horizon', False, []))

    def test_untrusted_same_geometry(self):
        cal, calls, redo = self.make(600.0, lane_cal(1.5), lane_w_m=5.0)   # 영점 폭 5.0 m → 불신, 기하는 같음
        out = engine._apply_lane_trust(cal, redo)
        self.assertEqual((out.lane_trust['trusted'], out.lane_trust['calib_changed']), (False, False))
        self.assertEqual(len(calls), 1)

    def test_to_dict_key(self):
        cal = calib()
        self.assertNotIn('lane_trust', cal.to_dict())      # --no-lane-trust: t40 meta 그대로
        cal.lane_trust = {'trusted': True}
        self.assertEqual(cal.to_dict()['lane_trust'], {'trusted': True})


class DepthGuardLaneTest(unittest.TestCase):
    """t48: 차선을 뺀 구간 (폭 범위 밖) 은 근접 보험 축척에서 차선 픽셀 (class 3) 을 쓰지 않는다 (t47 리뷰 재현 입력)."""

    def setup(self, applied):
        cal = calib(cam_h=1.5)
        cal.lane_trust = {'trusted': False, 'applied': applied}
        rows = np.arange(H, dtype=np.float32)[:, None]
        zg = np.where(rows > HORIZON + 1, cal.f_px * cal.cam_h / np.maximum(rows - HORIZON, 1), 100.0)
        depth = np.broadcast_to(zg * 0.25, (H, W)).astype(np.float32).copy()   # 지면과 맞는 Depth Pro 원값
        seg = np.zeros((H, W), np.uint8)
        seg[400:660, 300:980] = 3          # 차선 픽셀만 (도로 class 2 없음)
        return cal, depth, seg

    def test_lane_pixels_used_when_not_excluded(self):
        cal, depth, seg = self.setup(applied=False)
        out = gc.depth_guard(depth, seg, cal, [], [])
        self.assertEqual((out['status'], out['scale_src']), ('clear', 'frame'))
        self.assertAlmostEqual(out['scale'], 4.0, places=2)

    def test_lane_pixels_dropped_when_excluded(self):
        cal, depth, seg = self.setup(applied=True)
        self.assertEqual(gc.depth_guard(depth, seg, cal, [], [])['status'], 'no_scale')
        seg[400:660, 300:980] = 2          # 도로 픽셀은 그대로 쓴다
        self.assertEqual(gc.depth_guard(depth, seg, cal, [], [])['scale_src'], 'frame')


class LdwLaneTrustTest(Globals):
    def judge(self, lane_trust, measurable=True):
        engine.VEHICLE, engine.LDW_VARIANT = 'car', 'V0'
        cal = calib()
        k = 0.004
        cal.lane_baseline = ({'ldw_measurable': True, 'y_ref': 600, 'm_per_px': k, 'ldw_shift_px': 160.0,
                              'ego_w_px': round(gc.EGO_CAR_W_M / k, 1), 'conf': 'high', 'lane_w_m': 3.3}
                             if measurable else {'ldw_measurable': False, 'conf': 'failed'})
        if lane_trust is not None:
            cal.lane_trust = lane_trust
        engine.CAR_CAL = cal
        lane = {'ok': True, 'lines': [LEFT, RIGHT], 'reason': '', 'vp': (VP_X, HORIZON, 1.0)}
        return engine._ldw_car(lane_mask(), np.zeros((H, W), np.uint8), lane, cal)

    def test_untrusted_width(self):
        w = self.judge({'trusted': False, 'reason': 'lane_width_out_of_range', 'lane_w_m': 5.595,
                        'lane_w_m_check': 3.3})
        self.assertEqual((w.level, w.reason), (0, '[측정 불가] 차선 신뢰 낮음 (차로 폭 5.6m)'))
        self.assertTrue(w.extra['unreliable'])
        self.assertEqual(w.extra['lane_trust'], 'lane_width_out_of_range')
        w = self.judge({'trusted': False, 'reason': 'lane_width_out_of_range_check', 'lane_w_m': 3.3,
                        'lane_w_m_check': 1.125})
        self.assertEqual(w.reason, '[측정 불가] 차선 신뢰 낮음 (검산 차로 폭 1.1m)')

    def test_untrusted_zero_failed_and_no_horizon(self):
        w = self.judge({'trusted': False, 'reason': 'lane_zero_failed:no_horizon', 'lane_w_m': None}, measurable=False)
        self.assertEqual(w.reason, '[측정 불가] 차선 신뢰 낮음 (영점 실패)')
        w = self.judge({'trusted': False, 'reason': 'no_lane_free_horizon', 'lane_w_m': None})
        self.assertEqual(w.reason, '[측정 불가] 차선 신뢰 낮음 (검산 불가)')

    def test_trusted_same_as_without_gate(self):
        a = self.judge({'trusted': True, 'reason': 'ok', 'lane_w_m': 3.3})
        b = self.judge(None)
        self.assertEqual(a.__dict__, b.__dict__)
        self.assertEqual((a.level, a.reason), (1, '우측 차선 위 주행'))


class CliLaneTrustTest(Globals):
    NAMES = Globals.NAMES + ('LANE_TRUST',)

    def test_default_on_and_option(self):
        with patch.object(engine, 'run_video'):
            engine.main(['x.mp4', '--vehicle', 'car'])
            self.assertTrue(engine.LANE_TRUST)
            engine.main(['x.mp4', '--vehicle', 'car', '--no-lane-trust'])
            self.assertFalse(engine.LANE_TRUST)


if __name__ == '__main__':
    unittest.main()
