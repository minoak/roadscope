"""t41 LDW 방식 (V0 / VA / VB / VAB) 단위 테스트. 모델·영상·GPU 없이 합성 차선으로 확인한다.

python -m unittest tests.test_ldw_variants -v
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from roadscope import engine
from roadscope import geom_car as gc
from tools.evaluate import rejudge_ldw_vb as vb

W, H = 1280, 720
HORIZON, VP_X, HOOD = 300, 640, 680
# 소실점 (640, 300) 을 지나는 좌우 차선 (x = dydx·y + c). y=600 에서 x 415 / 865 (폭 450px)
LEFT = {'dydx': -0.75, 'c': 865.0}
RIGHT = {'dydx': 0.75, 'c': 415.0}


def calib(cam_h=2.33):
    return gc.CarCalib(W=W, H=H, horizon_y=HORIZON, vp_x=VP_X, horizon_ok=True, horizon_src='foe',
                       horizon_conf='high', vp_x_src='foe', hood_row=HOOD, hood_src='flow', cam_h=cam_h,
                       cam_h_src='vehicle+lane', f_px=600.0)


def lane_mask():
    m = np.zeros((H, W), np.uint8)
    for line in (LEFT, RIGHT):
        for y in range(HORIZON + 5, H):
            x = int(round(line['dydx'] * y + line['c']))
            m[y, max(0, x - 2):x + 3] = 3
    return m


class Globals(unittest.TestCase):
    """engine 모듈 설정을 테스트마다 되돌린다."""
    NAMES = ('VEHICLE', 'CAR_CAL', 'LDW_VARIANT', 'CAR_M2F', 'HOOD_RATIO', 'HOOD_MANUAL', 'RESCUE_LOWCONF',
             'MASK_BOXES', 'GHOST_BOXES', 'NEAR_OF_TWO', 'CAR_TTC', 'CAR_DEPTH_GUARD')

    def setUp(self):
        saved = {k: getattr(engine, k) for k in self.NAMES}
        self.addCleanup(lambda: [setattr(engine, k, v) for k, v in saved.items()])


class LaneZeroRelaxTest(unittest.TestCase):
    def run_zero(self, n_lane, n_total, relaxed):
        mask = lane_mask() == 3
        mask[HOOD:] = False   # 보닛 띠 아래 픽셀은 지운 것으로 준다
        frames = [[LEFT, RIGHT]] * n_lane + [[]] * (n_total - n_lane)
        return gc.lane_zero(calib(), frames, lambda j: mask, n_total, relaxed=relaxed)

    def test_relaxed_rescues_between_thresholds(self):
        # 400 프레임 중 25 프레임만 쌍: V0 필요 max(30, 40) = 40 → 실패, VA 필요 max(15, 20) = 20 → 성공
        strict, relaxed = self.run_zero(25, 400, False), self.run_zero(25, 400, True)
        self.assertEqual((strict['ldw_measurable'], strict['reason'], strict['required_frames']),
                         (False, 'insufficient_lane_pairs', 40))
        self.assertEqual((relaxed['ldw_measurable'], relaxed['reason'], relaxed['required_frames']), (True, 'ok', 20))
        self.assertEqual(relaxed['evidence_frames'], 25)

    def test_relaxed_floor_is_15_frames(self):
        self.assertEqual(self.run_zero(14, 100, True)['reason'], 'insufficient_lane_pairs')
        self.assertTrue(self.run_zero(15, 100, True)['ldw_measurable'])

    def test_relaxed_same_result_when_strict_passes(self):
        strict, relaxed = self.run_zero(50, 400, False), self.run_zero(50, 400, True)
        self.assertTrue(strict['ldw_measurable'])
        strict.pop('required_frames'), relaxed.pop('required_frames')
        self.assertEqual(strict, relaxed)

    def test_default_is_strict(self):
        mask = lane_mask() == 3
        frames = [[LEFT, RIGHT]] * 25 + [[]] * 375
        self.assertEqual(gc.lane_zero(calib(), frames, lambda j: mask, 400)['required_frames'], 40)


class LdwCarVariantTest(Globals):
    def judge(self, variant, conf, shift_px=160.0):
        engine.VEHICLE, engine.LDW_VARIANT = 'car', variant
        cal = calib()
        k = 0.004
        cal.lane_baseline = {'ldw_measurable': True, 'y_ref': 600, 'm_per_px': k, 'ldw_shift_px': shift_px,
                             'ego_w_px': round(gc.EGO_CAR_W_M / k, 1), 'conf': conf, 'lane_w_m': 1.8}
        engine.CAR_CAL = cal
        lane = {'ok': True, 'lines': [LEFT, RIGHT], 'reason': '', 'vp': (VP_X, HORIZON, 1.0)}
        return engine._ldw_car(lane_mask(), np.zeros((H, W), np.uint8), lane, cal)

    def test_v0_default_measures_low_conf_segment(self):
        self.assertEqual(engine.LDW_VARIANT, 'V0')
        w = self.judge('V0', 'low')   # 기준 열 800, 차로 중심 640, 폭 450 → 편향 0.356 > 0.3
        self.assertEqual((w.level, w.reason), (1, '우측 차선 위 주행'))
        self.assertFalse(w.extra.get('unreliable'))

    def test_vb_turns_off_low_conf_segment(self):
        for variant in ('VB', 'VAB'):
            w = self.judge(variant, 'low')
            self.assertEqual(w.level, 0)
            self.assertEqual(w.reason, '[측정 불가] 차선 영점 신뢰 낮음 (차로 폭 1.80m)')
            self.assertTrue(w.extra['unreliable'])
            self.assertIsNone(w.extra['ref'])

    def test_vb_keeps_high_conf_segment(self):
        self.assertEqual(self.judge('VB', 'high').__dict__, self.judge('V0', 'high').__dict__)

    def test_va_does_not_change_frame_judgement(self):
        self.assertEqual(self.judge('VA', 'low').__dict__, self.judge('V0', 'low').__dict__)

    def test_failed_zero_reason_unchanged(self):
        engine.VEHICLE, engine.LDW_VARIANT = 'car', 'VB'
        cal = calib()
        cal.lane_baseline = {'ldw_measurable': False, 'conf': 'failed'}
        w = engine._ldw_car(lane_mask(), np.zeros((H, W), np.uint8), {'ok': True, 'lines': []}, cal)
        self.assertEqual(w.reason, '[측정 불가] 차선 영점 실패')


class CliTest(Globals):
    def test_default_and_option(self):
        with patch.object(engine, 'run_video') as rv:
            engine.main(['x.mp4', '--vehicle', 'car'])
            self.assertEqual(engine.LDW_VARIANT, 'V0')
            engine.main(['x.mp4', '--vehicle', 'car', '--ldw-variant', 'VAB'])
            self.assertEqual(engine.LDW_VARIANT, 'VAB')
            self.assertEqual(rv.call_count, 2)
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            engine.main(['x.mp4', '--ldw-variant', 'VC'])


def record(ldw_level, fcw_level=0, seg=0):
    return {'frame_idx': 0, 't_sec': 0.0, 'bin': max(ldw_level, fcw_level), 'bin_label': '',
            'top_warning': 'FCW' if fcw_level >= ldw_level and fcw_level else ('LDW' if ldw_level else None),
            'top_reason': '', 'calib_seg': seg, 'lane_measurable': True,
            'warnings': {'FCW': {'level': fcw_level, 'reason': '정면 8.0m' if fcw_level else '', 'track_id': None},
                         'LDW': {'level': ldw_level, 'reason': '우측 차선 위 주행' if ldw_level else '',
                                 'offset_ratio': 0.35, 'ref': {'y': 600}},
                         'BSW': {'level': 0, 'reason': '', 'track_id': None, 'side': None},
                         'HEAD': {'level': 0, 'reason': '', 'track_id': None}}}


class OfflineVbTest(unittest.TestCase):
    LOW = {'ldw_measurable': True, 'conf': 'low', 'lane_w_m': 2.221}

    def test_low_conf_ldw_off_and_summary_refreshed(self):
        out = vb.apply_vb(record(1), {0: self.LOW})
        self.assertEqual(out['warnings']['LDW'], {'level': 0, 'reason': '[측정 불가] 차선 영점 신뢰 낮음 (차로 폭 2.22m)',
                                                  'offset_ratio': None, 'ref': None})
        self.assertFalse(out['lane_measurable'])
        self.assertEqual((out['bin'], out['bin_label'], out['top_warning'], out['top_reason']),
                         (0, engine.mvp.SEVERITY_TO_BIN[0], None, ''))

    def test_other_warning_becomes_top(self):
        r = record(1, fcw_level=1)
        out = vb.apply_vb(r, {0: self.LOW})
        self.assertEqual((out['bin'], out['top_warning'], out['top_reason']), (1, 'FCW', '정면 8.0m'))
        self.assertEqual(out['warnings']['FCW'], r['warnings']['FCW'])

    def test_high_conf_and_failed_unchanged(self):
        r = record(1)
        for lb in ({'ldw_measurable': True, 'conf': 'high', 'lane_w_m': 3.3}, {'ldw_measurable': False, 'conf': 'failed'}):
            self.assertEqual(vb.apply_vb(r, {0: lb}), r)

    def test_uses_frame_segment(self):
        r = record(1, seg=1)
        self.assertEqual(vb.apply_vb(r, {0: self.LOW, 1: {'ldw_measurable': True, 'conf': 'high'}}), r)

    def test_engine_and_offline_reason_match(self):
        """오프라인 사유 문구 = 엔진 VB 분기 문구 (화면 사유 user_reason 을 거친 뒤)."""
        engine_reason = engine.user_reason(engine.mvp.WarningResult('LDW', 0, '[측정 불가] 차선 영점 신뢰 낮음 (차로 폭 2.22m)'))
        self.assertEqual(vb.low_conf_reason(self.LOW), engine_reason)


if __name__ == '__main__':
    unittest.main()
