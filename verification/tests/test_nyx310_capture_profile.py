import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from nyx310_capture_profile import configure_capture


class CaptureProfileTests(unittest.TestCase):
    def sdk(self, rgb=(1600, 1200), fps=15):
        sdk = Mock()
        state = dict(Color=list(rgb), ToF=[640, 480], fps=fps)
        def check(status, operation):
            if status:
                raise RuntimeError(f'{operation}: {status}')
        sdk._check.side_effect = check
        for name in ('Color', 'ToF'):
            def get(device, width, height, name=name):
                width._obj.value, height._obj.value = state[name]
                return 0
            getattr(sdk.base, f'scGet{name}Resolution').side_effect = get
        def get_fps(device, value):
            value._obj.value = state['fps']
            return 0
        def set_rgb(device, width, height):
            state['Color'] = [width, height]
            return 0
        def set_fps(device, value):
            state['fps'] = value
            return 0
        sdk.base.scGetFrameRate.side_effect = get_fps
        sdk.base.scSetColorResolution.side_effect = set_rgb
        sdk.base.scSetFrameRate.side_effect = set_fps
        return sdk, state

    def test_apply_and_readback_preserve_tof(self):
        sdk, state = self.sdk(fps=30)
        result = configure_capture(sdk)
        self.assertEqual(result['before']['Color'], [1600, 1200])
        self.assertEqual(result['active'], dict(Color=[800, 600], ToF=[640, 480], fps=15))
        self.assertEqual(state, result['active'])
        sdk.base.scSetToFResolution.assert_not_called()
        sdk.base.scStartStream.assert_not_called()

    def test_already_correct_does_not_write(self):
        sdk, _ = self.sdk(rgb=(800, 600))
        configure_capture(sdk)
        sdk.base.scSetColorResolution.assert_not_called()
        sdk.base.scSetFrameRate.assert_not_called()

    def test_reject_silent_setter_failure(self):
        sdk, _ = self.sdk()
        sdk.base.scSetColorResolution.side_effect = None
        sdk.base.scSetColorResolution.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'readback mismatch'):
            configure_capture(sdk)
        sdk.base.scStartStream.assert_not_called()

    def test_sdk_error_stops_before_stream(self):
        sdk, _ = self.sdk()
        sdk.base.scSetColorResolution.side_effect = None
        sdk.base.scSetColorResolution.return_value = -1
        with self.assertRaisesRegex(RuntimeError, 'set RGB 800x600'):
            configure_capture(sdk)
        sdk.base.scStartStream.assert_not_called()


if __name__ == '__main__':
    unittest.main()
