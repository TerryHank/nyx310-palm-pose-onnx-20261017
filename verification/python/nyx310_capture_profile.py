"""Apply the validated NYX310 RGB profile before starting the stream."""
import ctypes as c


def configure_capture(sdk):
    for name in ('Color', 'ToF'):
        function = getattr(sdk.base, f'scGet{name}Resolution')
        function.argtypes = [c.c_void_p, c.POINTER(c.c_int32), c.POINTER(c.c_int32)]
        function.restype = c.c_int
    sdk.base.scGetFrameRate.argtypes = [c.c_void_p, c.POINTER(c.c_int32)]
    sdk.base.scGetFrameRate.restype = c.c_int
    sdk.base.scSetFrameRate.argtypes = [c.c_void_p, c.c_int32]
    sdk.base.scSetFrameRate.restype = c.c_int
    sdk.base.scSetColorResolution.argtypes = [c.c_void_p, c.c_int32, c.c_int32]
    sdk.base.scSetColorResolution.restype = c.c_int

    def read():
        values = {}
        for name in ('Color', 'ToF'):
            width, height = c.c_int32(), c.c_int32()
            sdk._check(getattr(sdk.base, f'scGet{name}Resolution')(
                sdk.device, c.byref(width), c.byref(height)), f'get {name} resolution')
            values[name] = [width.value, height.value]
        fps = c.c_int32()
        sdk._check(sdk.base.scGetFrameRate(sdk.device, c.byref(fps)), 'get frame rate')
        values['fps'] = fps.value
        return values

    before = read()
    if before['Color'] != [800, 600]:
        sdk._check(sdk.base.scSetColorResolution(sdk.device, 800, 600), 'set RGB 800x600')
    if before['fps'] != 15:
        sdk._check(sdk.base.scSetFrameRate(sdk.device, 15), 'set 15 FPS')
    after = read()
    if after['Color'] != [800, 600] or after['fps'] != 15 or after['ToF'] != before['ToF']:
        raise RuntimeError(f'NYX310 capture profile readback mismatch: {after}')
    return dict(before=before, active=after, applied_on_each_launch=True)
