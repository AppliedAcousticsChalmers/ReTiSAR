import sys
import timeit
from time import sleep

import numpy as np

from . import tools


def main():
    """
    Returns
    -------
    logging.Logger
        logger instance
    """
    # modify timer template to also receive evaluation results from `timeit` function call
    timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        result = {stmt}
    _t1 = _timer()
    return _t1 - _t0, result  # time, computation result
"""

    # _timeit_fft()
    _timeit_noise()
    # _timeit_sh()
    # _timeit_basic()

    return None


def _timeit(description, stmt, setup, _globals, reference=None, repeat=10, number=2000):
    print(description)

    # do timing and get results
    result = timeit.Timer(stmt=stmt, setup=setup, globals=_globals).repeat(repeat=repeat, number=number)
    result = (min(list(zip(*result))[0]), result[0][1])  # time, computation result

    # print conclusion
    print('time: {:-29.2f}s'.format(result[0]))
    sleep(.05)  # to get correct output order

    if reference:
        t = result[0] / reference[0]
        file = sys.stdout
        if abs(t - 1) < 2E-2:
            grade = 'EVEN'
        elif t < 1:
            grade = 'BETTER'
        else:
            grade = 'WORSE'
            file = sys.stderr
        print('time factor: {:-22.2f} ... {}'.format(t, grade), file=file)
        sleep(.05)  # to get correct output order

        # flip computation result, if matrices do not match
        if reference[1].shape != result[1].shape:
            reference = (reference[0], reference[1].T)
        if reference[1].shape != result[1].shape:
            print('result: {:22} {}'.format('', 'DIMENSION MISMATCH'), file=sys.stderr)
            sleep(.05)  # to get correct output order
            print()
            return result

        r = np.abs(np.sum(np.subtract(result[1], reference[1])))
        file = sys.stdout
        if r == 0:
            grade = 'PERFECT'
        elif r < 1E-10:
            grade = 'OKAY'
        else:
            grade = 'MISMATCH'
            file = sys.stderr
        print('result sum:  {:-22} ... {}'.format(r, grade), file=file)
        sleep(.05)  # to get correct output order

        r = np.abs(np.subtract(result[1], reference[1])).max()
        file = sys.stdout
        if r == 0:
            grade = 'PERFECT'
        elif r < 1E-10:
            grade = 'OKAY'
        else:
            grade = 'MISMATCH'
            file = sys.stderr
        print('result max:  {:-22} ... {}'.format(r, grade), file=file)
        sleep(.05)  # to get correct output order

    print()
    return result


def _timeit_fft():
    import pyfftw

    _TIMEIT_REPEAT = 5
    _TIMEIT_NUMBER = 1000

    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 50

    input_td = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH))
    # input_td = pyfftw.byte_align(input_td)  # no effect

    # input_td = pyfftw.empty_aligned((_CHANNEL_COUNT, _BLOCK_LENGTH), dtype=np.float64, n=8)
    # input_td[:] = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH))  # no effect

    ref = _timeit('numpy', 'result = fft.rfft(input_td)', 'import numpy.fft as fft',
                  locals(), None, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _rfft = pyfftw.builders.rfft(input_td, overwrite_input=True)
    _timeit('pyfftw overwrite', 'result = _rfft(input_td)', '', locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _rfft = pyfftw.builders.rfft(input_td, overwrite_input=True, planner_effort='FFTW_PATIENT')
    _timeit('pyfftw effort', 'result = _rfft(input_td)', '', locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _rfft = pyfftw.builders.rfft(input_td, overwrite_input=True, planner_effort='FFTW_PATIENT', threads=2)
    _timeit('pyfftw threads', 'result = _rfft(input_td)', '', locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _timeit('pyfftw numpy interface', 'result = fft.rfft(input_td)', 'import pyfftw.interfaces.numpy_fft as fft',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _timeit('pyfftw numpy interface effort', 'result = fft.rfft(input_td, planner_effort="FFTW_PATIENT")',
            'import pyfftw.interfaces.numpy_fft as fft', locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _timeit('pyfftw numpy interface threads', 'result = fft.rfft(input_td, planner_effort="FFTW_PATIENT", threads=2)',
            'import pyfftw.interfaces.numpy_fft as fft', locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _timeit('scipy', 'result = fft.rfft(input_td)', 'import scipy.fftpack as fft',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _timeit('pyfftw scipy interface', 'result = fft.rfft(input_td)', 'import pyfftw.interfaces.scipy_fftpack as fft',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)

    _timeit('mkl_fft', 'result = fft.rfft(input_td)', 'import mkl_fft as fft',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)


def _timeit_noise():
    _TIMEIT_REPEAT = 5
    _TIMEIT_NUMBER = 15

    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 110
    _AR_ORDER = 8
    _AR_POWER = 1

    coefficients = np.zeros(_AR_ORDER)
    coefficients[0] = 1
    for k in range(1, _AR_ORDER):
        coefficients[k] = (k - 1 - _AR_POWER / 2) * coefficients[k - 1] / k
    _ar_coefficients = coefficients[1:]

    _iir_b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    _iir_a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    _iir_t60 = int(np.log(1000.0) / (1.0 - np.abs(np.roots(_iir_a)).max())) + 1

    _normal_ref_t60 = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH + _iir_t60))
    _normal_ref = _normal_ref_t60[:, -_BLOCK_LENGTH:].T.copy()

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
shaped = np.zeros_like(normal)
for idx in range(_BLOCK_LENGTH):
    shaped[idx] = normal[idx] - (_ar_coefficients * _ar_buffer).sum(axis=0)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = shaped[idx]
result = shaped.T
"""
    _ar_coefficients = coefficients[1:, np.newaxis]
    ref = _timeit('AR original implementation', s, 'import numpy as np',
                  locals(), None, _TIMEIT_REPEAT, _TIMEIT_NUMBER)
    _ar_coefficients = coefficients[1:]

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
shaped = np.zeros_like(normal)
for idx in range(normal.shape[0]):
    shaped[idx] = normal[idx] - np.einsum('i,ij', _ar_coefficients, _ar_buffer)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = shaped[idx]
result = shaped.T
"""
    _timeit('AR use einsum()', s, 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # not good

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
shaped = np.zeros_like(normal)
for idx in range(_BLOCK_LENGTH):
    shaped[idx] = normal[idx] - np.inner(_ar_coefficients, _ar_buffer.T)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = shaped[idx]
result = shaped.T
"""
    _timeit('AR use inner()', s, 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # not good

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
shaped = np.zeros_like(normal)
for idx in range(_BLOCK_LENGTH):
    shaped[idx] = normal[idx] - np.dot(_ar_coefficients, _ar_buffer)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = shaped[idx]
result = shaped.T
"""
    _timeit('AR use dot()', s, 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # good (best AR)

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
for idx in range(_BLOCK_LENGTH):
    normal[idx] -= np.dot(_ar_coefficients, _ar_buffer)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = normal[idx]
result = normal.T
"""
    _timeit('AR use dot() 1 array', s, 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # not good

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
for n in normal:
    n -= np.dot(_ar_coefficients, _ar_buffer)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = n
result = normal.T
"""
    _timeit('AR use dot() 1 array iterate', s, 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # good (best AR)

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.T.copy()
shaped = np.zeros_like(normal)
with np.nditer([shaped, normal], flags=['external_loop'], order='F', op_flags=[['readwrite'], ['readonly']]) as it:
    for it_shaped, it_normal in it:
        it_shaped[...] = it_normal - np.dot(_ar_coefficients, _ar_buffer)
        _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
        _ar_buffer[0] = it_shaped
result = shaped
"""
    _ar_coefficients = coefficients[1:]
    _timeit('AR nditer', s, 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # not good

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.T.copy()
with np.nditer(normal, flags=['external_loop'], order='F', op_flags=['readwrite']) as it:
    for it_normal in it:
        it_normal[...] -= np.dot(_ar_coefficients, _ar_buffer)
        _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
        _ar_buffer[0] = it_normal
result = normal
"""
    _timeit('AR nditer 1 array', s, 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # not good

#     s = """\
# _ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
# normal = _normal_ref.T.copy()
# shaped = np.zeros_like(normal)
# with it.copy():
#     for it_shaped, it_normal in it:
#         it_shaped[...] = it_normal - np.dot(_ar_coefficients, _ar_buffer)
#         _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
#         _ar_buffer[0] = it_shaped
# result = shaped
# """
#     normal = _normal_ref.T.copy()
#     shaped = np.zeros_like(normal)
#     it = np.nditer([shaped, normal], flags=["external_loop"], order="F", op_flags=[["readwrite"], ['readonly']])
#     _timeit('AR nditer reused', s, 'import numpy as np',
#             locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # not working properly

    s = """\
normal = _normal_ref_t60.copy()
result = signal.lfilter(_iir_b, _iir_a, normal)
result = result[:, _iir_t60:]  # skip transient response
"""
    _timeit('IIR with lfilter()', s, 'from scipy import signal',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)  # good (best IIR)


def _timeit_sh():
    # print('system switcher interval is {}'.format(sys.getswitchinterval()))
    # print('system recursion limit is {}'.format(sys.getrecursionlimit()))
    # sys.setrecursionlimit(1500)
    # print('system recursion limit is {}'.format(sys.getrecursionlimit()))

    _TIMEIT_REPEAT = 10
    _TIMEIT_NUMBER = 2000

    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 110
    _SH_MAX_ORDER = 8

    _SH_COEFFICIENTS = pow(_SH_MAX_ORDER + 1, 2)

    spherical_harmonic_bases = tools.generate_noise((_SH_COEFFICIENTS, _CHANNEL_COUNT), is_complex=True)
    data = tools.generate_noise((_BLOCK_LENGTH, _CHANNEL_COUNT), is_complex=True)

    def _test_performance(shb, d, dtype):
        # cast type
        shb = shb.astype(dtype)
        d = d.astype(dtype)

        ref = _timeit('numpy.inner() as {}'.format(dtype), 'np.inner(shb, d.T)', 'import numpy as np',
                      locals(), None, _TIMEIT_REPEAT, int(_TIMEIT_NUMBER / (2 * _SH_COEFFICIENTS)))
        r1 = _timeit('numpy.einsum() as {}'.format(dtype), 'np.einsum("ij,jl", shb, d)', 'import numpy as np',
                     locals(), ref, _TIMEIT_REPEAT, int(_TIMEIT_NUMBER / (2 * _SH_COEFFICIENTS)))

        print()
        return [ref, r1]

    print('comparison of dtypes')
    data = data.T.copy()
    rs = [_test_performance(spherical_harmonic_bases, data, np.complex64),
          _test_performance(spherical_harmonic_bases, data, np.complex128)]
    for r in range(len(rs[0])):
        print('results {} max: {}'.format(r, np.abs(np.subtract(*rs[:][r][1])).max()))


def _timeit_basic():
    _TIMEIT_REPEAT = 10
    _TIMEIT_NUMBER = 1000

    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 50

    data = tools.generate_noise((_BLOCK_LENGTH, _CHANNEL_COUNT), is_complex=True)

    ref = _timeit('multiply', 'data * data', '',
                  locals(), None, _TIMEIT_REPEAT, _TIMEIT_NUMBER)
    _timeit('numpy.multiply()', 'np.multiply(data, data)', 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)
    print()

    ref = _timeit('abs', 'abs(data)', '',
                  locals(), None, _TIMEIT_REPEAT, _TIMEIT_NUMBER)
    _timeit('numpy.abs()', 'np.abs(data)', 'import numpy as np',
            locals(), ref, _TIMEIT_REPEAT, _TIMEIT_NUMBER)
