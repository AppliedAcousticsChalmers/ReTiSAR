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

    # _timeit_roll()
    # _timeit_fft()
    # _timeit_noise()
    # _timeit_sht()
    # _timeit_sp()
    # _timeit_basic()
    # _test_multiprocessing()
    # _test_client_name_length()
    _test_client_name_lock()

    return None


def _timeit(
    description,
    stmt,
    setup,
    _globals,
    reference=None,
    check_dtype=None,
    repeat=10,
    number=2000,
):
    print(description)

    # do timing and get results
    result = timeit.Timer(stmt=stmt, setup=setup, globals=_globals).repeat(
        repeat=repeat, number=number
    )
    result = (min(list(zip(*result))[0]), result[0][1])  # time, computation result

    # print conclusion
    print(f"time: {result[0]:-29.2f}s")
    sleep(0.05)  # to get correct output order

    if reference:
        t = result[0] / reference[0]
        file = sys.stdout
        if abs(t - 1) < 2e-2:
            grade = "EVEN"
        elif t < 1:
            grade = "BETTER"
        else:
            grade = "WORSE"
            file = sys.stderr
        print(f"time factor: {t:-22.2f} ... {grade}", file=file)
        sleep(0.05)  # to get correct output order

        # flip computation result, if matrices do not match
        if reference[1].shape != result[1].shape:
            reference = (reference[0], reference[1].T)
        if reference[1].shape != result[1].shape:
            print(f'result: {"":22} {"DIMENSION MISMATCH"}', file=sys.stderr)
            sleep(0.05)  # to get correct output order
            print()
            return result

        if check_dtype:
            # check date type
            if result[1].dtype == check_dtype:
                grade = "MATCH"
                file = sys.stdout
            else:
                grade = "MISMATCH"
                file = sys.stderr
            print(f"dtype: {str(result[1].dtype):>28} ... {grade}", file=file)
            sleep(0.05)  # to get correct output order

        r = np.abs(np.sum(np.subtract(result[1], reference[1])))
        file = sys.stdout
        if r == 0:
            grade = "PERFECT"
        elif r < 1e-10:
            grade = "OKAY"
        else:
            grade = "MISMATCH"
            file = sys.stderr
        print(f"result sum:  {r:-22} ... {grade}", file=file)
        sleep(0.05)  # to get correct output order

        r = np.abs(np.subtract(result[1], reference[1])).max()
        file = sys.stdout
        if r == 0:
            grade = "PERFECT"
        elif r < 1e-10:
            grade = "OKAY"
        else:
            grade = "MISMATCH"
            file = sys.stderr
        print(f"result max:  {r:-22} ... {grade}", file=file)
        sleep(0.05)  # to get correct output order

    print()
    return result


def _timeit_roll():
    _TIMEIT_REPEAT = 5
    _TIMEIT_NUMBER = 100

    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 50
    _BLOCK_COUNT = 10

    input_td = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH))
    buffer_ref = tools.generate_noise((_BLOCK_COUNT, _CHANNEL_COUNT, _BLOCK_LENGTH))

    buffer = buffer_ref.copy()
    s = """\
buffer = globals().get("buffer")  # I don't know why this is necessary...
buffer = np.roll(buffer, 1, axis=0)  # roll buffer to the right
buffer[0] = input_td  # update the first element
result = buffer
"""
    ref = _timeit(
        description="numpy roll",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    buffer = buffer_ref.copy()
    s = """\
buffer = globals().get("buffer")  # I don't know why this is necessary...
buffer[1:] = buffer[:-1]  # roll buffer to the right
buffer[0] = input_td  # update the first element
result = buffer
"""
    _timeit(
        description="slice indexing",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # slower

    import jack

    rb = jack.RingBuffer(buffer_ref.nbytes)
    s = """\
bytes_written = rb.write(input_td.tobytes())
if bytes_written != input_td.nbytes:
    print("OVERFLOW")
read = np.frombuffer(rb.read(input_td.nbytes), dtype=input_td.dtype).reshape(
    input_td.shape
)
if read.nbytes != input_td.nbytes:
    print("UNDERFLOW")
result = read
"""
    # this is not expected to deliver the same result as the roll methods above
    _timeit(
        description="Ringbuffer",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    rb = jack.RingBuffer(buffer_ref.nbytes)
    s = """\
if rb.write_space >= input_td.nbytes:
    rb.write_buffers[0][: input_td.nbytes] = input_td.tobytes()
    rb.write_advance(input_td.nbytes)
else:
    print("OVERFLOW")
read = np.frombuffer(
    rb.read_buffers[0][: input_td.nbytes], dtype=input_td.dtype
).reshape(input_td.shape)
rb.read_advance(read.nbytes)
if read.nbytes != input_td.nbytes:
    print("UNDERFLOW")
result = read
"""
    # this code does not run so far, I don't know how it is supposed to be done with
    # write_buffers() and read_buffers() independent of that I don't know why the timeit() code
    # does not run like that this is not expected to deliver the same result as the roll methods
    # above
    _timeit(
        description="Ringbuffer no copy",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )


def _timeit_fft():
    import pyfftw

    _TIMEIT_REPEAT = 5
    _TIMEIT_NUMBER = 1000

    # _BLOCK_LENGTH = 256
    # _CHANNEL_COUNT = 32
    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 110

    input_td = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH))
    # input_td = pyfftw.byte_align(input_td)  # no effect

    # input_td = pyfftw.empty_aligned((_BLOCK_LENGTH), dtype=np.float64, n=8)
    # input_td[:] = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH))  # no effect

    ref = _timeit(
        description="numpy.fft",
        stmt="result = fft.rfft(input_td)",
        setup="import numpy.fft as fft",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _rfft = pyfftw.builders.rfft(input_td, overwrite_input=True)
    _timeit(
        description="pyfftw overwrite",
        stmt="result = _rfft(input_td)",
        setup="",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _rfft = pyfftw.builders.rfft(
        input_td, overwrite_input=True, planner_effort="FFTW_PATIENT"
    )
    _timeit(
        description="pyfftw effort",
        stmt="result = _rfft(input_td)",
        setup="",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _rfft = pyfftw.builders.rfft(
        input_td, overwrite_input=True, planner_effort="FFTW_PATIENT", threads=2
    )
    _timeit(
        description="pyfftw 2 threads",
        stmt="result = _rfft(input_td)",
        setup="",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _timeit(
        description="pyfftw numpy interface",
        stmt="result = fft.rfft(input_td)",
        setup="import pyfftw.interfaces.numpy_fft as fft",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _timeit(
        description="pyfftw numpy interface effort",
        stmt='result = fft.rfft(input_td, planner_effort="FFTW_PATIENT")',
        setup="import pyfftw.interfaces.numpy_fft as fft",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _timeit(
        description="pyfftw numpy interface 2 threads",
        stmt='result = fft.rfft(input_td, planner_effort="FFTW_PATIENT", threads=2)',
        setup="import pyfftw.interfaces.numpy_fft as fft",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _timeit(
        description="scipy.fftpack",
        stmt="result = fft.rfft(input_td)",
        setup="import scipy.fftpack as fft",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    # in scipy >= 1.4.0
    _timeit(
        description="scipy.fft",
        stmt="result = fft.rfft(input_td)",
        setup="import scipy.fft as fft",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _timeit(
        description="pyfftw scipy interface",
        stmt="result = fft.rfft(input_td)",
        setup="import pyfftw.interfaces.scipy_fftpack as fft",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )

    _timeit(
        description="mkl_fft",
        stmt="result = fft.rfft(input_td)",
        setup="import mkl_fft as fft",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )


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

    ref = _timeit(
        description="noise white complex128",
        stmt='tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH), dtype="complex128")',
        setup="from ReTiSAR import tools",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )
    _timeit(
        description="noise white complex64",
        reference=ref,
        check_dtype=np.complex64,
        stmt='tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH), dtype="complex64")',
        setup="from ReTiSAR import tools",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )
    _timeit(
        description="noise white float64",
        reference=ref,
        check_dtype=np.float64,
        stmt='tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH), dtype="float64")',
        setup="from ReTiSAR import tools",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )
    _timeit(
        description="noise white float32",
        reference=ref,
        check_dtype=np.float32,
        stmt='tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH), dtype="float32")',
        setup="from ReTiSAR import tools",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )
    print(tools.SEPARATOR)

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
    ref = _timeit(
        description="AR original implementation",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )
    _ar_coefficients = coefficients[1:]

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
shaped = np.zeros_like(normal)
for idx in range(normal.shape[0]):
    shaped[idx] = normal[idx] - np.einsum("i,ij", _ar_coefficients, _ar_buffer)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = shaped[idx]
result = shaped.T
"""
    _timeit(
        description="AR use einsum()",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # not good

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
    _timeit(
        description="AR use inner()",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # not good

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
    _timeit(
        description="AR use dot()",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # good (best AR)

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
for idx in range(_BLOCK_LENGTH):
    normal[idx] -= np.dot(_ar_coefficients, _ar_buffer)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = normal[idx]
result = normal.T
"""
    _timeit(
        description="AR use dot() 1 array",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # not good

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.copy()
for n in normal:
    n -= np.dot(_ar_coefficients, _ar_buffer)
    _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
    _ar_buffer[0] = n
result = normal.T
"""
    _timeit(
        description="AR use dot() 1 array iterate",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # good (best AR)

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.T.copy()
shaped = np.zeros_like(normal)
with np.nditer(
    [shaped, normal],
    flags=["external_loop"],
    order="F",
    op_flags=[["readwrite"], ["readonly"]],
) as it:
    for it_shaped, it_normal in it:
        it_shaped[...] = it_normal - np.dot(_ar_coefficients, _ar_buffer)
        _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
        _ar_buffer[0] = it_shaped
result = shaped
"""
    _ar_coefficients = coefficients[1:]
    _timeit(
        description="AR nditer",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # not good

    s = """\
_ar_buffer = np.zeros((_ar_coefficients.shape[0], _CHANNEL_COUNT))
normal = _normal_ref.T.copy()
with np.nditer(
    normal, flags=["external_loop"], order="F", op_flags=["readwrite"]
) as it:
    for it_normal in it:
        it_normal[...] -= np.dot(_ar_coefficients, _ar_buffer)
        _ar_buffer = np.roll(_ar_buffer, 1, axis=0)
        _ar_buffer[0] = it_normal
result = normal
"""
    _timeit(
        description="AR nditer 1 array",
        stmt=s,
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # not good

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
    #     it = np.nditer(
    #         [shaped, normal],
    #         flags=["external_loop"],
    #         order="F",
    #         op_flags=[["readwrite"], ["readonly"]],
    #     )
    #     _timeit(
    #         description="AR nditer reused",
    #         stmt=s,
    #         setup="import numpy as np",
    #         _globals=locals(),
    #         reference=ref,
    #         repeat=_TIMEIT_REPEAT,
    #         number=_TIMEIT_NUMBER,
    #     )  # not working properly

    s = """\
normal = _normal_ref_t60.copy()
result = signal.lfilter(_iir_b, _iir_a, normal)
result = result[:, _iir_t60:]  # skip transient response
"""
    _timeit(
        description="IIR with lfilter()",
        stmt=s,
        setup="from scipy import signal",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )  # good (best IIR)


def _timeit_sp():
    _TIMEIT_REPEAT = 10

    _TIMEIT_NUMBER = 200
    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 110

    # _TIMEIT_NUMBER = 2000
    # _BLOCK_LENGTH = 256
    # _CHANNEL_COUNT = 32

    data = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH))

    def _test_performance(d, dtype, r):
        d = d.astype(dtype)  # makes a copy
        # tools.plot_ir_and_tf(d[0], fs=48000, is_show_blocked=False)
        return _timeit(
            description=f"numpy irfft(rfft()) as {d.dtype}",
            stmt="np.fft.irfft(np.fft.rfft(d, axis=-1), axis=-1)",
            setup="import numpy as np",
            _globals=locals(),
            reference=r,
            check_dtype=dtype,
            repeat=_TIMEIT_REPEAT,
            number=_TIMEIT_NUMBER,
        )
        # return _timeit(
        #     description=f"PYFFTW irfft(rfft()) as {d.dtype}",
        #     stmt="pyfftw.builders.irfft(pyfftw.builders.rfft(d, axis=-1), axis=-1)",
        #     setup="import pyfftw",
        #     _globals=locals(),
        #     reference=r,
        #     check_dtype=dtype,
        #     repeat=_TIMEIT_REPEAT,
        #     number=_TIMEIT_NUMBER,
        # )

    print("comparison of dtypes")
    ref = _test_performance(data, np.float64, None)
    _test_performance(data, np.float32, ref)
    _test_performance(data, np.float16, ref)


def _timeit_sht():
    # print(f"system switcher interval is {sys.getswitchinterval()}")
    # print(f"system recursion limit is {sys.getrecursionlimit()}")
    # sys.setrecursionlimit(1500)
    # print(f"system recursion limit is {sys.getrecursionlimit()}")

    _TIMEIT_REPEAT = 10
    _TIMEIT_NUMBER = 2000

    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 110
    _SH_MAX_ORDER = 8

    _SH_COEFFICIENTS = pow(_SH_MAX_ORDER + 1, 2)

    spherical_harmonic_bases = tools.generate_noise(
        (_SH_COEFFICIENTS, _CHANNEL_COUNT), dtype=np.complex128
    )
    data = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH), dtype=np.complex128)

    def _test_performance(shb, d, dtype):
        shb = shb.astype(dtype)  # makes a copy
        d = d.astype(dtype)  # makes a copy

        ref = _timeit(
            description=f"numpy.inner() as {d.dtype}",
            stmt="np.inner(shb, d.T)",
            setup="import numpy as np",
            _globals=locals(),
            repeat=_TIMEIT_REPEAT,
            number=_TIMEIT_NUMBER // (2 * _SH_COEFFICIENTS),
        )
        r1 = _timeit(
            description=f"numpy.einsum() as {d.dtype}",
            stmt='np.einsum("ij,jl", shb, d)',
            setup="import numpy as np",
            _globals=locals(),
            reference=ref,
            repeat=_TIMEIT_REPEAT,
            number=_TIMEIT_NUMBER // (2 * _SH_COEFFICIENTS),
        )
        r2 = _timeit(
            description=f"numpy.dot() as {d.dtype}",
            stmt="np.dot(shb, d)",
            setup="import numpy as np",
            _globals=locals(),
            reference=ref,
            repeat=_TIMEIT_REPEAT,
            number=_TIMEIT_NUMBER // (2 * _SH_COEFFICIENTS),
        )

        print()
        return [ref, r1, r2]

    print("comparison of dtypes")
    # noinspection PyUnusedLocal
    rs = [
        _test_performance(spherical_harmonic_bases, data, np.complex128),
        _test_performance(spherical_harmonic_bases, data, np.complex64),
    ]
    # for r in range(len(rs[0])):
    #     print(f"results {r} max: {np.abs(np.subtract(*rs[:][r][1])).max()}")


def _timeit_basic():
    _TIMEIT_REPEAT = 10
    _TIMEIT_NUMBER = 2000

    _BLOCK_LENGTH = 4096
    _CHANNEL_COUNT = 110

    data = tools.generate_noise((_CHANNEL_COUNT, _BLOCK_LENGTH), dtype=np.complex128)

    ref = _timeit(
        description="multiply",
        stmt="data * data",
        setup="",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )
    _timeit(
        description="numpy.multiply()",
        stmt="np.multiply(data, data)",
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )
    print()

    ref = _timeit(
        description="abs",
        stmt="abs(data)",
        setup="",
        _globals=locals(),
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )
    _timeit(
        description="numpy.abs()",
        stmt="np.abs(data)",
        setup="import numpy as np",
        _globals=locals(),
        reference=ref,
        repeat=_TIMEIT_REPEAT,
        number=_TIMEIT_NUMBER,
    )


def _test_multiprocessing():
    import jack

    client = jack.Client(name="PICKLE_TEST")
    print("--- CLIENT CREATED ---\n")

    # import dill
    #
    # print(dill.detect.badtypes(client, depth=1).keys())
    # print("--- DILL BADTYPES ---\n")
    #
    # dill.detect.trace(True)
    # print(dill.pickles(client))
    # print("--- DILL TRACED ---\n")

    import pickle

    # This fails since pickling is not possible for `jack.Client`, see `_multiprocessing`
    print(pickle.dumps(obj=client, protocol=pickle.HIGHEST_PROTOCOL))
    print("--- PICKLE DUMPED ---\n")

    client.activate()
    client.inports.register(f"input_{1}")
    client.outports.register(f"output_{1}")
    print("--- CLIENT ACTIVATED ---\n")

    # This fails since pickling is not possible for `jack.Port`, see `_multiprocessing`
    print(pickle.dumps(obj=client, protocol=pickle.HIGHEST_PROTOCOL))
    print("--- PICKLE DUMPED ---\n")

    client.deactivate()
    client.close()
    print("--- CLIENT CLOSED ---\n")


def _test_client_name_length():
    def _generate_name(le):
        nums = list(range(1, 10))
        nums.append(0)
        n = "".join("".join(str(i) for i in nums) for _ in range(le // 10))
        return f"{n}{''.join(str(i) for i in nums[:le % 10])}"

    from ._jack_client import JackClient

    # Test for different client name lengths
    # 27 used to be the maximum length due to semaphore length limitations on macOS
    # 63 seems to be the maximum length in more recent version of Jack on macOS
    for length in [10, 27, 28, 63, 64, 100]:

        # create client
        name = _generate_name(le=length)
        print(f'creating client with name "{name}"')
        client = JackClient(
            name=name,
            is_main_client=False,
            is_disable_file_logger=True,
            is_disable_logger=True,
        )

        # evaluate name length
        is_correct = len(client.name) == length
        print(
            f"correct client name length ({length}): {is_correct}\n",
            file=sys.stdout if is_correct else sys.stderr,
        )
        sleep(0.05)  # to get correct output order

        # terminate client
        try:
            client.terminate()
            client.join()
        except AttributeError:
            pass


def _test_client_name_lock():
    """
    Test whether arbitrarily many clients with the same name can be instantiated consecutively
    (the old client will be terminated before creating the new one).

    This should not be a problem or an unusual use case for Jack to handle. However, this test
    revealed some problems as documented in https://github.com/jackaudio/jack2/issues/658 and
    https://github.com/spatialaudio/jackclient-python/issues/98.

    On macOS, creating the 99th instance fails with a jack.JackOpenError when initializing the
    client. This occurred neither on Linux nor on Windows based on the same Jack version.

    After the failure occurs no clients with that name can be instantiated at all. This persists
    even through a restart of Jack. AFAIK only a system restart helps to resolve the lock.
    """
    import jack

    try:
        i = 0
        while True:
            i += 1

            # name = f"Client{i:d}"  # runs for arbitrarily many clients
            name = f"Client"  # fails for the 99th instance

            print(f'Test {i:d}: creating "{name}" ...')
            client = jack.Client(name=name)
            client.activate()
            client.deactivate()
            client.close()
            del client

    except KeyboardInterrupt:
        print("... interrupted by user.")
