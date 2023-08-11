#!/usr/bin/env python3

#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

#
# usage: generate_hw_testcases [-h] {write-configs,export-excel,export-csv} ...
#
# Create hardware test cases
#
# positional arguments:
#   {write-configs,export-excel,export-csv}
#     write-configs       Write test case configurations and sample data
#     export-excel        Write test cases as an Excel spreadsheet
#     export-csv          Write test cases as an CSV spreadsheet
#
# optional arguments:
#   -h, --help            show this help message and exit
#

from abc import ABC, abstractmethod, abstractproperty
import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import functools
import itertools
import importlib
import math
import operator
import pandas as pd
from pathlib import Path
import os
import re
import sys
import traceback
import warnings
from typing import Callable, List, Optional, Sequence, Union
import numpy as np
import numpy.ma as ma
from struct import pack, unpack
from numpy.random import default_rng
from openpyxl.styles.alignment import Alignment
from openpyxl.utils import get_column_letter
from scipy.sparse import rand as randSparse
from scipy.sparse import random as randomSparse
import bitarray
import cmath

# TODO: Fix this awful hack, whose purpose in life is to point to where you've
# checked out `NUMERICSBENCH_REPO`
import os
numericBenchPath = os.getenv('NUMERICSBENCH_PATH')
if numericBenchPath == None:
    print("env variable NUMERICSBENCH_PATH not set, Using default path for NumericsBench")
    sys.path.append(str(Path(__file__).parents[5] / 'NumericsBench'))
else:
    sys.path.append(numericBenchPath)

from operators.compute_core import bfloat16
from operators.platform.quantize_info import QuantizationInfo
from operators.platform.quantized_tensor import NBQuantized


def numerics_bench_imports(arch):
    global Add, Mult, Conv2DVPUX, MaxPool, AveragePool, PRelu, RequantFuseWithPRelu
    global HSwish, Sigmoid, Softmax, PlatformVPU
    arch = "vpu26"
    conv_arch = "37XX"
    PlatformVPU = getattr(importlib.import_module(f"operators.platform.{arch}"), f"Platform{arch.upper()}")
    Conv2DVPUX = getattr(importlib.import_module(f"operators.{arch}"), f"Conv2DVPUX{conv_arch}")
    numerics_bench_operators = importlib.import_module(f"operators.{arch}")
    sys.modules["NB_operators"] = numerics_bench_operators
    from NB_operators import Add, Mult, MaxPool, AveragePool, PRelu, RequantFuseWithPRelu
    from NB_operators import HSwish, Sigmoid, Softmax


class Architecture(Enum):
    VPUX30XX = 3700,
    VPUX37XX = 3720

    def __str__(self):
       return self.name


ALL_ARCHITECTURES = [Architecture.VPUX37XX]


class CompilerBackend(Enum):
    Flatbuffer = auto(),
    ELF = auto()


Orderer = Callable[[np.ndarray], np.ndarray]

def tohex(val, nbits):
    return hex((val + (1 << nbits)) % (1 << nbits))

def OrderNHWC(data: np.ndarray) -> np.ndarray:
    return np.concatenate([a.transpose(1, 2, 0).flatten() for a in data])

def OrderNWHC(data: np.ndarray) -> np.ndarray:
    return np.concatenate([a.transpose(2, 1, 0).flatten() for a in data])

def OrderNWCH(data: np.ndarray) -> np.ndarray:
    return np.concatenate([a.transpose(2, 0, 1).flatten() for a in data])

def OrderNCWH(data: np.ndarray) -> np.ndarray:
    return np.concatenate([a.transpose(0, 2, 1).flatten() for a in data])

def OrderNHCW(data: np.ndarray) -> np.ndarray:
    return np.concatenate([a.transpose(1, 0, 2).flatten() for a in data])

def OrderNCHW(data: np.ndarray) -> np.ndarray:
    return data

class Order(Enum):
    NHWC = 0    # ZXY
    NWHC = 1    # ZYX
    NWCH = 2    # YZX
    NCWH = 3    # YXZ
    NHCW = 4    # XZY
    NCHW = 5    # XYZ

SW2HWOrder = {
    Order.NHWC: 'ZXY',
    Order.NWHC: 'ZYX',
    Order.NWCH: 'YZX',
    Order.NCWH: 'YXZ',
    Order.NHCW: 'XZY',
    Order.NCHW: 'XYZ',
}

class MPE_MODE(Enum):
    CUBOID_16x16 = auto()
    CUBOID_8x16 = auto()
    CUBOID_4x16 = auto()

mpeCube2NTHW_NTK = {
    MPE_MODE.CUBOID_16x16: '16x4',
    MPE_MODE.CUBOID_4x16: '4x16',
    MPE_MODE.CUBOID_8x16: '8x8'
}

class SEGMENTATION(Enum):
    SOK = 1
    SOH = 2
    SOW = 3
    SOHW = 4
    SOHK = 5

def orderToOrderer(order: Order) -> np.ndarray:
    if order == Order.NHWC:
        return OrderNHWC
    elif order == Order.NWHC:
        return OrderNWHC
    elif order == Order.NWCH:
        return OrderNWCH
    elif order == Order.NCWH:
        return OrderNCWH
    elif order == Order.NHCW:
        return OrderNHCW
    elif order == Order.NCHW:
        return OrderNCHW
    else:
        raise ValueError('output order is not supported: %s', order.name.lower())

def PadNCHWChannels(data: np.ndarray) -> np.ndarray:
    data = data.reshape(data.shape[0], functools.reduce(operator.mul, data.shape[1:]))
    if data.shape[1] & 0xF:
        zeros = np.zeros((data.shape[0], 0x10 - (data.shape[1] & 0xF)), dtype=data.dtype)
        data = np.append(data, zeros, axis=1)
    return data


class Error(Exception):
    pass


class ComputationError(Error):
    pass


class ValidationError(Error):
    pass


class PaddingError(Error):
    pass


class EntropyError(Error):
    pass


class AlignmentError(Error):
    pass


def ValidatePaddings(kernel, paddings):
    # kernel size are width|height
    # The padding order is top|left|bottom|right
    # Regarding documentation (http://dub30.ir.intel.com/svn/TRUNK/keembay/docs/specification/pdf/Gen3_Intel_Movidius_VPU_3400VE-A0_Databook_v1.4.pdf KB databook (page 5558))
    # we have next paddings constraints:
    # When the kernel x dimension is odd, the PAD amount is [KERNEL_X-1]/2 on left and right
    # When the kernel y dimension is odd, the PAD amount is [KERNEL_Y-1]/2 on top and bottom
    # When the kernel x dimension is even, the PAD amount is [KERNEL_X]/2 on left and [KERNEL_X]/2-1 on right
    # When the kernel y dimension is even, the PAD amount is [KERNEL_Y]/2 on top and [KERNEL_Y]/2-1 on bottom

    kernel_y = kernel[0]
    kernel_x = kernel[1]

    top = paddings[0]
    left = paddings[1]
    bottom = paddings[2]
    right = paddings[3]

    if kernel_x % 2 != 0:
        if left > (kernel_x - 1) // 2:
            raise PaddingError(f'kernel.x ({kernel_x}) is odd, and left padding ({left}) > (kernel.x - 1) // 2 ({(kernel_x - 1) // 2})')
        if right > ((kernel_x - 1) // 2):
            raise PaddingError(f'kernel.x ({kernel_x}) is odd, and right padding ({right}) > (kernel.x - 1) // 2 ({(kernel_x - 1) // 2})')
    else:
        if left > kernel_x // 2:
            raise PaddingError(f'kernel.x ({kernel_x}) is even, and left padding ({left}) > kernel.x // 2 ({kernel_x // 2})')
        if right > kernel_x // 2:
            raise PaddingError(f'kernel.x ({kernel_x}) is even, and right padding ({right}) > kernel.x // 2 ({kernel_x // 2})')

    if kernel_y % 2 != 0:
        if top > (kernel_y - 1) // 2:
            raise PaddingError(f'kernel.y ({kernel_y}) is odd, and top padding ({top}) > (kernel.y - 1) // 2 ({(kernel_y - 1) // 2})')
        if bottom > (kernel_y - 1) // 2:
            raise PaddingError(f'kernel.y ({kernel_y}) is odd, and bottom padding ({bottom}) > (kernel.y - 1) // 2 ({(kernel_y - 1) // 2})')
    else:
        if top > kernel_y // 2:
            raise PaddingError(f'kernel.y ({kernel_y}) is even, and top padding ({top}) > kernel.y // 2 ({kernel_y // 2})')
        if bottom > kernel_y // 2:
            raise PaddingError(f'kernel.y ({kernel_y}) is even, and bottom padding ({bottom}) > kernel.y // 2 ({kernel_y // 2})')


def CheckHWAlignment(type, multipyer):
    if((type.bitsize * multipyer) % 128 != 0) :
        return False
    return True


def ValidateHWAlignment(type, multipyer):
    if(not CheckHWAlignment(type, multipyer)) :
        raise AlignmentError(f'type ({type}) has {type.bitsize} bits and unappropriate multipyer {multipyer})')


def ValidateSwizzlingKey(key):
    if key is not None and key not in range(1,6):
        raise ValidationError(f'swizzling key must be between values 1-5. Value provided: {key}')


def getSEParams(chanCount, seSize=None):
    if (seSize == None):
        seSize = chanCount
    (numSEs, rem)  = divmod(chanCount, seSize)
    if (rem != 0):
        raise Error("Incorrect Storage Element size")
    return (seSize, numSEs)

def compressSEs(sparseNotCompressed: np.ndarray, seSize=None) -> np.ndarray:
    shape = sparseNotCompressed.shape
    (seSize, numSEs) = getSEParams(shape[1], seSize)
    dtype = sparseNotCompressed.dtype
    compressedSETensor = np.zeros(shape, dtype)
    for N in range(shape[0]):
        for H in range(shape[2]):
            for W in range(shape[3]):
                for se in range(numSEs):
                    compressedSETensor[N,se*seSize:se*seSize + seSize,H,W] \
                    = sorted(sparseNotCompressed[N,se*seSize:se*seSize + seSize,H,W], key=lambda x: not x)
    return compressedSETensor

def decompressSEs(sparseTensor: np.ndarray, sparsityMap: np.ndarray, seSize=None) -> np.ndarray:
    shape = sparseTensor.shape
    (seSize, numSEs) = getSEParams(shape[1], seSize)
    dtype = sparseTensor.dtype
    denseTensor = np.zeros(shape, dtype)
    for N in range(shape[0]):
        for H in range(shape[2]):
            for W in range(shape[3]):
                for se in range(numSEs):
                    seSM = sparsityMap[N,se*seSize:se*seSize + seSize,H,W]
                    compressedIdx = 0
                    for smIdx, smElem in enumerate(seSM):
                        if (smElem):
                            denseTensor[N,smIdx+se*seSize,H,W] = sparseTensor[N,compressedIdx+se*seSize,H,W]
                            compressedIdx += 1
    return denseTensor

@dataclass
class Value:
    ttype: 'TType'
    filename: str
    data: np.ndarray
    bitwidth: int
    bitsize: int
    signed: bool
    orderer: Optional[Orderer]
    is_float: bool = field(init=False)
    scale: float = field(init=False, default=1.)
    zero: int = field(init=False, default=0)

    def __post_init__(self):
        self.is_float = self.ttype.is_float

    @property
    def low(self):
        low = 0
        if self.signed:
            low -= 2 ** self.bitwidth
        return low

    @property
    def high(self):
        return 2 ** self.bitwidth - 1

    def write_data(self, dir: Path, orderer: Orderer) -> None:
        if self.orderer:
            orderer = self.orderer
        data = orderer(self.data)
        data[data == np.NZERO] = np.PZERO
        self.ttype.pack(self, data).tofile(dir / self.filename)

    def check_entropy(self):
        self.ttype.check_entropy(self.data)

    @property
    def json_info(self):
        info = {
            'shape': self.data.shape,
            'dtype': self.ttype.stype,
            'quantization': {
                'scale': self.scale,
                'zeropoint': self.zero,
                'low_range': 0 if self.is_float else self.low,
                'high_range': 1 if self.is_float else self.high
            }
        }
        return info


class TType(ABC):
    def __init__(self, dtype: np.dtype, stype: str, qtype: str, bitwidth: int, signed: bool):
        self.dtype = dtype
        self.stype = stype
        self.qtype = qtype
        self.bitwidth = bitwidth
        self.signed = signed

    @abstractmethod
    def generate(self, filename, shape, rng) -> Value:
        pass

    @abstractmethod
    def check_entropy(self, data: np.ndarray):
        pass

    @abstractproperty
    def is_float(self) -> bool:
        pass

    @property
    def zero(self) -> Union[float, int]:
        return 0. if self.is_float else 0

    def pack(self, value: Value, data: np.ndarray) -> np.ndarray:
        return data

    def bias(self, data: np.ndarray) -> np.ndarray:
        return data - self.zero

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data

    @staticmethod
    def _check_entropy_eq(data: np.ndarray, value):
        count = np.sum(np.equal(data, value))
        if (data.size * .9) < count:
            raise EntropyError(f'got {count} elements == {value} in {data.size} elements')

    @staticmethod
    def _check_entropy_inf(data: np.ndarray):
        count = np.sum(np.isinf(data))
        if (data.size * .9) < count:
            raise EntropyError(f'got {count} infinite elements in {data.size} elements')

    @staticmethod
    def _check_entropy_nan(data: np.ndarray):
        if np.any(np.isnan(data)):
            raise EntropyError(f'got NaN elements')

def pack_u1(data: np.ndarray) -> np.ndarray:
    return np.packbits(np.array(data.flatten(), dtype=bool), bitorder='little')

def pack_int4(data: np.ndarray) -> np.ndarray:
    flat = data.flatten()
    result = []
    for idx in range(0, flat.size, 2):
        lsn = flat[idx + 0] & 0x0f
        msn = flat[idx + 1] & 0x0f
        datum = np.uint8(msn << 4 | lsn)
        result.append(datum)
    return np.array(result).astype(np.uint8)


class U1(TType):
    def __init__(self):
        super().__init__(np.uint8, 'u1', 'int8', 1, False)
        self.bitsize = 1
        self.low = np.uint8(0)
        self.high = np.uint8(1)

    def generate(self, filename: str, shape, rng, orderer=None) -> Value:
        return Value(self,
                     filename,
                     rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.uint8),
                     self.bitwidth,
                     self.bitsize,
                     False,
                     orderer)

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_eq(data, 0)
        self._check_entropy_eq(data, 1)

    @property
    def is_float(self) -> bool:
        return False

    def pack(self, value: Value, data: np.ndarray) -> np.ndarray:
        return pack_u1(data)

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(0, 1)

class UInt4(TType):
    def __init__(self, bitwidth=4):
        super().__init__(np.uint8, 'uint4', 'int8', bitwidth, False)
        self.bitsize = 4
        self.low = np.uint8(0)
        self.high = np.uint8((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> Value:
        return Value(self,
                     filename,
                     rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.uint8),
                     self.bitwidth,
                     self.bitsize,
                     False,
                     orderer)

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_eq(data, 0)
        self._check_entropy_eq(data, 15)

    @property
    def is_float(self) -> bool:
        return False

    def pack(self, value: Value, data: np.ndarray) -> np.ndarray:
        return pack_int4(data)

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(0, 15)


class Int4(TType):
    def __init__(self, bitwidth=3):
        super().__init__(np.int8, 'int4', 'int8', bitwidth, True)
        self.bitsize = 4
        self.low = np.int8(-(2 ** bitwidth))
        self.high = np.int8((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> Value:
        return Value(self,
                     filename,
                     rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.int8),
                     self.bitwidth,
                     self.bitsize,
                     True,
                     orderer)

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_eq(data, -8)
        self._check_entropy_eq(data, 0)
        self._check_entropy_eq(data, 7)

    @property
    def is_float(self) -> bool:
        return False

    def pack(self, value: Value, data: np.ndarray) -> np.ndarray:
        return pack_int4(data)

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(-8, 7)


class UInt8(TType):
    def __init__(self, bitwidth=8):
        super().__init__(np.uint8, 'uint8', 'uint8', bitwidth, False)
        self.bitsize = 8
        self.low = np.uint8(0)
        self.high = np.uint8((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> Value:
        return Value(self,
                     filename,
                     rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.uint8),
                     self.bitwidth,
                     self.bitsize,
                     False,
                     orderer)

    def generateSparse(self, filename: str, shape, rng, sparsity_factor=0.5, orderer=None) -> Value:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        matrix = randSparse(1, np.prod(shape), density=1-sparsity_factor, format="csr", random_state=rng)
        floatSparseData = np.around(matrix.toarray().reshape(shape) * 8.) / 8.
        data = rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.uint8)
        for N in range(shape[0]):
            for C in range(shape[1]):
                for H in range(shape[2]):
                    for W in range(shape[3]):
                        if floatSparseData[N][C][H][W] == 0:
                            data[N][C][H][W] = 0
        return Value(self,
                     filename,
                     data,
                     self.bitwidth,
                     self.bitsize,
                     False,
                     orderer)

    # Activation sparsity is different because all non zero values are written into beggining of SE and the rest are zeroes
    # example: SE = [12,13,14,15,0,0,0,0] and SM [1,0,1,0,1,0,0,1], if input is SOK then several SEs will be uses along axis
    def generateSparseAct(self, data_file: str, sm_file: str, shape, rng, sparsity_factor=0.5, orderer=None, seSize=None):
        sparseNotCompressed = self.generateSparse(data_file, shape, rng, sparsity_factor, orderer)
        compressedSETensor = compressSEs(sparseNotCompressed.data, seSize)

        return (Value(self, data_file, compressedSETensor, self.bitwidth, self.bitsize, False, orderer),
                Value(U1(), sm_file, np.array(sparseNotCompressed.data, dtype=bool), 1, 1, False, orderer))

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_eq(data, 0)
        self._check_entropy_eq(data, 255)

    @property
    def is_float(self) -> bool:
        return False

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(0, 255)

class Int8(TType):
    def __init__(self, bitwidth=7, value=None):
        super().__init__(np.int8, 'int8', 'int8', bitwidth, True)
        self.bitsize = 8
        self.low = np.int8(-(2 ** bitwidth))
        self.high = np.int8((2 ** bitwidth) - 1)
        self.value = value

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        if self.value:
            self.low = self.high = self.value

        return Value(self,
                      filename,
                      rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.int8),
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def generateSparse(self, filename: str, shape, rng, sparsity_factor=0.5, orderer=None) -> Value:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        matrix = randSparse(1, np.prod(shape), density=1-sparsity_factor, format="csr", random_state=rng)
        floatSparseData = np.around(matrix.toarray().reshape(shape) * 8.) / 8.
        data = rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.int8)
        for N in range(shape[0]):
            for C in range(shape[1]):
                for H in range(shape[2]):
                    for W in range(shape[3]):
                        if floatSparseData[N][C][H][W] == 0:
                            data[N][C][H][W] = 0
        return Value(self,
                    filename,
                    data,
                    self.bitwidth,
                    self.bitsize,
                    False,
                    orderer)

    def generateSparseAct(self, data_file: str, sm_file: str, shape, rng, sparsity_factor=0.5, orderer=None, seSize=None):
        sparseNotCompressed = self.generateSparse(data_file, shape, rng, sparsity_factor, orderer)
        compressedSETensor = compressSEs(sparseNotCompressed.data, seSize)

        return (Value(self, data_file, compressedSETensor, self.bitwidth, self.bitsize, False, orderer),
                Value(U1(), sm_file, np.array(sparseNotCompressed.data, dtype=bool), 1, 1, False, orderer))

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_eq(data, -128)
        self._check_entropy_eq(data, 0)
        self._check_entropy_eq(data, 127)

    @property
    def is_float(self) -> bool:
        return False

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(-128, 127)


class Int32(TType):
    def __init__(self, bitwidth=31):
        super().__init__(np.int32, 'int32', 'int32', bitwidth, True)
        self.bitsize = 32
        self.low = np.int32(-(2 ** bitwidth))
        self.high = np.int32((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        return Value(self,
                      filename,
                      rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.int32),
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_eq(data, self.low)
        self._check_entropy_eq(data, 0)
        self._check_entropy_eq(data, self.high)

    @property
    def is_float(self) -> bool:
        return False

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(self.low, self.high)


class FP16(TType):
    def __init__(self, bitwidth=16, value=None):
        super().__init__(np.float16, 'fp16', None, bitwidth, True)
        self.bitsize = 16
        self.value = value
    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        if self.value:
            data = np.around(rng.random(size=shape, dtype=np.float32) * 8.) / 8.
            data = (data * (2. ** self.bitwidth)).astype(np.float16)
            data[data == np.inf] = self.value
        else:
            data = np.around((2 * rng.random(size=shape, dtype=np.float32) - 1) * 8.) / 8.
            data = (data * (2. ** self.bitwidth)).astype(np.float16)
            data[data == np.NZERO] = np.PZERO

        return Value(self,
                      filename,
                      data,
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def generateSparse(self, filename: str, shape, rng, sparsity_factor=0.5, orderer=None) -> Value:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        matrix = randomSparse(1, np.prod(shape), density=1-sparsity_factor, format="csr", random_state=rng, data_rvs=lambda s: (2* rng.random(size=s, dtype=np.float32) - 1))
        data = np.around(matrix.toarray().reshape(shape) * 8.) / 8.

        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(np.float16),
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def generateSparseAct(self, data_file: str, sm_file: str, shape, rng, sparsity_factor=0.5, orderer=None, seSize=None):
        sparseNotCompressed = self.generateSparse(data_file, shape, rng, sparsity_factor, orderer)
        compressedSETensor = compressSEs(sparseNotCompressed.data, seSize)

        return (Value(self, data_file, compressedSETensor, self.bitwidth, self.bitsize, True, orderer),
                 Value(U1(), sm_file, np.array(sparseNotCompressed.data, dtype=bool), 1, 1, False, orderer))

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_inf(data)
        self._check_entropy_nan(data)

    @property
    def is_float(self) -> bool:
        return True

    def clip(self, data: np.ndarray) -> np.ndarray:
        # Translate negative zeros to positive zeros.
        data = data + 0.0

        # NB This is "Round to nearest ties to even" mode by default;
        #    we'll need to augment this if we want to test the other modes.
        return data.astype(np.float16)


class FP32(TType):
    def __init__(self, bitwidth=127):
        super().__init__(np.float32, 'fp32', None, bitwidth, True)
        self.bitsize = 32

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        data = np.around((2 * rng.random(size=shape, dtype=np.float32) - 1) * 8.) / 8.
        data[data == np.NZERO] = np.PZERO
        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(np.float32),
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_inf(data)
        self._check_entropy_nan(data)

    @property
    def is_float(self) -> bool:
        return True

    def clip(self, data: np.ndarray) -> np.ndarray:
        # Translate negative zeros to positive zeros.
        return data + 0.0


class BF16(TType):
    def __init__(self, bitwidth=127):
        super().__init__(bfloat16, 'bfloat16', None, bitwidth, True)
        self.bitsize = 16

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        data = np.around((2 * rng.random(size=shape, dtype=np.float32) - 1) * 8.) / 8.
        data[data == np.NZERO] = np.PZERO
        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(bfloat16),
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def generateSparse(self, filename: str, shape, rng, sparsity_factor=0.5, orderer=None) -> Value:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        matrix = randomSparse(1, np.prod(shape), density=1-sparsity_factor, format="csr", random_state=rng, data_rvs=lambda s: (2* rng.random(size=s, dtype=np.float32) - 1))
        data = np.around(matrix.toarray().reshape(shape) * 8.) / 8.

        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(bfloat16),
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def generateSparseAct(self, data_file: str, sm_file: str, shape, rng, sparsity_factor=0.5, orderer=None, seSize=None):
        sparseNotCompressed = self.generateSparse(data_file, shape, rng, sparsity_factor, orderer)
        compressedSETensor = compressSEs(sparseNotCompressed.data, seSize)

        return (Value(self, data_file, compressedSETensor, self.bitwidth, self.bitsize, True, orderer),
                Value(U1(), sm_file, np.array(sparseNotCompressed.data, dtype=bool), 1, 1, False, orderer))

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_inf(data)
        self._check_entropy_nan(data)

    @property
    def is_float(self) -> bool:
        return True


def idu(input: Value, weights: Value) -> "tuple[np.ndarray, np.ndarray]":
    """Models the hardware IDU"""
    if input.is_float or weights.is_float:
        return input.data.astype(np.float32), weights.data.astype(np.float32)

    def to_qint32(value: Value) -> Union[np.ndarray, NBQuantized]:
        return NBQuantized(value=value.data.astype(np.int32), scale=value.scale, zero_point=value.zero,
                           platform=PlatformVPU(), quantization_info=QuantizationInfo(value.ttype.qtype))

    return to_qint32(input), to_qint32(weights)

def iduConvCustom(input: Value, weights: Value) -> "tuple[np.ndarray, np.ndarray]":
    """Custom Model the hardware IDU that feet the NumericBench requirements for convolution operation"""
    if (input.data.dtype == np.float32) or (weights.data.dtype == np.float32) :
        # Issue link: https://gitlab.devtools.intel.com/iotgai/NumericsBench/-/issues/247
        raise Error(f'NumericBench\'s convolution operation doesn\'t support float32 datatype for inputs/weights')

    def to_qint32(value: Value) -> Union[np.ndarray, NBQuantized]:
        return NBQuantized(value=value.data.astype(np.int32), scale=value.scale, zero_point=value.zero,
                           platform=PlatformVPU(), quantization_info=QuantizationInfo(value.ttype.qtype))
    if not input.is_float and not weights.is_float :
        return to_qint32(input), to_qint32(weights)

    if input.data.dtype == weights.data.dtype :
        return input.data, weights.data

    # NumericBench requires activations and weights types are equal meanwhile VPUX37XX hardware supports different data types

    if (input.data.dtype == bfloat16) or (weights.data.dtype == bfloat16) :
        # docs link https://docs.intel.com/documents/MovidiusInternal/vpu27/common/SW/HLD/internal/02_04_NN_LayerMappingHLD.html#input-data-type-support
        raise Error(f'bfloat16 activations compatible with bfloat16 weights only')
    # NumericBench's convolution operation doesn't support float32 datatype for inputs/weights so we have to convert types to float16 dtype
    # but there's accuracy loss with conversion from int16/int32 -> fp16 is possible
    if not input.is_float and input.bitsize >= 16 :
        warnings.warn(f'Possible accuracy loss during conversion from {input.data.dtype} -> {np.float16}')
    if not weights.is_float and weights.bitsize >= 16 :
        warnings.warn(f'Possible accuracy loss during conversion from {weights.data.dtype} -> {np.float16}')
    return input.data.astype(np.float16), weights.data.astype(np.float16)

class Operation(ABC):
    def __init__(self, architecture: Architecture):
        self.architecture = architecture

    """Abstract base class for MPE operations."""
    def json_info(self, inputs) -> dict:
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractproperty
    def ident(self) -> str:
        pass

    @abstractproperty
    def orderer(self) -> Orderer:
        pass

    @abstractproperty
    def data(self) -> dict:
        pass

    @abstractmethod
    def generate_inputs(self, rng) -> List[Value]:
        pass

    @abstractmethod
    def apply_mpe(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def filter_issues(self, args) -> bool:
        pass

    def ppe(self, values: List[Value], output_ttype: TType, data: Union[np.ndarray, NBQuantized], activation=None) -> Value:
        """Models the hardware PPE"""
        def getValue(collection):
            return collection.value if isinstance(collection, NBQuantized) else collection

        ndarray = getValue(data)
        #replace np.nan values with cmath.nan  (0xFE00 -> 0x7E00 fp16 representation)
        ndarray = np.nan_to_num(ndarray, nan=cmath.nan)

        rescale = not output_ttype.is_float
        output_type = output_ttype
        output_scale = 1.
        output_zero_point = 0
        result_bitwidth = math.ceil(np.log2(np.amax(abs(np.nan_to_num(ndarray)))))
        bitshift = max(result_bitwidth - output_type.bitwidth, 0)

        if rescale:
            if np.issubdtype(ndarray.dtype, np.integer):
                ndarray = ndarray.astype(np.float64)

            if isinstance(activation, (PReLU, ReLUX)):
                activated = np.where(ndarray < 0, ndarray * activation.slope, ndarray)
                max_ = max(np.amax(activated), 0)
                min_ = min(np.amin(activated), 0)

                if activation.out_type is np.int8:
                    distance = max(-min_, max_)
                    zero_point = 0
                    scale = distance / 127.

                if activation.out_type is np.uint8:
                    length = max_ - min_
                    zero_point = np.round((-min_ * 255.) / length)
                    scale = length / 255.

                output_scale = scale
                output_zero_point = zero_point
            else:
                # replace with quantization (#-29828)
                ndarray /= (1. * (1 << bitshift))

        if activation:
            data = activation(data, np.float32(output_scale), activation.out_type(output_zero_point))
            ndarray = getValue(data)
            output_type = activation.get_out_type()

        ndarray = output_type.clip(ndarray).astype(output_type.dtype)
        value = Value(output_type, 'output-0.bin', ndarray, output_type.bitwidth, output_type.bitsize, output_type.signed, None)

        if rescale:
            if isinstance(data, NBQuantized):
                if isinstance(activation, (PReLU, ReLUX)):
                    value.zero = output_zero_point
                    value.scale = output_scale
                else:
                    # replace with quantization (#-29828)
                    value.zero = int(data.zero_point)
                    value.scale = (1 << bitshift)
            else:
                # replace with quantization (#-29828)
                value.scale = (1 << bitshift)

        return value

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        return [output]


def shape_to_str(shape: Sequence[int]) -> str:
    return 'x'.join([str(d) for d in shape])

def get_values_json_info(values):
    return [val.json_info for val in values]

class PReLU:
    def __init__(self, architecture, slope, out_type=np.float16):
        self.architecture = architecture

        self.slope = slope
        self.out_type = out_type
        if self.out_type is np.int8:
            self.out_type_desc = 'int8'
        elif self.out_type is np.uint8:
            self.out_type_desc = 'uint8'
        else:
            self.out_type_desc = 'fp16'

    def __str__(self):
        return 'prelu_{}_{}'.format(self.slope, self.out_type_desc)

    @property
    def json_info(self):
        return {'architecture': self.architecture.name, 'name': 'PReLU', 'alpha': self.slope, 'output_type': self.out_type_desc}

    def __call__(self, values, scale=None, zero_point=None):
        if isinstance(values, NBQuantized):
            return RequantFuseWithPRelu().inference(values, self.slope, scale, zero_point)
        else:
            return PRelu(output_data_type=np.float16).inference(values, self.slope)

    def get_out_type(self):
        if self.out_type is np.int8:
            return Int8()
        if self.out_type is np.uint8:
            return UInt8()
        if self.out_type is np.float16:
            return FP16()

class ReLUX:
    def __init__(self, architecture, maximum, out_type=np.float16):
        self.architecture = architecture

        self.slope = 0
        self.maximum = maximum
        self.out_type = out_type
        if self.out_type is np.int8:
            self.out_type_desc = 'int8'
        elif self.out_type is np.uint8:
            self.out_type_desc = 'uint8'
        else:
            self.out_type_desc = 'fp16'

    def __str__(self):
        return 'relux_{}_{}'.format(self.maximum, self.out_type_desc)

    @property
    def json_info(self):
        if self.out_type_desc == 'fp16':
            max = unpack('H', np.float16(self.maximum))[0]
        else:
            max = self.maximum
        return {'architecture': self.architecture.name, 'name': 'ReLUX', 'output_type': self.out_type_desc, 'max': max}

    def __call__(self, values, scale=None, zero_point=None):
        if isinstance(values, NBQuantized):
            data = RequantFuseWithPRelu().inference(values, self.slope, scale, zero_point)
            data.value = np.clip(data.value, None, self.maximum)
            return data
        else:
            data = PRelu(output_data_type=np.float16).inference(values, self.slope)
            data = np.clip(data, None, self.maximum)
            return data

    def get_out_type(self):
        if self.out_type is np.int8:
            return Int8()
        if self.out_type is np.uint8:
            return UInt8()
        if self.out_type is np.float16:
            return FP16()

class ZMajorConvolution(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'weights_swizzling_key'
    ]
    NAME = 'Z-Major'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

        self.issues = set()

    def json_info(self, inputs, outputs):
        json = {}
        json['case_type'] = 'ZMajorConvolution'
        json['input'] = [inputs[0].json_info]
        json['weight'] = [inputs[1].json_info]
        json['output'] = get_values_json_info(outputs)
        json['conv_op'] = {
            'stride': self.settings.kernel_strides,
            'pad': self.settings.kernel_pads,
            'group': 1,
            'dilation': 1,
            'compress': self.settings.compress,
            'mpe_mode': self.settings.mpe_mode.name
        }
        json['output_order'] = self.settings.output_order.name.lower()

        if self.settings.weights_swizzling_key:
            json['weights_swizzling_key'] = self.settings.weights_swizzling_key

        return json


    def validate(self):
        ValidatePaddings(self.settings.kernel_shape, self.settings.kernel_pads)
        # validate input tensor channels alignment
        # ValidateHWAlignment(self.settings.input_ttype, self.settings.input_shape[1])
        # validate weight tensor channels alignment
        # ValidateHWAlignment(self.settings.weight_ttype, self.settings.input_shape[1])
        # validate output tensor channels alignment
        ValidateHWAlignment(self.settings.output_ttype, self.settings.kernel_channels)
        ValidateSwizzlingKey(self.settings.weights_swizzling_key)

    @property
    def ident(self) -> str:
        name = str(self.architecture) + f'_zm_conv_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_{shape_to_str(self.settings.weight_shape)}x{self.settings.weight_ttype.stype}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}_kern_chan_{self.settings.kernel_channels}'
        if self.settings.output_order != Order.NHWC:
            name += '_' + self.settings.output_order.name.lower()
        if self.settings.compress:
            name += '_compressed'
        if self.settings.mpe_mode != MPE_MODE.CUBOID_16x16:
            name += '_' + self.settings.mpe_mode.name
        if self.settings.weights_swizzling_key:
            name += '_wswizz_' + str(self.settings.weights_swizzling_key)
        return name

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
        }

    def generate_inputs(self, rng) -> List[Value]:
        inputs = [self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
                  self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)]
        return inputs

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        if 'E#34451' in self.issues:
            return False
        return True

class SparseConvolution(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'sparsity_factor',
        'act_sparsity'
    ]
    NAME = 'Sparse'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

        self.issues = set()

    def json_info(self, inputs, outputs):

        weights_list = []
        if self.settings.act_sparsity:
            weights_list.append(inputs[2].json_info)
        else:
            weights_list.append(inputs[1].json_info)

        json = {}
        json['case_type'] = 'SparseZMajorConvolution'
        json['input'] = [inputs[0].json_info]
        if self.settings.act_sparsity:
            json['sparsity_map_input'] = [inputs[1].json_info]
        json['weight'] = weights_list
        json['output'] = get_values_json_info(outputs)
        json['conv_op'] = {
            'stride': self.settings.kernel_strides,
            'pad': self.settings.kernel_pads,
            'group': 1,
            'dilation': 1,
            'compress': self.settings.compress,
            'mpe_mode': self.settings.mpe_mode.name,
            'act_sparsity': self.settings.act_sparsity
        }
        json['output_order'] = self.settings.output_order.name.lower()
        return json

    def validate(self):
        ValidatePaddings(self.settings.kernel_shape, self.settings.kernel_pads)
        # validate input tensor channels allignement
        # ValidateHWAlignment(self.settings.input_ttype, self.settings.input_shape[1])
        # validate weight tensor channels allignement
        # ValidateHWAlignment(self.settings.weight_ttype, self.settings.input_shape[1])
        # validate output tensor channels allignement
        ValidateHWAlignment(self.settings.output_ttype, self.settings.kernel_channels)

    @property
    def ident(self) -> str:
        name = str(self.architecture) + (f'_sparse_conv_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_'
                                         f'{shape_to_str(self.settings.weight_shape)}x{self.settings.weight_ttype.stype}_'
                                         f'pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}_kern_chan_{self.settings.kernel_channels}')
        if self.settings.output_order != Order.NHWC:
            name += '_' + self.settings.output_order.name.lower()
        if self.settings.compress:
            name += '_compressed'
        if self.settings.mpe_mode != MPE_MODE.CUBOID_16x16:
            name += '_' + self.settings.mpe_mode.name
        if self.settings.act_sparsity:
            name += '_act'
        name += f'_weights_sparsity_{self.settings.sparsity_factor}'
        return name

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'MPE Mode': 'ZMajorConv',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Weights Type': self.settings.weight_ttype.stype,
            'Kernel Channels': str(self.settings.kernel_channels),
            'Kernel Shape': ', '.join([str(s) for s in self.settings.kernel_shape]),
            'Output Type': self.settings.output_ttype.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        inputs = []
        if self.settings.act_sparsity:
            input = self.settings.input_ttype.generateSparseAct('input-0.bin', 'input-1.bin', self.settings.input_shape, rng, sparsity_factor=self.settings.sparsity_factor, seSize=self.settings.input_shape[1])
            inputs.append(input[0])
            inputs.append(input[1])
        else:
            inputs.append(self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng))
        inputs.append(self.settings.weight_ttype.generateSparse('weights.dat', self.settings.weight_shape, rng, self.settings.sparsity_factor, orderer=OrderNCHW))
        return inputs


    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        if self.settings.act_sparsity: #values: inputs, sm_inputs, weights
            inputs = Value(values[0].ttype, 'dense-input-0.bin',
                        decompressSEs(values[0].data, values[1].data, self.settings.input_shape[1]),
                        values[0].bitwidth, values[0].bitsize, values[0].signed, values[0].orderer)
            weights = values[2]
        else:                          #values: inputs, weights
            inputs = values[0]
            weights = values[1]

        lhs, rhs = iduConvCustom(inputs, weights)
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                     pads = self.settings.kernel_pads,
                     strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        if 'E#34451' in self.issues:
            return False
        return True

class DepthWiseConv(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'kernel_strides',
        'kernel_pads'
    ]
    NAME = 'DW'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, 1] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'DepthWiseConv',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': self.settings.weight_shape[0],
                'dilation': 1
            }
        }

    def validate(self):
        ValidatePaddings(self.settings.kernel_shape, self.settings.kernel_pads)

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_dw_conv_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_{shape_to_str(self.settings.weight_shape)}x{self.settings.input_ttype.stype}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}_kern_chan_{self.settings.kernel_channels}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'C': self.settings.input_shape[1],
            'H': self.settings.input_shape[2],
            'W': self.settings.input_shape[3],
            'K': self.settings.kernel_channels,
            'K_h': self.settings.kernel_shape[0],
            'K_w': self.settings.kernel_shape[1],
            'S_h': self.settings.kernel_strides[1],
            'S_w': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.input_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=PadNCHWChannels)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides,
                        group = self.settings.weight_shape[0])
        return c2d.inference(lhs, rhs)

    def filter_issues(self, args) -> bool:
        return True


class DoubleZMajorConvolution(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'activation_swizzling_key',
        'act_sparsity'
    ]
    NAME = 'Double-Z-Major'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

        self.issues = set()

    def json_info(self, inputs, outputs):
        weights_list = []
        weights_list.append(inputs[1].json_info)
        weights_list.append(inputs[2].json_info)

        json = {}
        json['case_type'] = 'DoubleZMajorConvolution'
        json['input'] = [inputs[0].json_info]
        json['weight'] = weights_list
        json['output'] = get_values_json_info(outputs)
        json['conv_op'] = {
            'stride': self.settings.kernel_strides,
            'pad': self.settings.kernel_pads,
            'group': 1,
            'dilation': 1,
            'compress': self.settings.compress,
            'mpe_mode': self.settings.mpe_mode.name,
            'act_sparsity': self.settings.act_sparsity
        }
        json['output_order'] = self.settings.output_order.name.lower()

        if self.settings.activation_swizzling_key:
            json['activation_swizzling_key'] = self.settings.activation_swizzling_key

        return json


    def validate(self):
        ValidatePaddings(self.settings.kernel_shape, self.settings.kernel_pads)
        ValidateHWAlignment(self.settings.output_ttype, self.settings.kernel_channels)
        ValidateSwizzlingKey(self.settings.activation_swizzling_key)


    @property
    def ident(self) -> str:
        name = str(self.architecture) + (f'_zm_conv_double_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}'
            f'_{shape_to_str(self.settings.weight_shape)}x{self.settings.weight_ttype.stype}'
            f'_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}_kern_chan_{self.settings.kernel_channels}'
        )
        if self.settings.output_order != Order.NHWC:
            name += '_' + self.settings.output_order.name.lower()
        if self.settings.compress:
            name += '_compressed'
        if self.settings.mpe_mode != MPE_MODE.CUBOID_16x16:
            name += '_' + self.settings.mpe_mode.name
        if self.settings.activation_swizzling_key:
            name += '_actswizz_' + str(self.settings.activation_swizzling_key)
        if self.settings.act_sparsity:
            name+= f'_sparse_act_seSize_{self.settings.kernel_channels}'
        return name

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
        }

    def generate_inputs(self, rng) -> List[Value]:
        inputs = []
        inputs.append(self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng))
        if self.settings.act_sparsity: # weights are in dense format, without SM. They do contain 0's with the purpose to produce sparse output
            inputs.append(self.settings.weight_ttype.generateSparse('weights.dat', self.settings.weight_shape, rng, 0.9, orderer=OrderNCHW))
        else:
            inputs.append(self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW))
        # weights shape: OC IC KH KW
        # output of the first conv will be used as input for the 2nd conv, so
        # new weights are generated, with the shape alligned with the intermediary output
        weights_shape_conv_1 = self.settings.weight_shape.copy()
        weights_shape_conv_1[1] = self.settings.weight_shape[0]
        inputs.append(self.settings.weight_ttype.generate('weights1.dat', weights_shape_conv_1, rng, orderer=OrderNCHW))
        return inputs

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        result = result.astype('float16')
        input_values_conv2 = Value(self.settings.output_ttype, None, result, self.settings.output_ttype.bitwidth, self.settings.output_ttype.bitsize, self.settings.output_ttype.signed, None)
        lhs1, rhs1 = iduConvCustom(input_values_conv2, values[2])
        result = c2d.inference(lhs1, rhs1)
        return result

    def filter_issues(self, args) -> bool:
        if 'E#34451' in self.issues:
            return False
        return True

class EltwiseAdd(Operation):

    PARAMS = ['op_class', 'input_ttype', 'input_shape', 'output_ttype']
    NAME = 'Add'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'EltwiseAdd',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs)
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_ew_add_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}'

    @property
    def orderer(self) -> Orderer:
        return OrderNCHW

    @property
    def output_orderer(self) -> Orderer:
        return OrderNCHW

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'C': self.settings.input_shape[1],
            'H': self.settings.input_shape[2],
            'W': self.settings.input_shape[3],
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.input_ttype.generate('input-1.bin', self.settings.input_shape, rng)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        adder = Add()
        lhs, rhs = idu(values[0], values[1])
        if isinstance(lhs, NBQuantized) and isinstance(lhs, NBQuantized):
            # Workaround for NumericsBench's Add operation: when both values are
            # quantized and have roughly the same scale, NumericsBench just casts
            # them to uint32, adds them, clips to [0, 255], and returns them as
            # uint8 -- which isn't correct for anything other than uint8.
            # So we go through the underlying platform add operation instead,
            # which is what Add() does anyway when the scales don't quite match.
            return adder.add.inference(lhs, rhs)
        return adder.inference(lhs, rhs)

    def filter_issues(self, args) -> bool:
        return True


class EltwiseMult(Operation):

    PARAMS = ['op_class', 'input_ttype', 'input_shape', 'output_ttype']
    NAME = 'Mult'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'EltwiseMult',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs)
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_ew_mult_{self.settings.input_ttype.stype}'

    @property
    def orderer(self) -> Orderer:
        return OrderNCHW

    @property
    def output_orderer(self) -> Orderer:
        return OrderNCHW

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'C': self.settings.input_shape[1],
            'H': self.settings.input_shape[2],
            'W': self.settings.input_shape[3],
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.input_ttype.generate('input-1.bin', self.settings.input_shape, rng)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        multer = Mult()
        lhs, rhs = idu(values[0], values[1])
        if isinstance(lhs, NBQuantized) and isinstance(lhs, NBQuantized):
            # Workaround for NumericsBench's Mult operation: see EltwiseMult.apply_mpe()
            return multer.inference(lhs, rhs)
        return multer.functor(lhs, rhs)

    def filter_issues(self, args) -> bool:
        return True

class EltwiseSparse(Operation):

    PARAMS = ['op_class', 'input_ttype', 'input_shape', 'output_ttype', 'seSize']
    NAME = 'EltwiseSparse'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'EltwiseSparse',
            'input': [inputs[0].json_info],
            'sparsity_map_input': [inputs[1].json_info],
            'weight': [inputs[2].json_info],
            'sparsity_map_weights': [inputs[3].json_info],
            'output': get_values_json_info(outputs),
            'ew_op': {
                'se_size': self.settings.seSize
            }
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_ew_sparse_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_se_size_{self.settings.seSize}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'C': self.settings.input_shape[1],
            'H': self.settings.input_shape[2],
            'W': self.settings.input_shape[3],
            'SE Size' : self.settings.seSize
        }

    def generate_inputs(self, rng) -> List[Value]:
        input = self.settings.input_ttype.generateSparseAct('input-0.bin', 'input-1.bin', self.settings.input_shape, rng, sparsity_factor=0.5, seSize=self.settings.seSize)
        weights = self.settings.input_ttype.generateSparseAct('input-2.bin', 'input-3.bin', self.settings.input_shape, rng, sparsity_factor=0.5, seSize=self.settings.seSize)

        return [
            input[0],
            input[1],
            weights[0],
            weights[1],
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        adder = Add()
        # use dense excecution for reference, decompress SEs for reference inplementation
        input = Value(values[0].ttype, 'dense-input-0.bin',
                      decompressSEs(values[0].data, values[1].data, self.settings.seSize),
                      values[0].bitwidth, values[0].bitsize, values[0].signed, values[0].orderer)
        weights = Value(values[2].ttype, 'dense-input-1.bin',
                        decompressSEs(values[2].data, values[3].data, self.settings.seSize),
                        values[2].bitwidth, values[2].bitsize, values[2].signed, values[2].orderer)
        lhs, rhs = idu(input, weights)
        if isinstance(lhs, NBQuantized) and isinstance(lhs, NBQuantized):
            # Workaround for NumericsBench's Add operation: when both values are
            # quantized and have roughly the same scale, NumericsBench just casts
            # them to uint32, adds them, clips to [0, 255], and returns them as
            # uint8 -- which isn't correct for anything other than uint8.
            # So we go through the underlying platform add operation instead,
            # which is what Add() does anyway when the scales don't quite match.
            return adder.add.inference(lhs, rhs)
        return adder.inference(lhs, rhs)

    def filter_issues(self, args) -> bool:
        return True

class Maxpool(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = ['op_class', 'input_ttype', 'input_shape', 'kernel_shape', 'output_ttype', 'kernel_strides', 'kernel_pads']
    NAME = 'MaxPool'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'MaxPool',
            'input': get_values_json_info(inputs),
            'output': get_values_json_info(outputs),
            'pool_op': {
                'sub_type': 'max',
                'kernel_shape': self.settings.kernel_shape,
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads
            }
        }

    def validate(self):
        ValidatePaddings(self.settings.kernel_shape, self.settings.kernel_pads)

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_max_pool_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_{shape_to_str(self.settings.kernel_shape)}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'C': self.settings.input_shape[1],
            'H': self.settings.input_shape[2],
            'W': self.settings.input_shape[3],
            'K_h': self.settings.kernel_shape[0],
            'K_w': self.settings.kernel_shape[1],
            'S_h': self.settings.kernel_strides[1],
            'S_w': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = idu(values[0], values[0])
        maxpool = MaxPool(kernel_shape=self.settings.kernel_shape, strides=self.settings.kernel_strides, pads=self.settings.kernel_pads)
        return maxpool.inference(lhs)

    def filter_issues(self, args) -> bool:
        return True


class AvgPool(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = ['op_class', 'input_ttype', 'input_shape', 'kernel_shape', 'output_ttype', 'kernel_strides']
    NAME = 'AvgPool'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings

    def json_info(self, inputs, outputs):
        # NB We need to rescale the output if it's quantized, since this is how the system implements the
        # division for quantized pool outputs.

        return {
            'case_type': 'AvgPool',
            'input': get_values_json_info(inputs),
            'output': get_values_json_info(outputs),
            'pool_op': {
                'sub_type': 'avg',
                'kernel_shape': self.settings.kernel_shape,
                'stride': self.settings.kernel_strides
            }
        }

    def validate(self):
        return True

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_avg_pool_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_kernel_{shape_to_str(self.settings.kernel_shape)}_strides_{shape_to_str(self.settings.kernel_strides)}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'C': self.settings.input_shape[1],
            'H': self.settings.input_shape[2],
            'W': self.settings.input_shape[3],
            'K_h': self.settings.kernel_shape[0],
            'K_w': self.settings.kernel_shape[1],
            'S_h': self.settings.kernel_strides[1],
            'S_w': self.settings.kernel_strides[0],
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = idu(values[0], values[0])
        avgpool = AveragePool(kernel_shape=self.settings.kernel_shape, strides=self.settings.kernel_strides, pads=[0, 0, 0, 0])
        return avgpool.inference(lhs)

    def filter_issues(self, args) -> bool:
        return True

class ActivationType(Enum):
    HSwish = auto()
    Sigmoid = auto()
    Softmax = auto()
    vau_sigm = auto()
    vau_sqrt = auto()
    vau_tanh = auto()
    vau_log = auto()
    vau_exp = auto()
    vau_dp4 = auto()
    vau_dp4a = auto()
    vau_dp4m = auto()
    sau_dp4 = auto()
    sau_dp4a = auto()
    sau_dp4m = auto()
    lsu_b16 = auto()
    lsu_b16_vec = auto()

class ActKernel(Operation):
    PARAMS = ['op_class', 'input_ttype', 'input_shape', 'output_ttype', 'activation_type']
    NAME = 'ActKernel'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        self.issues = set()

    @property
    def ident(self) -> str:
        name = self.settings.activation_type[0].name
        ident =  str(self.architecture) + f'_act_kernel_{name}_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}'
        if self.settings.activation_type[0] == ActivationType.Softmax :
            ident += f'_axis_{self.settings.activation_type[1]}'
        return ident

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        ActivationType2String = {
            ActivationType.HSwish: 'HSwish',
            ActivationType.Sigmoid: 'Sigmoid',
            ActivationType.Softmax: 'Softmax',
            ActivationType.vau_sigm: 'vau_sigm',
            ActivationType.vau_sqrt: 'vau_sqrt',
            ActivationType.vau_tanh: 'vau_tanh',
            ActivationType.vau_log: 'vau_log',
            ActivationType.vau_exp: 'vau_exp',
            ActivationType.vau_dp4: 'vau_dp4',
            ActivationType.vau_dp4a: 'vau_dp4a',
            ActivationType.vau_dp4m: 'vau_dp4m',
            ActivationType.sau_dp4: 'sau_dp4',
            ActivationType.sau_dp4a: 'sau_dp4a',
            ActivationType.sau_dp4m: 'sau_dp4m',
            ActivationType.lsu_b16: 'lsu_b16',
            ActivationType.lsu_b16_vec: 'lsu_b16_vec',
        }

        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Type': ActivationType2String[self.settings.activation_type[0]],
            # ignore activation_type since only HSwish is allowed
        }

    def validate(self):
        return True

    def generate_inputs(self, rng) -> List[Value]:
        inputs_list = []
        inputs_list.append(self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng))
        if self.settings.activation_type[0] in {ActivationType.vau_dp4, ActivationType.vau_dp4a, ActivationType.vau_dp4m, \
                                                ActivationType.sau_dp4, ActivationType.sau_dp4a, ActivationType.sau_dp4m}:
            inputs_list.append(self.settings.input_ttype.generate('input-1.bin', self.settings.input_shape, rng))
        return inputs_list

    def json_info(self, inputs, outputs):
        if self.settings.activation_type[0] in {ActivationType.vau_dp4, ActivationType.vau_dp4a, ActivationType.vau_dp4m, \
                                                ActivationType.sau_dp4, ActivationType.sau_dp4a, ActivationType.sau_dp4m}:
            assert(not(inputs[0].is_float))
            assert(not(inputs[1].is_float))
            assert(not(outputs[0].is_float))
        else:
            assert(outputs[0].is_float)

        json = {}
        json['case_type'] = 'ActShave'
        json['input'] = get_values_json_info(inputs)
        json['output'] = get_values_json_info(outputs)
        json['activation'] = {}
        json['activation']['name'] = self.settings.activation_type[0].name

        if self.settings.activation_type[0] == ActivationType.Softmax :
            json['activation']['axis']=self.settings.activation_type[1]

        return json

    # unsigned = False: 8-bit signed integer multiplication with sum to 32-bit signed integer per four element sub-vector and optional accumulate
    # unsigned = True: 8-bit signed integer by 8-bit unsigned integer multiplication with sum to 32-bit signed integer per four element sub-vector
    # input_types=[Int8()]
    # output_types=[Int32()]
    def golden_reference_vau_dp4(self, values: List[Value], acc=False, unsigned=False) -> np.ndarray:
        N = self.settings.input_shape[0]
        C = self.settings.input_shape[1] // 4
        H = self.settings.input_shape[2]
        W = self.settings.input_shape[3]

        accIdx = 0
        accum = [0, 0, 0, 0]
        output_shape = [N, C, H, W]
        result = np.zeros(output_shape, dtype = self.settings.output_ttype.dtype)

        assert(not(values[0].data.size % 16))
        assert(not(C % 4))

        for n in range(0, N):
            for h in range(0, H):
                for w in range(0, W):
                    for c in range(0, C):
                        if unsigned == True:
                            b1 = np.int32(np.uint8(values[1].data[n][c * 4 + 0][h][w]))
                            b2 = np.int32(np.uint8(values[1].data[n][c * 4 + 1][h][w]))
                            b3 = np.int32(np.uint8(values[1].data[n][c * 4 + 2][h][w]))
                            b4 = np.int32(np.uint8(values[1].data[n][c * 4 + 3][h][w]))
                        else:
                            b1 = np.int32(values[1].data[n][c * 4 + 0][h][w])
                            b2 = np.int32(values[1].data[n][c * 4 + 1][h][w])
                            b3 = np.int32(values[1].data[n][c * 4 + 2][h][w])
                            b4 = np.int32(values[1].data[n][c * 4 + 3][h][w])

                        a1 = np.int32(values[0].data[n][c * 4 + 0][h][w])
                        a2 = np.int32(values[0].data[n][c * 4 + 1][h][w])
                        a3 = np.int32(values[0].data[n][c * 4 + 2][h][w])
                        a4 = np.int32(values[0].data[n][c * 4 + 3][h][w])

                        if acc == True:
                            val = accum[accIdx % 4] + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4
                            accum[accIdx % 4] = val
                            accIdx+=1
                        else:
                            val = a1*b1 + a2*b2 + a3*b3 + a4*b4

                        result[n][c][h][w] = val

        return result

    # unsigned = False: 8-bit signed integer multiplication with sum to 32-bit signed integer and optional accumulate
    # unsigned = True: 8-bit signed integer by 8-bit unsigned integer multiplication with sum to 32-bit signed integer
    # input_types=[Int32()]
    # output_types=[Int32()]
    def golden_reference_sau_dp4(self, values: List[Value], acc=False, unsigned=False) -> np.ndarray:
        result = np.zeros(self.settings.input_shape, dtype = self.settings.output_ttype.dtype)
        val = 0

        N = self.settings.input_shape[0]
        C = self.settings.input_shape[1]
        H = self.settings.input_shape[2]
        W = self.settings.input_shape[3]

        for n in range(0, N):
            for h in range(0, H):
                for w in range(0, W):
                    for c in range(0, C):
                        if unsigned == True:
                            b1 = np.int32(np.uint8((values[1].data[n][c][h][w] >> 24) & 0xff))
                            b2 = np.int32(np.uint8((values[1].data[n][c][h][w] >> 16) & 0xff))
                            b3 = np.int32(np.uint8((values[1].data[n][c][h][w] >> 8) & 0xff))
                            b4 = np.int32(np.uint8((values[1].data[n][c][h][w] >> 0) & 0xff))
                        else:
                            b1 = np.int32(np.int8((values[1].data[n][c][h][w] >> 24) & 0xff))
                            b2 = np.int32(np.int8((values[1].data[n][c][h][w] >> 16) & 0xff))
                            b3 = np.int32(np.int8((values[1].data[n][c][h][w] >> 8) & 0xff))
                            b4 = np.int32(np.int8((values[1].data[n][c][h][w] >> 0) & 0xff))

                        a1 = np.int32(np.int8((values[0].data[n][c][h][w] >> 24) & 0xff))
                        a2 = np.int32(np.int8((values[0].data[n][c][h][w] >> 16) & 0xff))
                        a3 = np.int32(np.int8((values[0].data[n][c][h][w] >> 8) & 0xff))
                        a4 = np.int32(np.int8((values[0].data[n][c][h][w] >> 0) & 0xff))

                        if acc == True:
                            val += a1*b1 + a2*b2 + a3*b3 + a4*b4
                        else:
                            val = a1*b1 + a2*b2 + a3*b3 + a4*b4

                        result[n][c][h][w] = val

        return result

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        if self.settings.activation_type[0] == ActivationType.HSwish :
            result = HSwish().inference(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.Sigmoid :
            result = Sigmoid().inference(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.Softmax :
            result = Softmax(axis=self.settings.activation_type[1]).inference(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.vau_sigm :
            result = Sigmoid().inference(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.vau_sqrt :
            with np.errstate(invalid='ignore'):
                result = np.sqrt(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.vau_tanh :
            result = np.tanh(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.vau_log :
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.log(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.vau_exp :
            result = np.exp(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.vau_dp4 :
            result = self.golden_reference_vau_dp4(values, acc=False, unsigned=False)
        elif self.settings.activation_type[0] == ActivationType.vau_dp4a :
            result = self.golden_reference_vau_dp4(values, acc=True, unsigned=False)
        elif self.settings.activation_type[0] == ActivationType.vau_dp4m :
            result = self.golden_reference_vau_dp4(values, acc=False, unsigned=True)
        elif self.settings.activation_type[0] == ActivationType.sau_dp4 :
            result = self.golden_reference_sau_dp4(values, acc=False, unsigned=False)
        elif self.settings.activation_type[0] == ActivationType.sau_dp4a :
            result = self.golden_reference_sau_dp4(values, acc=True, unsigned=False)
        elif self.settings.activation_type[0] == ActivationType.sau_dp4m :
            result = self.golden_reference_sau_dp4(values, acc=False, unsigned=True)
        elif self.settings.activation_type[0] == ActivationType.lsu_b16 :
            result = values[0].data.astype(np.float32)
            result = result.astype(bfloat16)
        elif self.settings.activation_type[0] == ActivationType.lsu_b16_vec :
            result = values[0].data.astype(np.float32)
            result = result.astype(bfloat16)
        else :
            raise Error(f'Unsupported Act-shave sub-type: {self.settings.activation_type[0].name}')

        result = ma.getdata(result)

        return result

    def filter_issues(self, args) -> bool:
        if 'E#29786' in self.issues:
            # Filter incorrect tensor serialization issues
            return False
        return True

class ReadAfterWriteACTDMA(Operation):
    PARAMS = ['op_class',
              'input_ttype',
              'input_shape',
              'output_ttype',
              'activation_type',
              'cluster_number',
              'iteration_count']

    NAME = 'ReadAfterWriteACTDMA'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings

    @property
    def ident(self) -> str:
        ident =  str(self.architecture) + f'_ReadAfterWriteACTDMA_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'
        return ident

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        ActivationType2String = {
            ActivationType.HSwish: 'HSwish',
            ActivationType.Sigmoid: 'Sigmoid',
        }

        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Type': ActivationType2String[self.settings.activation_type],
            'Cluster Number' : self.settings.cluster_number.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def validate(self):
        return True

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def json_info(self, inputs, outputs):
        assert(outputs[0].is_float)

        json = {
            'case_type': 'ReadAfterWriteACTDMA',
            'input': get_values_json_info(inputs),
            'output': get_values_json_info(outputs),
            'activation': {
                'name' : self.settings.activation_type.name
            },
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

        return json

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        count = (self.settings.iteration_count - 1) // 2
        result = values[0].data.astype(self.settings.input_ttype.dtype)
        for i in range(0, count) :
            if self.settings.activation_type[0] == ActivationType.HSwish :
                result = HSwish().inference(result)
            elif self.settings.activation_type[0] == ActivationType.Sigmoid :
                result = Sigmoid().inference(result)
            else :
                raise Error(f'Unsupported Act-shave sub-type: {self.settings.activation_type[0].name}')
            result = result.astype(self.settings.input_ttype.dtype)

        result = ma.getdata(result)
        return result

    def filter_issues(self, args) -> bool:
        if(self.settings.cluster_number == 1):
            # E#29786
            return False

        # E#29771
        return False

class ReadAfterWriteDMAACT(Operation):
    PARAMS = ['op_class',
              'input_ttype',
              'input_shape',
              'output_ttype',
              'activation_type',
              'cluster_number',
              'iteration_count']
    NAME = 'ReadAfterWriteDMAACT'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings

    @property
    def ident(self) -> str:
        ident =  str(self.architecture) + f'_ReadAfterWriteDMAACT_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'
        return ident

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        ActivationType2String = {
            ActivationType.HSwish: 'HSwish',
            ActivationType.Sigmoid: 'Sigmoid',
        }

        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Type': ActivationType2String[self.settings.activation_type],
            'Cluster Number' : self.settings.cluster_number.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def validate(self):
        return True

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def json_info(self, inputs, outputs):
        assert(outputs[0].is_float)

        json = {
            'case_type': 'ReadAfterWriteDMAACT',
            'input': get_values_json_info(inputs),
            'output': get_values_json_info(outputs),
            'activation': {
                'name' : self.settings.activation_type.name
            },
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

        return json

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, _ = idu(values[0], values[0])
        return lhs

    def filter_issues(self, args) -> bool:
        if(self.settings.cluster_number == 1):
            # E#29786
            return False

        return True

class ReadAfterWriteDPUDMA(Operation):

    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'cluster_number',
        'iteration_count'
    ]

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'ReadAfterWriteDPUDMA',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_ReadAfterWriteDPUDMA_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
            'Cluster Number' : self.settings.cluster_number.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        count = (self.settings.iteration_count - 1) // 2
        lhs, rhs = iduConvCustom(values[0], values[1])
        for i in range(0, count) :
            result = c2d.inference(lhs, rhs)
            ndarray = result.value if isinstance(result, NBQuantized) else result
            lhs = ndarray.astype(self.settings.input_ttype.dtype)

        return result

    def filter_issues(self, args) -> bool:
        return True

class ReadAfterWriteDMADPU(Operation):

    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'cluster_number',
        'iteration_count'
    ]

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'ReadAfterWriteDMADPU',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_ReadAfterWriteDMADPU_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
            'Cluster Number' : self.settings.cluster_number.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, _ = idu(values[0], values[0])
        return lhs

    def filter_issues(self, args) -> bool:
        return True

class ReadAfterWriteDPUACT(Operation):
    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'act_op',
        'cluster_number',
        'iteration_count'
    ]

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'ReadAfterWriteDPUACT',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'activation': {
                'name' : self.settings.act_op.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_ReadAfterWriteDPUACT_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
            'Cluster Number' : self.settings.cluster_number.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        count = (self.settings.iteration_count - 1) // 2
        lhs, rhs = iduConvCustom(values[0], values[1])
        for i in range(0, count) :
            result = c2d.inference(lhs, rhs)
            ndarray = result.value if isinstance(result, NBQuantized) else result
            lhs = ndarray.astype(self.settings.input_ttype.dtype)

        return result

    def filter_issues(self, args) -> bool:
        if(self.settings.cluster_number == 1):
            # E#29786
            return False

        return True

class ReadAfterWriteACTDPU(Operation):
    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'act_op',
        'cluster_number',
        'iteration_count'
    ]

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'ReadAfterWriteACTDPU',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'activation': {
                'name' : self.settings.act_op.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_ReadAfterWriteACTDPU_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
            'Cluster Number' : self.settings.cluster_number.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        count = (self.settings.iteration_count - 1) // 2
        result = values[0].data.astype(self.settings.input_ttype.dtype)
        for i in range(0, count) :
            if self.settings.act_op == ActivationType.HSwish:
                result = HSwish().inference(result)
            elif self.settings.act_op == ActivationType.Sigmoid :
                result = Sigmoid().inference(result)
            else :
                raise Error(f'Unsupported Act-shave sub-type: {self.settings.act_op}')
            result = result.astype(self.settings.input_ttype.dtype)

        result = ma.getdata(result)
        return result

    def filter_issues(self, args) -> bool:
        if(self.settings.cluster_number == 1):
            # E#29786
            return False

        return True

class MemoryLocation(Enum):
    DDR = auto()
    CMX0 = auto()
    CMX1 = auto()

class DMA(Operation):

    def __init__(self, architecture: Architecture, input_ttype, output_ttype, input_shape, src_memloc, dst_memloc, dma_engine, actCompressDenseMode = False, convert_datatype_en = False):
        super().__init__(architecture)
        settings = Settings()
        self.settings = settings
        self.settings.input_ttype = input_ttype
        self.settings.output_ttype = output_ttype
        self.settings.input_shape = input_shape
        self.settings.src_memloc = src_memloc
        self.settings.dst_memloc = dst_memloc
        self.settings.dma_engine = dma_engine
        self.settings.actCompressDenseMode = actCompressDenseMode
        self.settings.convert_datatype_en = convert_datatype_en
        self.issues = set()

    def json_info(self, input, outputs):
        json = {}
        json['case_type'] = 'DMAcompressAct' if self.settings.actCompressDenseMode == True else 'DMA'
        json['input'] = get_values_json_info(input)
        json['output'] = get_values_json_info(outputs)
        json['DMA_params'] = {}
        json['DMA_params']['src_memory_location'] = self.settings.src_memloc.name
        json['DMA_params']['dst_memory_location'] = self.settings.dst_memloc.name
        json['DMA_params']['dma_engine'] = self.settings.dma_engine

        return json

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        name =  str(self.architecture) + f'_DMA_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_from_{self.settings.src_memloc.name}_to_{self.settings.dst_memloc.name}_engine_{self.settings.dma_engine}'
        name += f'_{self.settings.output_ttype.stype}'
        return name

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        data={}
        data['Op Mode'] = 'MPE_DMA'
        data['Input Type'] = self.settings.input_ttype.stype
        data['Output Type'] = self.settings.output_ttype.stype
        data['Src location'] = self.settings.src_memloc.name
        data['Dst location'] = self.settings.dst_memloc.name
        data['Engine'] = self.settings.dma_engine
        return data

    def value(self):
        value = {'architecture': self.architecture.name}
        value = {**value, **self.json_info(self.inputs, self.outputs)}
        return value

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        if self.settings.convert_datatype_en is True:
            if self.settings.output_ttype.stype == "fp16":
                output = (values[0].data).astype(np.float16)
            elif self.settings.output_ttype.stype == "bfloat16":
                output = (values[0].data).astype(bfloat16)
            return output
        else:
            return values[0].data

    def compute_values(self):
        self.inputs = self.generate_inputs(default_rng(1))
        out = self.apply_mpe(self.inputs)
        value = Value(self.settings.output_ttype, 'output-0.bin', out, self.settings.output_ttype.bitwidth, self.settings.output_ttype.bitsize, self.settings.output_ttype.signed, None)
        self.outputs = [value]

    def write_data(self, dir: Path):
        orderer = OrderNCHW # write data as is
        for input in self.inputs:
            input.write_data(dir, orderer)
        for output in self.outputs:
            output.write_data(dir, orderer)

    def filter_issues(self, args) -> bool:
        if 'E#42566' in self.issues:
            return False
        return True

class DifferentClustersDPU(Operation):

    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'DPU_task_params'
    ]
    NAME = 'DifferentClustersDPU'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape
        self.issues = set()

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'DifferentClustersDPU',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'DPUTaskParams': {
                'input_cluster' : self.settings.DPU_task_params[0],
                'output_cluster' : self.settings.DPU_task_params[1],
                'weights_cluster' : self.settings.DPU_task_params[2],
                'weights_table_cluster' : self.settings.DPU_task_params[3]
            }
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        out_clusters_str = '_'.join([str(cluster) for cluster in self.settings.DPU_task_params[1]])
        return  str(self.architecture) + f'_DifferentClustersDPU_{self.settings.input_ttype.stype}_input_cluster_{self.settings.DPU_task_params[0]}_output_cluster_{out_clusters_str}_weights_cluster_{self.settings.DPU_task_params[2]}_weights_table_cluster_{self.settings.DPU_task_params[3]}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        # make a copy of the output for each cluster specified in params
        for idx, _ in enumerate(self.settings.DPU_task_params[1]):
            value = Value(
                output.ttype, "output-{}.bin".format(idx), output.data, output.bitwidth,
                output.bitsize, output.signed, output.orderer)
            outputs.append(value)

        return outputs

class MultiClustersDPU(Operation):

    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'MultiClustersDPU_params'
    ]
    NAME = 'MultiClustersDPU'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape
        self.issues = set()

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'MultiClustersDPU',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'DPUTaskParams': {
                'task_clusters' : self.settings.MultiClustersDPU_params[0],
                'segmentation': self.settings.MultiClustersDPU_params[1].name,
                'broadcast': self.settings.MultiClustersDPU_params[2]
            }
        }

    def validate(self):
        if self.settings.MultiClustersDPU_params[1] not in { SEGMENTATION.SOK, SEGMENTATION.SOH}:
            raise Exception(f'MultiClustersDPU, unsupported segmentation type: {self.settings.MultiClustersDPU_params[1].name}\n')

    @property
    def ident(self) -> str:
        task_clusters_str = '_'.join([str(cluster) for cluster in self.settings.MultiClustersDPU_params[0]])
        kernel_size_str = 'x'.join([str(dim) for dim in self.settings.kernel_shape])
        kernel_stride_str = 'x'.join([str(dim) for dim in self.settings.kernel_strides])
        return  str(self.architecture) + f'_MultiClustersDPU_{self.settings.input_ttype.stype}_task_cluster_{task_clusters_str}_{self.settings.MultiClustersDPU_params[1]}_broadcast_{self.settings.MultiClustersDPU_params[2]}_kern_chan_{self.settings.kernel_channels}_kern_sz_{kernel_size_str}_kern_stride_{kernel_stride_str}_in_shape_{shape_to_str(self.settings.input_shape)}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

    def get_out_cluster_data(self, orig_data, start, end, axis, step_per_cluster) -> np.ndarray:
        broadcast = self.settings.MultiClustersDPU_params[2]
        if (broadcast == True):
            return orig_data

        start[axis] = end[axis]
        remainder = orig_data.shape[axis] - end[axis]
        step = remainder if (remainder < step_per_cluster) else step_per_cluster
        end[axis] = start[axis] + step

        return orig_data[start[0]:end[0], start[1]:end[1], start[2]:end[2], start[3]:end[3]]

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        task_clusters = self.settings.MultiClustersDPU_params[0]
        num_clusters = len(task_clusters)
        tile_start = [0, 0, 0, 0]
        tile_end = list(output.data.shape)
        axis = self.settings.MultiClustersDPU_params[1].value
        tile_end[axis] = 0
        step_per_cluster = (output.data.shape[axis] + num_clusters - 1) // num_clusters

        # generate a reference for each output:
        #  1. for output broadcasting in multiple clusters, make a copy of the ref for each cluster
        #  2. for segmentation (K or H) without broadcast, slice the output from NB as follows - all slices are equal, except possibly the last one
        for idx, _ in enumerate(task_clusters):
            data = self.get_out_cluster_data(output.data, tile_start, tile_end, axis, step_per_cluster)
            value = Value(
                output.ttype, "output-{}.bin".format(idx), data, output.bitwidth,
                output.bitsize, output.signed, output.orderer)
            outputs.append(value)

        return outputs

class RaceConditionDMA(Operation):

    PARAMS = ['op_class', 'input_ttype', 'output_ttype', 'iteration_count']
    NAME = 'RaceConditionDMA'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings

    def json_info(self, input, outputs):
        return {
            'case_type': 'RaceConditionDMA',
            'input': get_values_json_info(input),
            'output': get_values_json_info(outputs),
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_DMA_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

    @property
    def orderer(self) -> Orderer:
        return OrderNCHW

    @property
    def output_orderer(self) -> Orderer:
        return OrderNCHW

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'Iteration Count': self.settings.iteration_count,
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', [1, 16, 16, 16], rng)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, _ = idu(values[0], values[0])
        return lhs

    def filter_issues(self, args) -> bool:
        return True

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        num_outputs = 2
        for idx in range(num_outputs):
            value = Value(
                output.ttype, "output-{}.bin".format(idx), output.data, output.bitwidth,
                output.bitsize, output.signed, output.orderer)
            outputs.append(value)

        return outputs


class RaceConditionDPU(Operation):

    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'iteration_count'
    ]
    NAME = 'RaceConditionDMA'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'RaceConditionDPU',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_DPU_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
            'Iteration Count': self.settings.iteration_count
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        num_outputs = 2
        for idx in range(num_outputs):
            value = Value(
                output.ttype, "output-{}.bin".format(idx), output.data, output.bitwidth,
                output.bitsize, output.signed, output.orderer)
            outputs.append(value)

        return outputs

class RaceConditionDPUDMA(Operation):

    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'iteration_count'
    ]

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'RaceConditionDPUDMA',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_DPU_DMA_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'MPE Mode': 'DPU_DMA_race_cond',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Weights Type': self.settings.weight_ttype.stype,
            'Kernel Channels': str(self.settings.kernel_channels),
            'Kernel Shape': ', '.join([str(s) for s in self.settings.kernel_shape]),
            'Output Type': self.settings.output_ttype.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        value0 = Value(
            output.ttype, "output-{}.bin".format(0), output.data, output.bitwidth,
            output.bitsize, output.signed, output.orderer)
        inputs = self.generate_inputs(default_rng(1))
        value1 = Value(
            inputs[0].ttype, "output-{}.bin".format(1), inputs[0].data, inputs[0].bitwidth,
            inputs[0].bitsize, inputs[0].signed, inputs[0].orderer)
        outputs.append(value0)
        outputs.append(value1)

        return outputs

class RaceConditionDPUDMAACT(Operation):
    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'act_op',
        'iteration_count'
    ]

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'RaceConditionDPUDMAACT',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'activation': {
                'name' : self.settings.act_op.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_DPU_DMA_ACT_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'MPE Mode': 'DPU_DMA_ACT_race_cond',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Weights Type': self.settings.weight_ttype.stype,
            'Kernel Channels': str(self.settings.kernel_channels),
            'Kernel Shape': ', '.join([str(s) for s in self.settings.kernel_shape]),
            'Output Type': self.settings.output_ttype.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        inputs = self.generate_inputs(default_rng(1))

        value0 = Value(
            output.ttype, "output-{}.bin".format(0), output.data, output.bitwidth,
            output.bitsize, output.signed, output.orderer)

        value1 = Value(
            inputs[0].ttype, "output-{}.bin".format(1), inputs[0].data, inputs[0].bitwidth,
            inputs[0].bitsize, inputs[0].signed, inputs[0].orderer)

        if self.settings.act_op == ActivationType.HSwish :
            act_kernel_data = HSwish().inference(inputs[0].data)
        elif self.settings.act_op == ActivationType.Sigmoid :
            act_kernel_data = Sigmoid().inference(inputs[0].data)
        else :
            raise Error(f'RaceConditionDPUDMAACT: Need to add act-shave as output: {self.settings.act_op.name}')

        act_kernel_data = ma.getdata(act_kernel_data)
        value2 = Value(
            inputs[0].ttype, "output-{}.bin".format(2), act_kernel_data, inputs[0].bitwidth,
            inputs[0].bitsize, inputs[0].signed, inputs[0].orderer)
        outputs.append(value0)
        outputs.append(value1)
        outputs.append(value2)

        return outputs

class RaceConditionDPUACT(Operation):
    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_cub',
        'act_op',
        'iteration_count'
    ]

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'RaceConditionDPUACT',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
            },
            'activation': {
                'name' : self.settings.act_op.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_DPU_ACT_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'MPE Mode': 'DPU_ACT_race_cond',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Weights Type': self.settings.weight_ttype.stype,
            'Kernel Channels': str(self.settings.kernel_channels),
            'Kernel Shape': ', '.join([str(s) for s in self.settings.kernel_shape]),
            'Output Type': self.settings.output_ttype.stype,
            'Iteration Count': self.settings.iteration_count.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        inputs = self.generate_inputs(default_rng(1))

        value0 = Value(
            output.ttype, "output-{}.bin".format(0), output.data, output.bitwidth,
            output.bitsize, output.signed, output.orderer)

        if self.settings.act_op == ActivationType.HSwish :
            act_kernel_data = HSwish().inference(inputs[0].data)
        elif self.settings.act_op == ActivationType.Sigmoid :
            act_kernel_data = Sigmoid().inference(inputs[0].data)
        else :
            raise Error(f'RaceConditionDPUDMAACT: Need to add act-shave as output: {self.settings.act_op.name}')

        act_kernel_data = ma.getdata(act_kernel_data)
        value1 = Value(
            inputs[0].ttype, "output-{}.bin".format(1), act_kernel_data, inputs[0].bitwidth,
            inputs[0].bitsize, inputs[0].signed, inputs[0].orderer)
        outputs.append(value0)
        outputs.append(value1)

        return outputs

class RaceCondition:
    PARAMS = ['operation', 'iteration_count', 'requested_cluster', 'requested_unit']

    def __init__(self, architecture: Architecture, operation, iter_count, requested_cluster, requested_unit):
        self.architecture = architecture
        self.operation = operation
        self.op = operation.op
        self.op.odu = self.odu
        self.iter_count = iter_count
        self.requested_cluster = requested_cluster
        self.requested_unit = requested_unit

    def json_info(self):
        return {
            'architecture': self.architecture.name,
            'case_type': 'RaceCondition',
            'iteration_count' : self.iter_count,
            'requested_clusters' : self.requested_cluster,
            'requested_units' : self.requested_unit,
        }

    def validate(self):
        self.operation.validate()

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_race_cond_{self.operation.ident}_iters_{self.iter_count}_clusters_{self.requested_cluster}_shaves_{self.requested_unit}'

    def compute_values(self):
        self.operation.compute_values()

    def write_data(self, dir: Path):
        self.operation.write_data(dir)

    def value(self):
        json = self.json_info()
        json['operation'] = self.operation.value()
        return json

    def filter_issues(self, args):
        return self.op.filter_issues(args)

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        for cluster_idx in range(self.requested_cluster):
          for unit_idx in range(self.requested_unit):
            value = Value(
                output.ttype, "output-{}.bin".format(cluster_idx*self.requested_unit + unit_idx), output.data, output.bitwidth,
                output.bitsize, output.signed, output.orderer)
            outputs.append(value)

        return outputs


class DualChannelDMA(Operation):

    PARAMS = ['op_class', 'input_ttype', 'output_ttype']
    NAME = 'DualChannelDMA'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings

    def json_info(self, input, outputs):
        return {
            'case_type': 'DualChannelDMA',
            'input': get_values_json_info(input),
            'output': get_values_json_info(outputs)
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_DMA_dual_channel_{self.settings.input_ttype.stype}'

    @property
    def orderer(self) -> Orderer:
        return OrderNCHW

    @property
    def output_orderer(self) -> Orderer:
        return OrderNCHW

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', [1, 16, 16, 16], rng)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, _ = idu(values[0], values[0])
        return lhs

    def filter_issues(self, args) -> bool:
        return True

    def odu(self, output: Value) -> List[Value]:
        """Models the hardware ODU"""
        outputs = list()

        num_outputs = 4
        for idx in range(num_outputs):
            value = Value(
                output.ttype, "output-{}.bin".format(idx), output.data, output.bitwidth,
                output.bitsize, output.signed, output.orderer)
            outputs.append(value)

        return outputs

class SETablePattern(Enum):
    SwitchLines = auto()
    OriginalInput = auto()

class StorageElementTableDPU(Operation):

    PARAMS = [
        'op_class',
        'input_ttype',
        'input_shape',
        'weight_ttype',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'output_order',
        'kernel_strides',
        'kernel_pads',
        'compress',
        'mpe_mode',
        'SE_table_pattern'
    ]
    NAME = 'StorageElementTableDPU'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'StorageElementTableDPU',
            'input': [inputs[0].json_info],
            'weight': [inputs[1].json_info],
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_mode': self.settings.mpe_mode.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'SE_table_pattern': self.settings.SE_table_pattern.name
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return  str(self.architecture) + f'_StorageElementTableDPU_input_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_weights_{shape_to_str(self.settings.weight_shape)}x{self.settings.weight_ttype.stype}_pattern_{self.settings.SE_table_pattern}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return orderToOrderer(self.settings.output_order)

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Input Scale': self.settings.input_ttype.scale if hasattr(self.settings.input_ttype, 'scale') else 1,
            'Input Zero Point': self.settings.input_ttype.zero if hasattr(self.settings.input_ttype, 'zero') else 0,
            'Weights Type': np.dtype(self.settings.weight_ttype),
            'Weights Scale': self.settings.weight_ttype.scale if hasattr(self.settings.weight_ttype, 'scale') else 1,
            'Weights Zero Point': self.settings.weight_ttype.zero if hasattr(self.settings.weight_ttype, 'zero') else 0,
            'Output Type': np.dtype(self.settings.output_ttype),
            'Output Scale': self.settings.output_ttype.scale if hasattr(self.settings.output_ttype, 'scale') else 1,
            'Output Zero Point': self.settings.output_ttype.zero if hasattr(self.settings.output_ttype, 'zero') else 0,
            'IC': self.settings.input_shape[1],
            'IH': self.settings.input_shape[2],
            'IW': self.settings.input_shape[3],
            'IK': self.settings.kernel_channels,
            'KH': self.settings.kernel_shape[0],
            'KW': self.settings.kernel_shape[1],
            'SH': self.settings.kernel_strides[1],
            'SW': self.settings.kernel_strides[0],
            'PT': self.settings.kernel_pads[0],
            'PB': self.settings.kernel_pads[2],
            'PL': self.settings.kernel_pads[1],
            'PR': self.settings.kernel_pads[3],
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_mode],
            'Output Permute': SW2HWOrder[self.settings.output_order],
            'Compression': int(self.settings.compress),
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        pattern = self.settings.SE_table_pattern
        orig_input = values[0].data
        input_data = np.empty_like(orig_input)

        # emulate input + se table and pass it to NumericsBench conv
        if pattern == SETablePattern.SwitchLines:
            height = input_data.shape[2] if input_data.shape[2] % 2 == 0 else input_data.shape[2] - 1
            for h in range(0, height, 2):
                input_data[:, :, h], input_data[:, :, h + 1] = orig_input[:, :, h + 1], orig_input[:, :, h]
        elif pattern == SETablePattern.OriginalInput:
            input_data = orig_input

        input_value = Value(
                values[0].ttype, "input-0-with-se-table.bin", input_data, values[0].bitwidth,
                values[0].bitsize, values[0].signed, values[0].orderer)

        lhs, rhs = iduConvCustom(input_value, values[1])
        c2d = Conv2DVPUX(kernel_shape=self.settings.kernel_shape,
                         pads = self.settings.kernel_pads,
                         strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True


class Settings:
    def __str__(self):
        return '\n  '.join([f'{name}={getattr(self, name)}' for name in dir(self) if not name.startswith('_')])


class DPUPipeline:
    def __init__(self, architecture: Architecture, option_names, option_values, activation=None):
        settings = Settings()
        self.settings = settings
        self.architecture = architecture
        self.issues = set()
        for name, value in zip(option_names, option_values):
            setattr(settings, name, value)

        self.op = settings.op_class(self.architecture, self.settings)
        self.activation = activation

    def compute_values(self):
        try:
            self.inputs = self.op.generate_inputs(default_rng(1))
            mpe_data = self.op.apply_mpe(self.inputs)
            self.mpe_data = mpe_data.value if isinstance(mpe_data, NBQuantized) else mpe_data
            ppe_value = self.op.ppe(self.inputs, self.settings.output_ttype, mpe_data, self.activation)
            self.outputs = self.op.odu(ppe_value)
            # skip entropy check for act kernel sqrt and log functions, as they tested with both positive/negative input data, resulting in NaN values
            if not (isinstance(self.op, ActKernel) and (self.settings.activation_type[0] == ActivationType.vau_sqrt or self.settings.activation_type[0] == ActivationType.vau_log)):
                for output in self.outputs:
                    output.check_entropy()
        except Exception as ex:
            raise ComputationError(f'computing {self.ident}') from ex

    def validate(self):
        try:
            self.op.validate()
        except Exception as ex:
            raise ValidationError(f'validating {self.ident}') from ex

    @property
    def name(self):
        return self.op.NAME

    @property
    def ident(self):
        activation_ident = str(self.activation or self.settings.output_ttype.stype)
        return f'{self.op.ident}_{activation_ident}'

    @property
    def data(self):
        data = {
            **self.op.data,
            'Type': self.op.NAME,
            'Content Creation': 0,
            'ActSpBitsIC': '',
            'WtSpBitsIC': '',
            'Issues': ', '.join(self.issues)
        }
        if self.activation:
            data['PPE'] = str(self.activation)
        return data

    def write_data(self, dir: Path):
        orderer = self.op.orderer
        for input in self.inputs:
            input.write_data(dir, orderer)
        for output in self.outputs:
            output.write_data(dir, self.op.output_orderer)
        orderer(self.mpe_data).tofile(dir / 'mpe_raw.bin')

    def value(self):
        value = {'architecture': self.op.architecture.name}
        value = {**value, **self.op.json_info(self.inputs, self.outputs)}
        if self.activation:
            value['activation'] = self.activation.json_info
        return value

    def filter_issues(self, args) -> bool:
        return not self.issues and self.op.filter_issues(args)

    def filter_issues(self, args):
        return self.op.filter_issues(args)


class Pad:
    # The padding order is top|left|bottom|right
    none = [[0]*4]

    def all(x):
        return [[p]*4 for p in range(1,x+1)]

    def top(x):
        return [[p,0,0,0] for p in range(1,x+1)]

    def left(x):
        return [[0,p,0,0] for p in range(1,x+1)]

    def bottom(x):
        return [[0,0,p,0] for p in range(1,x+1)]

    def right(x):
        return [[0,0,0,p] for p in range(1,x+1)]

    def top_bottom(x):
        return [[p,0,p,0] for p in range(1,x+1)]

    def left_right(x):
        return [[0,p,0,p] for p in range(1,x+1)]


def filter_issues(args, p: DPUPipeline) -> bool:
    # TODO: Add arguments to selectively filter by issues.
    return p.filter_issues(args)


_ZMCONV_VALID_WEIGHT_TYPES = {
    Int8: [Int8(3)],
    Int4: [Int8(3)],
    UInt8: [UInt8(3)],
    UInt4: [],
    FP16: [FP16(4), Int8(3)],
    BF16: [BF16(4)]
}


_PPE_VALID_OUTPUT_TYPES = {
    False: [Int32(), FP16(), UInt8(), Int8()],  # Integer MACs
    True: [FP32(), FP16(), BF16(), UInt8(), Int8()],    # FP MACs
}

_PPE_HAS_PERMUTATION_SUPPORT = {
    Int8: True,
    Int4: False,
    UInt8: True,
    UInt4: False,
    FP16: True,
    BF16: True,
    Int32: True,
    FP32: True
}


def genZMConvs(architectures=ALL_ARCHITECTURES,
               input_types=[UInt8(2)],
               input_shapes=[[1, 32, 16, 16]],
               weight_types=None,
               kernel_channels=[64],
               kernel_shapes=[[1, 1]],
               output_types=None,
               output_orders=[Order.NHWC],
               strides=[[1, 1]],
               pads=Pad.none,
               compress=[False],
               mpe_modes=[MPE_MODE.CUBOID_16x16],
               activations=[None],
               weights_swizzling_keys=[None]):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, activation, weights_swizzling_key, comp) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, activations, weights_swizzling_keys, compress):

        if weight_types is None:
            current_weight_types = _ZMCONV_VALID_WEIGHT_TYPES[input_type.__class__]

        else:
            current_weight_types = weight_types

        for weight_type in current_weight_types:

            if output_types is None:
                current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float or weight_type.is_float]
            else:
                current_output_types = output_types

            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, ZMajorConvolution.PARAMS, (ZMajorConvolution,
                                                                           input_type,
                                                                           input_shape,
                                                                           weight_type,
                                                                           kernel_channel,
                                                                           kernel_shape,
                                                                           output_type,
                                                                           output_order,
                                                                           stride,
                                                                           pad,
                                                                           comp,
                                                                           mpe_mode,
                                                                           weights_swizzling_key), activation)

def genSparseConvs( architectures=ALL_ARCHITECTURES,
                    input_types=[UInt8(2)],
                    input_shapes=[[1, 32, 16, 16]],
                    weight_types=None,
                    kernel_channels=[32],
                    kernel_shapes=[[1, 1]],
                    output_types=None,
                    output_orders=[Order.NHWC],
                    strides=[[1, 1]],
                    pads=Pad.none,
                    compress=[False],
                    mpe_modes=[MPE_MODE.CUBOID_16x16],
                    sparsity_factors=[0.5],
                    activation_sparsity=[False]):
    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, sparsity_factor, comp, act_sparsity) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, sparsity_factors, compress, activation_sparsity):

        if weight_types is None:
            current_weight_types = _ZMCONV_VALID_WEIGHT_TYPES[input_type.__class__]
        else:
            current_weight_types = weight_types

        for weight_type in current_weight_types:

            if output_types is None:
                current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float or weight_type.is_float]
            else:
                current_output_types = output_types

            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, SparseConvolution.PARAMS, (SparseConvolution,
                                                                            input_type,
                                                                            input_shape,
                                                                            weight_type,
                                                                            kernel_channel,
                                                                            kernel_shape,
                                                                            output_type,
                                                                            output_order,
                                                                            stride,
                                                                            pad,
                                                                            comp,
                                                                            mpe_mode,
                                                                            sparsity_factor,
                                                                            act_sparsity
                                                                            ))

def genDoubleZMConvs(architectures=ALL_ARCHITECTURES,
               input_types=[UInt8(2)],
               input_shapes=[[1, 32, 16, 16]],
               weight_types=None,
               kernel_channels=[64],
               kernel_shapes=[[1, 1]],
               output_types=None,
               output_orders=[Order.NHWC],
               strides=[[1, 1]],
               pads=Pad.none,
               compress=[False],
               mpe_modes=[MPE_MODE.CUBOID_16x16],
               activations=[None],
               activation_swizzling_keys=[None],
               activation_sparsity=[None]):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, activation, activation_swizzling_key, comp, act_sparsity) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, activations, activation_swizzling_keys, compress, activation_sparsity):

        if weight_types is None:
            current_weight_types = _ZMCONV_VALID_WEIGHT_TYPES[input_type.__class__]
        else:
            current_weight_types = weight_types

        for weight_type in current_weight_types:
            if output_types is None:
                current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float or weight_type.is_float]
            else:
                current_output_types = output_types

            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, DoubleZMajorConvolution.PARAMS, (DoubleZMajorConvolution,
                                                                                input_type,
                                                                                input_shape,
                                                                                weight_type,
                                                                                kernel_channel,
                                                                                kernel_shape,
                                                                                output_type,
                                                                                output_order,
                                                                                stride,
                                                                                pad,
                                                                                comp,
                                                                                mpe_mode,
                                                                                activation_swizzling_key,
                                                                                act_sparsity), activation)


def genEltwiseAdds(architectures=ALL_ARCHITECTURES,
                   input_types=[Int8(6)],
                   input_shapes=[[1, 256, 16, 16]],
                   output_types=None):
    for (architecture, input_type, input_shape) in itertools.product(architectures, input_types, input_shapes):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(architecture, EltwiseAdd.PARAMS, (EltwiseAdd,
                                                                input_type,
                                                                input_shape,
                                                                output_type))


def genEltwiseMults(architectures=ALL_ARCHITECTURES,
                    input_types=[Int8(6)],
                    input_shapes=[[1, 256, 16, 16]],
                    output_types=None):
    for (architecture, input_type, input_shape) in itertools.product(architectures, input_types, input_shapes):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(architecture, EltwiseMult.PARAMS, (EltwiseMult,
                                                                 input_type,
                                                                 input_shape,
                                                                 output_type))

def genEltwiseSparse(architectures=ALL_ARCHITECTURES,
                     input_types=[Int8(6)],
                     input_shapes=[[1, 16, 16, 32]],
                     output_types=None,
                     seSizes=None):
    if (seSizes == None):
        seSizes = [input_shapes[1]]
    for (architecture, input_type, input_shape, seSize) in itertools.product(architectures, input_types, input_shapes, seSizes):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(architecture, EltwiseSparse.PARAMS, (EltwiseSparse,
                                                                input_type,
                                                                input_shape,
                                                                output_type,
                                                                seSize))

def genMaxPools(architectures=ALL_ARCHITECTURES,
                input_types=[FP16(6)],
                input_shapes=[[1, 64, 16, 16]],
                kernel_shapes=[[2, 2]],
                output_types=None,
                strides=[[2, 2]],
                pads=Pad.none):
    for (architecture, input_type, input_shape, kernel_shape, stride, pad) in itertools.product(architectures, input_types, input_shapes, kernel_shapes, strides, pads):
        if output_types is None:
            if input_type.is_float:
                if input_type.__class__ is BF16:
                    current_output_types = [BF16()]
                else:
                    current_output_types = [FP16()]
            else:
                current_output_types = [Int32(), FP16(), UInt8(), Int8()]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(architecture, Maxpool.PARAMS, (Maxpool,
                                                             input_type,
                                                             input_shape,
                                                             kernel_shape,
                                                             output_type,
                                                             stride,
                                                             pad))


def genAvgPools(architectures=ALL_ARCHITECTURES,
                input_types=[FP16(6)],
                input_shapes=[[1, 64, 32, 32]],
                kernel_shapes=[[2, 2]],
                output_types=None,
                strides=[[2, 2]]):
    for (architecture, input_type, input_shape, kernel_shape, stride) in itertools.product(architectures, input_types, input_shapes, kernel_shapes, strides):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(architecture, AvgPool.PARAMS, (AvgPool,
                                                             input_type,
                                                             input_shape,
                                                             kernel_shape,
                                                             output_type,
                                                             stride))

def getValidOutputTypes(input_type, kernel_channels) :
    aviable_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
    output_types = []
    for output_type in aviable_output_types:
        if(CheckHWAlignment(output_type, kernel_channels)):
            output_types.append(output_type)
    return output_types

def genDepthWiseConvs(architectures=ALL_ARCHITECTURES,
                      input_types=[FP16(2)],
                      input_shapes=[[1, 16, 32, 32]],
                      kernel_channels=[16],
                      kernel_shapes=[[4, 4]],
                      output_types=None,
                      strides=[[1, 1]],
                      pads=Pad.none):
    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, stride, pad) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, strides, pads):

        if output_types is None:
            current_output_types = getValidOutputTypes(input_type, kernel_channel)
        else:
            current_output_types = output_types
        for output_type in current_output_types:
            yield DPUPipeline(architecture, DepthWiseConv.PARAMS, (DepthWiseConv,
                                                                   input_type,
                                                                   input_shape,
                                                                   kernel_channel,
                                                                   kernel_shape,
                                                                   output_type,
                                                                   stride,
                                                                   pad))

def genReadAfterWriteACTDMA(architectures=ALL_ARCHITECTURES,
                            input_types=[FP16(0)],
                            input_shapes=[[1, 10, 2, 3]],
                            output_types=[FP16()],
                            act_shave_subtypes=[ActivationType.HSwish],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):
                    for (architecture, input_type, input_shape, output_type, act_shave_subtype, cluster_number) in itertools.product(architectures, input_types, input_shapes, output_types, act_shave_subtypes, cluster_numbers):
                        yield DPUPipeline(architecture, ReadAfterWriteACTDMA.PARAMS, (ReadAfterWriteACTDMA, input_type, input_shape, output_type, act_shave_subtype, cluster_number, iteration_count))

def genReadAfterWriteDMAACT(architectures=ALL_ARCHITECTURES,
                            input_types=[FP16(0)],
                            input_shapes=[[1, 10, 2, 3]],
                            output_types=[FP16()],
                            act_shave_subtypes=[ActivationType.HSwish],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):
                    for (architecture, input_type, input_shape, output_type, act_shave_subtype, cluster_number) in itertools.product(architectures, input_types, input_shapes, output_types, act_shave_subtypes, cluster_numbers):
                        yield DPUPipeline(architecture, ReadAfterWriteDMAACT.PARAMS, (ReadAfterWriteDMAACT, input_type, input_shape, output_type, act_shave_subtype, cluster_number, iteration_count))

def genReadAfterWriteDPUDMA(architectures=ALL_ARCHITECTURES,
                            input_types=[FP16(0)],
                            input_shapes=[[1, 16, 16, 16]],
                            weight_types=[FP16(0)],
                            kernel_channels=[16],
                            kernel_shapes=[[1, 1]],
                            output_types=[FP16()],
                            output_orders=[Order.NHWC],
                            strides=[[1, 1]],
                            pads=Pad.none,
                            compress=False,
                            mpe_modes=[MPE_MODE.CUBOID_16x16],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):
    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, cluster_number) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, cluster_numbers):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, ReadAfterWriteDPUDMA.PARAMS, (ReadAfterWriteDPUDMA,
                                                                              input_type,
                                                                              input_shape,
                                                                              weight_type,
                                                                              kernel_channel,
                                                                              kernel_shape,
                                                                              output_type,
                                                                              output_order,
                                                                              stride,
                                                                              pad,
                                                                              compress,
                                                                              mpe_mode,
                                                                              cluster_number,
                                                                              iteration_count))

def genReadAfterWriteDMADPU(architectures=ALL_ARCHITECTURES,
                            input_types=[FP16(0)],
                            input_shapes=[[1, 16, 16, 16]],
                            weight_types=[FP16(0)],
                            kernel_channels=[16],
                            kernel_shapes=[[1, 1]],
                            output_types=[FP16()],
                            output_orders=[Order.NHWC],
                            strides=[[1, 1]],
                            pads=Pad.none,
                            compress=False,
                            mpe_modes=[MPE_MODE.CUBOID_16x16],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):
    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, cluster_number) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, cluster_numbers):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, ReadAfterWriteDMADPU.PARAMS, (ReadAfterWriteDMADPU,
                                                                              input_type,
                                                                              input_shape,
                                                                              weight_type,
                                                                              kernel_channel,
                                                                              kernel_shape,
                                                                              output_type,
                                                                              output_order,
                                                                              stride,
                                                                              pad,
                                                                              compress,
                                                                              mpe_mode,
                                                                              cluster_number,
                                                                              iteration_count))

def genReadAfterWriteDPUACT(architectures=ALL_ARCHITECTURES,
                            input_types=[FP16(0)],
                            input_shapes=[[1, 16, 8, 8]],
                            weight_types=[FP16(0)],
                            kernel_channels=[16],
                            kernel_shapes=[[1, 1]],
                            output_types=[FP16()],
                            output_orders=[Order.NHWC],
                            strides=[[1, 1]],
                            pads=Pad.none,
                            compress=False,
                            mpe_modes=[MPE_MODE.CUBOID_16x16],
                            act_types=[ActivationType.HSwish],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, act_type, cluster_number) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, act_types, cluster_numbers):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, ReadAfterWriteDPUACT.PARAMS, (ReadAfterWriteDPUACT,
                                                                              input_type,
                                                                              input_shape,
                                                                              weight_type,
                                                                              kernel_channel,
                                                                              kernel_shape,
                                                                              output_type,
                                                                              output_order,
                                                                              stride,
                                                                              pad,
                                                                              compress,
                                                                              mpe_mode,
                                                                              act_type,
                                                                              cluster_number,
                                                                              iteration_count))

def genReadAfterWriteACTDPU(architectures=ALL_ARCHITECTURES,
                            input_types=[FP16(0)],
                            input_shapes=[[1, 16, 8, 8]],
                            weight_types=[FP16(0)],
                            kernel_channels=[16],
                            kernel_shapes=[[1, 1]],
                            output_types=[FP16()],
                            output_orders=[Order.NHWC],
                            strides=[[1, 1]],
                            pads=Pad.none,
                            compress=False,
                            mpe_modes=[MPE_MODE.CUBOID_16x16],
                            act_types=[ActivationType.HSwish],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, act_type, cluster_number) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, act_types, cluster_numbers):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, ReadAfterWriteACTDPU.PARAMS, (ReadAfterWriteACTDPU,
                                                                              input_type,
                                                                              input_shape,
                                                                              weight_type,
                                                                              kernel_channel,
                                                                              kernel_shape,
                                                                              output_type,
                                                                              output_order,
                                                                              stride,
                                                                              pad,
                                                                              compress,
                                                                              mpe_mode,
                                                                              act_type,
                                                                              cluster_number,
                                                                              iteration_count))

def genDifferentClustersDPU(architectures=ALL_ARCHITECTURES,
                            input_types=[FP16(4)],
                            input_shapes=[[1, 32, 16, 16]],
                            weight_types=[FP16(4)],
                            kernel_channels=[64],
                            kernel_shapes=[[1, 1]],
                            output_types=[FP16(4)],
                            output_orders=[Order.NHWC],
                            strides=[[1, 1]],
                            pads=Pad.none,
                            compress=False,
                            mpe_modes=[MPE_MODE.CUBOID_16x16],
                            input_clusters = [0, 1],
                            output_clusters = [[0], [1], [0, 1]],
                            weights_clusters = [0, 1],
                            weights_table_clusters = [0, 1]):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, input_cluster, output_cluster, weights_cluster, weights_table_cluster) \
            in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, input_clusters, output_clusters, weights_clusters, weights_table_clusters):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                if(len(output_cluster) == 1 and input_cluster == output_cluster[0] == weights_cluster == weights_table_cluster):
                    print("skip, DPU uses the memory of the same cluster, cluster num:", input_cluster)
                    continue

                yield DPUPipeline(architecture, DifferentClustersDPU.PARAMS, (DifferentClustersDPU,
                                                                              input_type,
                                                                              input_shape,
                                                                              weight_type,
                                                                              kernel_channel,
                                                                              kernel_shape,
                                                                              output_type,
                                                                              output_order,
                                                                              stride,
                                                                              pad,
                                                                              compress,
                                                                              mpe_mode,
                                                                              [input_cluster, output_cluster, weights_cluster, weights_table_cluster]))

def genMultiClustersDPU(architectures=ALL_ARCHITECTURES,
                        input_types=[FP16(4)],
                        input_shapes=[[1, 32, 16, 16]],
                        weight_types=[FP16(4)],
                        kernel_channels=[64],
                        kernel_shapes=[[1, 1]],
                        output_types=[FP16(4)],
                        output_orders=[Order.NHWC],
                        strides=[[1, 1]],
                        pads=Pad.none,
                        compress=False,
                        mpe_modes=[MPE_MODE.CUBOID_16x16],
                        task_clusters=[[0, 1]],
                        segmentation=SEGMENTATION.SOK,
                        is_out_broadcasted=[True]):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, task_cluster, broadcast) \
            in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, task_clusters, is_out_broadcasted):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue

                yield DPUPipeline(architecture, MultiClustersDPU.PARAMS, (MultiClustersDPU,
                                                                          input_type,
                                                                          input_shape,
                                                                          weight_type,
                                                                          kernel_channel,
                                                                          kernel_shape,
                                                                          output_type,
                                                                          output_order,
                                                                          stride,
                                                                          pad,
                                                                          compress,
                                                                          mpe_mode,
                                                                          [task_cluster, segmentation, broadcast]))

def genActShave(input_types,
                input_shapes,
                output_types,
                act_shave_subtypes,
                architectures=ALL_ARCHITECTURES):
    for (architecture, input_type, input_shape, output_type, act_shave_subtype) in itertools.product(architectures, input_types, input_shapes, output_types, act_shave_subtypes):
        yield DPUPipeline(architecture, ActKernel.PARAMS, (ActKernel, input_type, input_shape, output_type, act_shave_subtype))

def genDMA(architectures=ALL_ARCHITECTURES, tensor_types=[FP16(2)], input_shapes=[[1, 16, 32, 32]], src_locations=[MemoryLocation.CMX0], dst_locations=[MemoryLocation.CMX0], dma_engines=[0], actCompressDenseMode = False, convert = [False]):
    for (architecture, tensor_type, input_shape, src_location, dst_location, dma_engine, convert_en) in itertools.product(architectures, tensor_types, input_shapes, src_locations, dst_locations, dma_engines, convert):
        yield DMA(architecture, tensor_type, tensor_type,
                                            input_shape,
                                            src_location,
                                            dst_location,
                                            dma_engine,
                                            actCompressDenseMode,
                                            convert_en)

def genDMAConvert(architectures=ALL_ARCHITECTURES, tensor_types=[FP16(2)], out_tensor_types=[FP16(2)], input_shapes=[[1, 16, 32, 32]], src_locations=[MemoryLocation.CMX0], dst_locations=[MemoryLocation.CMX0], dma_engines=[0], actCompressDenseMode = False, convert = [False]):
    for (architecture, tensor_type, out_tensor_type, input_shape, src_location, dst_location, dma_engine, convert_en) in itertools.product(architectures, tensor_types, out_tensor_types, input_shapes, src_locations, dst_locations, dma_engines, convert):
        yield DMA(architecture, tensor_type, out_tensor_type,
                                            input_shape,
                                            src_location,
                                            dst_location,
                                            dma_engine,
                                            actCompressDenseMode,
                                            convert_en)
def genRaceConditionDMA(architectures=ALL_ARCHITECTURES,
                        input_types=[FP16(2)],
                        output_types=[FP16(2)],
                        iteration_count=64):
    for (architecture, input_type, output_type) in itertools.product(architectures, input_types, output_types):
        yield DPUPipeline(architecture, RaceConditionDMA.PARAMS, (RaceConditionDMA,
                                                    input_type,
                                                    output_type,
                                                    iteration_count
                                                    ))
def genRaceCondition(ops,
                     architectures=ALL_ARCHITECTURES,
                     iteration_counts=[64],
                     requested_clusters=[1],
                     requested_units=[1]):
    for (architecture, op, iteration_count, requested_cluster, requested_unit) in itertools.product(architectures, ops, iteration_counts, requested_clusters, requested_units):
        yield RaceCondition(architecture, op, iteration_count, requested_cluster, requested_unit)

def genRaceConditionDPU(architectures=ALL_ARCHITECTURES,
                        input_types=[FP16(4)],
                        input_shapes=[[1, 32, 16, 16]],
                        weight_types=[FP16(4)],
                        kernel_channels=[64],
                        kernel_shapes=[[1, 1]],
                        output_types=[FP16(4)],
                        output_orders=[Order.NHWC],
                        strides=[[1, 1]],
                        pads=Pad.none,
                        compress=False,
                        mpe_modes=[MPE_MODE.CUBOID_16x16],
                        iteration_count = 64):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, RaceConditionDPU.PARAMS, (RaceConditionDPU,
                                                                          input_type,
                                                                          input_shape,
                                                                          weight_type,
                                                                          kernel_channel,
                                                                          kernel_shape,
                                                                          output_type,
                                                                          output_order,
                                                                          stride,
                                                                          pad,
                                                                          compress,
                                                                          mpe_mode,
                                                                          iteration_count))

def genRaceConditionDPUDMA(architectures=ALL_ARCHITECTURES,
                           input_types=[FP16(4)],
                           input_shapes=[[1, 32, 16, 16]],
                           weight_types=[FP16(4)],
                           kernel_channels=[64],
                           kernel_shapes=[[1, 1]],
                           output_types=[FP16(4)],
                           output_orders=[Order.NHWC],
                           strides=[[1, 1]],
                           pads=Pad.none,
                           compress=False,
                           mpe_modes=[MPE_MODE.CUBOID_16x16],
                           iteration_count = 64):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, RaceConditionDPUDMA.PARAMS, (RaceConditionDPUDMA,
                                                                             input_type,
                                                                             input_shape,
                                                                             weight_type,
                                                                             kernel_channel,
                                                                             kernel_shape,
                                                                             output_type,
                                                                             output_order,
                                                                             stride,
                                                                             pad,
                                                                             compress,
                                                                             mpe_mode,
                                                                             iteration_count))

def genRaceConditionDPUDMAACT(architectures=ALL_ARCHITECTURES,
                              input_types=[FP16(0)],
                              input_shapes=[[1, 32, 16, 16]],
                              weight_types=[FP16(0)],
                              kernel_channels=[64],
                              kernel_shapes=[[1, 1]],
                              output_types=[FP16(4)],
                              output_orders=[Order.NHWC],
                              strides=[[1, 1]],
                              pads=Pad.none,
                              compress=False,
                              mpe_modes=[MPE_MODE.CUBOID_16x16],
                              act_types=[ActivationType.HSwish],
                              iteration_count = 64):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, act_type) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, act_types):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, RaceConditionDPUDMAACT.PARAMS, (RaceConditionDPUDMAACT,
                                                                                input_type,
                                                                                input_shape,
                                                                                weight_type,
                                                                                kernel_channel,
                                                                                kernel_shape,
                                                                                output_type,
                                                                                output_order,
                                                                                stride,
                                                                                pad,
                                                                                compress,
                                                                                mpe_mode,
                                                                                act_type,
                                                                                iteration_count))

def genDualChannelDMA(architectures=ALL_ARCHITECTURES,
                        input_types=[FP16(2)],
                        output_types=[FP16(2)]):
    for (architecture, input_type, output_type) in itertools.product(architectures, input_types, output_types):
        yield DPUPipeline(architecture, DualChannelDMA.PARAMS, (DualChannelDMA,
                                                    input_type,
                                                    output_type))

def genStorageElementTableDPU(architectures=ALL_ARCHITECTURES,
                            input_types=[FP16(4)],
                            input_shapes=[[1, 32, 16, 16]],
                            weight_types=[FP16(4)],
                            kernel_channels=[32],
                            kernel_shapes=[[1, 1]],
                            output_types=[FP16(4)],
                            output_orders=[Order.NHWC],
                            strides=[[1, 1]],
                            pads=Pad.none,
                            compress=False,
                            mpe_modes=[MPE_MODE.CUBOID_16x16],
                            patterns=[SETablePattern.SwitchLines]):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_mode, pattern) \
            in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_modes, patterns):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue

                yield DPUPipeline(architecture, StorageElementTableDPU.PARAMS, (StorageElementTableDPU,
                                                                              input_type,
                                                                              input_shape,
                                                                              weight_type,
                                                                              kernel_channel,
                                                                              kernel_shape,
                                                                              output_type,
                                                                              output_order,
                                                                              stride,
                                                                              pad,
                                                                              compress,
                                                                              mpe_mode,
                                                                              pattern))

def genRaceConditionDPUACT(architectures=ALL_ARCHITECTURES,
                              input_types=[FP16(0)],
                              input_shapes=[[1, 32, 16, 16]],
                              weight_types=[FP16(0)],
                              kernel_channels=[64],
                              kernel_shapes=[[1, 1]],
                              output_types=[FP16(4)],
                              output_orders=[Order.NHWC],
                              strides=[[1, 1]],
                              pads=Pad.none,
                              compress=False,
                              mpe_cubs=[MPE_MODE.CUBOID_16x16],
                              act_types=[ActivationType.HSwish],
                              iteration_count = 64):

    for (architecture, input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, act_type) in itertools.product(architectures, input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, act_types):

        current_weight_types = weight_types
        for weight_type in current_weight_types:

            current_output_types = output_types
            for output_type in current_output_types:
                if(output_order != Order.NHWC and not _PPE_HAS_PERMUTATION_SUPPORT[output_type.__class__]) :
                    print("skip", output_order, output_type)
                    continue
                yield DPUPipeline(architecture, RaceConditionDPUACT.PARAMS, (RaceConditionDPUACT,
                                                                                input_type,
                                                                                input_shape,
                                                                                weight_type,
                                                                                kernel_channel,
                                                                                kernel_shape,
                                                                                output_type,
                                                                                output_order,
                                                                                stride,
                                                                                pad,
                                                                                compress,
                                                                                mpe_cub,
                                                                                act_type,
                                                                                iteration_count))

def generate_options(args):
    # Available CMX size per tile on VPU37XX: 2M (total) - 0.1M (reserved for runtime) ~= 1.9M
    # Available CMX size is declared in Bytes and is used in DMA_1x1x1* tests
    AVAILABLE_CMX_SIZE = 1024*1942 if args.architecture == "vpu27" else 1024*1432
    HALF_OF_CMX_SIZE = int(AVAILABLE_CMX_SIZE/2)

    return itertools.chain(

        # ActShave
        genActShave(
            input_types=[FP16(0)],
            input_shapes=[[1, 10, 2, 3],  [1, 1000, 1, 1], [1, 1, 1000, 1], [1, 1, 1, 1000]],
            output_types=[FP16()],
            act_shave_subtypes=[
                [ActivationType.HSwish],
                [ActivationType.Sigmoid], # - Sigmoid testcase fails on. Inference result is close enough to NumericBench reference, but not bit-exact
                [ActivationType.Softmax, 1],  # axis C
                [ActivationType.Softmax, 2],  # axis H
                [ActivationType.Softmax, 3],  # axis W
            ]),

        # NOTE: test must align tensor size according to vector size
        genActShave(
            input_types=[FP16(0)],
            input_shapes=[[1, 16, 2, 3],  [1, 1000, 1, 1], [1, 1, 1000, 1], [1, 1, 1, 1000]],
            output_types=[FP16()],
            act_shave_subtypes=[
                [ActivationType.vau_sigm],
                [ActivationType.vau_sqrt],
                [ActivationType.vau_tanh],
                [ActivationType.vau_log],
                [ActivationType.vau_exp],
            ],
            architectures=[Architecture.VPUX37XX]),

        genActShave(
            input_types=[BF16(0)],
            input_shapes=[[1, 16, 2, 3],  [1, 1000, 1, 1], [1, 1, 1000, 1], [1, 1, 1, 1000]],
            output_types=[BF16()],
            act_shave_subtypes=[
                [ActivationType.lsu_b16],
                [ActivationType.lsu_b16_vec],
            ],
            architectures=[Architecture.VPUX37XX]),

        genActShave(
            input_types=[Int32()],
            input_shapes=[[1, 16, 2, 3],  [1, 1000, 1, 1], [1, 1, 1000, 1], [1, 1, 1, 1000]],
            output_types=[Int32()],
            act_shave_subtypes=[
                [ActivationType.sau_dp4],
                [ActivationType.sau_dp4a],
                [ActivationType.sau_dp4m],
            ],
            architectures=[Architecture.VPUX37XX]),

        genActShave(
            input_types=[Int8()],
            input_shapes=[[1, 16, 2, 3], [1, 1008, 1, 1]],
            output_types=[Int32()],
            act_shave_subtypes=[
                [ActivationType.vau_dp4],
                [ActivationType.vau_dp4a],
                [ActivationType.vau_dp4m],
            ],
            architectures=[Architecture.VPUX37XX]),

        # Z-Major Convolution
        #
        # NB MoviSim seems to require uint8 activations when using uint8
        #    weights, and vice-versa.
        #
        # NB NumericsBench requires that if we're using quantized (integer)
        #    activations, the weights must also be quantized.  It also complains
        #    about mixing signed/unsigned values, or integer activations with
        #    fp16 weights.

        # Z-Major Convolution
        genZMConvs(input_types=[Int8(3), Int4(3), UInt8(3), UInt4(3), FP16(4), BF16(4)]),

        # Z-Major Convolution, uint8 activations with extended kernel shapes
        # NB The number of bits used is turned pretty far down, to avoid issues
        # with floating point rounding.
        genZMConvs(input_types=[UInt8(1)],
                   weight_types=[UInt8(1)],
                   kernel_shapes=[[r, c] for r in range(1, 12) for c in range(1, 12) if (r, c) != (1, 1)],
                   output_types=[FP16()],
                   compress=[True, False]),

        # Z-Major Convolution with strides
        genZMConvs(input_types=[FP16(3)],
                   weight_types=[Int8(3), FP16(3)],
                   output_types=[FP16()],
                   kernel_shapes=[[2, 2]],
                   strides=[[r, c] for r in range(1, 8) for c in range(1, 8)],
                   compress=[True, False]),

        # Z-Major Convolution, padding, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10], [11, 11]],
                   output_types=[UInt8()],
                   pads=Pad.none + Pad.all(5) + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5),
                   compress=[True, False]),

        # Z-Major Convolution, padding, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10], [11, 11]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.all(5) + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5),
                   compress=[True, False]),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[Int8(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int8()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5),
                   compress=[True, False]),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 32, 32, 32]],
                   weight_types=[Int8(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int8()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5),
                   compress=[True, False]),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[Int8(2)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5),
                   compress=[True, False]),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 32, 32, 32]],
                   weight_types=[Int8(2)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5),
                   compress=[True, False]),

        # Z-Major Convolution, padding, fp16
        genZMConvs(input_types=[FP16(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[FP16(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[FP16()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5),
                   compress=[True, False]),

        # Z-Major Convolution, padding, bf16
        genZMConvs(input_types=[BF16(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[BF16(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[BF16()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5),
                   compress=[True, False]),

        # Z-Major Convolution, padding, 4x6 kernel, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 8, 8]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[4, 6]],
                   output_types=[UInt8()],
                   pads=[[2,0,0,0],[0,3,0,0]],
                   compress=[True, False]),

        # Z-Major Convolution, padding, 5x5 kernel, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[8, 10]],
                   output_types=[UInt8()],
                   pads=[[4,0,0,0],[0,5,0,0]],
                   compress=[True, False]),

        # Z-Major Convolution, output order
        genZMConvs(input_types=[Int8(3), FP16(4)],
                   output_orders=[Order.NWHC, Order.NWCH, Order.NCWH, Order.NHCW, Order.NCHW],
                   compress=[True, False]),

        # Z-Major Convolution, integer cuboid combinations
        genZMConvs(input_types=[Int8(3)],
                   input_shapes=[[1, 16, 32, 64]],
                   weight_types=[Int8(2)],
                   output_types=[Int8()],
                   mpe_modes=[MPE_MODE.CUBOID_16x16, MPE_MODE.CUBOID_8x16, MPE_MODE.CUBOID_4x16],
                   compress=[True, False]),

        # Z-Major Convolution, fp cuboid combinations
        genZMConvs(input_types=[FP16(4)],
                   input_shapes=[[1, 16, 32, 64]],
                   weight_types=[FP16(2)],
                   output_types=[FP16()],
                   mpe_modes=[MPE_MODE.CUBOID_16x16, MPE_MODE.CUBOID_8x16, MPE_MODE.CUBOID_4x16],
                   compress=[True, False]),

        # Z-Major Convolution, sparse weights, sparse activations
        genSparseConvs(
            input_types=[Int8(4)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[Int8(4)],
            output_types=[Int8()],
            sparsity_factors = [0.1, 0.5, 0.9],
            compress=[True, False],
            activation_sparsity=[False, True]),

        genSparseConvs(
            input_types=[UInt8(4)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[UInt8(4)],
            output_types=[UInt8()],
            sparsity_factors = [0.1, 0.5, 0.9],
            compress=[True, False],
            activation_sparsity=[False, True]),


        genSparseConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[FP16(2), Int8(4)],
            output_types=[FP16()],
            sparsity_factors = [0.1, 0.5, 0.9],
            compress=[True, False],
            activation_sparsity=[False, True]),

        genSparseConvs(
            input_types=[BF16(2)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[BF16(2)],
            output_types=[BF16()],
            sparsity_factors = [0.1, 0.5, 0.9],
            compress=[True, False],
            activation_sparsity=[False, True]),

        # Double Z-Major conv, intermediary sparse activations
        genDoubleZMConvs(
            input_types=[FP16(3)],
            input_shapes=[[1, 16, 1, 1]],
            weight_types=[FP16(-3)],
            kernel_shapes=[[1, 1]],
            kernel_channels=[16, 32, 64, 128, 256, 512],
            output_types=[FP16()],
            compress=[True, False],
            activation_sparsity=[True]),

        # subnormal case
        genSparseConvs(
            input_types=[FP16(-15)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[FP16(-15)],
            output_types=[FP32()],
            sparsity_factors=[0.1, 0.5, 0.9]),

        genZMConvs(input_types=[FP16(2)],
                   input_shapes=[[1, 16, 16, 16]],
                   weight_types=[FP16(2)],
                   output_types=[FP16()],
                   compress=[True, False]),

        # Eltwise Add
        genEltwiseAdds(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                       input_shapes=[[1, 256, 16, 16]]),

        # Eltwise Mult
        genEltwiseMults(input_types=[Int8(3), UInt8(4), FP16(6), BF16(6)],
                        input_shapes=[[1, 1, 1, 64]]),

        # Eltwise Sparse
        genEltwiseSparse(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                         input_shapes=[[1, 256, 16, 16]], seSizes=[256, 128, 64, 32, 16]),

        # MaxPool
        genMaxPools(input_types=[UInt8(6), Int8(6), FP16(6), BF16(6)],
                    input_shapes=[[1, 64, 16, 16]],
                    pads=Pad.none + Pad.all(1) + Pad.top_bottom(1) + Pad.left_right(1)),

        genMaxPools(input_types=[UInt8(6)],
                    output_types=[UInt8()],
                    strides=[[r, c] for r in range(1, 8) for c in range(1, 8) if (r, c) != (2, 2)]),

        genMaxPools(input_types=[UInt8(6)],
                    output_types=[UInt8()],
                    kernel_shapes=[[r, c] for r in range(1, 12) for c in range(1, 12) if (r, c) != (2, 2)]),

        #maxpool with padding up to 5 in all directions
        genMaxPools(input_types=[Int8()],
                    output_types=[Int8()],
                    input_shapes=[[1, 64, 16, 16]],
                    kernel_shapes=[[11, 11]],
                    strides=[[1, 1]],
                    pads= Pad.all(5)
                    ),

        # AvgPool
        genAvgPools(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                    input_shapes=[[1, 64, 32, 32]]),

        # DepthWiseConv
        genDepthWiseConvs(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                          pads=[[0, 0, 0, 0], [1, 0, 0, 0]]),

        genDepthWiseConvs(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                          input_shapes=[[1, 32, 32, 32]],
                          kernel_channels=[32]),

        genDepthWiseConvs(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                          input_shapes=[[1, 64, 32, 32]],
                          kernel_channels=[64]),

        genDepthWiseConvs(input_types=[UInt8(6)],
                          output_types=[UInt8()],
                          strides=[[r, c] for r in range(1, 8) for c in range(1, 8) if (r, c) != (1, 1)]),

        genDepthWiseConvs(input_types=[UInt8(6)],
                          output_types=[UInt8()],
                          kernel_shapes=[[r, c] for r in range(1, 12) for c in range(1, 12) if (r, c) != (4, 4)]),

        #depthwise with padding up to 5 in all directions
        genDepthWiseConvs(input_types=[UInt8(6)],
                          output_types=[UInt8()],
                          kernel_shapes=[[11, 11]],
                          pads= Pad.all(5)
        ),

        # MobileNet ELTWISE, uint8
        genEltwiseAdds(input_types=[UInt8(2)],
                       input_shapes=[[1, 32, 56, 56],
                                     [1, 32, 28, 28],
                                     [1, 64, 14, 14]],
                       output_types=[UInt8()]),

        # MobileNet CONV (ZMajorConv)
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 32, 112, 112]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[16],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 112, 112]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[96],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 96, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 32, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[144],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 144, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 144, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 32, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[192],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 192, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 192, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[64],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 64, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[384],
                   output_types=[UInt8()],
                   compress=[True, False]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 384, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[64],
                   output_types=[UInt8()],
                   compress=[True, False]),

        # Z-Major Convolution, weights swizzling_key = 1-to-5
        # Weights swizzling implemented in ZMajorConv builder
        genZMConvs(input_types=[FP16(3)],
                   input_shapes=[[1, 16, 1, 1]],
                   weight_types=[FP16(-3)],
                   kernel_channels=[64,128,256,512,1024],
                   kernel_shapes=[[1, 1]],
                   output_types=[FP16()],
                   pads=Pad.none,
                   weights_swizzling_keys=[1,2,3,4,5],
                   compress=[True, False]),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 1, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            weights_swizzling_keys=[1,2,3,4,5],
            compress=[True, False]
        ),

        # Z-Major Convolution, activation swizzling_key = 1-to-5
        # Activation swizzling implemented in DoubleConv builder
        genDoubleZMConvs(input_types=[FP16(3)],
                   input_shapes=[[1, 16, 1, 1]],
                   weight_types=[FP16(-3)],
                   kernel_channels=[64,128,256,512],
                   kernel_shapes=[[1, 1]],
                   output_types=[FP16()],
                   pads=Pad.none,
                   activation_swizzling_keys=[1,2,3,4,5],
                   compress=[True, False]
        ),

        # Z-Major Continued Convolution, fp16
        genZMConvs(input_types=[FP16(3)],
                   input_shapes=[[1, 16*1024, 1, 1]],
                   weight_types=[FP16(-3)],
                   kernel_channels=[16],
                   kernel_shapes=[[1, 1]],
                   output_types=[FP16()],
                   pads=Pad.none,
                   compress=[True, False]),

        # Z-major compressed Convolution, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 16, 64, 64]],
                   weight_types=[Int8(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[5, 5]],
                   output_types=[Int8()],
                   strides=[[1, 1]],
                   pads=[[0, 0, 0, 0]],
                   compress=[True]),

        # Z-major first layer DPU optimization uint8
        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 1, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 2, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 3, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 4, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 5, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 6, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 7, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 8, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 9, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 10, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 11, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 12, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[4, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 13, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 14, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 15, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        # Z-major first layer DPU optimization fp16
        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 1, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 2, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 3, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 4, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),


        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 5, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 6, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 7, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 8, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 9, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 10, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 11, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 32, 2, 15]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            output_orders=[Order.NCHW],
            pads=[[0, 0, 0, 0]],
            compress=[True, False]
        ),

        genReadAfterWriteACTDMA(
            architectures=[Architecture.VPUX37XX],
            iteration_count=13
        ),

        genReadAfterWriteDMAACT(
            architectures=[Architecture.VPUX37XX],
            iteration_count=13
        ),

        genReadAfterWriteDPUACT(
            architectures=[Architecture.VPUX37XX],
            iteration_count=11
        ),

        genReadAfterWriteACTDPU(
            act_types=[ActivationType.Sigmoid],
            iteration_count=13
        ),

        genReadAfterWriteDPUDMA(
            iteration_count=13
        ),

        genReadAfterWriteDMADPU(
            iteration_count=13
        ),

        genDifferentClustersDPU(architectures=[Architecture.VPUX37XX]),
        genMultiClustersDPU(architectures=[Architecture.VPUX37XX],
                            kernel_channels=[64],
                            segmentation=SEGMENTATION.SOK,
                            is_out_broadcasted=[True, False]),
        genMultiClustersDPU(architectures=[Architecture.VPUX37XX],
                            input_shapes=[[1, 32, 32, 32]],
                            kernel_channels=[64],
                            strides=[[3, 3], [1, 1]],
                            segmentation=SEGMENTATION.SOH,
                            is_out_broadcasted=[True, False]),
        genMultiClustersDPU(architectures=[Architecture.VPUX37XX],
                            input_shapes=[[1, 32, 32, 32]],
                            kernel_channels=[64],
                            kernel_shapes=[[3, 3]],
                            strides=[[2, 2], [1, 1]],
                            segmentation=SEGMENTATION.SOH,
                            is_out_broadcasted=[True, False]),

        # # check all datatypes
        genDMA(
            tensor_types=[Int4(3), UInt4(3), Int8(3), UInt8(3), FP16(4), BF16(4)],
            input_shapes=[[1, 16, 32, 32]]
        ),

        # check all memory locations
        genDMA(
            architectures=[Architecture.VPUX37XX],
            tensor_types=[Int8(3)],
            input_shapes=[[1, 32, 16, 16]],
            src_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1, MemoryLocation.DDR],
            dst_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1, MemoryLocation.DDR],
            dma_engines=[0, 1]
        ),
        # check max available CMX
        genDMA(
            architectures=[Architecture.VPUX37XX],
            tensor_types=[UInt8(3)],
            input_shapes=[[1, 1, 1, HALF_OF_CMX_SIZE]],
            src_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1],
            dst_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1],
            dma_engines=[0, 1]
        ),

        genRaceConditionDMA(
            architectures=[Architecture.VPUX37XX],
            input_types=[FP16(2)],
            output_types=[FP16(2)],
            iteration_count=64 # 64 (max) barriers = 2 tiles x 32 barriers per tile
        ),

        genRaceConditionDPU(
            architectures=[Architecture.VPUX37XX],
            iteration_count=48 # 48 barriers = 2 tiles x 24 barriers per tile
        ),

        genRaceConditionDPUDMA(
            architectures=[Architecture.VPUX37XX],
            iteration_count=48 # 48 barriers = 2 tiles x 24 barriers per tile
        ),

        genRaceConditionDPUDMAACT(
            architectures=[Architecture.VPUX37XX],
            iteration_count=24 # single tile test, max 32 barriers per tile, test configures iteration_count+1=25 barriers
        ),

        genRaceConditionDPUACT(
            architectures=[Architecture.VPUX37XX],
            iteration_count=24
        ),

        genRaceCondition(
            ops = genActShave(
                architectures=[Architecture.VPUX37XX],
                input_types=[FP16(0)],
                input_shapes=[[1, 10, 2, 3]],
                output_types=[FP16()],
                act_shave_subtypes=[
                    [ActivationType.HSwish],
                    [ActivationType.Sigmoid], # Sigmoid testcase fails. Inference result is close enough to NumericBench reference, but not bit-exact
                    [ActivationType.Softmax, 1],  # axis C
                    [ActivationType.Softmax, 2],  # axis H
                    [ActivationType.Softmax, 3],  # axis W
                ]
            ),
            architectures=[Architecture.VPUX37XX],
            iteration_counts=[10], # number of barriers used in test is 10x2=20 < 32 (max number of barriers per tile)
            requested_clusters=[1, 2],
            requested_units=[1, 2]
        ),

        # NOTE: test must align tensor size according to vector size
        genRaceCondition(
            ops = genActShave(
                architectures=[Architecture.VPUX37XX],
                input_types=[FP16(0)],
                input_shapes=[[1, 16, 2, 3]],
                output_types=[FP16()],
                act_shave_subtypes=[
                    [ActivationType.vau_sigm],
                    [ActivationType.vau_sqrt],
                    [ActivationType.vau_tanh],
                    [ActivationType.vau_log],
                    [ActivationType.vau_exp],
                    [ActivationType.lsu_b16],
                    [ActivationType.lsu_b16_vec],
                ]
            ),
            architectures=[Architecture.VPUX37XX],
            iteration_counts=[10],
            requested_clusters=[1, 2],
            requested_units=[1, 2]
        ),

        genZMConvs(
            input_types=[Int8(2)],
            input_shapes=[[1, 16, 16, 16]],
            weight_types=[Int8(2)],
            kernel_channels=[16],
            kernel_shapes=[[1, 1]],
            output_types=[Int32()],
            pads=[[0, 0, 0, 0]],
            activations=[
                PReLU(args.architecture, 0.1, np.int8),
                PReLU(args.architecture, 0.1, np.uint8),
                PReLU(args.architecture, 0.5, np.int8),
                PReLU(args.architecture, 0.5, np.uint8),
                PReLU(args.architecture, 1, np.int8),
                PReLU(args.architecture, 1, np.uint8),
                PReLU(args.architecture, 1.5, np.int8),
                PReLU(args.architecture, 1.5, np.uint8),
            ]
        ),

        genZMConvs(
            input_types=[Int8(4)],
            input_shapes=[[1, 16, 16, 16]],
            weight_types=[Int8(4)],
            kernel_channels=[16],
            kernel_shapes=[[1, 1]],
            output_types=[Int32()],
            pads=[[0, 0, 0, 0]],
            activations=[
                ReLUX(args.architecture, 6, np.int8),
                ReLUX(args.architecture, 6, np.uint8),
            ]
        ),

        genZMConvs(
            input_types=[FP16(1)],
            input_shapes=[[1, 16, 16, 16]],
            weight_types=[FP16(1)],
            kernel_channels=[16],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            activations=[
                ReLUX(args.architecture, 6, np.int8),
                ReLUX(args.architecture, 6, np.uint8),
                ReLUX(args.architecture, 6),
            ]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 16, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[16],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            activations=[
                PReLU(args.architecture, 0.1),
                PReLU(args.architecture, 0.5),
                PReLU(args.architecture, 1),
                PReLU(args.architecture, 1.5),
                PReLU(args.architecture, -0.1),
                PReLU(args.architecture, -0.5),
                PReLU(args.architecture, -1),
                PReLU(args.architecture, -1.5),
            ]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 16, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[16],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            activations=[
                PReLU(args.architecture, 0.1, np.int8),
                PReLU(args.architecture, 0.1, np.uint8),
                PReLU(args.architecture, 0.5, np.int8),
                PReLU(args.architecture, 0.5, np.uint8),
                PReLU(args.architecture, 1, np.int8),
                PReLU(args.architecture, 1, np.uint8),
                PReLU(args.architecture, 1.5, np.int8),
                PReLU(args.architecture, 1.5, np.uint8),
            ]
        ),

        # when storage_elements are stacked in Z direction, they must be a power of 2 in size
        genStorageElementTableDPU(
            patterns=[SETablePattern.SwitchLines, SETablePattern.OriginalInput],
            input_shapes=[[1, 32, 16, 16]],
        ),

        genDualChannelDMA(
            architectures=[Architecture.VPUX37XX],
            input_types=[FP16(2)],
            output_types=[FP16(2)]
        )
    )

def create_config_files(args):
    tests_path = Path(args.filter)
    if tests_path.is_file():
        with open(args.filter) as file:
            ci_tests = file.readlines()
            ci_tests = [line.rstrip() for line in ci_tests]
            ci_tests = "|".join(ci_tests)
        args.filter = f"\\b(?:{ci_tests})\\b"
    filt = re.compile(args.filter)
    args.root.mkdir(parents=True, exist_ok=args.exist_ok)
    found = {}
    for option in generate_options(args):
        if option.architecture is not args.architecture:
            continue

        option.validate()
        ident = option.ident
        if ident in found:
            raise Exception(f'Duplicate option ident: {ident}:\n  {option.settings}')
        found[ident] = 1
        if not filter_issues(args, option):
            continue
        if not filt.match(ident):
            continue
        try:
            option.compute_values()
        except ComputationError as ce:
            if isinstance(ce.__cause__, EntropyError) and args.low_entropy_ok:
               traceback.print_exc(file=sys.stderr)
            else:
                raise
        path = args.root / option.ident
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'config.json').open('w') as outfile:
            option.write_data(path)
            descriptor = option.value()
            if 'operation' in descriptor:
                descriptor['operation']['compiler_backend'] = args.compiler_backend.name
            descriptor['compiler_backend'] = args.compiler_backend.name

            json.dump(descriptor, outfile, indent=4)

def export_excel(args):
    data = {}
    for case in generate_options(args):
        if case.name in data:
            data[case.name].append(case)
        else:
            data[case.name] = [case]

    def alignWidth(sheet, header):
        for index, column in enumerate(sheet.columns):
            maxWidth = len(header[index])
            for cell in column:
                maxWidth = max(maxWidth, len(str(cell.value)))
            adjustedWidth = (maxWidth + 2) * 1.2
            sheet.column_dimensions[get_column_letter(index + 1)].width = adjustedWidth

    def centralize(sheet):
        for column in sheet.columns:
            for cell in column:
                cell.alignment = Alignment(horizontal='center')

    with pd.ExcelWriter(args.filename) as writer:
        for name, values in data.items():
            header = list(values[0].data.keys())
            df = pd.DataFrame(value.data for value in values)
            df.to_excel(writer, sheet_name=name, columns=header, index=False)
            sheet = writer.sheets[name]
            centralize(sheet)
            alignWidth(sheet, header)


def export_csv(args):
    df = pd.DataFrame((opt.data for opt in generate_options(args)))
    df.to_csv(args.filename)


def get_architecture_type(arg):
    if arg == "vpu27":
        return Architecture.VPUX37XX
    else:
        raise argparse.ArgumentTypeError("value must be 'vpu27'")


def get_compiler_backend(arg):
    if arg == "flatbuffer":
        return CompilerBackend.Flatbuffer
    elif arg == "elf":
        return CompilerBackend.ELF
    else:
        raise argparse.ArgumentTypeError("value must be either 'flatbuffer' or 'elf', got '{}'".format(arg))


def main():
    parser = argparse.ArgumentParser(description='Create hardware test cases', prog='generate_hw_testcases')
    subparsers = parser.add_subparsers()

    parser_write_configs = subparsers.add_parser('write-configs', help='Write test case configurations and sample data')
    parser_write_configs.add_argument('root', type=Path, help='The directory where the test cases should be written')
    parser_write_configs.add_argument('--exist-ok', help='Reuse the contents of the root', action='store_true')
    parser_write_configs.add_argument('--low-entropy-ok', help='Ignore entropy errors', action='store_true')
    parser_write_configs.add_argument('--filter', help='Regex filter for the generated tests', default='.*')
    parser_write_configs.add_argument('--arch', help='Architecture for which to generate tests', default='vpu27', dest="architecture", type=get_architecture_type)
    args, remaining = parser.parse_known_args()
    compiler_backend_default = "elf" if args.architecture is Architecture.VPUX37XX else "flatbuffer"
    parser_write_configs.add_argument('--compiler-backend', help='Compiler backend for which to generate tests', default=compiler_backend_default, dest="compiler_backend", type=get_compiler_backend)
    parser_write_configs.set_defaults(func=create_config_files)

    parser_export_excel = subparsers.add_parser('export-excel', help='Write test cases as an Excel spreadsheet')
    parser_export_excel.add_argument('filename', type=Path, help='The spreadsheet to create')
    parser_export_excel.set_defaults(func=export_excel)

    parser_export_csv = subparsers.add_parser('export-csv', help='Write test cases as an CSV spreadsheet')
    parser_export_csv.add_argument('filename', type=Path, help='The spreadsheet to create')
    parser_export_csv.set_defaults(func=export_csv)

    args = parser.parse_args()
    numerics_bench_imports(args.architecture)
    args.func(args)


if __name__ == '__main__':
    main()
