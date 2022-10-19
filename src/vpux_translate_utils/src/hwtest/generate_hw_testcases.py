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
import math
import operator
import pandas as pd
from pathlib import Path
import re
import sys
import traceback
import warnings
from typing import Callable, List, Optional, Sequence, Union
import numpy.ma as ma
from openpyxl.styles.alignment import Alignment
from openpyxl.utils import get_column_letter
from scipy.sparse import rand as randSparse

import os
numericBenchPath = os.getenv('NUMERICSBENCH_PATH')
if numericBenchPath == None:
    print("env variable NUMERICSBENCH_PATH not set, Using default path for NumericsBench")
    sys.path.append(str(Path(__file__).parents[5] / 'NumericsBench'))
else:
    sys.path.append(numericBenchPath)

from operators.compute_core import bfloat16
from operators.vpu26 import Add, Mult, Conv2DVPUX37XX, MaxPool, AveragePool, PRelu, RequantFuseWithPRelu
from operators.vpu26 import HSwish, Sigmoid, Softmax
from operators.platform.quantize_info import QuantizationInfo
from operators.platform.quantized_tensor import NBQuantized
from operators.platform.vpu26 import PlatformVPU26

import numpy as np
from numpy.random import default_rng

class Architecture(Enum):
    VPUX30XX = 3700,
    VPUX37XX = 3720,
    VPUX40XX = 4000

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

class M2I_FMT(Enum):
    PL_YUV420_8 = auto()
    SP_NV12_8   = auto()
    PL_FP16_RGB = auto()
    PL_FP16_BGR = auto()
    PL_RGB24    = auto()
    PL_BGR24    = auto()
    IL_RGB888   = auto()
    IL_BGR888   = auto()

class MPE_CUBES(Enum):
    CUBOID_16x16 = auto()
    CUBOID_8x16 = auto()
    CUBOID_4x16 = auto()

mpeCube2NTHW_NTK = {
    MPE_CUBES.CUBOID_16x16: '16x4',
    MPE_CUBES.CUBOID_4x16: '4x16',
    MPE_CUBES.CUBOID_8x16: '8x8'
}

class SEGMENTATION(Enum):
    SOK = 1
    SOH = 2

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
    # Regarding documentation
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


def pack_int4(data: np.ndarray) -> np.ndarray:
    flat = data.flatten()
    result = []
    for idx in range(0, flat.size, 2):
        lsn = flat[idx + 0] & 0x0f
        msn = flat[idx + 1] & 0x0f
        datum = np.uint8(msn << 4 | lsn)
        result.append(datum)
    return np.array(result).astype(np.uint8)


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

    def generateSparse(self, filename: str, shape, rng, sparsity_factor=0.5, orderer=None) -> np.ndarray:
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

    def generateSparse(self, filename: str, shape, rng, sparsity_factor=0.5, orderer=None) -> np.ndarray:
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
            data = np.full(shape, self.value, dtype=np.float16)
        else:
            data = np.around(rng.random(size=shape, dtype=np.float32) * 8.) / 8.
            data = (data * (2. ** self.bitwidth)).astype(np.float16)

        return Value(self,
                      filename,
                      data,
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def generateSparse(self, filename: str, shape, rng, sparsity_factor=0.5, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        matrix = randSparse(1, np.prod(shape), density=1-sparsity_factor, format="csr", random_state=rng)
        data = np.around(matrix.toarray().reshape(shape) * 8.) / 8.

        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(np.float16),
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
        data = np.around(rng.random(size=shape, dtype=np.float32) * 8.) / 8.
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
        data = np.around(rng.random(size=shape, dtype=np.float32) * 8.) / 8.
        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(bfloat16),
                      self.bitwidth,
                      self.bitsize,
                      True,
                      orderer)

    def generateSparse(self, filename: str, shape, rng, sparsity_factor=0.5, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        matrix = randSparse(1, np.prod(shape), density=1-sparsity_factor, format="csr", random_state=rng)
        data = np.around(matrix.toarray().reshape(shape) * 8.) / 8.

        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(bfloat16),
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


def idu(input: Value, weights: Value) -> "tuple[np.ndarray, np.ndarray]":
    """Models the hardware IDU"""
    if input.is_float or weights.is_float:
        return input.data.astype(np.float32), weights.data.astype(np.float32)

    def to_qint32(value: Value) -> Union[np.ndarray, NBQuantized]:
        return NBQuantized(value=value.data.astype(np.int32), scale=value.scale, zero_point=value.zero,
                           platform=PlatformVPU26(), quantization_info=QuantizationInfo(value.ttype.qtype))

    return to_qint32(input), to_qint32(weights)

def iduConvCustom(input: Value, weights: Value) -> "tuple[np.ndarray, np.ndarray]":
    """Custom Model the hardware IDU that feet the NumericBench requirements for convolution operation"""
    if (input.data.dtype == np.float32) or (weights.data.dtype == np.float32) :
        raise Error(f'NumericBench\'s convolution operation doesn\'t support float32 datatype for inputs/weights')

    def to_qint32(value: Value) -> Union[np.ndarray, NBQuantized]:
        return NBQuantized(value=value.data.astype(np.int32), scale=value.scale, zero_point=value.zero,
                           platform=PlatformVPU26(), quantization_info=QuantizationInfo(value.ttype.qtype))
    if not input.is_float and not weights.is_float :
        return to_qint32(input), to_qint32(weights)

    if input.data.dtype == weights.data.dtype :
        return input.data, weights.data

    # NumericBench requires activations and weights types are equal meanwhile VPUX37XX hardware supports different data types

    if (input.data.dtype == bfloat16) or (weights.data.dtype == bfloat16) :
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
        rescale = not output_ttype.is_float
        output_type = output_ttype
        output_scale = 1.
        output_zero_point = 0

        result_bitwidth = math.ceil(np.log2(np.amax(abs(ndarray))))
        bitshift = max(result_bitwidth - output_type.bitwidth, 0)

        if rescale:
            if np.issubdtype(ndarray.dtype, np.integer):
                ndarray = ndarray.astype(np.float64)

            if isinstance(activation, PReLU):
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
                if isinstance(activation, PReLU):
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
        'mpe_cub'
    ]
    NAME = 'Z-Major'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'ZMajorConvolution',
            'input': [inputs[0].json_info],
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
            },
            'output_order': self.settings.output_order.name.lower()
        }

    def validate(self):
        ValidatePaddings(self.settings.kernel_shape, self.settings.kernel_pads)
        # validate input tensor channels alignment
        # ValidateHWAlignment(self.settings.input_ttype, self.settings.input_shape[1])
        # validate weight tensor channels alignment
        # ValidateHWAlignment(self.settings.weight_ttype, self.settings.input_shape[1])
        # validate output tensor channels alignment
        ValidateHWAlignment(self.settings.output_ttype, self.settings.kernel_channels)

    @property
    def ident(self) -> str:
        name = f'zm_conv_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_{shape_to_str(self.settings.weight_shape)}x{self.settings.weight_ttype.stype}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}_kern_chan_{self.settings.kernel_channels}'
        if self.settings.output_order != Order.NHWC:
            name += '_' + self.settings.output_order.name.lower()
        if self.settings.compress:
            name += '_compressed'
        if self.settings.mpe_cub != MPE_CUBES.CUBOID_16x16:
            name += '_' + self.settings.mpe_cub.name
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
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_cub],
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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
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
        'mpe_cub',
        'sparsity_factor'
    ]
    NAME = 'Sparse'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'SparseZMajorConvolution',
            'input': [inputs[0].json_info],
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
            },
            'output_order': self.settings.output_order.name.lower()
        }

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
        name = f'sparse_conv_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_{shape_to_str(self.settings.weight_shape)}x{self.settings.weight_ttype.stype}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}_kern_chan_{self.settings.kernel_channels}'
        if self.settings.output_order != Order.NHWC:
            name += '_' + self.settings.output_order.name.lower()
        if self.settings.compress:
            name += '_compressed'
        if self.settings.mpe_cub != MPE_CUBES.CUBOID_16x16:
            name += '_' + self.settings.mpe_cub.name
        name += f'_sparsity_{self.settings.sparsity_factor}'
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
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.weight_ttype.generateSparse('weights.dat', self.settings.weight_shape, rng, self.settings.sparsity_factor, orderer=OrderNCHW)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
                     pads = self.settings.kernel_pads,
                     strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
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
            'weight': inputs[1].json_info,
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
        return f'dw_conv_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_{shape_to_str(self.settings.weight_shape)}x{self.settings.input_ttype.stype}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}_kern_chan_{self.settings.kernel_channels}'

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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides,
                        group = self.settings.weight_shape[0])
        return c2d.inference(lhs, rhs)

    def filter_issues(self, args) -> bool:
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
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs)
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'ew_add_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}'

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
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs)
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'ew_mult_{self.settings.input_ttype.stype}'

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
        return f'max_pool_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_{shape_to_str(self.settings.kernel_shape)}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}'

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
        return f'avg_pool_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_kernel_{shape_to_str(self.settings.kernel_shape)}_strides_{shape_to_str(self.settings.kernel_strides)}'

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
        if self.settings.activation_type[0] == ActivationType.Sigmoid :
            self.issues.add('E#29771')
        if self.settings.activation_type[0] == ActivationType.Softmax :
            self.issues.add('E#29771')
            self.issues.add('E#29786')


    @property
    def ident(self) -> str:
        name = self.settings.activation_type[0].name
        ident = f'act_kernel_{name}_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}'
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
            result = np.sqrt(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.vau_tanh :
            result = np.tanh(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.vau_log :
            np.seterr(divide = 'ignore')
            result = np.log(values[0].data.astype(np.float32))
            result = np.nan_to_num(result)
            np.seterr(divide = 'warn')
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
        if 'E#29771' in self.issues:
            # Filter not bit-exact comparison issues
            return False
        if 'E#29786' in self.issues:
            # Filter incorrect tensor serialization issues
            return False
        return True

# dummy placeholder for NumericsBench-M2I-model, required to get output shape
def toyM2iReference(input_fmt, input_shape, out_fmt, out_type, out_sizes):
    N  = input_shape[0]

    if ((input_fmt == M2I_FMT.PL_FP16_RGB) or (input_fmt == M2I_FMT.PL_FP16_BGR) or
        (input_fmt == M2I_FMT.PL_RGB24)    or (input_fmt == M2I_FMT.PL_BGR24)):
        iC = input_shape[1] # Planar formats
        iH = input_shape[2]
        iW = input_shape[3]
    else: # includes I420, NV12 (see Openvino defs)
        iH = input_shape[1] # Interleaved
        iW = input_shape[2]
        iC = input_shape[3]

    # default output size
    oH = iH
    oW = iW

    # adjust output height for YUV-input, RGB-output
    if ((input_fmt == M2I_FMT.SP_NV12_8) or (input_fmt == M2I_FMT.PL_YUV420_8)):
        if ((out_fmt != M2I_FMT.SP_NV12_8) and (out_fmt != M2I_FMT.PL_YUV420_8)):
            oH = int(oH * 2 / 3)

    # custom resize for H,W dims
    if out_sizes:
        assert len(out_sizes)==2, 'Sizes must contain values H,W'
        oH = out_sizes[0]
        oW = out_sizes[1]

    if ((out_fmt == M2I_FMT.IL_RGB888) or (out_fmt == M2I_FMT.IL_BGR888)):
        out_shape = [N,oH,oW,3]
    else: # all other output formats are PLANAR
        out_shape = [N,3,oH,oW]

    # generate dummy data, not using input-data
    out = out_type.generate('fake_out_0.bin', out_shape, default_rng(1))
    return out


class M2iTask(Operation):

    def __init__(self, descr, architecture, input_fmt, input_shape, output_fmt, out_sizes, do_norm, norm_coefs):
        super().__init__(architecture)
        settings = Settings()
        self.settings = settings
        self.settings.input_fmt = input_fmt
        self.settings.output_fmt = output_fmt
        self.settings.output_sizes = out_sizes
        self.settings.input_shape = input_shape
        self.settings.do_norm = do_norm
        self.settings.norm_coefs = norm_coefs
        self.settings.input_type = self.get_elem_type(input_fmt)
        self.settings.output_type = self.get_elem_type(output_fmt)
        self.settings.descr = descr # just a string prefix (SG:single, F1:fuse_with_scale, F2:fuse_w/o_scale)

        # TBD in future cases: IL_RGB888 -> PL_RGB24 requires csc-flag ?
        self.settings.do_csc = True if (input_fmt != output_fmt) else False

        if do_norm and (len(norm_coefs) == 0):
            assert 0, 'TBD: generate coefs randomly if not provided'

    def get_elem_type(self, fmt):
        if ((fmt == M2I_FMT.SP_NV12_8) or (fmt == M2I_FMT.PL_YUV420_8) or
            (fmt == M2I_FMT.IL_RGB888) or (fmt == M2I_FMT.IL_BGR888)   or
            (fmt == M2I_FMT.PL_RGB24)  or (fmt == M2I_FMT.PL_BGR24)):
            return UInt8()
        elif ((fmt == M2I_FMT.PL_FP16_RGB) or (fmt == M2I_FMT.PL_FP16_BGR)):
            return FP16()
        else:
            assert 0, 'Unsupported M2iTask input format !'

    @property
    def ident(self) -> str:
        # 'output_shape' is implicity derived from 'input_shape' and 'output_format'
        ss = self.settings
        return f'm2i_{ss.descr}_in_{shape_to_str(ss.input_shape)}_{ss.input_fmt.name}_out_{ss.output_fmt.name}_outSz_{ss.output_sizes}_norm_{ss.do_norm}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    def validate(self):
        return True

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_type.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def json_info(self, inputs, outputs):
        json = {
            'case_type': 'M2iTask',
            'input': get_values_json_info(inputs),
            'output': get_values_json_info([outputs]),
            'm2i_params': {
                'input_fmt'    : self.settings.input_fmt.name,
                'output_fmt'   : self.settings.output_fmt.name,
                'output_sizes' : self.settings.output_sizes,
                'do_csc'       : self.settings.do_csc,
                'do_norm'      : self.settings.do_norm,
                'norm_coefs'   : self.settings.norm_coefs
            }
        }
        return json

    @property
    def data(self) -> dict:
        return {
            'Name': self.ident
        }

    def value(self):
        value = {'architecture': self.architecture.name}
        value = {**value, **self.json_info(self.inputs, self.o)}
        return value

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        # TBD: should be generated by NUMERIC_BENCH model !!!
        result = toyM2iReference(self.settings.input_fmt, self.settings.input_shape, \
                                 self.settings.output_fmt, self.settings.output_type, self.settings.output_sizes)
        return result

    def compute_values(self):
        self.inputs = self.generate_inputs(default_rng(1))
        self.o = self.apply_mpe(self.inputs)

    def write_data(self, dir: Path):
        orderer = OrderNCHW # write data as is
        for input in self.inputs:
            input.write_data(dir, orderer)
        self.o.write_data(dir, orderer)

    def filter_issues(self, args) -> bool:
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
        ident = f'ReadAfterWriteACTDMA_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'
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
        }

        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Type': ActivationType2String[self.settings.activation_type[0]],
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
                'name' : self.settings.activation_type[0].name
            },
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

        return json

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        count = (self.settings.iteration_count - 1) // 2
        result = values[0].data.astype(self.settings.input_ttype.dtype)
        for i in range(0, count) :
            result = HSwish().inference(result)
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
        ident = f'ReadAfterWriteDMAACT_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'
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
        }

        return {
            'Name': self.ident,
            'Input Type': np.dtype(self.settings.input_ttype),
            'Output Type': np.dtype(self.settings.output_ttype),
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Type': ActivationType2String[self.settings.activation_type[0]],
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
                'name' : self.settings.activation_type[0].name
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
        'mpe_cub',
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
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'ReadAfterWriteDPUDMA_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'

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
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_cub],
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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
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
        'mpe_cub',
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
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'ReadAfterWriteDMADPU_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'

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
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_cub],
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
        'mpe_cub',
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
            'weight': inputs[1].json_info,
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
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'ReadAfterWriteDPUACT_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'

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
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_cub],
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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
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
        'mpe_cub',
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
            'weight': inputs[1].json_info,
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
            'cluster_number' : self.settings.cluster_number,
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'ReadAfterWriteACTDPU_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}_cluster_{self.settings.cluster_number}'

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
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_cub],
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
            result = HSwish().inference(result)
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

    PARAMS = ['op_class', 'input_ttype', 'output_ttype', 'input_shape', 'src_memloc', 'dst_memloc', 'dma_engine']

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings

    def json_info(self, input, outputs):
        return {
            'case_type': 'DMA',
            'input': get_values_json_info(input),
            'output': get_values_json_info(outputs),
            'DMA_params': {
                'src_memory_location' : self.settings.src_memloc.name,
                'dst_memory_location' : self.settings.dst_memloc.name,
                'dma_engine' : self.settings.dma_engine
            }
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'DMA_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_from_{self.settings.src_memloc.name}_to_{self.settings.dst_memloc.name}_engine_{self.settings.dma_engine}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def output_orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        return {
            'Op Mode': 'MPE_DMA',
            'Input Type': self.settings.input_ttype.stype,
            'Output Type': self.settings.output_ttype.stype,
            'Src location': self.settings.src_memloc.name,
            'Dst location': self.settings.dst_memloc.name,
            'Engine': self.settings.dma_engine
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def apply_mpe(self, values: List[Value]) -> np.ndarray:
        return values[0].data

    def filter_issues(self, args) -> bool:
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
        'mpe_cub',
        'DPU_task_params'
    ]
    NAME = 'DifferentClustersDPU'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'DifferentClustersDPU',
            'input': [inputs[0].json_info],
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
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
        return f'DifferentClustersDPU_{self.settings.input_ttype.stype}_input_cluster_{self.settings.DPU_task_params[0]}_output_cluster_{out_clusters_str}_weights_cluster_{self.settings.DPU_task_params[2]}_weights_table_cluster_{self.settings.DPU_task_params[3]}'

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
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_cub],
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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
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
        'mpe_cub',
        'MultiClustersDPU_params'
    ]
    NAME = 'MultiClustersDPU'

    def __init__(self, architecture: Architecture, settings):
        super().__init__(architecture)

        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, outputs):
        return {
            'case_type': 'MultiClustersDPU',
            'input': [inputs[0].json_info],
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'DPUTaskParams': {
                'task_clusters' : self.settings.MultiClustersDPU_params[0],
                'segmentation': self.settings.MultiClustersDPU_params[1].name,
                'broadcast': self.settings.MultiClustersDPU_params[2]
            }
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        task_clusters_str = '_'.join([str(cluster) for cluster in self.settings.MultiClustersDPU_params[0]])
        kernel_size_str = 'x'.join([str(dim) for dim in self.settings.kernel_shape])
        kernel_stride_str = 'x'.join([str(dim) for dim in self.settings.kernel_strides])
        return f'MultiClustersDPU_{self.settings.input_ttype.stype}_task_cluster_{task_clusters_str}_{self.settings.MultiClustersDPU_params[1]}_broadcast_{self.settings.MultiClustersDPU_params[2]}_kern_chan_{self.settings.kernel_channels}_kern_sz_{kernel_size_str}_kern_stride_{kernel_stride_str}_in_shape_{shape_to_str(self.settings.input_shape)}'

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
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_cub],
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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
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
        return f'DMA_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

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
        'mpe_cub',
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
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'DPU_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

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
            'NTHW_NTK': mpeCube2NTHW_NTK[self.settings.mpe_cub],
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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

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
        'mpe_cub',
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
            'weight': inputs[1].json_info,
            'output': get_values_json_info(outputs),
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1,
                'compress': self.settings.compress,
                'mpe_cub': self.settings.mpe_cub.name
            },
            'output_order': self.settings.output_order.name.lower(),
            'iteration_count' : self.settings.iteration_count
        }

    def validate(self):
        pass

    @property
    def ident(self) -> str:
        return f'DPU_DMA_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

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
            'case_type': 'RaceConditionDPUDMAACT',
            'input': [inputs[0].json_info],
            'weight': inputs[1].json_info,
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
        return f'DPU_DMA_ACT_race_cond_{self.settings.input_ttype.stype}_iter_count_{self.settings.iteration_count}'

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
        c2d = Conv2DVPUX37XX(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

class RaceCondition:
    PARAMS = ['operation', 'iteration_count', 'requested_cluster', 'requested_unit']

    def __init__(self, architecture: Architecture, operation, iter_count, requested_cluster, requested_unit):
        self.architecture = architecture
        self.operation = operation
        self.op = operation.op
        self.iter_count = iter_count
        self.requested_cluster = requested_cluster
        self.requested_unit = requested_unit
        self.issues = set()
        if self.operation.settings.op_class == ActKernel and self.requested_cluster == 2 :
            self.issues.add('E#30347')

    def json_info(self):
        return {
            'architecture': self.architecture.name,
            'case_type': 'RaceCondition',
            'iteration_count' : self.iter_count,
            'requested_clusters' : self.requested_cluster,
            'requested_units' : self.requested_unit
        }

    def validate(self):
        self.operation.validate()

    @property
    def ident(self) -> str:
        return f'race_cond_{self.operation.ident}_iters_{self.iter_count}_clusters_{self.requested_cluster}_shaves_{self.requested_unit}'

    def compute_values(self):
        self.operation.compute_values()

    def write_data(self, dir: Path):
        self.operation.write_data(dir)

    def value(self):
        json = self.json_info()
        json['operation'] = self.operation.value()
        return json

    def filter_issues(self, args):
        if 'E#30347' in self.issues:
            # Filter unsupported cluster param
            return False
        return self.op.filter_issues(args)


class Settings:
    def __str__(self):
        return '\n  '.join([f'{name}={getattr(self, name)}' for name in dir(self) if not name.startswith('_')])


class DPUPipeline:
    def __init__(self, architecture: Architecture, option_names, option_values, activation=None):
        settings = Settings()
        self.settings = settings
        self.issues = set()
        for name, value in zip(option_names, option_values):
            setattr(settings, name, value)

        self.op = settings.op_class(architecture, settings)
        self.activation = activation

    def compute_values(self):
        try:
            self.inputs = self.op.generate_inputs(default_rng(1))
            mpe_data = self.op.apply_mpe(self.inputs)
            self.mpe_data = mpe_data.value if isinstance(mpe_data, NBQuantized) else mpe_data
            ppe_value = self.op.ppe(self.inputs, self.settings.output_ttype, mpe_data, self.activation)
            self.outputs = self.op.odu(ppe_value)
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
    Int8: [Int8(3), Int4(3)],
    Int4: [Int8(3), UInt4(3), Int4(3)],
    UInt8: [UInt8(3)],
    UInt4: [UInt4(3), Int4(3)],
    FP16: [FP16(4), Int8(3), Int4(3)],
    BF16: [BF16(4)]
}


_PPE_VALID_OUTPUT_TYPES = {
    False: [Int32(), FP16(), UInt8(), UInt4(), Int8(), Int4()],  # Integer MACs
    True: [FP32(), FP16(), BF16(), UInt8(), Int8(), Int4()],    # FP MACs
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


def genZMConvs(architecture,
               input_types=[UInt8(2)],
               input_shapes=[[1, 32, 16, 16]],
               weight_types=None,
               kernel_channels=[64],
               kernel_shapes=[[1, 1]],
               output_types=None,
               output_orders=[Order.NHWC],
               strides=[[1, 1]],
               pads=Pad.none,
               compress=False,
               mpe_cubs=[MPE_CUBES.CUBOID_16x16],
               activations=[None]):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, activation) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, activations):

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
                                                                           compress,
                                                                           mpe_cub), activation)

def genSparseConvs( architecture,
                    input_types=[UInt8(2)],
                    input_shapes=[[1, 32, 16, 16]],
                    weight_types=None,
                    kernel_channels=[32],
                    kernel_shapes=[[1, 1]],
                    output_types=None,
                    output_orders=[Order.NHWC],
                    strides=[[1, 1]],
                    pads=Pad.none,
                    compress=False,
                    mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                    sparsity_factors=[0.5]):

    print("genSparseConvs")
    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, sparsity_factor) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, sparsity_factors):

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
                                                                            compress,
                                                                            mpe_cub,
                                                                            sparsity_factor
                                                                            ))

def genEltwiseAdds(architecture,
                   input_types=[Int8(6)],
                   input_shapes=[[1, 256, 16, 16]],
                   output_types=None):
    for (input_type, input_shape) in itertools.product(input_types, input_shapes):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(architecture, EltwiseAdd.PARAMS, (EltwiseAdd,
                                                                input_type,
                                                                input_shape,
                                                                output_type))


def genEltwiseMults(architecture,
                    input_types=[Int8(6)],
                    input_shapes=[[1, 256, 16, 16]],
                    output_types=None):
    for (input_type, input_shape) in itertools.product(input_types, input_shapes):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(architecture, EltwiseMult.PARAMS, (EltwiseMult,
                                                                 input_type,
                                                                 input_shape,
                                                                 output_type))


def genMaxPools(architecture,
                input_types=[FP16(6)],
                input_shapes=[[1, 64, 16, 16]],
                kernel_shapes=[[2, 2]],
                output_types=None,
                strides=[[2, 2]],
                pads=Pad.none):
    for (input_type, input_shape, kernel_shape, stride, pad) in itertools.product(input_types, input_shapes, kernel_shapes, strides, pads):
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


def genAvgPools(architecture,
                input_types=[FP16(6)],
                input_shapes=[[1, 64, 32, 32]],
                kernel_shapes=[[2, 2]],
                output_types=None,
                strides=[[2, 2]]):
    for (input_type, input_shape, kernel_shape, stride) in itertools.product(input_types, input_shapes, kernel_shapes, strides):
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

def genDepthWiseConvs(architecture,
                      input_types=[FP16(2)],
                      input_shapes=[[1, 16, 32, 32]],
                      kernel_channels=[16],
                      kernel_shapes=[[4, 4]],
                      output_types=None,
                      strides=[[1, 1]],
                      pads=Pad.none):
    for (input_type, input_shape, kernel_channel, kernel_shape, stride, pad) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, strides, pads):

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

def genReadAfterWriteACTDMA(architecture,
                            input_types=[FP16(0)],
                            input_shapes=[[1, 10, 2, 3]],
                            output_types=[FP16()],
                            act_shave_subtypes=[
                                [ActivationType.HSwish],
                            ],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):
                    for (input_type, input_shape, output_type, act_shave_subtype, cluster_number) in itertools.product(input_types, input_shapes, output_types, act_shave_subtypes, cluster_numbers):
                        yield DPUPipeline(architecture, ReadAfterWriteACTDMA.PARAMS, (ReadAfterWriteACTDMA, input_type, input_shape, output_type, act_shave_subtype, cluster_number, iteration_count))

def genReadAfterWriteDMAACT(architecture,
                            input_types=[FP16(0)],
                            input_shapes=[[1, 10, 2, 3]],
                            output_types=[FP16()],
                            act_shave_subtypes=[
                                [ActivationType.HSwish],
                            ],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):
                    for (input_type, input_shape, output_type, act_shave_subtype, cluster_number) in itertools.product(input_types, input_shapes, output_types, act_shave_subtypes, cluster_numbers):
                        yield DPUPipeline(architecture, ReadAfterWriteDMAACT.PARAMS, (ReadAfterWriteDMAACT, input_type, input_shape, output_type, act_shave_subtype, cluster_number, iteration_count))

def genReadAfterWriteDPUDMA(architecture,
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
                            mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):
    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, cluster_number) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, cluster_numbers):

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
                                                                              mpe_cub,
                                                                              cluster_number,
                                                                              iteration_count))

def genReadAfterWriteDMADPU(architecture,
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
                            mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):
    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, cluster_number) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, cluster_numbers):

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
                                                                              mpe_cub,
                                                                              cluster_number,
                                                                              iteration_count))

def genReadAfterWriteDPUACT(architecture,
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
                            mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                            act_types=[ActivationType.HSwish],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, act_type, cluster_number) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, act_types, cluster_numbers):

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
                                                                              mpe_cub,
                                                                              act_type,
                                                                              cluster_number,
                                                                              iteration_count))

def genReadAfterWriteACTDPU(architecture,
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
                            mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                            act_types=[ActivationType.HSwish],
                            cluster_numbers=[0, 1],
                            iteration_count = 19):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, act_type, cluster_number) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, act_types, cluster_numbers):

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
                                                                              mpe_cub,
                                                                              act_type,
                                                                              cluster_number,
                                                                              iteration_count))

def genDifferentClustersDPU(architecture,
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
                            mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                            input_clusters = [0, 1],
                            output_clusters = [[0], [1], [0, 1]],
                            weights_clusters = [0, 1],
                            weights_table_clusters = [0, 1]):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, input_cluster, output_cluster, weights_cluster, weights_table_cluster) \
            in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, input_clusters, output_clusters, weights_clusters, weights_table_clusters):

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
                                                                              mpe_cub,
                                                                              [input_cluster, output_cluster, weights_cluster, weights_table_cluster]))

def genMultiClustersDPU(architecture,
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
                        mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                        task_clusters=[[0, 1]],
                        segmentation=SEGMENTATION.SOK,
                        is_out_broadcasted=[True]):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, task_cluster, broadcast) \
            in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, task_clusters, is_out_broadcasted):

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
                                                                          mpe_cub,
                                                                          [task_cluster, segmentation, broadcast]))



def genActShave(architecture,
                input_types,
                input_shapes,
                output_types,
                act_shave_subtypes):
    for (input_type, input_shape, output_type, act_shave_subtype) in itertools.product(input_types, input_shapes, output_types, act_shave_subtypes):
        yield DPUPipeline(architecture, ActKernel.PARAMS, (ActKernel, input_type, input_shape, output_type, act_shave_subtype))

def genM2iTask(descr,
               architecture,
               input_fmts,
               input_shapes,
               do_norm, norm_coefs,
               output_fmts,
               output_sizes):
    for (in_fmt, in_shape, out_fmt) in itertools.product(input_fmts, input_shapes, output_fmts):
        # OpenVINO defines SINGLE_PLANE buffs for NV12 and I420 as having H = Luma_H * 3/2
        # in order to include chroma buffs
        in_shp = in_shape[:] # a copy
        if ((in_fmt == M2I_FMT.SP_NV12_8) or (in_fmt == M2I_FMT.PL_YUV420_8)):
            assert ((in_shape[1] % 2) == 0), 'YUV input height must be EVEN number'
            in_shp[1] = int(in_shp[1] * 3 / 2)
        yield M2iTask(descr, architecture, in_fmt, in_shp, out_fmt, output_sizes, do_norm, norm_coefs)

AVAILABLE_CMX_SIZE = 1024*1942 # size in Bytes
HALF_OF_CMX_SIZE = int(AVAILABLE_CMX_SIZE/2)

def genDMA(architecture, tensor_types=[FP16(2)], input_shapes=[[1, 16, 32, 32]], src_locations=[MemoryLocation.CMX0], dst_locations=[MemoryLocation.CMX0], dma_engines=[0]):
    for (tensor_type, input_shape, src_location, dst_location, dma_engine) in itertools.product(tensor_types, input_shapes, src_locations, dst_locations, dma_engines):
        yield DPUPipeline(architecture, DMA.PARAMS, (DMA,
                                                     tensor_type,
                                                     tensor_type,
                                                     input_shape,
                                                     src_location,
                                                     dst_location,
                                                     dma_engine))

def genRaceConditionDMA(architecture,
                        input_types=[FP16(2)],
                        output_types=[FP16(2)],
                        iteration_count=64):
    for (input_type, output_type) in itertools.product(input_types, output_types):
        yield DPUPipeline(architecture, RaceConditionDMA.PARAMS, (RaceConditionDMA,
                                                    input_type,
                                                    output_type,
                                                    iteration_count
                                                    ))
def genRaceCondition(architecture,
                     ops,
                     iteration_counts=[64],
                     requested_clusters=[1],
                     requested_units=[1]):
    for (op, iteration_count, requested_cluster, requested_unit) in itertools.product(ops, iteration_counts, requested_clusters, requested_units):
        yield RaceCondition(architecture, op, iteration_count, requested_cluster, requested_unit)

def genRaceConditionDPU(architecture,
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
                        mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                        iteration_count = 64):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs):

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
                                                                          mpe_cub,
                                                                          iteration_count))

def genRaceConditionDPUDMA(architecture,
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
                           mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                           iteration_count = 64):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs):

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
                                                                             mpe_cub,
                                                                             iteration_count))

def genRaceConditionDPUDMAACT(architecture,
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
                              mpe_cubs=[MPE_CUBES.CUBOID_16x16],
                              act_types=[ActivationType.HSwish],
                              iteration_count = 64):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub, act_type) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs, act_types):

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
                                                                                mpe_cub,
                                                                                act_type,
                                                                                iteration_count))

def generate_options(args):
    return itertools.chain(

        # M2I
        genM2iTask('SG',  # CSC only
            architecture=Architecture.VPUX40XX,
            input_fmts=[M2I_FMT.SP_NV12_8, M2I_FMT.PL_YUV420_8],
            input_shapes=[[1, 240, 320, 1]], # N,H,W,C
            do_norm=False,
            norm_coefs = [],
            output_fmts=[M2I_FMT.IL_RGB888, M2I_FMT.IL_BGR888],
            output_sizes = []),

        genM2iTask('SG',  # Normalization only
            architecture=Architecture.VPUX40XX,
            input_fmts=[M2I_FMT.PL_FP16_RGB],
            input_shapes=[[1, 3, 256, 256]], # N,C,H,W
            do_norm=True,
            # A,B,C,D coefs for each of R,G,B channels
            norm_coefs = [10.0, 11.0, 12.0, 13.0, \
                          20.0, 21.0, 22.0, 23.0, \
                          30.0, 31.0, 32.0, 33.0],
            output_fmts=[M2I_FMT.PL_FP16_RGB],
            output_sizes = []),

        genM2iTask('SG',  # Resize only - PLANAR
            architecture=Architecture.VPUX40XX,
            input_fmts=[M2I_FMT.PL_FP16_RGB],
            input_shapes=[[1, 3, 256, 256]], # N,C,H,W
            do_norm=False, norm_coefs = [],
            output_fmts=[M2I_FMT.PL_FP16_RGB],
            output_sizes = [224,224]),

        genM2iTask('SG',  # Resize only - Interleaved
            architecture=Architecture.VPUX40XX,
            input_fmts=[M2I_FMT.IL_RGB888],
            input_shapes=[[1, 256, 256, 3]], # N,H,W,C
            do_norm=False, norm_coefs = [],
            output_fmts=[M2I_FMT.IL_RGB888],
            output_sizes = [224,224]),

        # Fuses types (with resize)
        #   [csc->convertU8toF16->resize->permute]
        #   [csc->resize->permute]
        #   [csc->resize]
        genM2iTask('F1',
            architecture=Architecture.VPUX40XX,
            input_fmts=[M2I_FMT.SP_NV12_8, M2I_FMT.PL_YUV420_8],
            input_shapes=[[1, 192, 256, 1]], # N,H,W,C
            do_norm=False,
            norm_coefs = [],
            output_fmts=[M2I_FMT.IL_RGB888,   M2I_FMT.IL_BGR888, \
                         M2I_FMT.PL_RGB24,    M2I_FMT.PL_BGR24,  \
                         M2I_FMT.PL_FP16_RGB, M2I_FMT.PL_FP16_BGR],
            output_sizes = [168, 224]),

        # Fuses types (no resize)
        #   [csc->permute]
        #   [csc->convertU8toF16->permute]
        genM2iTask('F2',
            architecture=Architecture.VPUX40XX,
            input_fmts=[M2I_FMT.SP_NV12_8, M2I_FMT.PL_YUV420_8],
            input_shapes=[[1, 168, 224, 1]], # N,H,W,C
            do_norm=False,
            norm_coefs = [],
            output_fmts=[M2I_FMT.PL_RGB24,    M2I_FMT.PL_BGR24,  \
                         M2I_FMT.PL_FP16_RGB, M2I_FMT.PL_FP16_BGR],
            output_sizes = []),

        # ActShave
        genActShave(
            architecture=Architecture.VPUX37XX,
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
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(0)],
            input_shapes=[[1, 16, 2, 3],  [1, 1000, 1, 1], [1, 1, 1000, 1], [1, 1, 1, 1000]],
            output_types=[FP16()],
            act_shave_subtypes=[
                [ActivationType.vau_sigm],
                [ActivationType.vau_sqrt],
                [ActivationType.vau_tanh],
                [ActivationType.vau_log],
                [ActivationType.vau_exp],
            ]),

        genActShave(
            architecture=Architecture.VPUX37XX,
            input_types=[BF16(0)],
            input_shapes=[[1, 16, 2, 3],  [1, 1000, 1, 1], [1, 1, 1000, 1], [1, 1, 1, 1000]],
            output_types=[BF16()],
            act_shave_subtypes=[
                [ActivationType.lsu_b16],
                [ActivationType.lsu_b16_vec],
            ]),

        genActShave(
            architecture=Architecture.VPUX37XX,
            input_types=[Int32()],
            input_shapes=[[1, 16, 2, 3],  [1, 1000, 1, 1], [1, 1, 1000, 1], [1, 1, 1, 1000]],
            output_types=[Int32()],
            act_shave_subtypes=[
                [ActivationType.sau_dp4],
                [ActivationType.sau_dp4a],
                [ActivationType.sau_dp4m],
            ]),

        genActShave(
            architecture=Architecture.VPUX37XX,
            input_types=[Int8()],
            input_shapes=[[1, 16, 2, 3], [1, 1008, 1, 1]],
            output_types=[Int32()],
            act_shave_subtypes=[
                [ActivationType.vau_dp4],
                [ActivationType.vau_dp4a],
                [ActivationType.vau_dp4m],
            ]),

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
        genZMConvs(architecture=Architecture.VPUX37XX, input_types=[Int8(3), Int4(3), UInt8(3), UInt4(3), FP16(4), BF16(4)]),

        # Z-Major Convolution, uint8 activations with extended kernel shapes
        # NB The number of bits used is turned pretty far down, to avoid issues
        # with floating point rounding.
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(1)],
                   weight_types=[UInt8(1)],
                   kernel_shapes=[[r, c] for r in range(1, 12) for c in range(1, 12) if (r, c) != (1, 1)],
                   output_types=[FP16()]),

        # Z-Major Convolution with strides
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[FP16(3)],
                   weight_types=[Int8(3), FP16(3)],
                   output_types=[FP16()],
                   kernel_shapes=[[2, 2]],
                   strides=[[r, c] for r in range(1, 8) for c in range(1, 8)]),

        # Z-Major Convolution, padding, uint8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10], [11, 11]],
                   output_types=[UInt8()],
                   pads=Pad.none + Pad.all(5) + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, uint8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10], [11, 11]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.all(5) + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[Int8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[Int8(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int8()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[Int8(2)],
                   input_shapes=[[1, 32, 32, 32]],
                   weight_types=[Int4(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int8()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[Int8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[Int8(2)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[Int8(2)],
                   input_shapes=[[1, 32, 32, 32]],
                   weight_types=[Int4(2)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, fp16
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[FP16(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[FP16(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[FP16()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, bf16
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[BF16(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[BF16(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[BF16()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, 4x6 kernel, uint8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 8, 8]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[4, 6]],
                   output_types=[UInt8()],
                   pads=[[2,0,0,0],[0,3,0,0]]),

        # Z-Major Convolution, padding, 5x5 kernel, uint8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[8, 10]],
                   output_types=[UInt8()],
                   pads=[[4,0,0,0],[0,5,0,0]]),

        # Z-Major Convolution, output order
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[Int8(3), FP16(4)],
                   output_orders=[Order.NWHC, Order.NWCH, Order.NCWH, Order.NHCW, Order.NCHW]),

        # Z-Major Convolution, integer cuboid combinations
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[Int8(3)],
                   input_shapes=[[1, 16, 32, 64]],
                   weight_types=[Int8(2)],
                   output_types=[Int8()],
                   mpe_cubs=[MPE_CUBES.CUBOID_16x16, MPE_CUBES.CUBOID_8x16, MPE_CUBES.CUBOID_4x16]),

        # Z-Major Convolution, fp cuboid combinations
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[FP16(4)],
                   input_shapes=[[1, 16, 32, 64]],
                   weight_types=[FP16(2)],
                   output_types=[FP16()],
                   mpe_cubs=[MPE_CUBES.CUBOID_16x16, MPE_CUBES.CUBOID_8x16, MPE_CUBES.CUBOID_4x16]),

        # Z-Major Convolution, sparse
        genSparseConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[Int8(4)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[Int8(4)],
            output_types=[Int8()],
            sparsity_factors = [0.1, 0.5, 0.9]),

        genSparseConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(4)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[UInt8(4)],
            output_types=[UInt8()],
            sparsity_factors = [0.1, 0.5, 0.9]),

        genSparseConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[FP16(2)],
            output_types=[FP16()],
            sparsity_factors = [0.1, 0.5, 0.9]),

        genSparseConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[BF16(2)],
            input_shapes=[[1, 128, 32, 32]],
            kernel_shapes=[[1, 1], [5, 5]],
            weight_types=[BF16(2)],
            output_types=[BF16()],
            sparsity_factors = [0.1, 0.5, 0.9]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[FP16(2)],
                   input_shapes=[[1, 16, 16, 16]],
                   weight_types=[FP16(2)],
                   output_types=[FP16()]),

        # Eltwise Add
        genEltwiseAdds(architecture=Architecture.VPUX37XX,
                       input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                       input_shapes=[[1, 256, 16, 16]]),

        # Eltwise Mult
        genEltwiseMults(architecture=Architecture.VPUX37XX,
                        input_types=[Int8(3), UInt8(4), FP16(6), BF16(6)],
                        input_shapes=[[1, 1, 1, 64]]),

        # MaxPool
        genMaxPools(architecture=Architecture.VPUX37XX,
                    input_types=[UInt8(6), Int8(6), FP16(6), BF16(6)],
                    input_shapes=[[1, 64, 16, 16]],
                    pads=Pad.none + Pad.all(1) + Pad.top_bottom(1) + Pad.left_right(1)),

        genMaxPools(architecture=Architecture.VPUX37XX,
                    input_types=[UInt8(6)],
                    output_types=[UInt8()],
                    strides=[[r, c] for r in range(2, 8) for c in range(2, 8) if (r, c) != (2, 2)]),

        genMaxPools(architecture=Architecture.VPUX37XX,
                    input_types=[UInt8(6)],
                    output_types=[UInt8()],
                    kernel_shapes=[[r, c] for r in range(2, 12) for c in range(2, 12) if (r, c) != (2, 2)]),

        # AvgPool
        genAvgPools(architecture=Architecture.VPUX37XX,
                    input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                    input_shapes=[[1, 64, 32, 32]]),

        # DepthWiseConv
        genDepthWiseConvs(architecture=Architecture.VPUX37XX,
                          input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                          pads=[[0, 0, 0, 0], [1, 0, 0, 0]]),

        genDepthWiseConvs(architecture=Architecture.VPUX37XX,
                          input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                          input_shapes=[[1, 32, 32, 32]],
                          kernel_channels=[32]),

        genDepthWiseConvs(architecture=Architecture.VPUX37XX,
                          input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                          input_shapes=[[1, 64, 32, 32]],
                          kernel_channels=[64]),

        genDepthWiseConvs(architecture=Architecture.VPUX37XX,
                          input_types=[UInt8(6)],
                          output_types=[UInt8()],
                          strides=[[r, c] for r in range(1, 8) for c in range(1, 8) if (r, c) != (1, 1)]),

        genDepthWiseConvs(architecture=Architecture.VPUX37XX,
                          input_types=[UInt8(6)],
                          output_types=[UInt8()],
                          kernel_shapes=[[r, c] for r in range(1, 12) for c in range(1, 12) if (r, c) != (4, 4)]),

        # MobileNet ELTWISE, uint8
        genEltwiseAdds(architecture=Architecture.VPUX37XX,
                       input_types=[UInt8(2)],
                       input_shapes=[[1, 32, 56, 56],
                                     [1, 32, 28, 28],
                                     [1, 64, 14, 14]],
                       output_types=[UInt8()]),

        # MobileNet CONV (ZMajorConv)
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 32, 112, 112]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[16],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 112, 112]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[96],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 96, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 32, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[144],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 144, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 144, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 32, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[192],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 192, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 192, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[64],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 64, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[384],
                   output_types=[UInt8()]),

        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[UInt8(2)],
                   input_shapes=[[1, 384, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[64],
                   output_types=[UInt8()]),

        # Z-Major Convolution, weights swizzling_key = 1-to-5
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[FP16(3)],
                   input_shapes=[[1, 16, 1, 1]],
                   weight_types=[FP16(-3)],
                   kernel_channels=[64,128,256,512,1024],
                   kernel_shapes=[[1, 1]],
                   output_types=[FP16()],
                   pads=Pad.none),

        # Z-Major Continued Convolution, fp16
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[FP16(3)],
                   input_shapes=[[1, 16*1024, 1, 1]],
                   weight_types=[FP16(-3)],
                   kernel_channels=[16],
                   kernel_shapes=[[1, 1]],
                   output_types=[FP16()],
                   pads=Pad.none),

        # Z-major compressed Convolution, int8
        genZMConvs(architecture=Architecture.VPUX37XX,
                   input_types=[Int8(2)],
                   input_shapes=[[1, 16, 64, 64]],
                   weight_types=[Int8(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[5, 5]],
                   output_types=[Int8()],
                   strides=[[1, 1]],
                   pads=[[0, 0, 0, 0]],
                   compress=True),

        # Z-major first layer DPU optimization uint8
        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 1, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 2, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 3, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 4, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 5, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 6, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 7, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 8, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 9, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 10, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 11, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 12, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[4, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 13, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 14, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[UInt8(2)],
            input_shapes=[[1, 15, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        # Z-major first layer DPU optimization fp16
        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 1, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 2, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 3, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 4, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),


        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 5, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 6, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 7, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 8, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 9, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 10, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 11, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 32, 2, 15]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            output_orders=[Order.NCHW],
            pads=[[0, 0, 0, 0]]
        ),

        genReadAfterWriteACTDMA(
            architecture=Architecture.VPUX37XX,
            iteration_count=13
        ),

        genReadAfterWriteDMAACT(
            architecture=Architecture.VPUX37XX,
            iteration_count=13
        ),

        genReadAfterWriteDPUACT(
            architecture=Architecture.VPUX37XX,
            iteration_count=11
        ),

        genReadAfterWriteACTDPU(
            architecture=Architecture.VPUX37XX,
            iteration_count=13
        ),

        genReadAfterWriteDPUDMA(
            architecture=Architecture.VPUX37XX,
            iteration_count=13
        ),

        genReadAfterWriteDMADPU(
            architecture=Architecture.VPUX37XX,
            iteration_count=13
        ),

        genDifferentClustersDPU(architecture=Architecture.VPUX37XX),
        genMultiClustersDPU(architecture=Architecture.VPUX37XX,
                            kernel_channels=[64],
                            segmentation=SEGMENTATION.SOK,
                            is_out_broadcasted=[True, False]),
        genMultiClustersDPU(architecture=Architecture.VPUX37XX,
                            input_shapes=[[1, 32, 32, 32]],
                            kernel_channels=[64],
                            strides=[[3, 3], [1, 1]],
                            segmentation=SEGMENTATION.SOH,
                            is_out_broadcasted=[True, False]),
        genMultiClustersDPU(architecture=Architecture.VPUX37XX,
                            input_shapes=[[1, 32, 32, 32]],
                            kernel_channels=[64],
                            kernel_shapes=[[3, 3]],
                            strides=[[2, 2], [1, 1]],
                            segmentation=SEGMENTATION.SOH,
                            is_out_broadcasted=[True, False]),

        # # check all datatypes
        genDMA(
            architecture=Architecture.VPUX37XX,
            tensor_types=[Int4(3), UInt4(3), Int8(3), UInt8(3), FP16(4), BF16(4)],
            input_shapes=[[1, 16, 32, 32]]
        ),

        # check all memory locations
        genDMA(
            architecture=Architecture.VPUX37XX,
            tensor_types=[Int8(3)],
            input_shapes=[[1, 32, 16, 16]],
            src_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1, MemoryLocation.DDR],
            dst_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1, MemoryLocation.DDR],
            dma_engines=[0, 1]
        ),
        # check max available CMX
        genDMA(
            architecture=Architecture.VPUX37XX,
            tensor_types=[UInt8(3)],
            input_shapes=[[1, 1, 1, HALF_OF_CMX_SIZE]],
            src_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1],
            dst_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1],
            dma_engines=[0, 1]
        ),

        genRaceConditionDMA(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            output_types=[FP16(2)],
            iteration_count=64 # 64 (max) barriers = 2 tiles x 32 barriers per tile
        ),

        genRaceConditionDMA(
            architecture=Architecture.VPUX40XX,
            input_types=[FP16(2)],
            output_types=[FP16(2)],
            iteration_count=96 # 96 (max) barriers = 6 tiles x 16 barriers per tile
        ),

        genRaceConditionDPU(
            architecture=Architecture.VPUX37XX,
            iteration_count=48 # 48 barriers = 2 tiles x 24 barriers per tile
        ),

        genRaceConditionDPU(
            architecture=Architecture.VPUX40XX,
            iteration_count=72 # 72 barriers = 6 tiles x 12 barriers per tile
        ),

        genRaceConditionDPUDMA(
            architecture=Architecture.VPUX37XX,
            iteration_count=48 # 48 barriers = 2 tiles x 24 barriers per tile
        ),

        genRaceConditionDPUDMA(
            architecture=Architecture.VPUX40XX,
            iteration_count=72 # 72 barriers = 6 tiles x 12 barriers per tile
        ),

        genRaceConditionDPUDMAACT(
            architecture=Architecture.VPUX37XX,
            iteration_count=24 # single tile test, max 32 barriers per tile, test configures iteration_count+1=25 barriers
        ),

        genRaceConditionDPUDMAACT(
            architecture=Architecture.VPUX40XX,
            iteration_count=12 # single tile test, max 16 barriers per tile, test configures iteration_count+1=13 barriers
        ),

        genRaceCondition(
            architecture=Architecture.VPUX37XX,
            ops = genActShave(
                architecture=Architecture.VPUX37XX,
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
            iteration_counts=[10],
            requested_clusters=[1, 2],
            requested_units=[1, 2]
        ),

        # NOTE: test must align tensor size according to vector size
        genRaceCondition(
            architecture=Architecture.VPUX37XX,
            ops = genActShave(
                architecture=Architecture.VPUX37XX,
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
            iteration_counts=[10],
            requested_clusters=[1, 2],
            requested_units=[1, 2]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[Int8(2)],
            input_shapes=[[1, 16, 16, 16]],
            weight_types=[Int8(2)],
            kernel_channels=[16],
            kernel_shapes=[[1, 1]],
            output_types=[Int32()],
            pads=[[0, 0, 0, 0]],
            activations=[
                PReLU(Architecture.VPUX37XX, 0.1, np.int8),
                PReLU(Architecture.VPUX37XX, 0.1, np.uint8),
                PReLU(Architecture.VPUX37XX, 0.5, np.int8),
                PReLU(Architecture.VPUX37XX, 0.5, np.uint8),
                PReLU(Architecture.VPUX37XX, 1, np.int8),
                PReLU(Architecture.VPUX37XX, 1, np.uint8),
                PReLU(Architecture.VPUX37XX, 1.5, np.int8),
                PReLU(Architecture.VPUX37XX, 1.5, np.uint8),
            ]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 16, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[16],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            activations=[
                PReLU(Architecture.VPUX37XX, 0.1),
                PReLU(Architecture.VPUX37XX, 0.5),
                PReLU(Architecture.VPUX37XX, 1),
                PReLU(Architecture.VPUX37XX, 1.5),
                PReLU(Architecture.VPUX37XX, -0.1),
                PReLU(Architecture.VPUX37XX, -0.5),
                PReLU(Architecture.VPUX37XX, -1),
                PReLU(Architecture.VPUX37XX, -1.5),
            ]
        ),

        genZMConvs(
            architecture=Architecture.VPUX37XX,
            input_types=[FP16(2)],
            input_shapes=[[1, 16, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[16],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]],
            activations=[
                PReLU(Architecture.VPUX37XX, 0.1, np.int8),
                PReLU(Architecture.VPUX37XX, 0.1, np.uint8),
                PReLU(Architecture.VPUX37XX, 0.5, np.int8),
                PReLU(Architecture.VPUX37XX, 0.5, np.uint8),
                PReLU(Architecture.VPUX37XX, 1, np.int8),
                PReLU(Architecture.VPUX37XX, 1, np.uint8),
                PReLU(Architecture.VPUX37XX, 1.5, np.int8),
                PReLU(Architecture.VPUX37XX, 1.5, np.uint8),
            ]
        ),
    )


def create_config_files(args):
    filt = re.compile(args.filter)
    args.root.mkdir(parents=True, exist_ok=args.exist_ok)
    found = {}
    for option in generate_options(args):
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


def main():
    parser = argparse.ArgumentParser(description='Create hardware test cases', prog='generate_hw_testcases')
    subparsers = parser.add_subparsers()

    parser_write_configs = subparsers.add_parser('write-configs', help='Write test case configurations and sample data')
    parser_write_configs.add_argument('root', type=Path, help='The directory where the test cases should be written')
    parser_write_configs.add_argument('--exist-ok', help='Reuse the contents of the root', action='store_true')
    parser_write_configs.add_argument('--low-entropy-ok', help='Ignore entropy errors', action='store_true')
    parser_write_configs.add_argument('--filter', help='Regex filter for the generated tests', default='.*')
    parser_write_configs.set_defaults(func=create_config_files)

    parser_export_excel = subparsers.add_parser('export-excel', help='Write test cases as an Excel spreadsheet')
    parser_export_excel.add_argument('filename', type=Path, help='The spreadsheet to create')
    parser_export_excel.set_defaults(func=export_excel)

    parser_export_csv = subparsers.add_parser('export-csv', help='Write test cases as an CSV spreadsheet')
    parser_export_csv.add_argument('filename', type=Path, help='The spreadsheet to create')
    parser_export_csv.set_defaults(func=export_csv)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
