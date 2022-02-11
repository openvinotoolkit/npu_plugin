#!/usr/bin/env python3

#
# Copyright Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
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


# TODO: Fix this awful hack, whose purpose in life is to point to where you've
# checked out ssh://git@gitlab.devtools.intel.com:29418/iotgai/NumericsBench.git.
import os
numericBenchPath = os.getenv('NUMERICSBENCH_PATH')
if numericBenchPath == None:
    print("env variable NUMERICSBENCH_PATH not set, Using default path for NumericsBench")
    sys.path.append(str(Path(__file__).parents[5] / 'NumericsBench'))
else:
    sys.path.append(numericBenchPath)

from operators.compute_core import bfloat16
from operators.vpu26 import Add, Mult, MTLConv2D, MaxPool, AveragePool
from operators.vpu26 import HSwish, Sigmoid, Softmax
from operators.platform.quantize_info import QuantizationInfo
from operators.platform.quantized_tensor import NBQuantized
from operators.platform.vpu26 import PlatformVPU26

import numpy as np
from numpy.random import default_rng


Orderer = Callable[[np.ndarray], np.ndarray]


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

class MPE_CUBES(Enum):
    CUBOID_16x16 = auto()
    CUBOID_8x16 = auto()
    CUBOID_4x16 = auto()

mpeCube2NTHW_NTK = {
    MPE_CUBES.CUBOID_16x16: '16x4',
    MPE_CUBES.CUBOID_4x16: '4x16',
    MPE_CUBES.CUBOID_8x16: '8x8'
}

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
    # we have next paddings constaints:
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

    def check_entropy(self, data: np.ndarray):
        self._check_entropy_eq(data, 0)
        self._check_entropy_eq(data, 255)

    @property
    def is_float(self) -> bool:
        return False

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(0, 255)

class Int8(TType):
    def __init__(self, bitwidth=7):
        super().__init__(np.int8, 'int8', 'int8', bitwidth, True)
        self.bitsize = 8
        self.low = np.int8(-(2 ** bitwidth))
        self.high = np.int8((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        return Value(self,
                      filename,
                      rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.int8),
                      self.bitwidth,
                      self.bitsize,
                      True,
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
    def __init__(self, bitwidth=16):
        super().__init__(np.float16, 'fp16', None, bitwidth, True)
        self.bitsize = 16

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        data = np.around(rng.random(size=shape, dtype=np.float32) * 8.) / 8.

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
        # Issue link: https://gitlab.devtools.intel.com/iotgai/NumericsBench/-/issues/247
        raise Error(f'NumericBench\'s convolution operation doesn\'t support float32 datatype for inputs/weights')

    def to_qint32(value: Value) -> Union[np.ndarray, NBQuantized]:
        return NBQuantized(value=value.data.astype(np.int32), scale=value.scale, zero_point=value.zero,
                           platform=PlatformVPU26(), quantization_info=QuantizationInfo(value.ttype.qtype))
    if not input.is_float and not weights.is_float :
        return to_qint32(input), to_qint32(weights)

    if input.data.dtype == weights.data.dtype :
        return input.data, weights.data

    # NumericBench requires activations and weights types are equal meanwhile MTL hardware supports different data types

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
    def apply(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def filter_issues(self, args) -> bool:
        pass


def shape_to_str(shape: Sequence[int]) -> str:
    return 'x'.join([str(d) for d in shape])

class ZMajorConvolution(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = [
        'mpe_op_class',
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

    def __init__(self, settings):
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, output):
        return {
            'case_type': 'ZMajorConvolution',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = MTLConv2D(kernel_shape=self.settings.kernel_shape,
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
        'mpe_op_class',
        'input_ttype',
        'input_shape',
        'kernel_channels',
        'kernel_shape',
        'output_ttype',
        'kernel_strides',
        'kernel_pads'
    ]
    NAME = 'DW'

    def __init__(self, settings):
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, 1] + settings.kernel_shape

    def json_info(self, inputs, output):
        return {
            'case_type': 'DepthWiseConv',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = MTLConv2D(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides,
                        group = self.settings.weight_shape[0])
        return c2d.inference(lhs, rhs)

    def filter_issues(self, args) -> bool:
        return True


class EltwiseAdd(Operation):

    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'output_ttype']
    NAME = 'Add'

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, inputs, output):
        return {
            'case_type': 'EltwiseAdd',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
            'output': output.json_info
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

    def apply(self, values: List[Value]) -> np.ndarray:
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

    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'output_ttype']
    NAME = 'Mult'

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, inputs, output):
        return {
            'case_type': 'EltwiseMult',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
            'output': output.json_info
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

    def apply(self, values: List[Value]) -> np.ndarray:
        multer = Mult()
        lhs, rhs = idu(values[0], values[1])
        if isinstance(lhs, NBQuantized) and isinstance(lhs, NBQuantized):
            # Workaround for NumericsBench's Mult operation: see EltwiseMult.apply()
            return multer.inference(lhs, rhs)
        return multer.functor(lhs, rhs)

    def filter_issues(self, args) -> bool:
        return True

class Maxpool(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'kernel_shape', 'output_ttype', 'kernel_strides', 'kernel_pads']
    NAME = 'MaxPool'

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, inputs, output):
        return {
            'case_type': 'MaxPool',
            'input': inputs[0].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = idu(values[0], values[0])
        maxpool = MaxPool(kernel_shape=self.settings.kernel_shape, strides=self.settings.kernel_strides, pads=self.settings.kernel_pads)
        return maxpool.inference(lhs)

    def filter_issues(self, args) -> bool:
        return True


class AvgPool(Operation):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'kernel_shape', 'output_ttype', 'kernel_strides']
    NAME = 'AvgPool'

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, inputs, output):
        # NB We need to rescale the output if it's quantized, since this is how the system implements the
        # division for quantized pool outputs.

        return {
            'case_type': 'AvgPool',
            'input': inputs[0].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = idu(values[0], values[0])
        avgpool = AveragePool(kernel_shape=self.settings.kernel_shape, strides=self.settings.kernel_strides, pads=[0, 0, 0, 0])
        return avgpool.inference(lhs)

    def filter_issues(self, args) -> bool:
        return True


class ActivationType(Enum):
    HSwish = auto()
    Sigmoid = auto()
    Softmax = auto()

class ActKernel(Operation):
    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'output_ttype', 'activation_type']
    NAME = 'ActKernel'

    def __init__(self, settings):
        self.settings = settings
        self.issues = set()
        if self.settings.activation_type[0] == ActivationType.Sigmoid :
            self.issues.add('EISW-29771')
        if self.settings.activation_type[0] == ActivationType.Softmax :
            self.issues.add('EISW-29771')
            self.issues.add('EISW-29786')


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
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def json_info(self, inputs, output):
        assert(output.is_float)

        json = {
            'case_type': 'ActShave',
            'input': inputs[0].json_info,
            'output': output.json_info,
            'activation': {
                'name' : self.settings.activation_type[0].name
            }
        }
        if self.settings.activation_type[0] == ActivationType.Softmax :
            json['activation']['axis']=self.settings.activation_type[1]

        return json

    def apply(self, values: List[Value]) -> np.ndarray:
        if self.settings.activation_type[0] == ActivationType.HSwish :
            result = HSwish().inference(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.Sigmoid :
            result = Sigmoid().inference(values[0].data.astype(np.float32))
        elif self.settings.activation_type[0] == ActivationType.Softmax :
            result = Softmax(axis=self.settings.activation_type[1]).inference(values[0].data.astype(np.float32))
        else :
            raise Error(f'Unsupported Act-shave sub-type: {self.settings.activation_type[0].name}')

        result = ma.getdata(result)

        return result

    def filter_issues(self, args) -> bool:
        if 'EISW-29771' in self.issues:
            # Filter not bit-exact comparision issues
            return False
        if 'EISW-29786' in self.issues:
            # Filter incorrect tensor serialization issues
            return False
        return True

class MemoryLocation(Enum):
    DDR = auto()
    CMX0 = auto()
    CMX1 = auto()

class DMA(Operation):

    PARAMS = ['mpe_op_class', 'input_ttype', 'output_ttype', 'input_shape', 'src_memloc', 'dst_memloc', 'dma_engine']

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, input, output):
        return {
            'case_type': 'DMA',
            'input': input[0].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        return values[0].data

    def filter_issues(self, args) -> bool:
        return True

class RaceConditionDMA(Operation):

    PARAMS = ['mpe_op_class', 'input_ttype', 'output_ttype', 'iteration_count']
    NAME = 'RaceConditionDMA'

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, input, output):
        return {
            'case_type': 'RaceConditionDMA',
            'input': input[0].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, _ = idu(values[0], values[0])
        return lhs

    def filter_issues(self, args) -> bool:
        return True

class RaceConditionDPU(Operation):

    PARAMS = [
        'mpe_op_class',
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

    def __init__(self, settings):
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, output):
        return {
            'case_type': 'RaceConditionDPU',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = MTLConv2D(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

class RaceConditionDPUDMA(Operation):

    PARAMS = [
        'mpe_op_class',
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

    def __init__(self, settings):
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, output):
        return {
            'case_type': 'RaceConditionDPUDMA',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = MTLConv2D(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

class RaceConditionDPUDMAACT(Operation):
    PARAMS = [
        'mpe_op_class',
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

    def __init__(self, settings):
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs, output):
        return {
            'case_type': 'RaceConditionDPUDMAACT',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
            'output': output.json_info,
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

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = iduConvCustom(values[0], values[1])
        c2d = MTLConv2D(kernel_shape=self.settings.kernel_shape,
                        pads = self.settings.kernel_pads,
                        strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def filter_issues(self, args) -> bool:
        return True

class RaceCondition:
    PARAMS = ['operation', 'iteration_count', 'requested_cluster', 'requested_unit']

    def __init__(self, operation, iter_count, requested_cluster, requested_unit):
        self.operation = operation
        self.mpe_op = operation.mpe_op
        self.iter_count = iter_count
        self.requested_cluster = requested_cluster
        self.requested_unit = requested_unit
        self.issues = set()
        if self.operation.settings.mpe_op_class == ActKernel and self.requested_cluster == 2 :
            self.issues.add('EISW-30347')

    def json_info(self):
        return {
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
        if 'EISW-30347' in self.issues:
            # Filter unsupported cluster param
            return False
        return self.mpe_op.filter_issues(args)

def ppe(values: List[Value], output_ttype: TType, data: Union[np.ndarray, NBQuantized], bitshift: int) -> Value:
    """Models the hardware PPE"""
    ndarray = data.value if isinstance(data, NBQuantized) else data

    rescale = not output_ttype.is_float
    if rescale:
        if np.issubdtype(ndarray.dtype, np.integer):
            ndarray = ndarray.astype(np.float64)
        ndarray /= (1. * (1 << bitshift))
        # if np.issubdtype(ndarray.dtype, np.integer):
        #     ndarray = np.right_shift(ndarray, bitshift)
        # else:
        #     ndarray /= (1. * (1 << bitshift))

    ndarray = output_ttype.clip(ndarray).astype(output_ttype.dtype)
    value = Value(output_ttype, 'output-0.bin', ndarray, output_ttype.bitwidth, output_ttype.bitsize, output_ttype.signed, None)

    if rescale:
        if isinstance(data, NBQuantized):
            value.zero = int(data.zero_point)
            value.scale = (1 << bitshift)
        else:
            value.scale = (1 << bitshift)

    return value


def odu(output: Value):
    """Models the hardware ODU"""
    return output


class Settings:
    def __str__(self):
        return '\n  '.join([f'{name}={getattr(self, name)}' for name in dir(self) if not name.startswith('_')])


class DPUPipeline:
    def __init__(self, option_names, option_values):
        settings = Settings()
        self.settings = settings
        self.issues = set()
        for name, value in zip(option_names, option_values):
            setattr(settings, name, value)

        self.mpe_op = settings.mpe_op_class(settings)

    def compute_values(self):
        try:
            self.inputs = self.mpe_op.generate_inputs(default_rng(1))
            mpe_data = self.mpe_op.apply(self.inputs)
            if isinstance(mpe_data, NBQuantized):
                self.mpe_data = mpe_data.value
            else:
                self.mpe_data = mpe_data
            result_bitwidth = math.ceil(np.log2(np.amax(abs(self.mpe_data))))
            bitshift = max(result_bitwidth - self.settings.output_ttype.bitwidth, 0)
            ppe_value = ppe(self.inputs, self.settings.output_ttype, mpe_data, bitshift)
            self.o = odu(ppe_value)
            self.o.check_entropy()
        except Exception as ex:
            raise ComputationError(f'computing {self.ident}') from ex

    def validate(self):
        try:
            self.mpe_op.validate()
        except Exception as ex:
            raise ValidationError(f'validating {self.ident}') from ex

    @property
    def name(self):
        return self.mpe_op.NAME

    @property
    def ident(self):
        return f'{self.mpe_op.ident}_{self.settings.output_ttype.stype}'

    @property
    def data(self):
        return {
            **self.mpe_op.data,
            'Type': self.mpe_op.NAME,
            'Content Creation': 0,
            'ActSpBitsIC': '',
            'WtSpBitsIC': '',
            'Issues': ', '.join(self.issues)
        }

    def write_data(self, dir: Path):
        orderer = self.mpe_op.orderer
        for input in self.inputs:
            input.write_data(dir, orderer)
        self.o.write_data(dir, self.mpe_op.output_orderer)
        orderer(self.mpe_data).tofile(dir / 'mpe_raw.bin')

    def value(self):
        return {
            **self.mpe_op.json_info(self.inputs, self.o),
        }

    def filter_issues(self, args):
        return self.mpe_op.filter_issues(args)


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


def genZMConvs(input_types=[UInt8(2)],
               input_shapes=[[1, 32, 16, 16]],
               weight_types=None,
               kernel_channels=[64],
               kernel_shapes=[[1, 1]],
               output_types=None,
               output_orders=[Order.NHWC],
               strides=[[1, 1]],
               pads=Pad.none,
               compress=False,
               mpe_cubs=[MPE_CUBES.CUBOID_16x16]):

    for (input_type, input_shape, kernel_channel, kernel_shape, output_order, stride, pad, mpe_cub) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, output_orders, strides, pads, mpe_cubs):

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
                yield DPUPipeline(ZMajorConvolution.PARAMS, (ZMajorConvolution,
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
                                                             mpe_cub
                                                             ))


def genEltwiseAdds(input_types=[Int8(6)],
                   input_shapes=[[1, 256, 16, 16]],
                   output_types=None):
    for (input_type, input_shape) in itertools.product(input_types, input_shapes):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(EltwiseAdd.PARAMS, (EltwiseAdd,
                                                  input_type,
                                                  input_shape,
                                                  output_type
                                                  ))


def genEltwiseMults(input_types=[Int8(6)],
                    input_shapes=[[1, 256, 16, 16]],
                    output_types=None):
    for (input_type, input_shape) in itertools.product(input_types, input_shapes):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(EltwiseMult.PARAMS, (EltwiseMult,
                                                   input_type,
                                                   input_shape,
                                                   output_type
                                                   ))


def genMaxPools(input_types=[FP16(6)],
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
            yield DPUPipeline(Maxpool.PARAMS, (Maxpool,
                                               input_type,
                                               input_shape,
                                               kernel_shape,
                                               output_type,
                                               stride,
                                               pad
                                               ))


def genAvgPools(input_types=[FP16(6)],
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
            yield DPUPipeline(AvgPool.PARAMS, (AvgPool,
                                               input_type,
                                               input_shape,
                                               kernel_shape,
                                               output_type,
                                               stride
                                               ))

def getValidOutputTypes(input_type, kernel_channels) :
    aviable_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
    output_types = []
    for output_type in aviable_output_types:
        if(CheckHWAlignment(output_type, kernel_channels)):
            output_types.append(output_type)
    return output_types

def genDepthWiseConvs(input_types=[FP16(2)],
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
            yield DPUPipeline(DepthWiseConv.PARAMS, (DepthWiseConv,
                                                     input_type,
                                                     input_shape,
                                                     kernel_channel,
                                                     kernel_shape,
                                                     output_type,
                                                     stride,
                                                     pad
                                                     ))

def genActShave(input_types,
               input_shapes,
               output_types,
               act_shave_subtypes):
    for (input_type, input_shape, output_type, act_shave_subtype) in itertools.product(input_types, input_shapes, output_types, act_shave_subtypes):
        yield DPUPipeline(ActKernel.PARAMS, (ActKernel, input_type, input_shape, output_type, act_shave_subtype))

AVAILABLE_CMX_SIZE = 1024*1942 # size in Bytes
HALF_OF_CMX_SIZE = int(AVAILABLE_CMX_SIZE/2)

def genDMA(tensor_types=[FP16(2)], input_shapes=[[1, 16, 32, 32]], src_locations=[MemoryLocation.CMX0], dst_locations=[MemoryLocation.CMX0], dma_engines=[0]):
    for (tensor_type, input_shape, src_location, dst_location, dma_engine) in itertools.product(tensor_types, input_shapes, src_locations, dst_locations, dma_engines):
        yield DPUPipeline(DMA.PARAMS, (DMA,
                                       tensor_type,
                                       tensor_type,
                                       input_shape,
                                       src_location,
                                       dst_location,
                                       dma_engine
                                       ))

def genRaceConditionDMA(input_types=[FP16(2)],
                        output_types=[FP16(2)],
                        iteration_count=64):
    for (input_type, output_type) in itertools.product(input_types, output_types):
        yield DPUPipeline(RaceConditionDMA.PARAMS, (RaceConditionDMA,
                                                    input_type,
                                                    output_type,
                                                    iteration_count
                                                    ))
def genRaceCondition(ops,
                    iteration_counts=[64],
                    requested_clusters=[1],
                    requested_units=[1]):
    for (op, iteration_count, requested_cluster, requested_unit) in itertools.product(ops, iteration_counts, requested_clusters, requested_units):
        yield RaceCondition(op, iteration_count, requested_cluster, requested_unit)

def genRaceConditionDPU(input_types=[FP16(4)],
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
                yield DPUPipeline(RaceConditionDPU.PARAMS, (RaceConditionDPU,
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
                                                            iteration_count
                                                            ))

def genRaceConditionDPUDMA(input_types=[FP16(4)],
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
                yield DPUPipeline(RaceConditionDPUDMA.PARAMS, (RaceConditionDPUDMA,
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
                                                               iteration_count
                                                               ))

def genRaceConditionDPUDMAACT(input_types=[FP16(0)],
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
                yield DPUPipeline(RaceConditionDPUDMAACT.PARAMS, (RaceConditionDPUDMAACT,
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
                                                               iteration_count
                                                               ))

def generate_options(args):
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
                   output_types=[FP16()]),

        # Z-Major Convolution with strides
        genZMConvs(input_types=[FP16(3)],
                   weight_types=[Int8(3), FP16(3)],
                   output_types=[FP16()],
                   kernel_shapes=[[2, 2]],
                   strides=[[r, c] for r in range(1, 8) for c in range(1, 8)]),

        # Z-Major Convolution, padding, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10], [11, 11]],
                   output_types=[UInt8()],
                   pads=Pad.none + Pad.all(5) + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10], [11, 11]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.all(5) + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[Int8(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int8()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 32, 32, 32]],
                   weight_types=[Int4(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int8()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[Int8(2)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 32, 32, 32]],
                   weight_types=[Int4(2)],
                   kernel_channels=[32],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int4(), UInt4()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, fp16
        genZMConvs(input_types=[FP16(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[FP16(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[FP16()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, bf16
        genZMConvs(input_types=[BF16(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[BF16(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[BF16()],
                   pads=Pad.none + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, 4x6 kernel, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 8, 8]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[4, 6]],
                   output_types=[UInt8()],
                   pads=[[2,0,0,0],[0,3,0,0]]),

        # Z-Major Convolution, padding, 5x5 kernel, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[8, 10]],
                   output_types=[UInt8()],
                   pads=[[4,0,0,0],[0,5,0,0]]),

        # Z-Major Convolution, output order
        genZMConvs(input_types=[Int8(3), FP16(4)],
                   output_orders=[Order.NWHC, Order.NWCH, Order.NCWH, Order.NHCW, Order.NCHW]),

        # Z-Major Convolution, integer cuboid combinations
        genZMConvs(input_types=[Int8(3)],
                   input_shapes=[[1, 16, 32, 64]],
                   weight_types=[Int8(2)],
                   output_types=[Int8()],
                   mpe_cubs=[MPE_CUBES.CUBOID_16x16, MPE_CUBES.CUBOID_8x16, MPE_CUBES.CUBOID_4x16]),

        # Z-Major Convolution, fp cuboid combinations
        genZMConvs(input_types=[FP16(4)],
                   input_shapes=[[1, 16, 32, 64]],
                   weight_types=[FP16(2)],
                   output_types=[FP16()],
                   mpe_cubs=[MPE_CUBES.CUBOID_16x16, MPE_CUBES.CUBOID_8x16, MPE_CUBES.CUBOID_4x16]),

        # Eltwise Add
        genEltwiseAdds(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                       input_shapes=[[1, 256, 16, 16]]),

        # Eltwise Mult
        genEltwiseMults(input_types=[Int8(3), UInt8(4), FP16(6), BF16(6)],
                        input_shapes=[[1, 1, 1, 64]]),

        # MaxPool
        genMaxPools(input_types=[UInt8(6), Int8(6), FP16(6), BF16(6)],
                    input_shapes=[[1, 64, 16, 16]],
                    pads=Pad.none + Pad.all(1) + Pad.top_bottom(1) + Pad.left_right(1)),

        genMaxPools(input_types=[UInt8(6)],
                    output_types=[UInt8()],
                    strides=[[r, c] for r in range(2, 8) for c in range(2, 8) if (r, c) != (2, 2)]),

        genMaxPools(input_types=[UInt8(6)],
                    output_types=[UInt8()],
                    kernel_shapes=[[r, c] for r in range(2, 12) for c in range(2, 12) if (r, c) != (2, 2)]),

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
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 112, 112]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[96],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 96, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 32, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[144],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 144, 56, 56]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 144, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 32, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[192],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 192, 28, 28]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[32],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 192, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[64],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 64, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[384],
                   output_types=[UInt8()]),

        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 384, 14, 14]],
                   weight_types=[UInt8(2)],
                   kernel_channels=[64],
                   output_types=[UInt8()]),

        # Z-Major Convolution, weights swizzling_key = 1-to-5
        genZMConvs(input_types=[FP16(3)],
                   input_shapes=[[1, 16, 1, 1]],
                   weight_types=[FP16(-3)],
                   kernel_channels=[64,128,256,512,1024],
                   kernel_shapes=[[1, 1]],
                   output_types=[FP16()],
                   pads=Pad.none),

        # Z-Major Continued Convolution, fp16
        genZMConvs(input_types=[FP16(3)],
                   input_shapes=[[1, 16*1024, 1, 1]],
                   weight_types=[FP16(-3)],
                   kernel_channels=[16],
                   kernel_shapes=[[1, 1]],
                   output_types=[FP16()],
                   pads=Pad.none),

        # Z-major compressed Convolution, int8
        genZMConvs(input_types=[Int8(2)],
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
            input_types=[UInt8(2)],
            input_shapes=[[1, 1, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 2, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 3, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 4, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 5, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 6, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 7, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 8, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 9, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 10, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 11, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 12, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[4, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 13, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[UInt8(2)],
            input_shapes=[[1, 14, 16, 16]],
            weight_types=[UInt8(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[UInt8()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
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
            input_types=[FP16(2)],
            input_shapes=[[1, 1, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 2, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 3, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 4, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),


        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 5, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 6, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 2]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 7, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 8, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 9, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 10, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 11, 16, 16]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[2, 4]],
            output_types=[FP16()],
            pads=[[0, 0, 0, 0]]
        ),

        genZMConvs(
            input_types=[FP16(2)],
            input_shapes=[[1, 32, 2, 15]],
            weight_types=[FP16(2)],
            kernel_channels=[64],
            kernel_shapes=[[1, 1]],
            output_types=[FP16()],
            output_orders=[Order.NCHW],
            pads=[[0, 0, 0, 0]]
        ),
        # check all datatypes
        genDMA(
            tensor_types=[Int4(3), UInt4(3), Int8(3), UInt8(3), FP16(4), BF16(4)],
            input_shapes=[[1, 16, 32, 32]]
        ),

        # check all memory locations
        genDMA(
            tensor_types=[Int8(3)],
            input_shapes=[[1, 32, 16, 16]],
            src_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1, MemoryLocation.DDR],
            dst_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1, MemoryLocation.DDR],
            dma_engines=[0, 1]
        ),
        # check max aviable CMX
        genDMA(
            tensor_types=[UInt8(3)],
            input_shapes=[[1, 1, 1, HALF_OF_CMX_SIZE]],
            src_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1],
            dst_locations=[MemoryLocation.CMX0, MemoryLocation.CMX1],
            dma_engines=[0, 1]
        ),
        genRaceConditionDMA(
            input_types=[FP16(2)],
            output_types=[FP16(2)],
            iteration_count=64
        ),

        genRaceConditionDPU(
            iteration_count=48
        ),

        genRaceConditionDPUDMA(
            iteration_count=48
        ),

        genRaceConditionDPUDMAACT(
            iteration_count=24
        ),

        genRaceCondition(
            ops = genActShave(
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
        )
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
