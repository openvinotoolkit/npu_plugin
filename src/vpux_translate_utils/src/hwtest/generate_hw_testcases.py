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
from enum import Enum
import json
import itertools
import pandas as pd
from pathlib import Path
import re
import sys
from typing import Callable, List, Optional, Sequence, Union

# TODO: Fix this awful hack, whose purpose in life is to point to where you've
# checked out ssh://git@gitlab.devtools.intel.com:29418/iotgai/NumericsBench.git.
import os
numericBenchPath = os.getenv('NUMERICSBENCH_PATH')
if numericBenchPath == None:
    print("env variable NUMERICSBENCH_PATH not set, Using default path for NumericsBench")
    sys.path.append(str(Path(__file__).parents[5] / 'NumericsBench'))
else:
    sys.path.append(numericBenchPath)

from operators.op_utils.bfloat16 import bfloat16
from operators.vpu26 import Add, Mult, Conv2d, MaxPool, AveragePool
from operators.platform.quantize_info import QuantizationInfo
from operators.platform.quantized_tensor import NBQuantized
from operators.platform.vpu26 import PlatformVPU26

import numpy as np
from numpy.random import default_rng


Orderer = Callable[[np.ndarray], np.ndarray]


def OrderNCHW(data: np.ndarray) -> np.ndarray:
    return data


def OrderNHWC(data: np.ndarray) -> np.ndarray:
    return np.concatenate([a.transpose(1, 2, 0).flatten() for a in data])


class ComputationError(Exception):
    pass


class ValidationError(Exception):
    pass


class PaddingError(Exception):
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

    kernel_x = kernel[0]
    kernel_y = kernel[1]

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


@dataclass
class Value:
    ttype: 'TType'
    filename: str
    data: np.ndarray
    bitwidth: int
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

    @property
    def json_info(self):
        info = {
            'shape': self.data.shape,
            'dtype': self.ttype.stype,
            'quantization': {
                'scale': self.scale,
                'zeropoint': self.zero,
                'low_range': self.low,
                'high_range': self.high
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


def pack_int4(data: np.ndarray) -> np.ndarray:
    flat = data.flatten()
    result = []
    for idx in range(0, flat.size, 2):
        lsn = flat[idx] & 0x0f
        msn = flat[idx + 1] & 0x0f
        datum = np.uint8(msn << 4 | lsn)
        result.append(datum)
    return np.array(result).astype(np.uint8)


class UInt4(TType):
    def __init__(self, bitwidth=4):
        super().__init__(np.uint8, 'uint4', 'int8', bitwidth, False)
        self.low = np.uint8(0)
        self.high = np.uint8((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> Value:
        return Value(self,
                      filename,
                      rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.uint8),
                      self.bitwidth,
                      False,
                      orderer)

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
        self.low = np.int8(-(2 ** bitwidth))
        self.high = np.int8((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> Value:
        return Value(self,
                      filename,
                      rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.int8),
                      self.bitwidth,
                      True,
                      orderer)

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
        self.low = np.uint8(0)
        self.high = np.uint8((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> Value:
        return Value(self,
                      filename,
                      rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.uint8),
                      self.bitwidth,
                      False,
                      orderer)

    @property
    def is_float(self) -> bool:
        return False

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(0, 255)

class Int8(TType):
    def __init__(self, bitwidth=7):
        super().__init__(np.int8, 'int8', 'int8', bitwidth, True)
        self.low = np.int8(-(2 ** bitwidth))
        self.high = np.int8((2 ** bitwidth) - 1)

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        return Value(self,
                      filename,
                      rng.integers(self.low, self.high, endpoint=True, size=shape, dtype=np.int8),
                      self.bitwidth,
                      True,
                      orderer)

    @property
    def is_float(self) -> bool:
        return False

    def clip(self, data: np.ndarray) -> np.ndarray:
        return data.round().clip(-128, 127)

class FP16(TType):
    def __init__(self, bitwidth=16):
        super().__init__(np.float16, 'fp16', None, bitwidth, True)

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        data = np.around(rng.random(size=shape, dtype=np.float32) * 8.) / 8.

        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(np.float16),
                      self.bitwidth,
                      True,
                      orderer)

    @property
    def is_float(self) -> bool:
        return True

    def clip(self, data: np.ndarray) -> np.ndarray:
        # Clip in this case is interesting: we want to take the data as fp32,
        # and round it after the first ten bits of the fractional part (since
        # that's what can be expressed in fp16).  The fun part is that if the
        # eleventh bit is 1, we always round up, because we're using the
        # ties-to-even rounding mode; if we naively cast to fp16, we would
        # simply truncate the fractional component past ten bits, which is not
        # how the DPU implements rounding.  So we have to do some tricky things.
        (frac, exp) = np.frexp(data.astype(np.float32))
        shift_factor = np.power(2, 11)
        (rounded, preserved) = np.modf(frac * shift_factor)
        carry = (rounded >= .5).astype(np.float32)
        result = np.ldexp((carry + preserved) / shift_factor, exp)
        return result


class FP32(TType):
    def __init__(self, bitwidth=127):
        super().__init__(np.float32, 'fp32', None, bitwidth, True)

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        data = np.around(rng.random(size=shape, dtype=np.float32) * 8.) / 8.
        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(np.float32),
                      self.bitwidth,
                      True,
                      orderer)

    @property
    def is_float(self) -> bool:
        return True


class BF16(TType):
    def __init__(self, bitwidth=127):
        super().__init__(bfloat16, 'bfloat16', None, bitwidth, True)

    def generate(self, filename: str, shape, rng, orderer=None) -> np.ndarray:
        # NB For now, we restrict the number of bits in our floats in order
        #    to ensure we're not running into rounding issues.
        data = np.around(rng.random(size=shape, dtype=np.float32) * 8.) / 8.
        return Value(self,
                      filename,
                      (data * (2. ** self.bitwidth)).astype(bfloat16),
                      self.bitwidth,
                      True,
                      orderer)

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


class MPE(ABC):
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
    def result_bitwidth(self, lhs: Value, rhs: Value) -> int:
        pass


def shape_to_str(shape: Sequence[int]) -> str:
    return 'x'.join([str(d) for d in shape])


class ZMajorConvolution(MPE):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'weight_ttype', 'kernel_channels', 'kernel_shape', 'output_ttype', 'kernel_strides', 'kernel_pads']

    def __init__(self, settings):
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, settings.input_shape[1]] + settings.kernel_shape

    def json_info(self, inputs):
        return {
            'case_type': 'ZMajorConvolution',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
            'conv_op': {
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads,
                'group': 1,
                'dilation': 1
            }
        }

    def validate(self):
        ValidatePaddings(self.settings.kernel_shape, self.settings.kernel_pads)

    @property
    def ident(self) -> str:
        return f'zm_conv_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_{shape_to_str(self.settings.weight_shape)}x{self.settings.weight_ttype.stype}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}_kern_chan_{self.settings.kernel_channels}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

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
            self.settings.weight_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = idu(values[0], values[1])
        c2d = Conv2d(kernel_shape=self.settings.kernel_shape,
                     pads = self.settings.kernel_pads,
                     strides = self.settings.kernel_strides)
        result = c2d.inference(lhs, rhs)
        return result

    def result_bitwidth(self, values: List[Value]) -> int:
        # NB zm_conv_int8_fp16_16_uint8 seems to have a cliff behavior at 14
        #    extra bits: at 13, we see 0 and FF as outputs; at 14, we see just 0.
        return values[0].bitwidth + values[1].bitwidth + self.settings.input_shape[2].bit_length()  + self.settings.input_shape[3].bit_length() - 1


class DepthWiseConv(MPE):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'kernel_channels', 'kernel_shape', 'output_ttype', 'kernel_strides', 'kernel_pads']

    def __init__(self, settings):
        self.settings = settings
        settings.weight_shape = [settings.kernel_channels, 1] + settings.kernel_shape

    def json_info(self, inputs):
        return {
            'case_type': 'DepthWiseConv',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info,
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
    def data(self) -> dict:
        return {
            'MPE Mode': 'DepthWiseConv',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Weights Type': self.settings.input_ttype.stype,
            'Kernel Channels': str(self.settings.kernel_channels),
            'Kernel Shape': ', '.join([str(s) for s in self.settings.kernel_shape]),
            'Output Type': self.settings.output_ttype.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng),
            self.settings.input_ttype.generate('weights.dat', self.settings.weight_shape, rng, orderer=OrderNCHW)
        ]

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = idu(values[0], values[1])
        c2d = Conv2d(kernel_shape=self.settings.kernel_shape,
                     pads = self.settings.kernel_pads,
                     strides = self.settings.kernel_strides,
                     group = self.settings.weight_shape[0])
        return c2d.inference(lhs, rhs)

    def result_bitwidth(self, values: List[Value]) -> int:
        return values[0].bitwidth + values[1].bitwidth + self.settings.input_shape[2].bit_length()  + self.settings.input_shape[3].bit_length() - 1



class EltwiseAdd(MPE):

    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'output_ttype']

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, inputs):
        return {
            'case_type': 'EltwiseAdd',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info
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
    def data(self) -> dict:
        return {
            'MPE Mode': 'EltwiseAdd',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Output Type': self.settings.output_ttype.stype
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

    def result_bitwidth(self, values: List[Value]) -> int:
        return values[0].bitwidth + 1


class EltwiseMult(MPE):

    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'output_ttype']

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, inputs):
        return {
            'case_type': 'EltwiseMult',
            'input': inputs[0].json_info,
            'weight': inputs[1].json_info
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
    def data(self) -> dict:
        return {
            'MPE Mode': 'EltwiseMult',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Output Type': self.settings.output_ttype.stype
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

    def result_bitwidth(self, values: List[Value]) -> int:
        return values[0].bitwidth + 1


class Maxpool(MPE):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'output_ttype', 'kernel_strides', 'kernel_pads']
    kernel_shape = [2, 2]

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, inputs):
        return {
            'case_type': 'MaxPool',
            'input': inputs[0].json_info,
            'pool_op': {
                'sub_type': 'max',
                'kernel_shape': self.kernel_shape,
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads
            }
        }

    def validate(self):
        ValidatePaddings(self.kernel_shape, self.settings.kernel_pads)

    @property
    def ident(self) -> str:
        return f'max_pool_{shape_to_str(self.settings.input_shape)}x{self.settings.input_ttype.stype}_pads_{shape_to_str(self.settings.kernel_pads)}_strides_{shape_to_str(self.settings.kernel_strides)}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        return {
            'MPE Mode': 'MaxPool',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Output Type': self.settings.output_ttype.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = idu(values[0], values[0])
        maxpool = MaxPool(kernel_shape=self.kernel_shape, strides=self.settings.kernel_strides, pads=self.settings.kernel_pads)
        return maxpool.inference(lhs)


    def result_bitwidth(self, values: List[Value]) -> int:
        return values[0].bitwidth


class AvgPool(MPE):

    # kernel_strides are x|y directions
    # The padding order is top|left|bottom|right
    PARAMS = ['mpe_op_class', 'input_ttype', 'input_shape', 'output_ttype', 'kernel_strides', 'kernel_pads']
    kernel_shape = [2, 2]

    def __init__(self, settings):
        self.settings = settings

    def json_info(self, inputs):
        return {
            'case_type': 'AvgPool',
            'input': inputs[0].json_info,
            'pool_op': {
                'sub_type': 'avg',
                'kernel_shape': self.kernel_shape,
                'stride': self.settings.kernel_strides,
                'pad': self.settings.kernel_pads
            }
        }

    def validate(self):
        ValidatePaddings(self.kernel_shape, self.settings.kernel_pads)

    @property
    def ident(self) -> str:
        return f'avg_pool_{self.settings.input_ttype.stype}'

    @property
    def orderer(self) -> Orderer:
        return OrderNHWC

    @property
    def data(self) -> dict:
        return {
            'MPE Mode': 'AvgPool',
            'Input Type': self.settings.input_ttype.stype,
            'Input Shape': ', '.join([str(s) for s in self.settings.input_shape]),
            'Output Type': self.settings.output_ttype.stype
        }

    def generate_inputs(self, rng) -> List[Value]:
        return [
            self.settings.input_ttype.generate('input-0.bin', self.settings.input_shape, rng)
        ]

    def apply(self, values: List[Value]) -> np.ndarray:
        lhs, rhs = idu(values[0], values[0])
        avgpool = AveragePool(self.kernel_shape, strides = self.settings.kernel_strides, pads = self.settings.kernel_pads)
        return avgpool.inference(lhs)

    def result_bitwidth(self, values: List[Value]) -> int:
        return values[0].bitwidth


def ppe(values: List[Value], output_ttype: TType, data: Union[np.ndarray, NBQuantized], bitshift: int) -> Value:
    """Models the hardware PPE"""
    if isinstance(data, NBQuantized):
        ndarray = data.value / (2 << bitshift)
    else:
        ndarray = data
    ndarray = output_ttype.clip(ndarray).astype(output_ttype.dtype)
    value = Value(output_ttype, 'output-0.bin', ndarray, output_ttype.bitwidth, output_ttype.signed, None)

    if isinstance(data, NBQuantized):
        value.scale = float(data.scale) / (2 << bitshift)
        value.zero = int(data.zero_point)

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
            if value.__class__ in [Int4, UInt4]:
                self.issues.add('EISW-13321')  # Int4 / UInt4 not supported

        self.mpe_op = settings.mpe_op_class(settings)

        if settings.mpe_op_class is EltwiseAdd and settings.input_ttype.__class__ in [FP16, BF16]:
            self.issues.add('EISW-6666')  # Double expected outputs for eltwise add with fp16 or bfloat16 inputs

        if settings.mpe_op_class is Maxpool and settings.input_ttype.__class__ in [FP16, BF16]:
            self.issues.add('EISW-15074')  # MaxPool produces zeros with fp16 and bf16 inputs

    def compute_values(self):
        try:
            self.inputs = self.mpe_op.generate_inputs(default_rng(1))
            mpe_data = self.mpe_op.apply(self.inputs)
            if isinstance(mpe_data, NBQuantized):
                self.mpe_data = mpe_data.value
            else:
                self.mpe_data = mpe_data
            result_bitwidth = self.mpe_op.result_bitwidth(self.inputs)
            bitshift = max(result_bitwidth - self.settings.output_ttype.bitwidth, 0)
            ppe_value = ppe(self.inputs, self.settings.output_ttype, mpe_data, bitshift)
            self.o = odu(ppe_value)
        except Exception as ex:
            raise ComputationError(f'computing {self.ident}') from ex

    def validate(self):
        try:
            self.mpe_op.validate()
        except Exception as ex:
            raise ValidationError(f'validating {self.ident}') from ex

    @property
    def ident(self):
        return f'{self.mpe_op.ident}_{self.settings.output_ttype.stype}'

    @property
    def data(self):
        return {
            'Issues': ', '.join(self.issues),
            **self.mpe_op.data
        }

    def write_data(self, dir: Path):
        orderer = self.mpe_op.orderer
        for input in self.inputs:
            input.write_data(dir, orderer)
        self.o.write_data(dir, orderer)
        orderer(self.mpe_data).tofile(dir / 'mpe_raw.bin')

    def value(self):
        return {
            **self.mpe_op.json_info(self.inputs),
            'output': self.o.json_info,
            'activation': {
                'name': None
            }
        }


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
    return 'EISW-13321' not in p.issues  # Filter int4


_ZMCONV_VALID_WEIGHT_TYPES = {
    Int8: [Int8(3), Int4(3)],
    Int4: [Int8(3), UInt4(3), Int4(3)],
    UInt8: [UInt8(3)],
    UInt4: [UInt4(3), Int4(3)],
    FP16: [FP16(4), Int8(3), Int4(3)],
    BF16: [BF16(4)]
}


_PPE_VALID_OUTPUT_TYPES = {
    False: [FP16(), UInt8(), UInt4(), Int8(), Int4()],  # Integer MACs
    True: [FP32(), FP16(), BF16(), UInt8(), Int8(), Int4()],    # FP MACs
}


def genZMConvs(input_types=[UInt8(2)],
               input_shapes=[[1, 32, 16, 16]],
               weight_types=None,
               kernel_channels=[64],
               kernel_shapes=[[1, 1]],
               output_types=None,
               strides=[[1, 1]],
               pads=Pad.none):

    for (input_type, input_shape, kernel_channel, kernel_shape, stride, pad) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, strides, pads):

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
                yield DPUPipeline(ZMajorConvolution.PARAMS, (ZMajorConvolution,
                                                             input_type,
                                                             input_shape,
                                                             weight_type,
                                                             kernel_channel,
                                                             kernel_shape,
                                                             output_type,
                                                             stride,
                                                             pad
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


_MAXPOOL_PPE_VALID_OUTPUT_TYPES = {
    False: [FP16(), UInt8(), UInt4(), Int8(), Int4()],  # Integer
    True: [FP16(), BF16()],  # FP
}


def genMaxPools(input_types=[FP16(6)],
                input_shapes=[[1, 64, 16, 16]],
                output_types=None,
                strides=[[2, 2]],
                pads=Pad.none):
    for (input_type, input_shape, stride, pad) in itertools.product(input_types, input_shapes, strides, pads):
        if output_types is None:
            current_output_types = _MAXPOOL_PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(Maxpool.PARAMS, (Maxpool,
                                               input_type,
                                               input_shape,
                                               output_type,
                                               stride,
                                               pad
                                               ))


def genAvgPools(input_types=[FP16(6)],
                input_shapes=[[1, 64, 32, 32]],
                output_types=None,
                strides=[[2, 2]],
                pads=Pad.none):
    for (input_type, input_shape, stride, pad) in itertools.product(input_types, input_shapes, strides, pads):
        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
        else:
            current_output_types = output_types

        for output_type in current_output_types:
            yield DPUPipeline(AvgPool.PARAMS, (AvgPool,
                                               input_type,
                                               input_shape,
                                               output_type,
                                               stride,
                                               pad
                                               ))


def genDepthWiseConvs(input_types=[FP16(2)],
                      input_shapes=[[1, 16, 32, 32]],
                      kernel_channels=[16],
                      kernel_shapes=[[4, 4]],
                      output_types=None,
                      strides=[[1, 1]],
                      pads=Pad.none):
    for (input_type, input_shape, kernel_channel, kernel_shape, stride, pad) in itertools.product(input_types, input_shapes, kernel_channels, kernel_shapes, strides, pads):

        if output_types is None:
            current_output_types = _PPE_VALID_OUTPUT_TYPES[input_type.is_float]
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


def generate_options(args):
    return itertools.chain(
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

        # Z-Major Convolution, padding, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10], [11, 11]],
                   output_types=[Int4(), UInt4(), UInt8()],
                   pads=Pad.none + Pad.all(5) + Pad.top(5) + Pad.left(5) + Pad.bottom(5) + Pad.right(5)),

        # Z-Major Convolution, padding, int8
        genZMConvs(input_types=[Int8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[Int4(2), Int8(2)],
                   kernel_channels=[16],
                   kernel_shapes=[[10, 10]],
                   output_types=[Int4(), UInt4(), Int8()],
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
                   pads=[[2,0,0,0],[3,0,0,0]]),

        # Z-Major Convolution, padding, 5x5 kernel, uint8
        genZMConvs(input_types=[UInt8(2)],
                   input_shapes=[[1, 16, 32, 32]],
                   weight_types=[UInt8(1)],
                   kernel_channels=[16],
                   kernel_shapes=[[8, 10]],
                   output_types=[UInt8()],
                   pads=[[4,0,0,0],[5,0,0,0]]),

        # Eltwise Add
        genEltwiseAdds(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                       input_shapes=[[1, 256, 16, 16]]),

        # Eltwise Mult
        #
        # NB: When changing the input shape here, make sure to verify that
        #     the corresponding C++ test builder code has been updated to
        #     work with the new shape.  (I.e. [1, 16, 16, 32] failed to
        #     produce a correct output.)
        genEltwiseMults(input_types=[Int8(3), UInt8(4), FP16(6), BF16(6)],
                        input_shapes=[[1, 1, 1, 8192]]),

        # MaxPool
        genMaxPools(input_types=[UInt8(6), Int8(6), FP16(6), BF16(6)],
                    input_shapes=[[1, 64, 16, 16]],
                    pads=Pad.none + Pad.all(1) + Pad.top_bottom(1) + Pad.left_right(1)),

        # AvgPool
        genAvgPools(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)],
                    input_shapes=[[1, 64, 32, 32]]),

        # DepthWiseConv
        genDepthWiseConvs(input_types=[Int8(6), UInt8(6), FP16(6), BF16(6)]),

        # MobileNet DepthWiseConv, uint8
        genDepthWiseConvs(input_types=[UInt8(2)],
                          input_shapes=[[1, 32, 112, 112]],
                          kernel_channels=[32],
                          kernel_shapes=[[3, 3]],
                          output_types=[UInt8()],
                          pads=[[1, 1, 1, 1]]),

        # MobileNet DepthWiseConv, uint8
        genDepthWiseConvs(input_types=[UInt8(2)],
                          input_shapes=[[1, 96, 112, 112]],
                          kernel_channels=[96],
                          kernel_shapes=[[3, 3]],
                          output_types=[UInt8()],
                          pads=[[1, 1, 1, 1]]),

        # MobileNet DepthWiseConv, uint8
        genDepthWiseConvs(input_types=[UInt8(2)],
                          input_shapes=[[1, 144, 56, 56]],
                          kernel_channels=[144],
                          kernel_shapes=[[3, 3]],
                          output_types=[UInt8()],
                          pads=[[1, 1, 1, 1]]),

        # MobileNet DepthWiseConv, uint8
        genDepthWiseConvs(input_types=[UInt8(2)],
                          input_shapes=[[1, 192, 28, 28]],
                          kernel_channels=[192],
                          kernel_shapes=[[3, 3]],
                          output_types=[UInt8()],
                          pads=[[1, 1, 1, 1]]),

        # MobileNet DepthWiseConv, uint8
        genDepthWiseConvs(input_types=[UInt8(2)],
                          input_shapes=[[1, 192, 28, 28]],
                          kernel_channels=[192],
                          kernel_shapes=[[3, 3]],
                          output_types=[UInt8()],
                          strides=[[2, 2]],
                          pads=[[1, 0, 1, 0]]),

        # MobileNet DepthWiseConv, uint8
        genDepthWiseConvs(input_types=[UInt8(2)],
                          input_shapes=[[1, 384, 14, 14]],
                          kernel_channels=[384],
                          kernel_shapes=[[3, 3]],
                          output_types=[UInt8()],
                          pads=[[1, 1, 1, 1]]),

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
        option.compute_values()
        path = args.root / option.ident
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'config.json').open('w') as outfile:
            option.write_data(path)
            descriptor = option.value()
            json.dump(descriptor, outfile, indent=4)


def export_excel(args):
    df = pd.DataFrame((opt.data for opt in generate_options(args)))
    df.to_excel(args.filename, sheet_name='DPU Test Cases', index=False)


def export_csv(args):
    df = pd.DataFrame((opt.data for opt in generate_options(args)))
    df.to_csv(args.filename)


def main():
    parser = argparse.ArgumentParser(description='Create hardware test cases', prog='generate_hw_testcases')
    subparsers = parser.add_subparsers()

    parser_write_configs = subparsers.add_parser('write-configs', help='Write test case configurations and sample data')
    parser_write_configs.add_argument('root', type=Path, help='The directory where the test cases should be written')
    parser_write_configs.add_argument('--exist-ok', help='Reuse the contents of the root', action='store_true')
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
