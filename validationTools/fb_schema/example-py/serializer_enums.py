from enum import Enum

"""
    To be expanded
"""

class MemoryLocation(Enum):
    NULL = 0
    ProgrammableInput = 1
    ProgrammableOutput = 2
    VPU_DDR_Heap = 3
    GraphFile = 4
    VPU_CMX_NN  = 5
    VPU_CMX_UPA = 6

class PPELayerType(Enum):
    NONE = 0

    LEAKY_RELU = 1
    LEAKY_PRELU = 2
    LEAKY_RELU_X = 3

    ADD = 4
    SUB = 5
    MULTIPLY = 6

    MAXIMUM = 7
    MINIMUM = 8
    CEILING = 9
    FLOOR = 10

    AND = 11
    OR = 12
    XOR = 13
    NOT = 14
    ABS = 15
    NEG = 16

    POW = 17
    EXP = 18
    SIGMOID = 19
    TANH = 20
    SQRT = 21
    RSQRT = 22
    FLEXARB = 23

    _STORE = 24
    _LOAD = 25
    _CLEAR = 26
    _NOOP = 27
    _HALT = 28


class DPULayerType(Enum):
    CONV = 0
    DWCONV = 1
    MAXPOOL = 2
    AVEPOOL = 3
    FCL = 4
    ELTWISE = 5


class NCE1LayerType(Enum):
    CONV = 0
    MAXPOOL = 1
    AVEPOOL = 2
    FCL = 3


class SoftwareLayerType(Enum):
    CONV = 0
    MAXPOOL = 1
    AVEPOOL = 2
    FCL = 3
    LRN = 4
    SOFTMAX = 5


class DType(Enum):
    NOT_SET = 0
    FP32 = 1
    FP16 = 2
    FP8 = 3
    U32 = 4
    U16 = 5
    U8 = 6
    I32 = 7
    I16 = 8
    I8 = 9
    I4 = 10
    I2 = 11
    I4X = 12
    BIN = 13
    LOG = 14