import unittest
import os
import sys
import numpy as np
from serializer_enums import DPULayerType, DType, PPELayerType

class HostRepr():
    def __init__(self, which=0):
        self.versionMaj = 3
        self.versionMin = 0
        self.versionPat = 1
        self.githash = "1b30ae2e04abf47f98a2ebc14d125f5cab9e7288"

        self.shaveMask = 1
        self.nce1Mask = 2
        self.dpuMask = 3
        self.leonCMX = 1000
        self.nnCMX = 2000
        self.ddrScratch = 1234

        self.dims = [1, 3, 256, 256]
        self.strides = [2, 6, 512, 1536, 393216]

        self.taskAmount = 321
        self.layerAmount = 123

        self.structure = [
            ("input", "conv1"),
            ("conv1", "output")
        ]

        if which == 0:
            # print("fakeDPULayer")
            self.network = [
                [
                    fakeDPULayer()
                ]
            ]
        elif which == 1:
            # print("fakeMvTensorLayerx3")
            self.network = [
                [
                    fakeMvTensorLayer(which=1),
                    fakeMvTensorLayer(which=0),
                ],
                [
                    fakeMvTensorLayer(which=2),
                ]
            ]
        else:
            print("Invalid selection")

        self.binaryData = [

            np.array([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1], dtype=np.float16),   # Conv Bias
            np.array([  # Just Data for test..
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            ], dtype=np.float16),
            np.ones((1*1*64*64), dtype=np.uint8),   # Conv Weights
            np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            ], dtype=np.uint8),
            np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            ], dtype=np.float16),

        ]


class fakeDPULayer():
    def __init__(self, which=0):

        # Resnet Example based off of csv
        self.referenceIDs = ["conv1"]
        self.dependency = []
        self.consumers = []
        self.taskID = 123

        self.op = DPULayerType.CONV # hardcoded for now
        self.clusterID = 0
        self.kH = 1
        self.kW = 1
        self.kStrideH = 1
        self.kStrideW = 1
        self.in_data = fakeTensor(offset=0x31000)
        self.out_data = fakeTensor(offset=0x0)
        self.param_data = fakeTensor(offset=0x62000)
        # THese PPE Layers Nones are just for testing. they are invalid
        self.ppe_param = ([PPELayerType.NONE.value, PPELayerType.NONE.value, PPELayerType.NONE.value], 1, 1, fakeTensor(offset=0x62000), None)

        self.padTop = [0, 0, 1, 0, 0]
        self.padBottom = [0, 0, 1, 0, 0]
        self.padLeft = [0, 0, 1, 0, 0]
        self.padRight = [0, 0, 1, 0, 0]
        self.dpuID = [0, 1, 2, 3, 4]
        self.mpe = [1, 0, 1, 1, 0]
        self.oXs = [0, 0, 0, 32, 32]
        self.oYs = [0, 20, 30, 0, 28]
        self.oZs = [0, 0, 0, 0, 0]


class fakeDMALayer():
    def __init__(self):

        self.referenceIDs = ["conv1"]
        self.src = 1
        self.dst = 2
        self.str = 100

class fakeNNDMALayer():
    def __init__(self):
        self.referenceIDs = ["conv1"]
        self.src = 1
        self.dst = 2
        self.str = 100
        self.sparsity = False
        self.compress = False


class fakeNCELayer():
    def __init__(self):
        self.referenceIDs = ["conv1"]
        self.h_split = 1
        self.ic_split = 1
        self.ic_pad = 32
        self.oc_pad = 64
        self.iw_pad = 224
        self.ih_pad = 224
        self.descMask = 1

        self.mode = [1, 1]
        self.processedI = [1, 1]
        self.processedO = [1, 1]
        self.startlineI = [1, 1]
        self.startlineO = [1, 1]


class fakeMvTensorLayer():
    def __init__(self, which=0):
        if which == 0:
            self.referenceIDs = ["conv1"]
            self.dependency = [1]
            self.consumers = [100]
            self.taskID = 10
            self.mvTensorID = 0

            self.radixX = 1
            self.radixY = 1
            self.radixStrideX = 1
            self.radixStrideY = 1
            self.padX = 0
            self.padY = 0
            self.padStyle = 0
            self.dilation = 1
            self.input = fakeTensor(
                x=32,
                y=32,
                z=1,
                x_s=2,
                y_s=2,
                z_s=64
                # offset=32*32*2
            )
            self.output = fakeTensor(
                x=28,
                y=28,
                z=6,
                x_s=2,
                y_s=12,
                z_s=336
                # offset=(32*32*2)*2
            )
            self.tap = fakeTensor(
                x=1,
                y=1,
                z=6,
                x_s=2,
                y_s=2,
                z_s=2,
                offset=2
            )
            self.bias = fakeTensor(
                x=1,
                y=1,
                z=6,
                x_s=2,
                y_s=2,
                z_s=2,
                offset=0
            )
        elif which == 1:

            self.referenceIDs = ["input"]
            self.dependency = []
            self.consumers = [10]
            self.taskID = 1
            self.mvTensorID = 5

            self.input = fakeTensor(
                x=32,
                y=32,
                z=1,
                x_s=2,
                y_s=2,
                z_s=64
            )
            self.output = fakeTensor(
                x=32,
                y=32,
                z=1,
                x_s=2,
                y_s=2,
                z_s=64,
                offset=32*32*2
            )
        elif which == 2:
            self.referenceIDs = ["output"]
            self.dependency = [10]
            self.consumers = []
            self.taskID = 100
            self.mvTensorID = 5

            self.input = fakeTensor(
                x=28,
                y=28,
                z=6,
                x_s=2,
                y_s=12,
                z_s=336,
                offset=(32*32*2)*2
            )
            self.output = fakeTensor(
                x=28,
                y=28,
                z=6,
                x_s=2,
                y_s=12,
                z_s=336,
                offset=((32 * 32 * 1 * 2) + (28 * 28 * 6)) * 2
            )
        else:
            print("invalid layer")
            quit()


class fakeTensor():
    def __init__(
        self,
        x=0,
        y=0,
        z=0,
        x_s=0,
        y_s=0,
        z_s=0,
        locale=0,
        offset=0,
        dtype=0,
        order=0
    ):
        self.x = x
        self.y = y
        self.z = z
        self.x_s = x_s
        self.y_s = y_s
        self.z_s = z_s
        self.locale = locale
        self.offset = offset
        self.dtype = dtype
        self.order = order


class fakeNNShvLayer():
    def __init__(self):
        # Example FCL
        self.referenceIDs = ["conv1"]
        self.input = fakeTensor()
        self.output = fakeTensor()
        self.tap = fakeTensor()
        self.bias = fakeTensor()


class fakeControllerLayer():
    def __init__(self):
        self.referenceIDs = ["conv1"]
        self.workID = 0


class fakeCustomLayer():
    def __init__(self):
        self.referenceIDs = ["conv1"]
        self.customLayerID = 0
        self.customLayerData = [1, 2, 3, 4]
