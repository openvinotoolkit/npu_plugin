#!/usr/bin/env python3

# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.


import Models.Layouts as Layouts
import numpy as np
from .Layer import Layer
from .Pooling import Pooling
from .Convolution2D import Convolution2D, Deconvolution, ConvolutionDepthWise2D
from .ReLU import ReLU
from .PReLU import PReLU
from .ELU import ELU
from .Eltwise import Eltwise
from .InnerProduct import InnerProduct
from Controllers.Tensor import UnpopulatedTensor, PopulatedTensor
from Controllers.TensorFormat import TensorFormat
from Controllers.PingPong import getManualHwSchedule
from Controllers.NCE import NCE_Scheduler
import Controllers.Globals as GLOBALS
from Models.EnumDeclarations import ISIStrategy
from Controllers.SplitLogic import split_over_h, split_over_k
from Controllers.Parsers.Parser.Layer import MangledName, OriginalName


from operator import mul
from functools import reduce
import math

class HwOp(Layer):
    # Casting from Convolution (I miss C++ ...)
    def __init__(self, parental_obj):
        for attr_name in parental_obj.__dict__:
            setattr(self, attr_name, getattr(parental_obj, attr_name))
        self.isHW = True
        self.split_over_c = 1
        self.split_over_h = False
        self.streaming = False
        self.cmx_unload = False
        self.post_op = []
        self.pre_op = []
        self.compatible_layouts = [Layouts.NCHW, Layouts.NHCW]
        self.hardware_solution = None
        self.is_concat = False
        self.scale = 1
        self.relu_scale = 1

    def setSplitOverC(self,split):
        if isinstance(split, int):
            self.split_over_c = split
        elif isinstance(split, list):
            self.split_over_c = split[0]
        else:
            raise ValueError("Incompatible type for split_over_c: {}".format(type(split)))

    def getSplitOverC(self):
        return self.split_over_c

    def setSplitOverH(self, split):
        if not isinstance(split, bool):
            raise ValueError("Incompatible type for split_over_c: {}".format(type(split)))
        self.split_over_h = split

    def isSplitOverH(self):
        return self.split_over_h

    def setStreaming(self, streaming):
        self.streaming = streaming

    def isStreaming(self):
        return self.streaming

    def setSolution(self, solution):
        self.solution = solution

    def getSolution(self):
        return self.solution

    def setPingPong(self, pingpong):
        self.ping_pong = pingpong

    def getPingPong(self):
        return self.ping_pong

    def setCMXforStreaming(self, cmx_size):
        self.cmx_size = 1024* cmx_size

    def getCMXforStreaming(self):
        return self.cmx_size

    def fitInCMX(self, cmx):
        input_size = sum([t.cluster_size for t in self.getInputTensors()])
        output_size = sum([t.cluster_size for t in self.getOutputTensors()])
        if self.hasWeights():
            weights_size = self.getWeights().getTopEncloserRecursive().cluster_size
        else:
            weights_size = 0

        return (input_size+output_size+weights_size)< cmx

    def splitOp(self, cmx):
        input_size = sum([t.size for t in self.getInputTensors()])
        output_size = sum([t.size for t in self.getOutputTensors()])
        if self.hasWeights():
            weights_size = self.getWeights().getTopEncloserRecursive().size
        else:
            weights_size = 0

        if True:
            return split_over_h(self, cmx)
        else:
            return split_over_k(self, cmx)



    def setImplementation(self, scheduler, bypass):
        if bypass:
            pingPongPair = getManualHwSchedule()
            location, _ , rud = pingPongPair[self.getName().stringifyOriginalName()]
            if pingPongPair.isStreamedAndSplit(self.getName().stringifyOriginalName()):
                ss = 'SS'
            elif pingPongPair.isStreamed(self.getName().stringifyOriginalName()):
                ss = 'SX'
            else:
                ss = 'XX'
            cmx_pos = pingPongPair.streamingCmxPos(self.getName().stringifyOriginalName())
            cmx_stream = pingPongPair.cmxForStreaming(self.getName().stringifyOriginalName())

            streamingConf = (cmx_pos, cmx_stream//1024)
        else:
            ss, location, _, rud, streamingConf = scheduler.ordered_dict[self.getName().stringifyName()]

        self.setPingPong((ss, location, _, rud, streamingConf))
        self.solution = scheduler.solution[self.getName().stringifyName()]
        self.setSplitOverC(scheduler.split[self.getName().stringifyName()])
        self.rud = rud

        self.setStreaming(True if ss[0].lower() == 's' else False)
        self.setSplitOverH(True if ss[1].lower() == 's' else False)

        self.setCMXforStreaming(streamingConf[1])

    def setCmxUnload(self, cmxUnload):
        if not isinstance(cmxUnload, bool):
            raise ValueError("Incompatible type for cmxUnload: {}".format(type(cmxUnload)))
        self.cmx_unload = cmxUnload

    def isUnload(self):
        return self.cmx_unload

    def scaleEnabled(self):
        if np.isscalar(self.scale) and np.isscalar(self.relu_scale) and \
            self.scale == 1 and self.relu_scale == 1:
            return False
        return True

    def isHardwarizeable(conv):
        return False

    def setHwSolution(self, hwSolution):
        self.hardware_solution = hwSolution

    def getHwSolution(self):
        return self.hardware_solution

    def setPostOp(self, postOp):
        self.post_op.append(postOp)

    def getPostOp(self):
        return self.post_op

    def setConcatParameters(self, concatOffset, concatOutputSize, totalOutChans):
        self.concatOffset = concatOffset
        self.concatOutputSize = concatOutputSize
        self.totalOutChans = totalOutChans
        self.is_concat = True

    def isConcat(self):
        return self.is_concat

    def getConcatParameters(self):
        concatOffset = 0
        concatOutputSize = 0
        if hasattr(self, 'concatOffset'):
            concatOffset = self.concatOffset
        if hasattr(self, 'concatOutputSize'):
            concatOutputSize = self.concatOutputSize

        return concatOffset, concatOutputSize

    def setPreOp(self, preOp):
        self.pre_op.append(preOp)

    def getPreOp(self):
        return self.pre_op

    def hasWeights(self):
        return False

    def setScale(self, scale):
        self.scale = scale

    def setReLUScale(self, scale):
        self.relu_scale = scale

    def getWindowsSize(self):
        if not type(self) in [HwPooling, HwDwConvolution, HwConvolution]:
            raise ValueError("Impossible to caluculate windows layer type {}".format(type(layer)))

        window_size = 64
        mpe_active = 16
        ky, kx = self.getKernelSize()
        sx, sy = self.getStrideSize()

        # Random check just in case...
        assert(sx == sy)

        def windows_calc(mpe):
            if (sx <= kx):
                return kx + sx * (mpe - 1)
            else:
                return kx * mpe

        mpe_active, window_size = max([(mpe, windows_calc(mpe)) for mpe in [1, 2, 4, 8, 16] if windows_calc(mpe) <= 32])

        return mpe_active, window_size

    def round_up(self, x, mult):
        return ((x + mult -1) // mult) * mult

    def generic_dpu_cost(self, workloads, nDPU, accumulate_input_channels, kernel_size):

        def __workload_cost(shapes, context, nDPU, accumulate_input_channels, kernel_size):
            context_cost = np.prod(kernel_size).astype(np.int32)
            if accumulate_input_channels:
                context_cost *= self.getInputTensors()[0].shape[1]
            # Get the min number of contexts
            n_contexts_in_ot = sum([np.prod([np.ceil(x/y) for x, y in zip(shape, context)]).astype(np.int32) for shape in shapes])
            return np.ceil(n_contexts_in_ot*context_cost/nDPU).astype(np.int32)

        # Calculate the cost of this operation in cycles

        def isListEmpty(inList):
            # https://stackoverflow.com/questions/1593564/python-how-to-check-if-a-nested-list-is-essentially-empty
            # Credit: Ashwin Nanjappa
            if isinstance(inList, list): # Is a list
                return all( map(isListEmpty, inList) )
            return False # Not a list

        if not isListEmpty(workloads):
            # More accurate calulation
            ot = self.getOutputTensors()[0]

            if any(isinstance(wl, list) for wl in workloads):
                # Max over different clusters. Pick the slowest
                return max([max([__workload_cost(wl.getShapes(ot), wl.getCtx(), 1, accumulate_input_channels, kernel_size) for wl in wl_cluster]) for wl_cluster in workloads])
            else:
                # max because layer cost is the cost of the slowest workload
                return max([__workload_cost(wl.getShapes(ot), wl.getCtx(), 1, accumulate_input_channels, kernel_size) for wl in workloads])

        else:
            # coarse grain estimation
            contexts = [[1, 16, 4, 4], [1, 16, 1, 16]]
            ot_shapes = [t.shape for t in self.getOutputTensors()]
            # min because I assume after I can pick the best configuration
            return min([__workload_cost(ot_shapes, ctx, nDPU, accumulate_input_channels, kernel_size) for ctx in contexts])


    def _adapt_tensors(self, tensors, shapes):
        # Adapt input tensors
        change = False
        min_shape = tuple([max([shape[i] for shape in tt.proposed_shapes]) for tt in tensors for i in range(4)])
        for idx, (tensor, shape) in enumerate(zip(tensors,shapes)):
            new_shape = tuple([self.round_up(shape[i], min_shape[i]) for i in range(len(shape))])
            if shape !=  new_shape:
                change = True
                print('Change shape from {} to {}'.format(shape, new_shape))
                # shapes[idx] = new_shape
                new_tensor = UnpopulatedTensor(new_shape)
                new_tensor.setLayout(tensor.getLayout())
                new_tensor.setName(tensor.getName() + "_pad")
                new_tensor.setDatatype(np.float16)
                tensor.place(new_tensor, UnpopulatedTensor.ORIGIN)

        return change

    def adaptTensors(self):
        # self._adapt_tensors(self.inputTensors, [x.getTopEncloserRecursive().getShape() for x in self.inputTensors])
        # self._adapt_tensors(self.outputTensors, [x.getTopEncloserRecursive().getShape() for x in self.outputTensors])
        pass

    def compile(self):
        from Controllers.HwCompiler import hwCompile
        if isinstance(self, HwConvolution):
            if self.sliced:
                i = self.getInputTensors()[0].getSecondTopEncloserRecursive().shape
            else:
                i = self.getInputTensors()[0].getTopEncloserRecursive().shape
            o = self.getOutputTensors()[0].shape

            # Ensure compatibility with old code
            stage = {'kx': self.kernelWidth, 'ky': self.kernelHeight,
                     'sx': self.strideWidth, 'sy': self.strideHeight,
                     'ic': i[1], 'oc': o[1],
                     'is': [i[3], i[2]],
                     'os': [o[3], o[2]],
                     'pkx': 1, 'pky':1, 'psx':1, 'psy':1}
            if isinstance(self, HwConvolutionPooling):
                stage['pkx'] = self.pooling_kernelWidth
                stage['pky'] = self.pooling_kernelHeight
                stage['psx'] = self.pooling_strideWidth
                stage['psy'] = self.pooling_strideHeight

            scheduler = NCE_Scheduler()
            solution, splits = scheduler.optimize_convolution(stage = stage)

            if solution == []:
                raise ValueError('Tensor adaptation generates an hw solution that cannot be implemented')

            if self.getSolution()[0]*self.getSplitOverC() != solution[0]*splits[0]:
                print("Update  {}".format(self.getName().stringifyName()))
                self.setSolution(solution)
                self.setSplitOverC(splits)

        self.hardware_solution = hwCompile(self)

    @property
    def DW_SOH_A0_workaround(self):
        # Bug 33243: Depthwise tensor segmentation issues invalid read at split boundary
        # Trigger conditions:
        #   - layer padding left is odd
        #   - layer is DW convolution
        #   - layer is splitted over H
        return  isinstance(self, HwDwConvolution) and \
                all([t.strategy == ISIStrategy.SplitOverH for t in self.getOutputTensors()]) and \
                self.getPadding()[1][0]%2 != 0 and self.getKernelSize()[0] > 1

class HwConvolution(HwOp, Convolution2D):

    def __init__(self, *args):
        super().__init__(*args)

        if GLOBALS.USING_KMB:
            ChannelMajor = TensorFormat(Layouts.ChannelMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
            ZMajor = TensorFormat(Layouts.ZMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))

            if(self.getInputTensors()[0].shape[1] >= 16):
                self.formatPool = [(ZMajor, ZMajor), (ZMajor, ChannelMajor), (ChannelMajor, ZMajor), (ChannelMajor, ChannelMajor),]
            else:
                self.formatPool = [(ChannelMajor, ZMajor), (ChannelMajor, ChannelMajor),]
        else:
            # Set the supported layouts
            restricted_planar = TensorFormat(Layouts.NCHW, (1, 2, 3), axesAlignment=(1, 1, 1, 8))
            restricted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, 1, 1, 8))
            self.formatPool = [
                (restricted_row_interleaved, restricted_row_interleaved),
                (restricted_row_interleaved, restricted_planar),
                (restricted_planar, restricted_row_interleaved),
                (restricted_planar, restricted_planar),
            ]

        if self.dilationFactor > 1:
            # Adapt kernel
            self.kernelHeight += (self.kernelHeight -1)*(self.dilationFactor -1)
            self.kernelWidth += (self.kernelWidth -1)*(self.dilationFactor -1)

            new_weights = np.zeros((self.weights.data.shape[0], self.weights.data.shape[1], self.kernelHeight, self.kernelWidth))
            new_weights[:, :, ::self.dilationFactor, ::self.dilationFactor] = self.weights.data

            self.setWeights(new_weights)

        if self.groupSize != 1:
            new_weights = np.zeros((self.weights.data.shape[0], self.groupSize*self.weights.data.shape[1], self.kernelHeight, self.kernelWidth))

            stepk = self.weights.shape[0]//self.groupSize
            stepc = self.weights.shape[1]
            for idx, group in enumerate(range(self.groupSize)):
                group_N_weights = self.weights.data[self.weights.data.shape[0] // self.groupSize * group:
                                   self.weights.data.shape[0] // self.groupSize * (group + 1), ]

                new_weights[stepk*idx:stepk*(idx+1), stepc*idx:stepc*(idx+1), ...] = group_N_weights

            self.setWeights(new_weights)
            self.groupSize = 1


        def round(x, base=16):
            return int(base * math.ceil(float(x)/base))

        w = self.getWeights()
        c = reduce(mul, w.shape[1:], 1)

        # Reshape to kmb representation (oC, WeightSet)
        # CM conv vs Zmajor Conv
        if(self.getInputTensors()[0].shape[1] >= 16):
            w.data = w.data.transpose(Layouts.ZMajor)
        else:
            w.data = w.data.transpose(Layouts.ChannelMajor)

        w.reshape((w.shape[0], 1, 1, c))

        if c%16:    # Check alignment
            # Align to 64bit (TODO: Use datatype in calculation)
            aligned_shape = (w.shape[0], 1, 1,  round(c))
            # Pad the weights
            aligned_weights = np.zeros(aligned_shape).astype(w.dtype)
            aligned_weights[:, :, :, :c] = w.data
            aligned_tensor = PopulatedTensor(aligned_weights)
            aligned_tensor.setLayout(w.getLayout())
            n = MangledName(OriginalName("AlignContainer_"+w.getName().stringifyOriginalName()))
            aligned_tensor.setName(n)
            aligned_tensor.setDatatype(w.getDatatype())
            aligned_tensor.quantization = w.quantization
            self.weights = aligned_tensor

    def adjustFormatPoolForSchedule(self, inputChannels):
        # Set the supported layouts
        adjusted_planar = TensorFormat(Layouts.NCHW, (1, 2, 3), axesAlignment=(1, inputChannels, 1, 8))
        adjusted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, inputChannels, 1, 8))
        restricted_planar = TensorFormat(Layouts.NCHW, (1, 2, 3), axesAlignment=(1, 1, 1, 8))
        restricted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, 1, 1, 8))
        self.formatPool = [
            (adjusted_row_interleaved, restricted_row_interleaved),
            (adjusted_row_interleaved, restricted_planar),
            (adjusted_planar, restricted_row_interleaved),
            (adjusted_planar, restricted_planar),
        ]


    def isHardwarizeable(conv, scheduler):
        if GLOBALS.USING_KMB:
            return True

        try:
            lname = conv.getStringifiedOriginalName()
            entry = scheduler.manual_schedule.ordered_dict[lname]

            if not entry[0]:
                print("Manually falling back to Software Implementation")
                return False
        except:
            pass

        # Check for supported padding
        if (conv.paddingWidth != 0 and conv.paddingWidth != (conv.kernelWidth // 2)) or \
           (conv.paddingHeight != 0 and conv.paddingHeight != (conv.kernelHeight // 2)):
            return False

        # Check for supported kernel sizes
        if conv.kernelHeight> 15 or conv.kernelWidth> 15:
            return False
        #Check for supported strides
        if conv.strideHeight > 8 or conv.strideWidth>8:
            return False

        kh = conv.kernelHeight
        kw = conv.kernelWidth
        if hasattr(conv, 'dilationFactor'):
            if conv.dilationFactor > 1:
                kh += (conv.kernelHeight -1)*(conv.dilationFactor -1)
                kw += (conv.kernelWidth -1)*(conv.dilationFactor -1)

        if not (conv.paddingHeight == 0 or conv.paddingHeight == (kh // 2)):
            # Only support same and none for padding in hw for now
            return False

        i = conv.getInputTensors()[0].shape
        o = conv.getOutputTensors()[0].shape

        # Ensure compatibility with old code
        stage = {'kx': conv.kernelWidth, 'ky': conv.kernelHeight,
                 'sx': conv.strideWidth, 'sy': conv.strideHeight,
                 'ic': i[1], 'oc': o[1],
                 'is': [i[3], i[2]],
                 'os': [o[3], o[2]],
                 'pk':1, 'ps':1}

        (inC, outC, solution), splits = scheduler.optimize_convolution(stage = stage)

        scheduler.solution[conv.getName().stringifyName()] = (inC, outC, solution)
        scheduler.split[conv.getName().stringifyName()] = splits

        if solution == []:
            return False
        return True

    def hasWeights(self):
        if self.getWeights() is not None:
            return True
        else:
            return False

    def cost(self, workloads, nDPU):
        return self.generic_dpu_cost(workloads, nDPU, True, [self.kernelHeight, self.kernelWidth])

class HwDwConvolution(HwOp, ConvolutionDepthWise2D):

    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        ChannelMajor = TensorFormat(Layouts.ChannelMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
        ZMajor = TensorFormat(Layouts.ZMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
        self.formatPool = [
            (ZMajor, ZMajor), (ZMajor, ChannelMajor)
        ]


        def round(x, base=16):
            return int(base * math.ceil(float(x)/base))

        w = self.getWeights()
        c = reduce(mul, w.shape[2:], 1)

        if c%16:    # Check alignment
            # Reshape to kmb representation (oC, ic, WeightSet)
            w.reshape((w.shape[1], 1, 1, c))

            # Align to 64bit (TODO: Use datatype in calculation)
            aligned_shape = (w.shape[0], 1, 1,  round(c))
            # Pad the weights
            aligned_weights = np.zeros(aligned_shape).astype(w.dtype)
            aligned_weights[:, :, :, :c] = w.data
            aligned_tensor = PopulatedTensor(aligned_weights)
            aligned_tensor.setLayout(w.getLayout())
            n = MangledName(OriginalName("AlignContainer_"+w.getName().stringifyOriginalName()))
            aligned_tensor.setName(n)
            aligned_tensor.setDatatype(w.getDatatype())
            aligned_tensor.quantization = w.quantization
            self.weights = aligned_tensor

    def adjustFormatPoolForSchedule(self, inputChannels):
        # Set the supported layouts
        return


    # TODO: implement me!
    def isHardwarizeable(fc, scheduler):
        return True

    def hasWeights(self):
        if self.getWeights() is not None:
            return True
        else:
            return False

    def cost(self, workloads, nDPU):
        return self.generic_dpu_cost(workloads, nDPU, False, [self.kernelHeight, self.kernelWidth])

class HwPooling(HwOp, Pooling):

    def __init__(self, *args):
        super().__init__(*args)

        if GLOBALS.USING_KMB:
            ChannelMajor = TensorFormat(Layouts.ChannelMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
            ZMajor = TensorFormat(Layouts.ZMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
            self.formatPool = [
                (ZMajor, ZMajor), (ZMajor, ChannelMajor)
            ]
        else:

            restricted_planar = TensorFormat(Layouts.NCHW, (1, 2, 3), axesAlignment=(1, 16, 1, 8))
            restricted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, 16, 1, 8))

            self.formatPool = [
                (restricted_row_interleaved, restricted_row_interleaved),
                (restricted_row_interleaved, restricted_planar),
                (restricted_planar, restricted_row_interleaved),
                (restricted_planar, restricted_planar),
            ]

    # TODO: Perform actual check here.
    def isHardwarizeable(pool, scheduler):
        # Checks if pooling can be implemented in hw

        # Check minumum line
        # min_lines = pool.kernelHeight + pool.strideHeight + 2
        # i = pool.getInputTensors()[0].getShape()
        # space_required = min_lines * 2 * i[1] * (((i[2] + 15) // 16)*16)

        try:
            lname = pool.getStringifiedOriginalName()
            entry = scheduler.manual_schedule.ordered_dict[lname]
            if not entry[0]:
                print("Manually falling back to Software Implementation")
                return False
        except:
            pass


        return True

    def adjustFormatPoolForSchedule(self, inputChannels):
        # Set the supported layouts
        adjusted_planar = TensorFormat(Layouts.NCHW, (1, 2, 3), axesAlignment=(1, inputChannels, 1, 8))
        adjusted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, inputChannels, 1, 8))
        restricted_planar = TensorFormat(Layouts.NCHW, (1, 2, 3), axesAlignment=(1, inputChannels, 1, 8))
        restricted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, inputChannels, 1, 8))
        self.formatPool = [
            (adjusted_row_interleaved, restricted_row_interleaved),
            (adjusted_row_interleaved, restricted_planar),
            (adjusted_planar, restricted_row_interleaved),
            (adjusted_planar, restricted_planar),
        ]

    def cost(self, workloads, nDPU):
        return self.generic_dpu_cost(workloads, nDPU, False, [self.kernelHeight, self.kernelWidth])

class HwEltwise(HwOp, Eltwise):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        ChannelMajor = TensorFormat(Layouts.ChannelMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
        ZMajor = TensorFormat(Layouts.ZMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
        # self.formatPool = [
        #     (ChannelMajor, ChannelMajor), (ChannelMajor, ZMajor),
        # ]
        self.formatPool = [
            (ZMajor, ZMajor), (ChannelMajor, ChannelMajor), (ZMajor, ChannelMajor), (ChannelMajor, ZMajor)
        ]

    def isHardwarizeable(eltwise, scheduler):
        return True

    def cost(self, workloads, nDPU):
        return self.generic_dpu_cost(workloads, nDPU, False, [1, 1])


class HwFC(HwOp, InnerProduct):

    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        if GLOBALS.USING_KMB:
            ChannelMajor = TensorFormat(Layouts.ChannelMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
            ZMajor = TensorFormat(Layouts.ZMajor, (1, 2, 3), axesAlignment=(1, 1, 1, 1))
            self.formatPool = [
                (ZMajor, ZMajor), (ChannelMajor, ChannelMajor), (ZMajor, ChannelMajor), (ChannelMajor, ZMajor)
            ]
        else:
            restricted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, 1, 1, 8))
            self.formatPool = [
                (restricted_row_interleaved, restricted_row_interleaved),
            ]

    def adjustFormatPoolForSchedule(self, inputChannels):
        # Set the supported layouts
        adjusted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, inputChannels, 1, 8))
        restricted_row_interleaved = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, 1, 1, 8))
        self.formatPool = [
            (adjusted_row_interleaved, restricted_row_interleaved),
        ]

    # TODO: implement me!
    def isHardwarizeable(fc, scheduler):

        try:
            lname = fc.getStringifiedOriginalName()
            entry = scheduler.manual_schedule.ordered_dict[lname]
            if not entry[0]:
                print("Manually falling back to Software Implementation")
                return False
        except:
            pass


        # Check if fc layer can be implemented in hw
        s = fc.inputTensors[0].getTopEncloserRecursive().getShape()
        if len(s) - 1 != s.count(1):
            # 3D FC cannot be implemented in HW
            return False
        return True

    def hasWeights(self):
        if self.getWeights() is not None:
            return True
        else:
            return False

    def cost(self, workloads, nDPU):
        return self.generic_dpu_cost(workloads, nDPU, True, [1, 1])

class HwConvolutionPooling(HwConvolution):
    def __init__(self, parental_obj):
        for attr_name in parental_obj.__dict__:
            setattr(self, attr_name, getattr(parental_obj, attr_name))

    def setPoolingParameter(self, poolingLayer):
        if not isinstance(poolingLayer, HwPooling):
            raise ValueError("Incompatible type for poolingLayer: {}".format(type(poolingLayer)))
        self.type = poolingLayer.type
        self.globalPooling = poolingLayer.globalPooling
        self.pooling_kernelHeight = poolingLayer.kernelHeight
        self.pooling_kernelWidth = poolingLayer.kernelWidth
        self.pooling_strideHeight = poolingLayer.strideHeight
        self.pooling_strideWidth = poolingLayer.strideWidth
        self.pooling_paddingHeight = poolingLayer.paddingHeight
        self.pooling_paddingWidth = poolingLayer.paddingWidth

    def isGlobal(self):
        return self.globalPooling

    def getType(self):
        return self.type

    def canFuse(conv, pool):
        # Check that pooling is non-overlapping
        if (pool.kernelHeight > pool.strideHeight) or (pool.kernelWidth >pool.strideWidth):
            return False

        # Fusing is it supported only for this mode when split over width is implemented
        if conv.kernelHeight != 3 or conv.kernelWidth != 3 or \
           conv.strideHeight != 1 or conv.strideWidth != 1 or \
           conv.paddingHeight != 1 or conv.paddingWidth != 1:
            return False

        i = conv.getInputTensors()[0].shape
        o = pool.getOutputTensors()[0].shape

        # Try to fuse and check if the solution exists and split over C is == 1
        stage = {'kx': conv.kernelWidth, 'ky': conv.kernelHeight,
                 'sx': conv.strideWidth, 'sy': conv.strideHeight,
                 'ic': i[1], 'oc': o[1],
                 'is': [i[3], i[2]],
                 'os': [o[3], o[2]],
                 'pkx':pool.kernelWidth, 'pky':pool.kernelHeight,
                 'pkx':pool.strideWidth, 'pky':pool.strideHeight}

        scheduler = NCE_Scheduler()
        solution, splits = scheduler.optimize_convolution(stage = stage)

        if isinstance(splits, list):
            splits = max(splits)

        if solution == [] or splits != 1:
            return False

        return True

    def cost(self, workloads, nDPU):
        raise ValueError("Fusing Conv and Pooling is not valid in VPU2")
