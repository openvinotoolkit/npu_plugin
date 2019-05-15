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

from enum import Enum
from Controllers.Parsers.Parser.Layer import MangledName, OriginalName
import numpy as np

from ordered_set import OrderedSet

class VPUTask():
    def __init__(self):
        self.input_tensors = []
        self.output_tensors = []
        self.parameters = []
        self.cost = 0
        self.wait_barriers = []
        self.update_barriers = []

    def getInputTensors(self):
        return self.input_tensors

    def getParameters(self):
        return self.parameters

    def getInputAndParameters(self):
        return self.input_tensors + self.parameters

    def getInputAndParametersEnclosers(self):
        return self.getInputTensorsEnclosers() + self.getParameterTensorsEnclosers()

    def getInputTensorsEnclosers(self):
        return list(OrderedSet([t.getTopEncloserRecursive() for t in self.input_tensors]))

    def getParameterTensorsEnclosers(self):
        return list(OrderedSet([t.getTopEncloserRecursive() for t in self.parameters]))

    def getOutputTensors(self):
        return self.output_tensors

    def getOutputTensorsEnclosers(self):
        return list(OrderedSet([t.getTopEncloserRecursive() for t in self.output_tensors]))

    def getPadding(self, cluster=None):
        (pl, pr), (pt, pb) = self.op.getPadding()
        if cluster == None or all([it.broadcast for it in self.input_tensors]):
            return (pl, pr), (pt, pb)
        else:
            st = self.output_tensors[0].broadcast_subtensor(cluster).original_tensor
            # padding at cluster level 
            pt = pt if hasattr(st, "offset") and st.offset[2] == 0 else 0
            pl = pl if hasattr(st, "offset") and st.offset[3] == 0 else 0

            pb = pb if hasattr(st, "offset") and st.shape[2] + st.offset[2] == self.output_tensors[0].dimensions[2] else 0
            pr = pr if hasattr(st, "offset") and st.shape[3] + st.offset[3] == self.output_tensors[0].dimensions[3] else 0

            return (pl, pr), (pt, pb)

    @property
    def output_size(self):
        return self.__tensors_size(self.output_tensors)

    @property
    def input_size(self):
        return self.__tensors_size(self.input_tensors)

    @property
    def size(self):
        return self.__tensors_size(self.tensors)

    @property
    def cluster_size(self):
        return self.__tensors_cluster_size(self.tensors)

    @property
    def tensors(self):
        return self.output_tensors + self.input_tensors + self.parameters

    def __tensors_size(self, tensors):
        return sum([t.size for t in tensors])

    def __tensors_cluster_size(self, tensors):
        return sum([t.cluster_size for t in tensors])

    def setWaitBarriers(self, wait_barriers):
        self.wait_barriers = wait_barriers

    def getWaitBarriers(self):
        return self.wait_barriers

    def setUpdateBarriers(self, update_barriers):
        self.update_barriers = update_barriers

    def getUpdateBarriers(self):
        return self.update_barriers

    def find_in_ddr(self, ig_map, names):
        return ig_map['ddr_heap'].find(names) + ig_map['ddr_bss'].find(names)

    def find_in_cmx(self, ig_map, names):
        return ig_map['nn_cmx'].find(names)

    def resolveTensors(self, ig_map):
        self.input_tensors = self.find_in_cmx(ig_map, [t.name for t in self.input_tensors])
        self.output_tensors = self.find_in_cmx(ig_map, [t.name for t in self.output_tensors])
        self.parameters = self.find_in_cmx(ig_map, [t.name for t in self.getParameterTensorsEnclosers()])

    def align_size_to(self, size, num_bytes):
        aligned = size
        if (aligned % num_bytes):
            aligned = num_bytes + num_bytes * (aligned//num_bytes)
        return aligned

class DPUTask(VPUTask):
    def __init__(self, layer, nDPU):
        super(DPUTask, self).__init__()
        self.name = layer.name.stringifyName()
        self.input_tensors = list(layer.getInputTensors())
        self.output_tensors = list(layer.getOutputTensors())
        self.workloads = []
        self.op = layer
        self.nDPU = nDPU
        self.cost = self.__calulate_op_cost()
        self.type = type(self)
        self.in_place = layer.getInPlace()
        self.implicit = layer.getImplicit()

        from Controllers.Parsers.Parser.Hw import HwPooling, HwDwConvolution
        if type(layer) in [HwPooling, HwDwConvolution]:
            self.MPEModes = [(1, 16)]
        else:
            self.MPEModes = [(4, 4), (1, 16)]

        if self.hasWeights():
            self.op.weights.setLayout(self.input_tensors[0].getLayout(), set_subtensors=True)
            self.parameters.append(self.op.weights)
            if self.op.weights.sparse:
                self.parameters.append(self.op.weights.sparsity)
        if self.hasWeightsTable():
            self.parameters.append(self.op.weights_table)
        if self.hasSparseDW():
            self.parameters.append(self.op.sparse_dw)
        if self.hasBias():
            self.parameters.append(self.op.bias)
            if self.op.bias.sparse:
                self.parameters.append(self.op.bias.sparsity)

    def __calulate_op_cost(self):
        # Estimate the operation cost
        if hasattr(self.op, 'cost'):
            return self.op.cost(self.workloads, self.nDPU)
        return 0

    def getWeightSize(self):
        if self.op.hasWeights():
            return self.op.getWeights().size
        else:
            return 0

    def getWeights(self):
        if self.hasWeights():
            return [p for p in self.parameters if p.name.stringifyName() == self.op.getWeights().name.stringifyName()][0]
        else:
            None

    def getWeightsEnclosure(self):
        if self.hasWeights():
            return [p for p in self.parameters if p.name.stringifyName() == self.op.getWeights().getTopEncloserRecursive().name.stringifyName()][0]
        else:
            None

    def getBias(self):
        if self.hasBias():
            return [p for p in self.parameters if p.name.stringifyName() == self.op.getBias().name.stringifyName()][0]
        else:
            None

    def getWeightsTable(self):
        if self.hasWeightsTable():
            return [p for p in self.parameters if p.name.stringifyName() == self.op.weights_table.name.stringifyName()][0]
        else:
            None

    def getSparseDW(self):
        if self.hasSparseDW():
            return [p for p in self.parameters if p.name.stringifyName() == self.op.sparse_dw.name.stringifyName()][0]
        else:
            None

    def getSparseDWChannelLen(self):
        if self.hasSparseDW():
            if hasattr(self.op, "per_channel_bit_pattern"):
                return self.op.per_channel_bit_pattern
            else:
                return 0
        else:
            return 0

    def hasWeights(self):
        return hasattr(self.op, "weights")

    def hasWeightsTable(self):
        return hasattr(self.op, "weights_table")

    def hasSparseDW(self):
        # Only for DW op
        return hasattr(self.op, "sparse_dw")

    def hasBias(self):
        # the DPU task have bias only if is not encoded into a weight table
        return hasattr(self.op, "bias") and (not self.hasWeightsTable())

    def hasScale(self):
        return hasattr(self.op, "q_scale")

    def hasShift(self):
        return hasattr(self.op, "q_shift")

    def setWorkloads(self, workloads):
        if any(isinstance(wl, list) for wl in workloads):
            self.workloads = [[wl for wl in sorted(workloads[cluster], key=lambda wl: wl.execution_cycles(), reverse=True)] for cluster in range(len(workloads))]
        else:
            self.workloads = [wl for wl in sorted(workloads, key=lambda wl: wl.execution_cycles(), reverse=True)]
        self.cost = self.__calulate_op_cost()

    def resolveWeightTable(self, ig_map, nClusters):

        from Controllers.Parsers.Parser.Hw import HwPooling, HwDwConvolution

        def swap_indexes(index, layout):
            return tuple([index[idx] for idx in layout])

        # All (quantized) the layer should have this
        if not self.hasWeightsTable():
            return
        weights_table = self.find_in_ddr(ig_map, [self.op.weights_table.name])[0]

        if self.hasWeights():
            weights = self.find_in_cmx(ig_map, [self.op.weights.getTopEncloserRecursive().name])[0]
            if self.op.weights.sparse:
                sparsity = self.find_in_cmx(ig_map, [self.op.weights.sparsity.name])[0]
            elif self.hasSparseDW():
                sparsity = self.find_in_cmx(ig_map, [self.op.sparse_dw.name])[0]
            else:
                sparsity = [None] * nClusters
        else:
            weights = [None] * nClusters
            if self.hasSparseDW():
                sparsity = self.find_in_cmx(ig_map, [self.op.sparse_dw.name])[0]
            else:
                sparsity = [None] * nClusters

        for cluster in range(nClusters):
            k_idx = weights_table[cluster].layout.index(0)
            for k in range(weights_table[cluster].dimensions[0]):
                if weights[cluster]:
                    if type(self.op) in [HwDwConvolution, HwPooling]:
                        # Weight sets for Depthwise are not inclusive of channels
                        sk = weights[cluster].dimensions[-1] * weights[cluster].dimensions[-2]
                    else:
                        sk = weights[cluster].strides[weights[cluster].layout.index(0)]
                    # Offset = k * stride_k
                    if weights.sparse:
                        weights_table[cluster].data[swap_indexes([k,0,0,0], weights_table[cluster].layout)] = weights[cluster].address + sum(weights[cluster].compressed_kernel_sizes[:k])
                    else:
                        weights_table[cluster].data[swap_indexes([k,0,0,0], weights_table[cluster].layout)] = weights[cluster].address + k * self.align_size_to(sk, 16)

                if sparsity[cluster]:
                    sk = sparsity[cluster].strides[sparsity[cluster].layout.index(0)]
                    weights_table[cluster].data[swap_indexes([k,0,0,1], weights_table[cluster].layout)] = sparsity[cluster].address + k * self.align_size_to(sk, 16)
                else:
                    #place the weight_sparse address outside of the weight table, so we don't end up reading sparse pattern in dense case
                    weights_table[cluster].data[swap_indexes([k,0,0,1], weights_table[cluster].layout)] = 0xFFFFFF

class DMATask(VPUTask):
    class Type(Enum):
        DDR2CMX = 'DDR2CMX'
        CMX2DDR = 'CMX2DDR'
        CMX2UPA = 'CMX2UPA'
        UPA2CMX = 'UPA2CMX'

    def __init__(self, tensors, direction):
        super(DMATask, self).__init__()
        if not direction in [DMATask.Type.CMX2DDR, DMATask.Type.DDR2CMX,
                             DMATask.Type.CMX2UPA, DMATask.Type.UPA2CMX]:
            raise ValueError("Invalid DMA transaction: {}".format(direction))

        self.name = "{}_{}".format("_".join([t.name.stringifyName() for t in tensors]), direction.value)
        if  direction in [DMATask.Type.CMX2DDR, DMATask.Type.CMX2UPA]:
            self.input_tensors = tensors
        else:
            self.output_tensors = tensors
        self.type = direction
        DMA_BYTE_PER_CYCLES = 20/0.7; # 20 GByte/s / 0.700 GHz -> Byte/cycle
        # cost in clock cycles
        self.cost =  self.cluster_size / DMA_BYTE_PER_CYCLES

    def resolveTensors(self, ig_map):
        if  self.type == DMATask.Type.CMX2DDR:
            self.input_tensors = self.find_in_cmx(ig_map, [t.name for t in self.input_tensors])
            self.output_tensors = self.find_in_ddr(ig_map, [t.name for t in self.input_tensors])
        elif self.type == DMATask.Type.DDR2CMX:
            self.input_tensors = self.find_in_ddr(ig_map, [t.name for t in self.output_tensors])
            self.output_tensors = self.find_in_cmx(ig_map, [t.name for t in self.output_tensors])
        else:
            raise ValueError("UPA CMX not supported yet")

    def getMatchingTensors(self):
        def canFlatten(in_tensor, out_tensors):
            return in_tensor.sparse and all([t.sparse for t in out_tensors])

        def tryFlatten(in_tensor, out_tensors):
            if canFlatten(in_tensor, out_tensors) and False: # TODO: enable this when multiple subtensors are supported
                return (in_tensor.flatten(), [t.flatten() for t in out_tensors])
            return (in_tensor, out_tensors)

        return [tryFlatten(it, [ot for ot in self.output_tensors if ot.name == it.name]) for it in self.input_tensors]

class LeonTask(VPUTask):
    class Type(Enum):
        Placeholder = 'Placeholder'
        DEALLOC = 'DEALLOC'
        Barrier = 'Barrier'

    def __init__(self, tensors, op_type):
        super(LeonTask, self).__init__()
        self.type = op_type
        self.name = self.name = "{}_{}".format("_".join([t.name.stringifyName() for t in tensors]), op_type.value)
        if self.type == LeonTask.Type.DEALLOC:
            self.input_tensors = tensors
        elif self.type == LeonTask.Type.Barrier:
            self.setup_barrier = tensors
