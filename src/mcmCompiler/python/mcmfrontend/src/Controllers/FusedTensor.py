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

from copy import copy
import numpy as np
import uuid
import os

from Controllers.Tensor import PopulatedTensor, ResolvedTensor, addCoordinates
from Controllers.Parsers.Parser.Layer import MangledName, OriginalName
from Controllers.Math import prod

# Fused Tensor class is enclosing multiple tensors toghetehr as a single
# entity in order to fuse them at serialization stage
class FusedTensor():

    def __init__(self, tensors):
        if any([not isinstance(tensor, PopulatedTensor)
                for tensor in tensors]):
            raise ValueError("Only PopulatedTensors can be fused together")
        if any([t.sparse for t in tensors]):
            # Place the sparse tensors at the end because weight table address
            # has to be the same in each cluster
            self.fused_tensors = [t for t in tensors if not t.sparse]
            self.fused_tensors += [t for t in tensors if t.sparse]
        else:
            self.fused_tensors = tensors

        # Get the name in common
        common = os.path.commonprefix([t.name.stringifyOriginalName() for t in self.fused_tensors])
        name = common + "_and".join([t.name.stringifyOriginalName()[len(common):]
                                           for t in self.fused_tensors])

        self.name = MangledName(OriginalName(name))
        self.__ID = uuid.uuid4()

        # Can be fused only to uint8
        self.dtype = np.uint8
        self.layout = (0, 1, 2, 3)

        alias_list = list(set([t.alias for t in self.fused_tensors if hasattr(t, "alias")]))
        if len(alias_list) == 1:
            self.alias = alias_list[0]

    def getTopEncloserRecursive(self):
        return self

    @property
    def shape(self):
        return (self.base_size, 1, 1, 1)

    @property
    def size(self):
        return sum([t.size for t in self.fused_tensors])

    @property
    def base_size(self):
        return sum([t.base_size for t in self.fused_tensors])

    @property
    def cluster_size(self):
        return sum([t.cluster_size for t in self.fused_tensors])

    @property
    def base_cluster_size(self):
        return sum([t.base_cluster_size for t in self.fused_tensors])

    def setAddress(self, address):
        current_address = address
        subtensor_offset = []
        for t in self.fused_tensors:
            t.setAddress(current_address, subtensor_offset)
            # Update address and offsets
            subtensor_offset = [st.size - t.cluster_size for st in t.subtensors]
            current_address += t.cluster_size

    def resolve(self, resolve_subtensors=False, codec = False):
        rt = ResolvedFusedTensor(self, resolve_subtensors=resolve_subtensors, codec = codec)
        return rt

    def __deepcopy__(self, memo):
        return copy(self)

    def getStrideInBytes(self, axis):
        return np.dtype(self.dtype).itemsize * self.getStrideInElements(axis)

    def getStrideInElements(self, axis):

        # Increase the dimension in the axis of interest, to
        # ensure we will not fall off the edge.
        increment = [0] * len(self.shape)
        increment[axis] = 1
        increment = tuple(increment)
        tensorShape = addCoordinates(self.shape, increment)

        tensorOrigin = (0, 0, 0, 0)
        tensorOriginNeighbour = addCoordinates(tensorOrigin, increment)

        d1 = self.__getDistanceFromOrigin(tensorOrigin, tensorShape)
        d2 = self.__getDistanceFromOrigin(tensorOriginNeighbour, tensorShape)

        return d2 - d1

    def __getDistanceFromOrigin(self, absoluteCoordinate, topTensorShape):

        absPos = self.__shapeInCanonicalLayout(absoluteCoordinate)
        reverseCanonicalShape = tuple(
            reversed(self.__shapeInCanonicalLayout(topTensorShape)))

        dist = 0
        for idx, coord in enumerate(reversed(absPos)):
            dist += coord * prod(reverseCanonicalShape[0:idx])

        return dist

    def __shapeInCanonicalLayout(self, shape):
        '''For a shape given in a canonical format, convert it to
           according to the (non-canonical) layout provided. The
           result is a new shape whose layout is canonical'''

        assert(len(shape) == len(self.layout))

        permutation = [self.layout.index(i) for i in self.layout]
        newShape = tuple([shape[self.layout[i]] for i in permutation])
        return newShape


class ResolvedFusedTensor():
    def __init__(self, tensor, resolve_subtensors=False, codec=False):
        self.__uniqueID = uuid.uuid4()
        self.fused_tensors = [ResolvedTensor(t, resolve_subtensors ,codec)
                              for t in tensor.fused_tensors]
        self.name = tensor.name
        self.original_tensor = tensor
        self.__subtensors = []
        self.__dtype = tensor.dtype
        self.__layout = tensor.layout
        self.__codec = codec
        self.__strides = [tensor.getStrideInBytes(axis)
                          for axis in range(len(tensor.shape))]
        elem_size = np.dtype(tensor.dtype).itemsize
        self.__alternate_strides = [elem_size] + [tensor.getStrideInBytes(axis)
                                                  for axis in range(len(tensor.shape))]

        # Place subtensors
        for t in tensor.fused_tensors:
            for st in t.subtensors:
                if resolve_subtensors and (st.enclosure is None):
                    st.place(t, st.offset)

        # Get the number of subtensors
        n_subtensors = list(set([len(t.subtensors) for t in self.fused_tensors]))
        assert(len(n_subtensors) == 1)
        for n in range(n_subtensors[0]):
            self.__subtensors.append(ResolvedFusedTensor(FusedTensor([t.subtensors[n] 
                                     for t in tensor.fused_tensors]), codec=codec))

    def __prepare_fused_tensor_data(self, tensor):
        data = tensor.data.transpose(tensor.layout)
        if tensor.sparse:
            # Compress the data
            compress_data, self.__compressed_kernel_sizes = tensor.flatnonzero(data)
            data = np.array(compress_data).astype(tensor.dtype)
        data = data.flatten().view(self.dtype)
        if self.__codec:
            data = self.__codec.compress(data)
        return data

    def getStridesAndSize(self):
        return self.__alternate_strides

    def __get_sparse_tensor(self):
        sparse_tensors =[t for t in self.fused_tensors
                          if t.sparse]

        if len(sparse_tensors) > 1:
            raise ValueError("Only one tensor can be sparse with DMA fusion")
        elif len(sparse_tensors) == 1:
            return sparse_tensors[0]
        else:
            return None

    @property
    def size(self):
        return self.compute_data().size * np.dtype(self.dtype).itemsize

    @property
    def dimensions(self):
        return (self.size, 1, 1, 1)

    @property
    def opaque(self):
        return True

    @property
    def layout(self):
        return self.__layout

    @property
    def dtype(self):
        return self.__dtype

    def getTopEncloserRecursive(self):
        return self

    @property
    def sparsity(self):
        sparse_tensor = self.__get_sparse_tensor()
        if sparse_tensor:
            return sparse_tensor.sparsity
        return None

    @property
    def se(self):
        sparse_tensor = self.__get_sparse_tensor()
        if sparse_tensor:
            return sparse_tensor.se
        return None

    @property
    def sparse(self):
        return any([t.sparse for t in self.fused_tensors])

    def __getitem__(self, key):
        return self.__subtensors[key] if key < len(
            self.__subtensors) and not self.broadcast else self

    def broadcast_subtensor(self, key):
        return self.__subtensors[key] if key < len(self.__subtensors) else self

    @property
    def broadcast(self):
        total_broadcast = list(set([t.broadcast for t in self.fused_tensors]))
        assert(len(total_broadcast)==1)
        return total_broadcast[0]

    @property
    def address(self):
        return min([t.address for t in self.fused_tensors])

    @property
    def strides(self):
        return self.__strides

    @property
    def compressed_kernel_sizes(self):
        if not self.sparse:
            raise ValueError("Dense tensor do not have compressed kernels")
        return self.__compressed_kernel_sizes

    def compute_data(self):
        # Compute only once
        if hasattr(self, "final_data"):
            return self.final_data
        self.final_data = np.concatenate([self.__prepare_fused_tensor_data(t)
                               for t in self.original_tensor.fused_tensors],
                               axis=-1)
        return self.final_data
    @property
    def data(self):
        # Compute the actual data
        self.__data = self.compute_data()
        self.__data.reshape(self.dimensions)
        return self.__data

    @property
    def uniqueID(self):
        return self.__uniqueID




