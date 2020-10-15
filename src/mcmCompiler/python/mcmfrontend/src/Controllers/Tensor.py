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
import math
from Models import Layouts
from Controllers.Parsers.Parser.Layer import MangledName, OriginalName
from Controllers.Math import prod


def addCoordinates(c1, c2):
    return tuple([sum(x) for x in zip(c1, c2)])


class Tensor():
    ORIGIN = (0, 0, 0, 0)

    def __str__(self):
        return str(self.name)

    def __init__(self, shape):
        # Support only 4D tensors. No particular reason for this
        # limitation (in terms of Tensor support).

        shape = tuple(shape)

        assert(len(shape) == 4)

        self.__ID = uuid.uuid4()
        self.shape = shape
        self.placedTensors = []
        self.enclosure = None
        self.relativeCoordinates = Tensor.ORIGIN
        self.proposed_shapes = []
        self.address = 0
        self.sparse = False
        self.dtype = np.float16
        self.broadcast = True
        self.subtensors = []

    def __deepcopy__(self, memo):
        """
            NetworkX performs a deep copy in several operations (e.g. edge contraction).
            We do not want this, this we may end up having multiple copies of an object.
        """
        return copy(self)

    def setName(self, name):
        if isinstance(name, str):
            self.name = MangledName(OriginalName(name))
        else:
            self.name = name

    def getName(self):
        return self.name

    def getShape(self):
        return self.shape

    def setDatatype(self, dtype):
        self.dtype = dtype

    def setSparsity(self, sparsity, sok_splits=1):
        def calulate_sparsity_map(data):
            # Calculate the sparsity map
            cmp_tensor = np.zeros(data.shape).astype(np.uint8)
            zero_point_tensor = np.stack([self.getSparsityZeroPoint(
                k) * np.ones(data.shape[1:]) for k in range(data.shape[0])]).astype(data.dtype)
            cmp_tensor[data != zero_point_tensor] = 1
            sparsity_map = np.squeeze(
                np.array(
                    np.split(
                        cmp_tensor,
                        math.ceil(
                            self.shape[3] /
                            8),
                        axis=3)))
            sparsity_map = np.einsum(
                'bka,a->kb', sparsity_map, np.array([1 << n for n in range(8)]))

            # Reshape and pad
            padded_cxy = ((sparsity_map.shape[1] + 15) // 16) * 16
            padded_sparsity_map = np.zeros(
                shape=[
                    sparsity_map.shape[0],
                    1,
                    1,
                    padded_cxy]).astype(
                np.uint8)
            for k in range(sparsity_map.shape[0]):
                padded_sparsity_map[k, 0, 0,
                                    :sparsity_map.shape[1]] = sparsity_map[k, :]

            return padded_sparsity_map

        if sparsity:
            if self.layout != Layouts.ZMajor:
                raise ValueError("Sparsity requires ZMajor layout")
            self.se = UnpopulatedTensor(
                shape=[
                    self.shape[0],
                    sok_splits,
                    self.shape[2],
                    self.shape[3]])
            self.se.setDatatype(np.int32)
            self.se.setLayout(self.getLayout())
            self.se.name = MangledName(
                OriginalName(
                    self.name.stringifyOriginalName() +
                    "_se"))
            # Effectively calculate this from data
            if hasattr(self, 'data'):
                self.sparsity = PopulatedTensor(
                    calulate_sparsity_map(self.data))
            else:
                self.sparsity = UnpopulatedTensor(shape=[
                                                  self.shape[0],
                                                  math.ceil(self.shape[1] / (8 * np.dtype(self.dtype).itemsize)),
                                                  self.shape[2], self.shape[3]
                                                  ])
                self.sparsity.setDatatype(np.uint8)
            self.sparsity.setLayout(Layouts.SparsityTensorLayout)
            self.sparsity.name = MangledName(OriginalName(
                self.name.stringifyOriginalName() + "_sm"))
        self.sparse = sparsity

        for st in self.subtensors:
            st.setSparsity(sparsity)

    def proposeShape(self, shape):
        self.proposed_shapes.append(shape)

    def getDatatype(self):
        return self.dtype

    def setLayout(self, layout, set_subtensors=False):
        self.layout = layout
        if set_subtensors:
            for st in self.subtensors:
                st.setLayout(layout)

    def setLayoutRecursivly(self, layout):
        self.layout = layout
        if self.enclosure:
            return self.enclosure.setLayoutRecursivly(layout)
        else:
            return self

    def getLayout(self):
        return self.layout

    def getStrides(self):
        return [self.getStrideInBytes(axis) for axis in range(len(self.shape))]

    def storePlacedTensor(self, smallerTensor):
        # TODO: Check that there is no intersection with other tensors
        # that have already been placed.

        self.placedTensors.append(smallerTensor)

    def getTopEncloserRecursive(self):
        if self.enclosure:
            return self.enclosure.getTopEncloserRecursive()
        else:
            return self

    def getSecondTopEncloserRecursive(self):
        if self.enclosure:
            enc = self.enclosure.getTopEncloserRecursive()
            if enc.enclosure is None:
                return self
            else:
                return self.enclosure.getTopEncloserRecursive()
        else:
            return self

    def getTopEncloser(self):
        if self.enclosure:
            return self.getTopEncloserRecursive()
        else:
            return None

    def getAbsolutePosition(self):
        if self.enclosure:
            enclosureCoords = self.enclosure.getAbsolutePosition()
            return addCoordinates(enclosureCoords, self.relativeCoordinates)

        else:
            return self.relativeCoordinates

    def __shapeInCanonicalLayout(self, shape):
        '''For a shape given in a canonical format, convert it to
           according to the (non-canonical) layout provided. The
           result is a new shape whose layout is canonical'''

        assert(len(shape) == len(self.layout))

        permutation = [self.layout.index(i) for i in self.layout]
        newShape = tuple([shape[self.layout[i]] for i in permutation])
        return newShape

    def __getDistanceFromOrigin(self, absoluteCoordinate, topTensorShape):

        absPos = self.__shapeInCanonicalLayout(absoluteCoordinate)
        reverseCanonicalShape = tuple(
            reversed(self.__shapeInCanonicalLayout(topTensorShape)))

        dist = 0
        for idx, coord in enumerate(reversed(absPos)):
            dist += coord * prod(reverseCanonicalShape[0:idx])

        return dist

    def getDistanceFromOrigin(self):
        topEncloser = self.getTopEncloserRecursive()
        topTensorShape = topEncloser.getShape()
        return self.__getDistanceFromOrigin(
            self.getAbsolutePosition(), topTensorShape)

    def getStrideInElements(self, axis):
        topEncloser = self.getTopEncloserRecursive()
        topTensorShape = topEncloser.getShape()

        # Increase the dimension in the axis of interest, to
        # ensure we will not fall off the edge.
        increment = [0] * len(self.shape)
        increment[axis] = 1
        increment = tuple(increment)
        topTensorShape = addCoordinates(topTensorShape, increment)

        tensorOrigin = self.getAbsolutePosition()
        tensorOriginNeighbour = addCoordinates(tensorOrigin, increment)

        d1 = self.__getDistanceFromOrigin(tensorOrigin, topTensorShape)
        d2 = self.__getDistanceFromOrigin(tensorOriginNeighbour, topTensorShape)

        return d2 - d1

    def getStrideInBytes(self, axis):
        return np.dtype(self.dtype).itemsize * self.getStrideInElements(axis)

    def setQuantizationParameters(self, quantization):
        self.quantization = quantization

    def getQuantizationParameters(self):
        return self.quantization

    def isQuantized(self):
        return hasattr(
            self,
            "quantization") and (
            self.dtype not in [
                np.float16,
                np.float32])

    def place(self, largerTensor, topCornerInLargerTensor):
        """
        Place this tensor inside another.

        arguments:
        @ largerTensor: Tensor which to place this tensor in.
        @ topCornerInLargerTensor: position placed (co-ord)
        """
        # For the placement to be valid, larger and smaller tensors
        # must have the same layout
        assert(self.getLayout() == largerTensor.getLayout())
        assert(self.getDatatype() == largerTensor.getDatatype())

        # You cannot place a tensor where it will not fit.
        assert(all([x1 <= x2 for x1, x2 in zip(self.shape, largerTensor.shape)]))

        # Check that the current tensor fits into the larger tensor
        # Start and End element of the larger tensor follows [, ) convention.
        # assert (largerTensor.getShape() >= tuple([sum(x) for x in zip(self.getShape(),topCornerInLargerTensor)]))

        self.relativeCoordinates = topCornerInLargerTensor
        largerTensor.storePlacedTensor(self)
        assert(self.enclosure is None)
        self.enclosure = largerTensor

    def resolve(self, resolve_subtensors=False, codec=False):
        rt = ResolvedTensor(self, resolve_subtensors=resolve_subtensors, codec=codec)
        return rt

    def pprint(self, recursively=False):
        UNDEF = "<undefined>"
        layout = self.layout if hasattr(self, "layout") else UNDEF
        dist = self.getDistanceFromOrigin() if hasattr(self, "layout") else UNDEF
        """
        Pretty Print the Buffer
        """
        print("""
                X       ID = {}
                X
                X       Shape = {}
                X       Layout = {}
                X       Distance from Origin = {}
               X X
             XX   XX
          XXX       XXX
        XX    Tensor   XX
            """.format(self.ID, self.shape, layout, dist)
              )

        if recursively:
            if self.enclosure:
                print()
                print('Enclosing Tensor:')
                self.enclosure.pprint(recursively)

    # Flat the non-zero (!= from zeroPoint) elements in a tensor
    # NB: the sparsity map should be padded to 16 Bytes over the dimension
    # with bigger strides (top left)
    def flatnonzero(self, data):
        def round_up(size, alignment=16):
            return ((size + alignment - 1) // alignment) * alignment

        if not hasattr(self, "compressed_data"):
            self.compressed_data = []
            for k in range(data.shape[0]):
                compressed_kernel = list(data[k, ...].flatten(
                )[data[k, ...].flatten() != self.getSparsityZeroPoint(k)])
                data_size = len(compressed_kernel) * \
                    np.dtype(data.dtype).itemsize
                self.compressed_data.append(
                    compressed_kernel + (round_up(data_size, 16) - data_size) * [self.getSparsityZeroPoint(k)])

            self.address_offset = [
                len(part) *
                np.dtype(
                    data.dtype).itemsize for part in self.compressed_data]
            self.compressed_data = sum(self.compressed_data, [])

        return self.compressed_data, self.address_offset

    # Count the number of elements that are not equal of the zeroPoint (for
    # each output channel)
    def count_nonzero(self):
        return len(self.flatnonzero(self.data)[0])

    def getSparsityZeroPoint(self, output_channel):
        if hasattr(self, "quantization"):
            return self.quantization.getZeroPoint(output_channel)
        else:
            return 0

    # ID must be read-only
    @property
    def ID(self):
        return self.__ID

    @property
    def size(self):
        def round_up(size, alignment=16):
            return ((size + alignment - 1) // alignment) * alignment

        shp = self.getTopEncloserRecursive().shape
        tensor_size = np.prod(np.array(list(shp))) * np.dtype(
            self.getDatatype()).itemsize
        if self.sparse:
            if hasattr(self, "data"):
                size = self.count_nonzero() * np.dtype(self.getDatatype()).itemsize
            else:
                # Assume sparse data contains only zeroes
                size = tensor_size + self.se.size + self.sparsity.size
        else:
            size = tensor_size
        # CMX is aligned to 16 Bytes (128 bit lane size...)

        return round_up(size)

    @property
    def cluster_size(self):
        if not self.broadcast and len(self.subtensors) > 0:
            return max([t.size for t in self.subtensors])
        else:
            return self.size

    @property
    def base_size(self):
        def round_up(size, alignment=16):
            return ((size + alignment - 1) // alignment) * alignment

        shp = self.shape
        tensor_size = np.prod(np.array(list(shp))) * np.dtype(
            self.getDatatype()).itemsize
        if self.sparse:
            if hasattr(self, "data"):
                size = self.count_nonzero() * np.dtype(self.getDatatype()).itemsize
            else:
                # Assume sparse data contains only zeroes
                size = tensor_size + self.se.size + self.sparsity.size
        else:
            size = tensor_size
        # CMX is aligned to 16 Bytes (128 bit lane size...)

        return round_up(size)

    @property
    def base_cluster_size(self):
        if not self.broadcast and len(self.subtensors) > 0:
            return max([t.base_size for t in self.subtensors])
        else:
            return self.base_size

    def setAddress(self, address, subtensor_offset=[]):
        self.address = address
        if self.sparse:
            # data, se pointer and sparsity map are contiguous in memory
            if not hasattr(self, "data"):
                # Only for unpopulated tensor
                self.se.address = address + \
                    (self.size - self.se.size - self.sparsity.size)
                self.sparsity.address = address + \
                    (self.size - self.sparsity.size)

        if subtensor_offset == []:
            # Default subtensor offset is zero
            subtensor_offset = [0] * len(self.subtensors)

        for st, st_offset in zip(self.subtensors, subtensor_offset):
            st.setAddress(address + st_offset)

    def reshape(self, shape):
        # Check that the reshape is acceptable by numpy
        self.shape = shape

    def reorder(self, layout):
        """
        """
        self.shape = tuple([self.shape[i] for i in layout])


class UnpopulatedTensor(Tensor):
    def place(self, largerTensor, topCornerInLargerTensor):
        assert(isinstance(largerTensor, UnpopulatedTensor))
        super().place(largerTensor, topCornerInLargerTensor)

    def splitAcrossClusters(self, wl_list, split_over_h=False, multicast=False):
        def __gen_tile(wl, cluster):
            if len(wl.rect_lst) != 1:
                raise ValueError(
                    "More than one rectangle ({}) in the workload.. impossible to generate a subtensor".format(
                        len(
                            wl.rect_lst)))
            rr = wl.rect_lst[0]

            if split_over_h:
                subtensor = UnpopulatedTensor(
                    [self.shape[0], self.shape[1], rr.height, rr.width])
                subtensor.offset = (0, 0, rr.bl[0], rr.bl[1])
            else:
                subtensor = UnpopulatedTensor(
                    [rr.height, rr.width, self.shape[2], self.shape[3]])
                subtensor.offset = (rr.bl[0], rr.bl[1], 0, 0)

            subtensor.cluster = cluster
            subtensor.layout = self.layout
            subtensor.dtype = self.dtype
            if self.isQuantized():
                subtensor.quantization = self.quantization
            subtensor.name = MangledName(
                OriginalName(self.name.stringifyOriginalName()))
            # When SOK, storage element pointer size is equale than H*W*Splits_K
            sok_splits = len(wl_list) if not split_over_h else 1
            subtensor.setSparsity(self.sparse, sok_splits)
            if not split_over_h or multicast:
                # offset need to be zero on the channel axis because the concat is resolved by the ODU
                offset = [o if idx != 1 else 0 for idx, o in enumerate(subtensor.offset)]
                subtensor.place(self, offset)
            return subtensor

        self.wl_list = wl_list
        self.subtensors = [__gen_tile(wl, idx)
                           for idx, wl in enumerate(wl_list)]
        self.broadcast = True if not split_over_h or multicast else False
        # Need to adjust also the se table of the parent tensor
        if not split_over_h and hasattr(self, "se") and len(wl_list) > 1:
            self.se = UnpopulatedTensor(shape=[
                                            self.se.shape[0],
                                            len(wl_list),
                                            self.se.shape[2],
                                            self.se.shape[3]])
            self.se.setDatatype(np.int32)
            self.se.setLayout(self.getLayout())
            self.se.name = MangledName(
                OriginalName(
                    self.name.stringifyOriginalName() +
                    "_se"))


class PopulatedTensor(Tensor):
    def __init__(self, data, isFirstAxisBatchNumber=False, name=None):
        # Check that the data being loaded matches the layout
        self.isFirstAxisBatchNumber = isFirstAxisBatchNumber
        if data.dtype == np.float32:
            print("Warning: data in fp32 format. Cast to fp16")
            data = data.astype(np.float16)
        self.data = data

        # Canonicalize shape to match the dimension of origin
        origShape = self.data.shape
        assert(len(origShape) <= len(Tensor.ORIGIN))

        diff = len(Tensor.ORIGIN) - len(origShape)
        if isFirstAxisBatchNumber:
            newShape = tuple(origShape[0]) + \
                tuple([1] * (diff - 1)) + origShape[1:]
        else:
            newShape = origShape + tuple([1] * diff)

        self.data = self.data.reshape(newShape)

        # Set the canonical layout
        self.setLayout(tuple(range(len(Tensor.ORIGIN))))

        if name is not None:
            self.name = MangledName(OriginalName(name))

        super().__init__(self.data.shape)
        self.dtype = data.dtype


    def place(self, largerTensor, topCornerInLargerTensor):
        # assert(isinstance(largerTensor, UnpopulatedTensor)) # not hold for
        # split over K
        super().place(largerTensor, topCornerInLargerTensor)

    def reshape(self, shape):
        self.data = self.data.reshape(shape)
        self.shape = shape

    # def __canonicalizeShape(self, shape):
    def splitAcrossClusters(self, wl_list, split_over_h=False):
        def __gen_tile(wl, cluster):
            if len(wl.rect_lst) != 1:
                raise ValueError(
                    "More than one rectangle ({}) in the workload.. impossible to generate a subtensor".format(
                        len(
                            wl.rect_lst)))
            rr = wl.rect_lst[0]

            if split_over_h:
                splitted_data = self.data
            else:
                splitted_data = self.data[rr.bl[0]:rr.bl[0] +
                    rr.height, rr.bl[1]:rr.bl[1] + rr.width, :, :]
            subtensor = PopulatedTensor(splitted_data)

            subtensor.offset = (rr.bl[0], rr.bl[1], 0, 0)
            subtensor.layout = self.layout
            subtensor.cluster = cluster

            if self.isQuantized():
                subtensor.quantization = self.quantization
            subtensor.name = MangledName(
                OriginalName(self.name.stringifyOriginalName()))
            subtensor.setSparsity(self.sparse)
            return subtensor

        self.wl_list = wl_list
        self.subtensors = [__gen_tile(wl, idx)
                           for idx, wl in enumerate(wl_list)]
        self.broadcast = True if split_over_h else False


class ResolvedTensor():
    """
        A Resolved Tensor is an read-only representation of a tensor
    """

    def __init__(self, tensor, resolve_subtensors=False, codec=False):
        self.__uniqueID = uuid.uuid4()
        self.__topID = tensor.getTopEncloserRecursive().ID
        self.__original_tensor = tensor
        self.__dimensions = tensor.shape
        self.__strides = [tensor.getStrideInBytes(
            axis) for axis in range(len(tensor.shape))]
        elem_size = np.dtype(self.original_tensor.getDatatype()).itemsize
        self.__alternate_strides = [
            elem_size] + [tensor.getStrideInBytes(axis) for axis in range(len(tensor.shape))]
        self.__layout = tensor.getLayout()
        self.__dtype = tensor.dtype
        self.__name = tensor.name  # in kmb each tensor has a name
        self.__sparse = tensor.sparse
        self.__size = tensor.size
        self.__broadcast = tensor.broadcast
        if self.__sparse:
            self.__se = tensor.se
            self.__sparsity = tensor.sparsity

        self.__subtensors = []
        for st in tensor.subtensors:
            if resolve_subtensors and (st.enclosure is None):
                st.place(tensor, st.offset)
            self.__subtensors.append(ResolvedTensor(st, codec=codec))

        if isinstance(tensor, PopulatedTensor):
            self.__opaque = True
            # Apply the layout. Convert from canonical format to the specified
            # one.
            self.__data = tensor.data.transpose(self.__layout)
            if self.__sparse:
                # Compress the data
                compress_data, self.__compressed_kernel_sizes = tensor.flatnonzero(
                    self.__data)
                self.__data = np.array(compress_data).astype(tensor.dtype)
        else:
            if hasattr(
                    self.original_tensor,
                    "placedTensors") and len(
                    self.original_tensor.placedTensors) > 0:
                # A Populated Tensor was placed inside an Unpopulated Tensor
                # Note: Placement should be ensuring that populated tensors will
                # not overlap.
                self.__opaque = True
                for p in tensor.placedTensors:
                    if (isinstance(p, UnpopulatedTensor)):
                        self.__opaque = False
                        self.__data = np.zeros(
                            tensor.shape).transpose(
                            self.__layout)
                    else:
                        d = p.data
                        offset = p.getAbsolutePosition()
                        self.__data = np.zeros(
                            tensor.shape).transpose(
                            self.__layout).astype(
                            p.getDatatype())
                        self.__data[offset[0]:d.shape[0],
                                    offset[1]:d.shape[1],
                                    offset[2]:d.shape[2],
                                    offset[3]:d.shape[3]] = d

                        # TODO: How does this work
                        if p.sparse:
                            self.__se = p.se
                            self.__sparsity = p.sparsity
                            self.__sparse = p.sparse
            else:
                self.__opaque = False
                self.__data = np.zeros(tensor.shape).transpose(self.__layout)
        self.__local_offset = tensor.getDistanceFromOrigin()
        self.__address = tensor.address + \
            self.__local_offset * np.dtype(self.__dtype).itemsize

        if tensor.isQuantized():
            self.__quantization = tensor.quantization
        else:
            self.__quantization = None

    def getStridesAndSize(self):
        return self.__alternate_strides

    def getTopEncloserRecursive(self):
        return self.original_tensor.getTopEncloserRecursive()

    def can_flatten(self):
        """
        Return True if tensor can be tranformed into 1d tensor
        """
        if len(self.__subtensors) > 0:
            return False  # Subtensors not supported yet
        elem_size = np.dtype(self.original_tensor.getDatatype()).itemsize
        dim_size = np.prod(self.__dimensions) * elem_size
        max_stride_index = self.__strides.index(max(self.__strides))
        stride_size = self.__strides[max_stride_index] * \
            self.__dimensions[max_stride_index]
        assert(stride_size >= dim_size)
        return dim_size == stride_size

    def flatten(self):
        """
        Return 1d version of tensor
        """
        assert(self.can_flatten())
        ret = copy(self)

        new_data = self.__data.flatten()
        elem_size = np.dtype(self.original_tensor.getDatatype()).itemsize
        ret.__subtensors = []
        ret.__dimensions = [len(new_data)]
        ret.__strides = [elem_size]
        ret.__data = new_data
        return ret

    def __getitem__(self, key):
        return self.__subtensors[key] if key < len(
            self.__subtensors) and not self.__broadcast else self

    def broadcast_subtensor(self, key):
        return self.__subtensors[key] if key < len(self.__subtensors) else self

    @property
    def broadcast(self):
        return self.__broadcast

    # From here on is read-only decorations
    @property
    def original_tensor(self):
        return self.__original_tensor

    @property
    def name(self):
        return self.__name

    @property
    def dimensions(self):
        return self.__dimensions

    @property
    def size(self):
        return self.__size

    @property
    def strides(self):
        return self.__strides

    @property
    def layout(self):
        return self.__layout

    @property
    def dtype(self):
        return self.__dtype

    @property
    def address(self):
        return self.__address

    @property
    def sparse(self):
        return self.__sparse

    @property
    def quantization(self):
        return self.__quantization

    @property
    def data(self):
        return self.__data

    @property
    def compressed_kernel_sizes(self):
        if not self.sparse:
            raise ValueError("Dense tensor do not have compressed kernels")
        return self.__compressed_kernel_sizes

    @property
    def local_offset(self):
        return self.__local_offset

    @property
    def uniqueID(self):
        return self.__uniqueID

    @property
    def topID(self):
        return self.__topID

    @property
    def se(self):
        return self.__se

    @property
    def sparsity(self):
        return self.__sparsity

    @property
    def opaque(self):
        """
            A "transparent" resolved tensor is one that contains positional information only.
            An "opaque" tensor is one that has data inside it.
        """
        return self.__opaque

    @property
    def subtensors(self):
        return self.__subtensors
