import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../flatbuffers/python'))
import flatbuffers
import numpy as np

import copy

# TODO: import these nicer.
import MVCNN.Version
import MVCNN.Resources
import MVCNN.SummaryHeader
import MVCNN.TensorReference
import MVCNN.IndirectDataReference
import MVCNN.SourceStructure
import MVCNN.GraphFile
import MVCNN.TaskList
import MVCNN.Task
import MVCNN.NNDMATask
import MVCNN.UPADMATask
import MVCNN.NCE1Task
import MVCNN.NCE2Task
import MVCNN.MvTensorTask
import MVCNN.PPETask
import MVCNN.PPEFixedFunction
import MVCNN.ControllerTask
import MVCNN.NCEInvariantFields
import MVCNN.ControllerSubTask
import MVCNN.NCEVariantFields
import MVCNN.DPULayerType
import MVCNN.SpecificTask
import MVCNN.BinaryData
import MVCNN.Link
import MVCNN.BarrierReference
import MVCNN.Barrier
import MVCNN.BarrierConfigurationTask

from serializer_enums import *

from hostStructure import fakeDMALayer, fakeDPULayer, fakeNNDMALayer, fakeNCELayer
from hostStructure import fakeMvTensorLayer, fakeNNShvLayer, fakeControllerLayer, fakeCustomLayer


def FinishFileID(builder, rootTable, fid):
    """
        This is a temporary function to workaround a bug in the
        current release of flatbuffers. When it is resolved by
        the maintainers, this can be removed.

        - Writes some header fields that were missing including
        the magic number identfier

        @param builder - flat buffer builder object
        @param rootTable - flat buffer current object
        @param fid - magic number for identification, must be 4 chars and ascii encoded

    """
    N = flatbuffers.number_types
    encode = flatbuffers.encode
    flags = N.Uint8Flags
    prepSize = N.Uint8Flags.bytewidth * len(fid)
    builder.Prep(builder.minalign, prepSize + len(fid))
    for i in range(len(fid) - 1, -1, -1):
        builder.head = builder.head - flags.bytewidth
        encode.Write(flags.packer_type, builder.Bytes, builder.Head(), fid[i])
    builder.Finish(rootTable)


class FieldDependencyError(Exception):
    """
        Custom Exception.
        If this is thrown, you have called functions out-of-order
        and something required was uninitialized.
    """
    pass


class Serializer():
    def __init__(self):
        """
            @fbb - flatbuffer builder object
            Here we initialize fields that will be populated later on
            with None, to allow FieldDependencyExceptions to be ignored for debug
        """
        self.fbb = flatbuffers.Builder(0)
        self.version = None
        self.resources = None
        self.inputTensors = None
        self.outputTensors = None
        self.summaryHeader = None
        self.graphFile = None
        self.bin_data = None
        self.taskLists = None
        self.summaryHeader = None

    def setVersion(self, major, minor, patch, githash):
        """
            Version of the graph file.
            @param Major (uint) Major architectural change
            @param Minor (uint) Non-backwards compatible change
            @param Patch (uint) backwards comaptible change
            @param GitHash (string) A string to precisely identify generation time.
                Can be things other than a githash, such as a timestamp or overall
                bundle release string
        """

        s = self.fbb.CreateString(githash)

        MVCNN.Version.VersionStart(self.fbb)
        MVCNN.Version.VersionAddMajorV(self.fbb, major)
        MVCNN.Version.VersionAddMinorV(self.fbb, minor)
        MVCNN.Version.VersionAddPatchV(self.fbb, patch)
        MVCNN.Version.VersionAddHash(self.fbb, s)
        self.version = MVCNN.Version.VersionEnd(self.fbb)

    def setResources(self, shaveMask, nce1Mask, dpuMask, leonCMX, nnCMX, ddrScratch):
        """
            Resources Available to the device for inference.
            Note that here we provide 'limits' rather than exact specifications.
            The device itself is responsible for deciding where things go

            @param shaveMask - amount of shave vector processors to use
                (TBD difference between UPA and NN shaves)
            @param nce1Mask - amount of NCE1 processors to use (0/1/2). Only available for
                Myriad-X based systems
            @param dpuMask - amount of NCE2 processing elements to use. Only available for
                VPU3 based systems
            @param leonCMX - amount of UPA CMX memory to use
            @param nnCMX - amount of NN CMX memory to use
            @param ddrScratch - amount of DDR Scratch memory to use for intermediary memory
        """
        MVCNN.Resources.ResourcesStart(self.fbb)
        MVCNN.Resources.ResourcesAddShaveMask(self.fbb, shaveMask)  # TODO: Number, not mask
        MVCNN.Resources.ResourcesAddNce1Mask(self.fbb, nce1Mask)    # TODO: Number, not mask
        MVCNN.Resources.ResourcesAddDpuMask(self.fbb, dpuMask)      # ""
        MVCNN.Resources.ResourcesAddLeonCmx(self.fbb, leonCMX)
        MVCNN.Resources.ResourcesAddNnCmx(self.fbb, nnCMX)
        MVCNN.Resources.ResourcesAddDdrScratch(self.fbb, ddrScratch)
        self.resources = MVCNN.Resources.ResourcesEnd(self.fbb)

    def setInputTensors(self, arr):
        """
            set the descriptions of input tensors to a graph.
            @param arr - a list containing SerialTensor objects.
        """
        st_arr = []
        for x in arr:
            st_arr.append(x._fb(self.fbb))

        sz = len(st_arr)
        MVCNN.SummaryHeader.SummaryHeaderStartNetInputVector(self.fbb, sz)
        for x in st_arr:
            self.fbb.PrependUOffsetTRelative(x)
        self.inputTensors = self.fbb.EndVector(sz)

    def setOutputTensors(self, arr):
        """
            set the descriptions of output tensors to a graph.
            @param arr - a list containing SerialTensor objects.
        """
        st_arr = []
        for x in arr:
            st_arr.append(x._fb(self.fbb))

        sz = len(st_arr)
        MVCNN.SummaryHeader.SummaryHeaderStartNetOutputVector(self.fbb, sz)
        for x in st_arr:
            self.fbb.PrependUOffsetTRelative(x)
        self.outputTensors = self.fbb.EndVector(sz)

    def setSummaryHeader(
        self,
        taskAmount=0,

        layerAmount=0
    ):
        """
            set the summary header section of the graph.
            Requires some values to be pre-calculated by other serializer
            code to function. Otherwise will throw custom exception.

            @param taskAmount - the amount of tasks in the graph
                (optional as to be eventually internalized)
            @param layerAmount - the amount of layers in the graph
                (optional as to be eventually internalized)
        """
        if self.version is None or \
            self.inputTensors is None or \
            self.outputTensors is None or \
            self.resources is None or \
            self.sourceStructure is None:
            raise FieldDependencyError()

        MVCNN.SummaryHeader.SummaryHeaderStart(self.fbb)
        MVCNN.SummaryHeader.SummaryHeaderAddVersion(self.fbb, self.version)
        MVCNN.SummaryHeader.SummaryHeaderAddNetInput(self.fbb, self.inputTensors)
        MVCNN.SummaryHeader.SummaryHeaderAddNetOutput(self.fbb, self.outputTensors)
        MVCNN.SummaryHeader.SummaryHeaderAddTaskCount(self.fbb, taskAmount)
        MVCNN.SummaryHeader.SummaryHeaderAddLayerCount(self.fbb, layerAmount)
        MVCNN.SummaryHeader.SummaryHeaderAddResources(self.fbb, self.resources)
        MVCNN.SummaryHeader.SummaryHeaderAddOriginalStructure(self.fbb, self.sourceStructure)
        self.summaryHeader = MVCNN.SummaryHeader.SummaryHeaderEnd(self.fbb)

    def getSourceStructure(self):
        """
            Returns an object containing the original structure of a graph.
            This can be passed to some other serialization functions requiring
            this information
        """
        return self.reference_structure

    def setSourceStructure(self, pairs):
        """
            sets up the original structure definition inside the serializer.

            @param pairs - an array of pairs, each pair referring to a dependency relationship.
                These do not necessarily need to be in order.
                A node cannot be in the graph without being at least once in these pairs.
        """

        # Get Unique nodes
        nodes = set()
        for a, b in pairs:
            nodes.add(a)
            nodes.add(b)
        nodes = list(nodes)

        self.reference_structure = nodes

        links = []

        for n in nodes:
            s = self.fbb.CreateString(n)
            sinks = []
            sources = []
            for p1, p2 in pairs:
                if p1 == p2:
                    assert 0, "Self-reference"
                if n == p2:
                    sinks.append(nodes.index(p1))
                if n == p1:
                    sources.append(nodes.index(p2))

            sz = len(sinks)
            MVCNN.Link.LinkStartSinkIDVector(self.fbb, sz)
            for x in sinks:
                self.fbb.PrependUint32(x)
            b_sinks = self.fbb.EndVector(sz)


            sz = len(sources)
            MVCNN.Link.LinkStartSourceIDVector(self.fbb, sz)
            for x in sources:
                self.fbb.PrependUint32(x)
            b_srcs = self.fbb.EndVector(sz)


            MVCNN.Link.LinkStart(self.fbb)
            MVCNN.Link.LinkAddName(self.fbb, s)
            MVCNN.Link.LinkAddThisID(self.fbb, nodes.index(n))
            MVCNN.Link.LinkAddSinkID(self.fbb, b_sinks)
            MVCNN.Link.LinkAddSourceID(self.fbb, b_srcs)
            ll = MVCNN.Link.LinkEnd(self.fbb)
            links.append(ll)


        sz = len(links)
        MVCNN.SourceStructure.SourceStructureStartLinksVector(self.fbb, sz)
        for x in links:
            self.fbb.PrependUOffsetTRelative(x)
        b_links = self.fbb.EndVector(sz)

        MVCNN.SourceStructure.SourceStructureStart(self.fbb)
        MVCNN.SourceStructure.SourceStructureAddFirstID(self.fbb, 0)
        MVCNN.SourceStructure.SourceStructureAddLinks(self.fbb, b_links)

        self.sourceStructure = MVCNN.SourceStructure.SourceStructureEnd(self.fbb)


    def setTaskLists(self, arr):
        """
            attaches the serialTaskLists provided into the serializer.

            @param arr - an array containing serialTaskLists.
        """
        st_arr = []
        for x in arr:
            st_arr.append(x._fb(self.fbb))

        sz = len(st_arr)
        MVCNN.GraphFile.GraphFileStartTaskListsVector(self.fbb, sz)
        for x in st_arr:
            self.fbb.PrependUOffsetTRelative(x)
        self.taskLists = self.fbb.EndVector(sz)

    def setBinaryData(self, arr):
        """
            Sets up the binary data for the graph file.
            Assumes that the provided data has been pre-aligned and other requirements fufilled.
            Ensures that the base pointer will be aligned to 64. (TODO: Confirm)

            @param arr - a list of numpy arrays, which can be different data types.
                Currently supported Fp16, uint8. More to come... :)
        """
        self.bin_data = []
        self.binary_data_vec = None

        for a in arr:

            data_fp64 = data_fp32 = data_fp16 = data_fp8 = \
            data_i64 = data_i32 = data_i16 = data_i8 = \
            data_u64 = data_u32 = data_u16 = data_u8 = \
            data_i4x = data_bin = data_log = -1

            # Floating Point
            if(a.dtype == np.float64):
                MVCNN.BinaryData.BinaryDataStartFp64Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.float64)
                    self.fbb.PrependFloat64(int_x)
                data_fp64 = self.fbb.EndVector(len(a))

            elif(a.dtype == np.float32):
                MVCNN.BinaryData.BinaryDataStartFp32Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.float32)
                    self.fbb.PrependFloat32(int_x)
                data_fp32 = self.fbb.EndVector(len(a))

            elif(a.dtype == np.float16):
                MVCNN.BinaryData.BinaryDataStartFp16Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.uint16)
                    self.fbb.PrependUint16(int_x)
                data_fp16 = self.fbb.EndVector(len(a))

            # TBD:  https://github.com/movidius/Fathom/issues/52
            # elif(a.dtype == np.float8):
            #     MVCNN.BinaryData.BinaryDataStartFp16Vector(self.fbb, len(a))
            #     for x in reversed(a):
            #         int_x = x.view(dtype=np.uint16)
            #         self.fbb.PrependUint16(int_x)
            #     data_fp16 = self.fbb.EndVector(len(a))

            # Signed Int

            elif(a.dtype == np.int64):
                MVCNN.BinaryData.BinaryDataStartI64Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.int64)
                    self.fbb.PrependInt64(int_x)
                data_i64 = self.fbb.EndVector(len(a))

            elif(a.dtype == np.int32):
                MVCNN.BinaryData.BinaryDataStartI32Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.int32)
                    self.fbb.PrependInt32(int_x)
                data_i32 = self.fbb.EndVector(len(a))

            elif(a.dtype == np.int16):
                MVCNN.BinaryData.BinaryDataStartI16Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.int16)
                    self.fbb.PrependInt16(int_x)
                data_i16 = self.fbb.EndVector(len(a))

            elif(a.dtype == np.int8):
                MVCNN.BinaryData.BinaryDataStartI8Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.int8)
                    self.fbb.PrependInt8(int_x)
                data_i8 = self.fbb.EndVector(len(a))

            # Unsigned Int

            elif(a.dtype == np.uint64):
                MVCNN.BinaryData.BinaryDataStartU64Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.uint64)
                    self.fbb.PrependUint64(int_x)
                data_u64 = self.fbb.EndVector(len(a))

            elif(a.dtype == np.uint32):
                MVCNN.BinaryData.BinaryDataStartU32Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.uint32)
                    self.fbb.PrependUint32(int_x)
                data_u32 = self.fbb.EndVector(len(a))

            elif(a.dtype == np.uint16):
                MVCNN.BinaryData.BinaryDataStartU16Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.uint16)
                    self.fbb.PrependUint16(int_x)
                data_u16 = self.fbb.EndVector(len(a))

            elif(a.dtype == np.uint8):
                MVCNN.BinaryData.BinaryDataStartU8Vector(self.fbb, len(a))
                for x in reversed(a):
                    int_x = x.view(dtype=np.uint8)
                    self.fbb.PrependUint8(int_x)
                data_u8 = self.fbb.EndVector(len(a))

            # Special Data Types

            # TBD:  https://github.com/movidius/Fathom/issues/52
            # elif(a.dtype == np.uint8):
            #     MVCNN.BinaryData.BinaryDataStartI4xVectorVector(self.fbb, len(a))
            #     for x in reversed(a):
            #         int_x = x.view(dtype=np.uint8)
            #         self.fbb.PrependUint8(int_x)
            #     data_u8 = self.fbb.EndVector(len(a))

            # elif(a.dtype == np.uint8):
            #     MVCNN.BinaryData.BinaryDataStartBinVector(self.fbb, len(a))
            #     for x in reversed(a):
            #         int_x = x.view(dtype=np.uint8)
            #         self.fbb.PrependUint8(int_x)
            #     data_u8 = self.fbb.EndVector(len(a))

            # elif(a.dtype == np.uint8):
            #     MVCNN.BinaryData.BinaryDataStartLogVector(self.fbb, len(a))
            #     for x in reversed(a):
            #         int_x = x.view(dtype=np.uint8)
            #         self.fbb.PrependUint8(int_x)
            #     data_u8 = self.fbb.EndVector(len(a))

            else:
                assert 0, "Not implemented" + type(a)

            MVCNN.BinaryData.BinaryDataStart(self.fbb)
            if data_fp64 != -1:
                # print("+fp64")
                MVCNN.BinaryData.BinaryDataAddFp64(self.fbb, data_fp64)
            elif data_fp32 != -1:
                # print("+fp32")
                MVCNN.BinaryData.BinaryDataAddFp32(self.fbb, data_fp32)
            elif data_fp16 != -1:
                # print("+fp16")
                MVCNN.BinaryData.BinaryDataAddFp16(self.fbb, data_fp16)
            elif data_fp8 != -1:
                # print("+fp8")
                MVCNN.BinaryData.BinaryDataAddFp8(self.fbb, data_fp8)

            elif data_i64 != -1:
                MVCNN.BinaryData.BinaryDataAddI64(self.fbb, data_i64)
            elif data_i32 != -1:
                MVCNN.BinaryData.BinaryDataAddI32(self.fbb, data_i32)
            elif data_i16 != -1:
                MVCNN.BinaryData.BinaryDataAddI16(self.fbb, data_i16)
            elif data_i8 != -1:
                # print("+i8")
                MVCNN.BinaryData.BinaryDataAddI8(self.fbb, data_i8)

            elif data_u64 != -1:
                MVCNN.BinaryData.BinaryDataAddU64(self.fbb, data_u64)
            elif data_u32 != -1:
                MVCNN.BinaryData.BinaryDataAddU32(self.fbb, data_u32)
            elif data_u16 != -1:
                MVCNN.BinaryData.BinaryDataAddU16(self.fbb, data_u16)
            elif data_u8 != -1:
                # print("+u8")
                MVCNN.BinaryData.BinaryDataAddU8(self.fbb, data_u8)

            elif data_i4x != -1:
                MVCNN.BinaryData.BinaryDataAddI4x(self.fbb, data_i4x)
            elif data_bin != -1:
                MVCNN.BinaryData.BinaryDataAddBin(self.fbb, data_bin)
            elif data_log != -1:
                MVCNN.BinaryData.BinaryDataAddLog(self.fbb, data_log)
            else:
                print("Warning: Tensor Not Serialized - Datatype failure")


            v = MVCNN.BinaryData.BinaryDataEnd(self.fbb)
            self.bin_data.append(v)

        sz = len(self.bin_data)
        MVCNN.GraphFile.GraphFileStartBinaryDataVector(self.fbb, sz)
        for x in reversed(self.bin_data):
            self.fbb.PrependUOffsetTRelative(x)
        self.binary_data_vec = self.fbb.EndVector(sz)

    def setBarrierTable(self, barriers):
        """
            Adds Barrier Table to the graphfile.

            @param barriers - a list of barriers to be written into this section.
                Order is assumed to be correct w.r.t other references (such as in Tasks).
        """


        b_arr = []
        for x in barriers:
            b_arr.append(x._fb(self.fbb))

        sz = len(b_arr)
        MVCNN.GraphFile.GraphFileStartBarrierTableVector(self.fbb, sz)
        for x in reversed(b_arr):
            self.fbb.PrependUOffsetTRelative(x)

        self.barrier_table = self.fbb.EndVector(sz)


    def finish(self, allow_missing_sections=False):
        """
            Completes the building of the graphfile, by instantiating the final
            outer 'graphfile' container.
            @param allow_missing_sections - do not error when a field is unpopulated
        """

        if allow_missing_sections is False:
            if self.binary_data_vec is None or \
                    self.summaryHeader is None or \
                    self.taskLists is None:
                raise FieldDependencyError()

        MVCNN.GraphFile.GraphFileStart(self.fbb)

        if self.summaryHeader is not None:
            MVCNN.GraphFile.GraphFileAddHeader(self.fbb, self.summaryHeader)

        if self.taskLists is not None:
            MVCNN.GraphFile.GraphFileAddTaskLists(self.fbb, self.taskLists)

        if self.binary_data_vec is not None:
            MVCNN.GraphFile.GraphFileAddBinaryData(self.fbb, self.binary_data_vec)

        if self.barrier_table is not None:
            MVCNN.GraphFile.GraphFileAddBarrierTable(self.fbb, self.barrier_table)

        self.graphFile = MVCNN.GraphFile.GraphFileEnd(self.fbb)


    def toFile(self, file_name):
        """
            Writes a completed graphfile to the file system.
        """

        if self.graphFile is None:
            raise FieldDependencyError()

        FinishFileID(self.fbb, self.graphFile, "BLOB".encode())

        buf = self.fbb.Output()

        with open(file_name, 'wb') as output:
            output.write(buf)

    def getBuilder(self):
        """
            *** advanced debug option only. ***
            gets the flat buffer builder object out of the
            serializing code for messing around with
        """
        return self.fbb

    def setBuilder(self, fbb):
        """
            *** advanced debug option only. ***
            force sets the flat buffer builder object after
            being manipulated externally.
        """
        self.fbb = fbb


class SerialTensor():
    """
        Helper class for tensors
    """
    def __init__(self,
                 dims,
                 strides,
                 sparsity_map_offset = 0,
                 data_offset = 0,
                 location = MemoryLocation.NULL.value,
                 data_dtype = DType.NOT_SET.value
                 ):
        """
            @param dims - list of dimensions (supports up to N)
                Dimensions should be invariant of layout
            @param strides - list of strides (supports up to N)
                For coverage, we include strides starting at the datatype level
                and up to if a second identical tensor would be present.
                i.e. for N dims, we should have N+1 strides.
                These are variant to layout and while each field corresponds to
                the similarly indexed dimension, they are not nessicarily always
                incrementing or decrementing.
                Through strides, one can figure out layout/ordering.
            @param data_offset - The offset at which one can find the data related to
                this Tensor. The serializer does not provide assisting functions for this
                field's calculation
            @param sparsity_map_offset - similar to the field above, this provides the
                offset required to find the Tensor's sparsity map
            @data_dtype - type of the data stored in tensor
        """
        self.dims = dims
        self.strides = strides
        self.sparsity_map = sparsity_map_offset
        self.data_offset = data_offset
        self.location = location
        self.data_dtype = data_dtype

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        MVCNN.IndirectDataReference.IndirectDataReferenceStart(fbb)
        if (self.sparsity_map is not None):
            MVCNN.IndirectDataReference.IndirectDataReferenceAddSparsityIndex(fbb, self.sparsity_map)
        MVCNN.IndirectDataReference.IndirectDataReferenceAddDataIndex(fbb, self.data_offset)
        idr = MVCNN.IndirectDataReference.IndirectDataReferenceEnd(fbb)

        MVCNN.TensorReference.TensorReferenceStartDimensionsVector(fbb, len(self.dims))
        for x in self.dims:
            fbb.PrependUint32(x)
        dd = fbb.EndVector(len(self.dims))

        MVCNN.TensorReference.TensorReferenceStartStridesVector(fbb, len(self.strides))
        for x in self.strides:
            fbb.PrependUint32(x)
        ss = fbb.EndVector(len(self.strides))

        MVCNN.TensorReference.TensorReferenceStart(fbb)
        MVCNN.TensorReference.TensorReferenceAddDimensions(fbb, dd)
        MVCNN.TensorReference.TensorReferenceAddStrides(fbb, ss)
        MVCNN.TensorReference.TensorReferenceAddLeadingOffset(fbb, 0)
        MVCNN.TensorReference.TensorReferenceAddTrailingOffset(fbb, 0)
        MVCNN.TensorReference.TensorReferenceAddData(fbb, idr)
        MVCNN.TensorReference.TensorReferenceAddLocale(fbb, self.location)
        MVCNN.TensorReference.TensorReferenceAddDataDtype(fbb, self.data_dtype)
        return MVCNN.TensorReference.TensorReferenceEnd(fbb)


class serialPPETask():
    """
        Helper for Serializing a PPE Generic Task
    """
    def __init__(
        self,
        ops,
        ClampHi=9999,
        ClampLo=0,
        PreLuAlpha=None,
        scale_data=None
    ):
        """
            @param ops - list of post op types,
            @param ClampHi
            @param ClampLo
            @param PreLuAlpha,
            @param scale_data
        """
        self.ops = ops
        self.ClampHi = ClampHi
        self.ClampLo = ClampLo
        self.PreLuAlpha = PreLuAlpha
        self.scale_data = scale_data

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        # ops vector
        sz = len(self.ops)
        MVCNN.PPEFixedFunction.PPEFixedFunctionStartOpsVector(fbb, sz)
        for op in self.ops:
            fbb.PrependUint8(op)
        ppe_ops_vec = fbb.EndVector(sz)

        # TODO: Only supporting one entry at the moment
        MVCNN.PPEFixedFunction.PPEFixedFunctionStart(fbb)
        MVCNN.PPEFixedFunction.PPEFixedFunctionAddOps(fbb, ppe_ops_vec)
        MVCNN.PPEFixedFunction.PPEFixedFunctionAddClampHigh(fbb, self.ClampHi)
        MVCNN.PPEFixedFunction.PPEFixedFunctionAddClampLow(fbb, self.ClampLo)
        f = MVCNN.PPEFixedFunction.PPEFixedFunctionEnd(fbb)
        # MVCNN.PPEFixedFunction.PPEFixedFunctionAddPreLuAlpha(fbb, self.PreLuAlpha)
        f = [f]

        sz = len(f)
        MVCNN.Task.TaskStartSourceTaskIDsVector(fbb, sz)
        for fx in f:
            fbb.PrependUOffsetTRelative(fx)
        ref_vec = fbb.EndVector(sz)


        scale_serialized = None
        if self.scale_data is not None:
            scale_serialized = self.scale_data._fb(fbb)

        MVCNN.PPETask.PPETaskStart(fbb)
        if scale_serialized is not None:
            MVCNN.PPETask.PPETaskAddScaleData(fbb, scale_serialized)
        MVCNN.PPETask.PPETaskAddFixedFunction(fbb, ref_vec)
        ppe_generic_task = MVCNN.PPETask.PPETaskEnd(fbb)

        return ppe_generic_task


class serialNCE2Task():
    """
        Helper for Serializing a DPU Task
    """
    def __init__(
        self,
        op,
        clusterID,
        kH,
        kW,
        kStrideH,
        kStrideW,
        padTop,
        padBottom,
        padLeft,
        padRight,
        in_data,
        out_data,
        param_data,
        bias_data,
        mpe,
        ppe_param,
        dpuID,
        oXs,
        oYs,
        oZs
    ):
        """
            Invariant fields are those which do not change within a layer set
            of DPU tasks.

            Variant fields may change from NCE2Task to NCE2Task

            Invariant:
                @param op - op type for DPE,
                @param clusterID - cluster identification (not directly mapped),
                @param inWidth - input Dimension,
                @param inHeight - input Dimension,
                @param inChan - input Dimension,
                @param outChan - output Dimension,
                @param kH - kernel size,
                @param kW - kernel size,
                @param kStrideH - kernel stride height value,
                @param kStrideW - kernel stride width value,
                @param padTop - padding over height,
                @param padBottom - padding over height,
                @param padLeft - padding over width,
                @param padRight - padding over width,
                @param in_data - SerialTensor object for this data,
                @param out_data - SerialTensor object for this data,
                @param param_data - SerialTensor object for this data,
                @param mpe - mode for DPU operation,
                @param ppe_param - serialPPETask,
            Variant:
                @param dpuID - an ID for DPU (not directly mapped),
                @param oXs - start location of output on the X axis ,
                @param oYs - start location of output on the Y axis ,
                @param oZs - start location of output on the Z axis ,
        """
        self.op = op
        self.clusterID = clusterID
        self.kH = kH
        self.kW = kW
        self.kStrideH = kStrideH
        self.kStrideW = kStrideW
        self.padTop = padTop
        self.padBottom = padBottom
        self.padLeft = padLeft
        self.padRight = padRight
        self.in_data = in_data
        self.out_data = out_data
        self.param_data = param_data
        self.bias_data = bias_data
        self.mpe = mpe
        self.ppe_param = ppe_param
        self.dpuID = dpuID
        self.oXs = oXs
        self.oYs = oYs
        self.oZs = oZs

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        it = self.in_data._fb(fbb)
        ot = self.out_data._fb(fbb)
        pt = None
        bd = None
        #some layers do not have weighs or bias
        if (self.param_data):
            pt = self.param_data._fb(fbb)
        if (self.bias_data):
            bd = self.bias_data._fb(fbb)
        ppe_task_serial = self.ppe_param._fb(fbb)

        MVCNN.NCEInvariantFields.NCEInvariantFieldsStart(fbb)
        MVCNN.NCEInvariantFields.NCEInvariantFieldsAddDpuTaskType(fbb, self.op)
        MVCNN.NCEInvariantFields.NCEInvariantFieldsAddKernelH(fbb, self.kH)
        MVCNN.NCEInvariantFields.NCEInvariantFieldsAddKernelW(fbb, self.kW)
        MVCNN.NCEInvariantFields.NCEInvariantFieldsAddKernelStrideH(fbb, self.kStrideH)
        MVCNN.NCEInvariantFields.NCEInvariantFieldsAddKernelStrideW(fbb, self.kStrideH)
        MVCNN.NCEInvariantFields.NCEInvariantFieldsAddInputData(fbb, it)
        MVCNN.NCEInvariantFields.NCEInvariantFieldsAddOutputData(fbb, ot)
        if pt is not None:
            MVCNN.NCEInvariantFields.NCEInvariantFieldsAddWeightsData(fbb, pt)
        if bd is not None:
            MVCNN.NCEInvariantFields.NCEInvariantFieldsAddBiasData(fbb, bd)

        MVCNN.NCEInvariantFields.NCEInvariantFieldsAddPpeTask(fbb, ppe_task_serial)
        iv = MVCNN.NCEInvariantFields.NCEInvariantFieldsEnd(fbb)

        v = []
        for x in range(len(self.dpuID)):
            MVCNN.NCEVariantFields.NCEVariantFieldsStart(fbb)

            MVCNN.NCEVariantFields.NCEVariantFieldsAddClusterID(fbb, self.clusterID)
            MVCNN.NCEVariantFields.NCEVariantFieldsAddWorkloadID(fbb, self.dpuID[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddMpeMode(fbb, self.mpe[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddPadTop(fbb, self.padTop[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddPadBottom(fbb, self.padBottom[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddPadLeft(fbb, self.padLeft[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddPadRight(fbb, self.padRight[x])

            MVCNN.NCEVariantFields.NCEVariantFieldsAddWorkloadStartX(fbb, self.oXs[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddWorkloadStartY(fbb, self.oYs[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddWorkloadStartZ(fbb, self.oZs[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddWorkloadEndX(fbb, self.oXs[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddWorkloadEndY(fbb, self.oYs[x])
            MVCNN.NCEVariantFields.NCEVariantFieldsAddWorkloadEndZ(fbb, self.oZs[x])
            v.append(MVCNN.NCEVariantFields.NCEVariantFieldsEnd(fbb))

        sz = len(v)
        MVCNN.NCE2Task.NCE2TaskStartVariantVector(fbb, sz)
        for i in v:
            fbb.PrependUOffsetTRelative(i)
        v_vec = fbb.EndVector(sz)

        MVCNN.NCE2Task.NCE2TaskStart(fbb)
        MVCNN.NCE2Task.NCE2TaskAddInvariant(fbb, iv)
        MVCNN.NCE2Task.NCE2TaskAddVariant(fbb, v_vec)

        task = MVCNN.NCE2Task.NCE2TaskEnd(fbb)

        return task

class serialBarrierConfigurationTask():
    def __init__(self, serialBarrier):
        self.sb = serialBarrier
        # self.associated_barrer = associated_barrer

    def _fb(self, fbb):

        ref = self.sb._fb(fbb)

        MVCNN.BarrierConfigurationTask.BarrierConfigurationTaskStart(fbb)
        MVCNN.BarrierConfigurationTask.BarrierConfigurationTaskAddTarget(fbb, ref)
        return MVCNN.BarrierConfigurationTask.BarrierConfigurationTaskEnd(fbb)

class serialBarrier():
    def __init__(self, consumer_count, producer_count, manual_barrier_id=None):
        """
            @param consumer_count - Amount of tasks that rely on this barrier to be freed
                so that they can execute
            @param producer_count - Amount of tasks that this barrier is waiting to be
                complete before unlocking
            @param manual_barrier_id - An optional field to specify an identifier overriding the default
                assignment in the device.
        """
        self.manual_barrier_id = manual_barrier_id
        self.consumer_count = consumer_count
        self.producer_count = producer_count

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        MVCNN.Barrier.BarrierStart(fbb)
        if self.manual_barrier_id is not None:
            MVCNN.Barrier.BarrierAddBarrierId(fbb, self.manual_barrier_id)

        MVCNN.Barrier.BarrierAddConsumerCount(fbb, self.consumer_count)
        MVCNN.Barrier.BarrierAddProducerCount(fbb, self.producer_count)
        return MVCNN.Barrier.BarrierEnd(fbb)


class serialBarrierReference():
    """
        Helper class for barrier objects
    """
    def __init__(self, barrier_to_wait_on, barriers_to_update):
        """
            @param barrier_to_wait_on - The barrier index (singular) that the parent task will wait upon
                before freeing itself. Provided value None for no waiting barrier reference
            @param barriers_to_update - The barrier indexes that the parent task will signal to
                when it has been freed. Provide empty list to indicate no consumers
        """
        if barrier_to_wait_on is not None:
            self.barrier_to_wait_on = barrier_to_wait_on + 1
        else:
            self.barrier_to_wait_on = 0

        self.barriers_to_update = barriers_to_update


class serialTask():
    """
        Helper class for generic Task objects (need a child object)
    """
    def __init__(
        self,
        thisID,
        referenceIDs,
        childTask,
        referenceTable,
        barrierInfo
    ):
        """
            @param thisID - A unique id for this task
            @param referenceIDs - a list of the ids of the original
                layers this task came from
            @param childTask - The specific task to perform (consider
                this similar to a subclass)
            @referenceTable - an object containing the original structure
                (used to lookup the aforementioned referenceIDs )
            @barrierInfo - a serialBarrierReference containing information about
                barrier waiting and updating
        """
        self.thisID = thisID
        self.referenceIDs = referenceIDs
        self.childTask = childTask
        self.referenceTable = referenceTable
        self.barrierInfo = barrierInfo

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        childserial = self.childTask._fb(fbb)

        sz = len(self.referenceIDs)
        MVCNN.Task.TaskStartSourceTaskIDsVector(fbb, sz)
        for ref in self.referenceIDs:
            entry = self.referenceTable.index(ref)
            fbb.PrependUOffsetTRelative(entry)
        ref_vec = fbb.EndVector(sz)

        szC = len(self.barrierInfo.barriers_to_update)
        if szC > 0:
            MVCNN.BarrierReference.BarrierReferenceStartUpdateBarriersVector(fbb, szC)
            for x in range(szC):
                fbb.PrependUint32(self.barrierInfo.barriers_to_update[x])
            cons = fbb.EndVector(szC)
        else:
            cons = 0

        MVCNN.BarrierReference.BarrierReferenceStart(fbb)
        MVCNN.BarrierReference.BarrierReferenceAddWaitBarrier(fbb, self.barrierInfo.barrier_to_wait_on)
        MVCNN.BarrierReference.BarrierReferenceAddUpdateBarriers(fbb, cons)
        dep = MVCNN.BarrierReference.BarrierReferenceEnd(fbb)

        MVCNN.Task.TaskStart(fbb)
        MVCNN.Task.TaskAddSourceTaskIDs(fbb, ref_vec)
        MVCNN.Task.TaskAddAssociatedBarriers(fbb, dep)

        if isinstance(self.childTask, serialNCE2Task):
            MVCNN.Task.TaskAddTaskType(fbb, MVCNN.SpecificTask.SpecificTask().NCE2Task)
        elif isinstance(self.childTask, serialSoftwareTask):
            MVCNN.Task.TaskAddTaskType(fbb, MVCNN.SpecificTask.SpecificTask().MvTensorTask)
        elif isinstance(self.childTask, serialNNDMATask):
            MVCNN.Task.TaskAddTaskType(fbb, MVCNN.SpecificTask.SpecificTask().NNDMATask)
        elif isinstance(self.childTask, serialControllerTask):
            MVCNN.Task.TaskAddTaskType(fbb, MVCNN.SpecificTask.SpecificTask().ControllerTask)
        else:
            assert False, "Invalid task type"

        MVCNN.Task.TaskAddTask(fbb, childserial)
        return MVCNN.Task.TaskEnd(fbb)


class serialNCE1Task():
    """
        Helper class for NCE1 Tasks
    """
    def __init__(self):
        pass

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        MVCNN.NCE1Task.NCE1TaskStart(fbb)
        return MVCNN.NCE1Task.NCE1TaskEnd(fbb)


class serialSoftwareTask():
    """
        Helper Class for software Tasks
        TODO: Possibly distinguish between UPA and NN software tasks
    """
    def __init__(self, **kwargs):
        # Because layers dont nessicarily share properties,
        # parse attrs generically
        self.__dict__.update(kwargs)

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """

        try:
            v = self.softwareEnum
        except:
            print("serialization not supported. MvTensorID not present")
            raise Exception

        spec = None

        if v == 0:  # Conv2D

            it = self.input._fb(fbb)
            ot = self.output._fb(fbb)
            pt = self.params._fb(fbb)
            bt = self.bias._fb(fbb)

            MVCNN.Conv2D.Conv2DStart(fbb)
            MVCNN.Conv2D.Conv2DAddRadixX(fbb, self.radixX)
            MVCNN.Conv2D.Conv2DAddRadixY(fbb, self.radixY)
            MVCNN.Conv2D.Conv2DAddStrideX(fbb, self.radixStrideX)
            MVCNN.Conv2D.Conv2DAddStrideY(fbb, self.radixStrideY)
            MVCNN.Conv2D.Conv2DAddPadX(fbb, self.padX)
            MVCNN.Conv2D.Conv2DAddPadY(fbb, self.padY)
            MVCNN.Conv2D.Conv2DAddPadStyle(fbb, self.padStyle)
            MVCNN.Conv2D.Conv2DAddDilation(fbb, self.dilation)
            MVCNN.Conv2D.Conv2DAddInput(fbb, it)
            MVCNN.Conv2D.Conv2DAddOutput(fbb, ot)
            MVCNN.Conv2D.Conv2DAddWeight(fbb, pt)
            MVCNN.Conv2D.Conv2DAddBias(fbb, bt)
            spec = MVCNN.Conv2D.Conv2DEnd(fbb)

        elif v == 5:
            it = self.input._fb(fbb)
            ot = self.output._fb(fbb)

            MVCNN.Passthrough.PassthroughStart(fbb)
            MVCNN.Passthrough.PassthroughAddInput(fbb, it)
            MVCNN.Passthrough.PassthroughAddOutput(fbb, ot)
            spec = MVCNN.Passthrough.PassthroughEnd(fbb)
        else:
            print("no serial for software id", v)

        MVCNN.MvTensorTask.MvTensorTaskStart(fbb)
        if (v == 0):
            MVCNN.MvTensorTask.MvTensorTaskAddLayerType(fbb, MVCNN.SoftwareLayer.SoftwareLayer().Conv2D)
        if (v == 5):
            MVCNN.MvTensorTask.MvTensorTaskAddLayerType(fbb, MVCNN.SoftwareLayer.SoftwareLayer().Passthrough)
        MVCNN.MvTensorTask.MvTensorTaskAddLayer(fbb, spec)
        return MVCNN.MvTensorTask.MvTensorTaskEnd(fbb)


class serialControllerTask():
    """
        Helper class for Leon NN Controller.
    """
    def __init__(self, subtask):
        """
            @param subtask - The specific task to perform
        """
        self.subtask = subtask

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        childserial = self.subtask._fb(fbb)

        MVCNN.ControllerTask.ControllerTaskStart(fbb)

        if isinstance(self.subtask, serialBarrierConfigurationTask):
            MVCNN.ControllerTask.ControllerTaskAddTaskType(fbb, MVCNN.ControllerSubTask.ControllerSubTask().BarrierConfigurationTask)
        else:
            assert False, "Invalid subtask type"

        MVCNN.ControllerTask.ControllerTaskAddTask(fbb, childserial)
        return MVCNN.ControllerTask.ControllerTaskEnd(fbb)

class serialBarrierConfigurationTask():
    def __init__(self, barrier):
        """
            @param barrier - barrier to update
        """
        self.barrier = barrier

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        serialbarrier = self.barrier._fb(fbb)

        MVCNN.BarrierConfigurationTask.BarrierConfigurationTaskStart(fbb)
        MVCNN.BarrierConfigurationTask.BarrierConfigurationTaskAddTarget(fbb, serialbarrier)
        return MVCNN.BarrierConfigurationTask.BarrierConfigurationTaskEnd(fbb)


class serialNNDMATask():
    """
        Helper Class for DMA Tasks
        TODO: Have one for NN, one for UPA
    """
    def __init__(self, src, dst):
        """
            @param src - where to read from (offset)
            @param dst - where to write (offset)
            @param stride - how much data to skip (placeholder field)
        """
        self.src = src


        # MVCNN.BinaryData.BinaryDataStartFp16Vector(self.fbb, len(a))
        #         for x in reversed(a):
        #             int_x = x.view(dtype=np.uint16)
        #             self.fbb.PrependUint16(int_x)
        #         data_fp16 = self.fbb.EndVector(len(a))


        self.dst = dst

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """
        s = self.src._fb(fbb)

        db = []
        for d in self.dst:
            db.append(d._fb(fbb))

        sz = len(db)
        MVCNN.NNDMATask.NNDMATaskStartDstVector(fbb, sz)
        for x in db:
            fbb.PrependUOffsetTRelative(x)
        content = fbb.EndVector(sz)

        MVCNN.NNDMATask.NNDMATaskStart(fbb)
        MVCNN.NNDMATask.NNDMATaskAddSrc(fbb, s)
        MVCNN.NNDMATask.NNDMATaskAddDst(fbb, content)
        return MVCNN.NNDMATask.NNDMATaskEnd(fbb)


class serialUPADMATask():
    """
        Helper Class for DMA Tasks
        TODO: Have one for NN, one for UPA
    """
    def __init__(self, src, dst):
        """
            @param src - where to read from (offset)
            @param dst - where to write (offset)
            @param stride - how much data to skip (placeholder field)
        """
        self.src = src
        self.dst = dst

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """

        s = self.src._fb(fbb)
        d = self.dst._fb(fbb)

        MVCNN.UPADMATask.UPADMATaskStart(fbb)
        MVCNN.UPADMATask.UPADMATaskAddSrc(fbb, s)
        MVCNN.UPADMATask.UPADMATaskAddDst(fbb, d)
        return MVCNN.UPADMATask.UPADMATaskEnd(fbb)


def calc_strides_zyx(data):
    return \
        [
            1,
            data.x,
            data.x * data.y,
            data.x * data.y * data.z
        ]

def choose_task(task):
    """
        This function is a mapping from the fake guest classes we have
        in the test suite to the serializer task classes.
        It serves as a good example on how you might map your own
        classes to the serial ones.
    """
    if(isinstance(task, fakeDPULayer)):

        bias_data = SerialTensor([1], [1],
                                 data_offset=task.param_data.offset,
                                 location=MemoryLocation.GraphFile.value)
        scale_data = SerialTensor([1], [1],
                                  data_offset=task.param_data.offset,
                                  location=MemoryLocation.GraphFile.value)
        task.ppe_param = serialPPETask(task.ppe_param[0], task.ppe_param[1], task.ppe_param[2], bias_data, scale_data)

        in_t = SerialTensor([task.in_data.x, task.in_data.y, task.in_data.z],
                            calc_strides_zyx(task.in_data),
                            data_offset=task.in_data.offset,
                            location=MemoryLocation.ProgrammableInput.value,
                            data_dtype=task.in_data.dtype)

        out_t = SerialTensor([task.out_data.x, task.out_data.y, task.out_data.z],
                            calc_strides_zyx(task.out_data),
                            data_offset=task.out_data.offset,
                            location=MemoryLocation.ProgrammableOutput.value,
                            data_dtype=task.out_data.dtype)

        param_t = SerialTensor([task.param_data.x, task.param_data.y, task.param_data.z],
                               calc_strides_zyx(task.param_data),
                               data_offset=task.param_data.offset,
                               location=MemoryLocation.GraphFile.value,
                               data_dtype=task.param_data.dtype)

        t = serialNCE2Task(
            task.op.value,
            task.clusterID,
            task.kH,
            task.kW,
            0,
            0,
            task.padTop,
            task.padBottom,
            task.padLeft,
            task.padRight,
            in_t,
            out_t,
            param_t,
            param_t, # TODO
            task.mpe,
            task.ppe_param,
            task.dpuID,
            task.oXs,
            task.oYs,
            task.oZs
        )
    elif(isinstance(task, fakeDMALayer)):
        t = serialUPADMATask(
            task.src,
            task.dst,
            task.str)
    elif(isinstance(task, fakeNNDMALayer)):
        t = serialNNDMATask(
            task.src,
            task.dst,
            task.str)
    elif(isinstance(task, fakeNCELayer)):
        t = serialNCE1Task(
        )
    elif(isinstance(task, fakeMvTensorLayer)):
        if(task.mvTensorID == 0):   # Conv2D

            in_t = SerialTensor(
                [task.input.x, task.input.y, task.input.z],
                [task.input.x_s, task.input.y_s, task.input.z_s],
                data_offset=task.input.offset)
            out_t = SerialTensor(
                [task.output.x, task.output.y, task.output.z],
                [task.output.x_s, task.output.y_s, task.output.z_s],
                data_offset=task.output.offset)
            param_t = SerialTensor(
                [task.tap.x, task.tap.y, task.tap.z],
                [task.tap.x_s, task.tap.y_s, task.tap.z_s],
                data_offset=task.tap.offset)
            bias_t = SerialTensor(
                [task.bias.x, task.bias.y, task.bias.z],
                [task.bias.x_s, task.bias.y_s, task.bias.z_s],
                data_offset=task.bias.offset)

            t = serialSoftwareTask(
                referenceIDs=task.referenceIDs,
                softwareEnum=task.mvTensorID,
                radixX=task.radixX,
                radixY=task.radixX,
                radixStrideX=task.radixStrideX,
                radixStrideY=task.radixStrideY,
                padX=task.padX,
                padY=task.padY,
                padStyle=task.padStyle,
                dilation=task.dilation,
                input=in_t,
                output=out_t,
                params=param_t,
                bias=bias_t
            )
        elif(task.mvTensorID == 5):

            in_t = SerialTensor(
                [task.input.x, task.input.y, task.input.z],
                [task.input.x_s, task.input.y_s, task.input.z_s],
                data_offset=task.input.offset)
            out_t = SerialTensor(
                [task.output.x, task.output.y, task.output.z],
                [task.output.x_s, task.output.y_s, task.output.z_s],
                data_offset=task.output.offset)

            t = serialSoftwareTask(
                referenceIDs=task.referenceIDs,
                softwareEnum=task.mvTensorID,
                input=in_t,
                output=out_t,
            )
        else:
            print("Invalid Software Layer")

    elif(isinstance(task, fakeNNShvLayer)):
        t = serialSoftwareTask(
        )
    elif(isinstance(task, fakeControllerLayer)):
        t = serialControllerTask(
        )

    else:
        print("Invalid Task Type", type(task))
        quit()

    return t


class serialTaskList():
    """
        Helper class for a serialized list of tasks
    """
    def __init__(self, arr):
        """
            @param arr - a list of SerialTask objects
        """

        self.serialTasks = []
        for x in arr:
            self.serialTasks.append(x)

    def _fb(self, fbb):
        """
            Helper for Serializing the Object
            @param fbb - flat buffer builder
        """

        task_bin = []
        for x in self.serialTasks:
            task_bin.append(x._fb(fbb))

        sz = len(self.serialTasks)
        MVCNN.TaskList.TaskListStartContentVector(fbb, sz)
        for x in task_bin:
            fbb.PrependUOffsetTRelative(x)
        content = fbb.EndVector(sz)

        MVCNN.TaskList.TaskListStart(fbb)
        MVCNN.TaskList.TaskListAddContent(fbb, content)
        return MVCNN.TaskList.TaskListEnd(fbb)



def serialize(info, file_name):
    """
        Example function that shows the flow of instructions to
        serialize from a guest object to a file.
    """

    ss = Serializer()

    ss.setVersion(
        info.versionMaj,
        info.versionMin,
        info.versionPat,
        info.githash
    )
    ss.setResources(
        info.shaveMask,
        info.nce1Mask,
        info.dpuMask,
        info.leonCMX,
        info.nnCMX,
        info.ddrScratch
    )

    tensor = SerialTensor(info.dims, info.strides)

    ss.setInputTensors([tensor, tensor])
    ss.setOutputTensors([tensor, tensor, tensor])

    ss.setSourceStructure(info.structure)

    ref = ss.getSourceStructure()

    # Convert to serial task type
    ident = 0
    serial_net = copy.deepcopy(info.network)

    # a dictionary of the form:
    #   key -> value
    #   where
    #   key = Barrier Index (guestClass contains IDs rather than indexes right now)
    #   value = (consumer_count, producer_count)
    #

    barriers = {}

    def barrierAdd(isDependency, original_pair):
        if isDependency:
            return (original_pair[0], original_pair[1] + 1)
        else:
            return (original_pair[0] + 1, original_pair[1])


    for x in range(len(info.network)):
        for y in range(len(info.network[x])):
            my_task_class = info.network[x][y]
            subtask = choose_task(my_task_class)

            # Setting up Barriers for later
            if len(my_task_class.dependency) > 0:
                for d in my_task_class.dependency:
                    barriers[d] = barrierAdd(True, barriers[d])  if d in barriers else (0, 1)

            if len(my_task_class.consumers) > 0:
                for d in my_task_class.consumers:
                    barriers[d] = barrierAdd(False, barriers[d]) if d in barriers else (1, 0)


            # References to these barriers

            if (len(my_task_class.dependency) > 0):
                prod = my_task_class.dependency[0]
            else:
                prod = 0

            barrierInfo = serialBarrierReference(
                # Should refer to barrier, rather than nodes.
                # Since this is a toy example, just assume there is equal barriers and nodes
                prod,
                my_task_class.consumers
            )

            serial_net[x][y] = serialTask(
                my_task_class.taskID,
                info.network[x][y].referenceIDs,
                subtask,
                ref,
                barrierInfo
            )

    tl = []
    for x in serial_net:
        tl.append(serialTaskList(x))


    ss.setSummaryHeader(
        taskAmount=len(serial_net),
        layerAmount=sum(len(x) for x in serial_net)
    )
    ss.setTaskLists(tl)

    ss.setBinaryData(info.binaryData)

    serial_barriers = []
    for k, v in barriers.items():
        serial_barriers.append(serialBarrier(v[0], v[1]))

    ss.setBarrierTable(serial_barriers)

    ss.finish()

    ss.toFile(file_name)
