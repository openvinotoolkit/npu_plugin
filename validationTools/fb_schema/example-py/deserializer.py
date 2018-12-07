import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../flatbuffers/python'))
import flatbuffers
import numpy as np
import MVCNN.Version
import MVCNN.Resources
import MVCNN.SummaryHeader
import MVCNN.TensorReference
import MVCNN.IndirectDataReference
import MVCNN.SourceStructure
import MVCNN.GraphFile
import MVCNN.Conv2D
import MVCNN.SoftwareLayer
import MVCNN.Passthrough

from enum import Enum
from guestStructure import GuestRepr, GuestLayer

class MemoryLocation(Enum):
    NULL = 0
    ProgrammableInput = 1
    ProgrammableOutput = 2
    VPU_DDR_Heap = 3
    GraphFile = 4


def deserialize(file_name):

    builder = flatbuffers.Builder(1024)

    f = open(file_name, 'rb')
    # file_id = f.read(4)
    buf = f.read()
    f.close()

    g = MVCNN.GraphFile.GraphFile.GetRootAsGraphFile(buf, 0)

    gr = GuestRepr()

    gr.githash = g.Header().Version().Hash().decode('ascii')
    v1 = g.Header().Version().MajorV()
    v2 = g.Header().Version().MinorV()
    v3 = g.Header().Version().PatchV()
    gr.version = [v1, v2, v3]

    gr.binaryData = []
    for y in range(g.BinaryDataLength()):

        data_found = False

        try:
            if g.BinaryData(y).Fp16Length() != 0:
                d = g.BinaryData(y).Fp16AsNumpy().view(np.float16)
                gr.binaryData.append(d)
                data_found = True
        except Exception as e:
            pass

        try:
            if g.BinaryData(y).U8Length() != 0:
                d = g.BinaryData(y).U8AsNumpy().view(np.uint8)
                gr.binaryData.append(d)
                data_found = True
            else:
                assert 0, "Unsupported"

        except Exception as e:
            pass

        if not data_found:
            assert 0, "NO DATA"


    gr.NCE1 = g.Header().Resources().Nce1Mask()
    gr.DPU = g.Header().Resources().DpuMask()
    gr.SHV = g.Header().Resources().ShaveMask()
    gr.LEONCMX = g.Header().Resources().LeonCmx()
    gr.NNCMX = g.Header().Resources().NnCmx()
    gr.DDR = g.Header().Resources().DdrScratch()

    gr.tasks = []
    gr.deplist = []
    for y in range(g.TaskListsLength()):
        for z in range(g.TaskLists(y).ContentLength()):
            task = g.TaskLists(y).Content(z)
            dep = g.TaskLists(y).Content(z).AssociatedBarriers()
            gr.deplist.append([
                # dep.NodeID(),
                # dep.DependencyCount(),
                # dep.ConsumersAsNumpy()
            ])

            if task.TaskType() == MVCNN.SpecificTask.SpecificTask().MvTensorTask:
                mvt = MVCNN.MvTensorTask.MvTensorTask()
                mvt.Init(task.Task().Bytes, task.Task().Pos)
                gl = None

                if mvt.LayerType() == MVCNN.SoftwareLayer.SoftwareLayer().Conv2D:
                    l = MVCNN.Conv2D.Conv2D()
                    l.Init(mvt.Layer().Bytes, mvt.Layer().Pos)

                    gl = GuestLayer(
                        layerId=0,
                        radixX=l.RadixX(),
                        inTensorOffset=l.Input().Data().DataIndex(),
                        outTensorOffset=l.Output().Data().DataIndex(),
                        paramTensorOffset=l.Weight().Data().DataIndex(),
                        paramTensorDims=l.Weight().DimensionsAsNumpy(),
                        biasTensorOffset=l.Bias().Data().DataIndex(),
                        biasTensorDims=l.Bias().DimensionsAsNumpy()
                    )

                elif mvt.LayerType() == MVCNN.SoftwareLayer.SoftwareLayer().Passthrough:
                    l = MVCNN.Passthrough.Passthrough()
                    l.Init(mvt.Layer().Bytes, mvt.Layer().Pos)

                    gl = GuestLayer(
                        layerId=5,
                        inTensorOffset=l.Input().Data().DataIndex(),
                        inTensorDims=l.Input().DimensionsAsNumpy(),
                        outTensorOffset=l.Output().Data().DataIndex(),
                        outTensorDims=l.Output().DimensionsAsNumpy(),

                    )
                else:
                    print("Deserialization not written for MVT Layer", mvt.LayerType())
                    print(MVCNN.SoftwareLayer.SoftwareLayer().Passthrough)
                    print(MVCNN.SoftwareLayer.SoftwareLayer().Conv2D)
                    # assert 0, "Deserialization not written for MVT Layer"

                gr.tasks.append(gl)
            else:
                print("Deserialization not written for Task: ", task.TaskType())

    return gr