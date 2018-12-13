import unittest
import os
import sys
from serializer import serialize
from deserializer import deserialize
sys.path.append(os.path.join(os.path.dirname(__file__), '../flatbuffers/python'))
import flatbuffers

import numpy as np

from hostStructure import HostRepr, fakeMvTensorLayer, fakeDPULayer

class TestFlatbuffers(unittest.TestCase):

    # def setUp(self):
    @classmethod
    def setUpClass(cls):

        cls.filename = "VPU3.py.bin"


        print("##################### Compile CSV Example ########################")

        # First example based off of the dpu csv.
        cls.hostS_dpu = HostRepr()
        serialize(cls.hostS_dpu, cls.filename)
        cls.guestS_dpu = deserialize(cls.filename)

        print("############# Compile Software Multi-Layer Example ###############")

        cls.filename = "VPU3-Stress.py.bin"

        # Second Stress example
        cls.hostS_stress = HostRepr(which=1)
        serialize(cls.hostS_stress, cls.filename)
        cls.guestS_stress = deserialize(cls.filename)


    def test_version(self):
        self.assertEqual(self.hostS_dpu.versionMaj, self.guestS_dpu.version[0])
        self.assertEqual(self.hostS_dpu.versionMin, self.guestS_dpu.version[1])
        self.assertEqual(self.hostS_dpu.versionPat, self.guestS_dpu.version[2])
        self.assertEqual(self.hostS_dpu.githash, self.guestS_dpu.githash)

    def test_resources(self):
        self.assertEqual(self.hostS_dpu.nce1Mask, self.guestS_dpu.NCE1)
        self.assertEqual(self.hostS_dpu.dpuMask, self.guestS_dpu.DPU)
        self.assertEqual(self.hostS_dpu.shaveMask, self.guestS_dpu.SHV)
        self.assertEqual(self.hostS_dpu.leonCMX, self.guestS_dpu.LEONCMX)
        self.assertEqual(self.hostS_dpu.nnCMX, self.guestS_dpu.NNCMX)
        self.assertEqual(self.hostS_dpu.ddrScratch, self.guestS_dpu.DDR)

    def test_bindata(self):
        for x in range(len(self.hostS_dpu.binaryData)):
            self.assertTrue(np.allclose(self.hostS_dpu.binaryData[x], self.guestS_dpu.binaryData[x]))

    def test_layer(self):
        # Check that all host structures were serialized

        flattened_guest_network = [item for sublist in self.hostS_stress.network for item in sublist]

        for t1 in flattened_guest_network:
            match = False
            for t2 in self.guestS_stress.tasks:
                if isinstance(t1, fakeMvTensorLayer):
                    if hasattr(t2, 'layerId'):
                        if(t1.mvTensorID == t2.layerId):
                            match = True

            self.assertTrue(match)
        self.assertTrue(len(flattened_guest_network) == len(self.guestS_stress.tasks))


    def test_tensor_connection(self):
        # From Host
        flattened_binSection = [item for sublist in self.hostS_stress.binaryData for item in sublist]

        flattened_host_network = [item for sublist in self.hostS_stress.network for item in sublist]

        inLayer = flattened_host_network[0]
        convLayer = flattened_host_network[1]
        outLayer = flattened_host_network[2]

        tap = convLayer.tap
        convTapSize = tap.x * tap.y * tap.z
        convTaps = flattened_binSection[tap.offset: tap.offset + convTapSize]

        bias = convLayer.bias
        convBiasSize = bias.x * bias.y * bias.z
        convBias = flattened_binSection[bias.offset: bias.offset + convBiasSize]

        # --------------------------------------

        # From Blob
        hBinData = self.guestS_dpu.binaryData
        hBinData = [item for sublist in hBinData for item in sublist]

        task = self.guestS_stress.tasks[1]

        td = task.paramTensorOffset
        bd = task.biasTensorOffset

        TapSize = int(np.prod(task.paramTensorDims))
        Taps2 = hBinData[td: td + TapSize]

        BiasSize = int(np.prod(task.biasTensorDims))
        Bias2 = hBinData[bd: bd + BiasSize]

        self.assertTrue(np.allclose(convTaps, Taps2))
        self.assertTrue(np.allclose(convBias, Bias2))

    # def test_structure_and_dependencies(self):
    #     simulated = []
    #     for x in self.guestS_stress.deplist:
    #         simulated.append(x[0])
    #         if x[2] != 0:
    #             for y in x[2]:  # Check depencies are in order
    #                 self.assertTrue(y in simulated)

if __name__ == '__main__':
    unittest.main()