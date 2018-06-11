import numpy as np
import unittest
from ctypes import *
import composition_api as ca
import filecmp

class TestComposition(unittest.TestCase):

    def test_simple(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_empty_om(self):
        g = ca.getOM()
        self.assertFalse(g.isValid())

    def test_SWIG_connection(self):
        self.assertEqual(1, ca.testSWIG())

    def test_minimal_om(self):
        g = ca.getOM()
        shape = ca.getShape(1, 32, 32, 3)

        in_ = ca.input(g, shape)
        out_ = ca.output(g, in_)

        self.assertTrue(g.isValid())

    def test_convolution(self):
        g = ca.getOM()

        shape = ca.getShape(1, 32, 32, 3)

        arr = [
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27
        ]
        weightData = ca.getData(np.array(arr).astype(np.float32))

        weights = ca.getConstantTensor(ca.getShape(3, 3, 1, 3), weightData)

        in_ = ca.input(g, shape)

        c_ = ca.conv(g, in_, weights, 4, 4, 1, 1)
        out_ = ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(c_, 4, 4, 1, 1), 0)

    @unittest.skip("unimplemented")
    def test_maxpool(self):
        pass

    @unittest.skip("unimplemented")
    def test_avepool(self):
        pass

    @unittest.skip("unimplemented")
    def test_concat(self):
        pass

    def test_serialize_convolution_01(self):
        """
            32x32x1
                |
               < >  (3x3x1x1)
                |
            32x32x1

        """

        g = ca.getOM()
        shape = ca.getShape(1, 32, 32, 1)

        arr = [
            0.1111, 0.1121, 0.1131, 0.1141, 0.1151, 0.1161, 0.1171, 0.1181, 0.1191
        ]
        weightData = ca.getData(np.array(arr).astype(np.float32))

        weights = ca.getConstantTensor(ca.getShape(3, 3, 1, 1), weightData)

        in_ = ca.input(g, shape)

        c_ = ca.conv(g, in_, weights, 4, 4, 0, 0)
        out_ = ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(c_, 4, 4, 0, 0), 0)

        fs = ca.serialize(g)
        print(fs)

        self.assertTrue(filecmp.cmp("cpp.blob", "../tests/data/gold_01.blob"))

    def test_serialize_convolution_02(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """

        g = ca.getOM()
        shape = ca.getShape(1, 32, 32, 3)

        arr = [
            0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109,
            0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119,
            0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129
        ]
        weightData = ca.getData(np.array(arr).astype(np.float32))

        weights = ca.getConstantTensor(ca.getShape(3, 3, 1, 3), weightData)

        in_ = ca.input(g, shape)

        c_ = ca.conv(g, in_, weights, 4, 4, 0, 0)
        out_ = ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(c_, 4, 4, 0, 0), 0)

        fs = ca.serialize(g)
        print(fs)

        self.assertTrue(filecmp.cmp("cpp.blob", "../tests/data/gold_02.blob"))


    def test_serialize_convolution_03(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """

        g = ca.getOM()
        shape = ca.getShape(1, 256, 256, 3)

        arr = [
            0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109,
            0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119,
            0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129
        ]
        weightData = ca.getData(np.array(arr).astype(np.float32))

        weights = ca.getConstantTensor(ca.getShape(3, 3, 1, 3), weightData)

        in_ = ca.input(g, shape)

        c_ = ca.conv(g, in_, weights, 2, 2, 0, 0)
        out_ = ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(c_, 2, 2, 0, 0), 0)

        fs = ca.serialize(g)
        print(fs)

        self.assertTrue(filecmp.cmp("cpp.blob", "../tests/data/gold_03.blob"))

    def test_serialize_convolution_04(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """

        g = ca.getOM()
        shape = ca.getShape(1, 256, 256, 3)

        arr = [
            0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
            0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
            0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
            0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
            0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
            0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
            0.7, 0.71, 0.72, 0.73, 0.74
        ]
        weightData = ca.getData(np.array(arr).astype(np.float32))

        weights = ca.getConstantTensor(ca.getShape(5, 5, 1, 3), weightData)

        in_ = ca.input(g, shape)

        c_ = ca.conv(g, in_, weights, 2, 2, 0, 0)
        out_ = ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(c_, 2, 2, 0, 0), 0)

        fs = ca.serialize(g)
        print(fs)

        self.assertTrue(filecmp.cmp("cpp.blob", "../tests/data/gold_04.blob"))


    def test_serialize_convolution_05(self):
        """
            256x256x1
                |
               < >  (3x3x1x1)
                |
               < >  (3x3x1x1)
                |
            256x256x1

        """

        g = ca.getOM()
        shape = ca.getShape(1, 256, 256, 1)
        in_ = ca.input(g, shape)

        k1data = [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ]
        k1blurData = ca.getData(np.array(k1data).astype(np.float32))
        bweights = ca.getConstantTensor(ca.getShape(3, 3, 1, 1), k1blurData)


        k2data = [ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 ]
        k2edgeData = ca.getData(np.array(k2data).astype(np.float32))
        eweights = ca.getConstantTensor(ca.getShape(3, 3, 1, 1), k2edgeData)

        c1_ = ca.conv(g, in_, bweights, 1, 1, 0, 0)
        c2_ = ca.conv(g, c1_, eweights, 1, 1, 0, 0)
        out_ = ca.output(g, c2_)

        self.assertTrue(g.isValid())

        fs = ca.serialize(g)

        self.assertTrue(filecmp.cmp("cpp.blob", "../tests/data/gold_05.blob"))




if __name__ == '__main__':
    unittest.main()