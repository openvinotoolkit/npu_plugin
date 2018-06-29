# TODO Improve this temporary solution
import sys
sys.path.append('../api/')

import numpy as np
import unittest
import os
import composition_api as ca
import filecmp
import xmlrunner


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
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        ca.output(g, in_)

        self.assertTrue(g.isValid())

    def test_convolution(self):
        g = ca.getOM()

        shape = ca.getShape(32, 32, 3)

        weightData = ca.getData(np.arange(3 * 3 * 3 * 3).astype(np.float32))
        weights = ca.constant(g, weightData, ca.getShape(3, 3, 3, 3))

        in_ = ca.input(g, shape)

        c_ = ca.conv2D(g, in_, weights, 4, 4, 1, 1)
        cOp_ = ca.getSourceOp(g, c_)
        ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(cOp_, 4, 4, 1, 1), 0)

    def test_maxpool(self):
        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        mx_ = ca.maxpool2D(g, in_, 2, 2, 2, 2, 0, 0)
        ca.output(g, mx_)

        self.assertTrue(g.isValid())

    def test_concat(self):

        g = ca.getOM()
        shape = ca.getShape(256, 256, 1)
        in_ = ca.input(g, shape)

        k1data = [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ]
        k1blurData = ca.getData(np.array(k1data).astype(np.float32))
        bweights = ca.constant(g, k1blurData, ca.getShape(3, 3, 1, 1))


        k2data = [ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 ]
        k2edgeData = ca.getData(np.array(k2data).astype(np.float32))
        eweights = ca.constant(g, k2edgeData, ca.getShape(3, 3, 1, 1))

        c1_ = ca.conv2D(g, in_, bweights, 1, 1, 0, 0)
        c2_ = ca.conv2D(g, in_, eweights, 1, 1, 0, 0)
        cc_ = ca.concat(g, c1_, c2_)
        ca.output(g, cc_)

        self.assertTrue(g.isValid())

    def test_avgpool(self):

        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        kernels = ca.get2DVector(7, 7)
        stride = ca.get2DVector(1, 1)
        padding = ca.get4DVector(0, 0, 0, 0)
        mx_ = ca.avgpool2D(g, in_, kernels, stride, padding)
        ca.output(g, mx_)

        self.assertTrue(g.isValid())

    def test_fullyConnected(self):

        g = ca.getOM()
        shape = ca.getShape(100, 4)

        in_ = ca.input(g, shape)
        weightData = ca.getData(np.arange(100 * 100).astype(np.float32))
        weights_ = ca.constant(g, weightData, ca.getShape(100, 100))
        mx_ = ca.fullyConnected(g, in_, weights_)
        ca.output(g, mx_)

        self.assertTrue(g.isValid())

    def test_batchNorm(self):

        g = ca.getOM()
        shape = ca.getShape(28, 28, 4)

        in_ = ca.input(g, shape)
        meanData = ca.getData(np.arange(28 * 28 * 4).astype(np.float32))
        mean_ = ca.constant(g, meanData, ca.getShape(28, 28, 4))

        varData = ca.getData(np.arange(28 * 28 * 4).astype(np.float32))
        variance_ = ca.constant(g, varData, ca.getShape(28, 28, 4))

        offsetData = ca.getData(np.arange(28 * 28 * 4).astype(np.float32))
        offset_ = ca.constant(g, offsetData, ca.getShape(28, 28, 4))

        scaleData = ca.getData(np.arange(28 * 28 * 4).astype(np.float32))
        scale_ = ca.constant(g, scaleData, ca.getShape(28, 28, 4))

        eps_ = 2

        mx_ = ca.batchNorm(g, in_, mean_, variance_, offset_, scale_, eps_)
        ca.output(g, mx_)

        self.assertTrue(g.isValid())

    def test_scale(self):
        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        weightData = ca.getData(np.arange(32 * 32 * 3).astype(np.float32))

        scaleMatrix_ = ca.constant(g, weightData, ca.getShape(32, 32, 3))
        mx_ = ca.scale(g, in_, scaleMatrix_)
        ca.output(g, mx_)

        self.assertTrue(g.isValid())

    def test_relu(self):

        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        mx_ = ca.relu(g, in_)
        ca.output(g, mx_)

        self.assertTrue(g.isValid())

    def test_softmax(self):

        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        mx_ = ca.softmax(g, in_)
        ca.output(g, mx_)

        self.assertTrue(g.isValid())

    def test_add(self):
        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        mx_ = ca.maxpool2D(g, in_, 1, 1, 1, 1, 0, 0)
        ad_ = ca.add(g, in_, mx_)
        ca.output(g, ad_)

        self.assertTrue(g.isValid())

    def test_bias(self):
        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)

        data = ca.getData(np.arange(3).astype(np.float32))
        bias = ca.constant(g, data, ca.getShape(3))
        bi_ = ca.bias(g, in_, bias)
        ca.output(g, bi_)

        self.assertTrue(g.isValid())

    def test_subtract(self):
        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        mx_ = ca.maxpool2D(g, in_, 1, 1, 1, 1, 0, 0)

        sb_ = ca.subtract(g, in_, mx_)
        ca.output(g, sb_)

        self.assertTrue(g.isValid())

    def test_multiply(self):
        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        mx_ = ca.maxpool2D(g, in_, 1, 1, 1, 1, 0, 0)

        mu_ = ca.multiply(g, in_, mx_)
        ca.output(g, mu_)

        self.assertTrue(g.isValid())

    def test_divide(self):
        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        mx_ = ca.maxpool2D(g, in_, 1, 1, 1, 1, 0, 0)
        mu_ = ca.divide(g, in_, mx_)

        ca.output(g, mu_)

        self.assertTrue(g.isValid())

    def test_reshape(self):
        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(g, shape)
        new_shape = ca.getShape(32, 32, 3)
        mx_ = ca.reshape(g, in_, new_shape)
        ca.output(g, mx_)

        self.assertTrue(g.isValid())


    def test_serialize_convolution_01(self):
        """
            32x32x1
                |
               < >  (3x3x1x1)
                |
            32x32x1

        """

        g = ca.getOM()
        shape = ca.getShape(32, 32, 1)

        arr = [
            0.1111, 0.1121, 0.1131, 0.1141, 0.1151, 0.1161, 0.1171, 0.1181, 0.1191
        ]
        weightData = ca.getData(np.array(arr).astype(np.float32))

        weights = ca.constant(g, weightData, ca.getShape(3, 3, 1, 1))

        in_ = ca.input(g, shape)

        c_ = ca.conv2D(g, in_, weights, 4, 4, 0, 0)
        cOp_ = ca.getSourceOp(g, c_)
        ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(cOp_, 4, 4, 0, 0), 0)

        fs = ca.serialize(g)
        print(fs)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_01.blob"))

    def test_serialize_convolution_02(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """

        g = ca.getOM()
        shape = ca.getShape(32, 32, 3)

        weightData = ca.getData(np.linspace(0.101, 0.101 + (3 * 3 * 3 * 3 - 1) * 0.001, 3 * 3 * 3 * 3).astype(np.float32))

        weights = ca.constant(g, weightData, ca.getShape(3, 3, 3, 3))

        in_ = ca.input(g, shape)

        c_ = ca.conv2D(g, in_, weights, 4, 4, 0, 0)
        cOp_ = ca.getSourceOp(g, c_)
        out_ = ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(cOp_, 4, 4, 0, 0), 0)

        fs = ca.serialize(g)
        print(fs)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_02.blob"))


    def test_serialize_convolution_03(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """

        g = ca.getOM()
        shape = ca.getShape(256, 256, 3)

        weightData = ca.getData(np.linspace(0.101, 0.101 + (3 * 3 * 3 * 3 - 1) * 0.001, 3 * 3 * 3 * 3).astype(np.float32))

        weights = ca.constant(g, weightData, ca.getShape(3, 3, 3, 3))

        in_ = ca.input(g, shape)

        c_ = ca.conv2D(g, in_, weights, 2, 2, 0, 0)
        cOp_ = ca.getSourceOp(g, c_)
        out_ = ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(cOp_, 2, 2, 0, 0), 0)

        fs = ca.serialize(g)
        print(fs)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_03.blob"))

    def test_serialize_convolution_04(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """

        g = ca.getOM()
        shape = ca.getShape(256, 256, 3)

        weightData = ca.getData(np.linspace(0.101, 0.101 + (5 * 5 * 3 * 3 - 1) * 0.001, 5 * 5 * 3 * 3).astype(np.float32))

        weights = ca.constant(g, weightData, ca.getShape(5, 5, 3, 3))

        in_ = ca.input(g, shape)

        c_ = ca.conv2D(g, in_, weights, 2, 2, 0, 0)
        cOp_ = ca.getSourceOp(g, c_)
        ca.output(g, c_)

        self.assertTrue(g.isValid())
        self.assertEqual(ca.testConv(cOp_, 2, 2, 0, 0), 0)

        fs = ca.serialize(g)
        print(fs)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_04.blob"))


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
        shape = ca.getShape(256, 256, 1)
        in_ = ca.input(g, shape)

        k1data = [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ]
        k1blurData = ca.getData(np.array(k1data).astype(np.float32))
        bweights = ca.constant(g, k1blurData, ca.getShape(3, 3, 1, 1))


        k2data = [ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 ]
        k2edgeData = ca.getData(np.array(k2data).astype(np.float32))
        eweights = ca.constant(g, k2edgeData, ca.getShape(3, 3, 1, 1))

        c1_ = ca.conv2D(g, in_, bweights, 1, 1, 0, 0)
        c2_ = ca.conv2D(g, c1_, eweights, 1, 1, 0, 0)
        ca.output(g, c2_)

        self.assertTrue(g.isValid())

        ca.serialize(g)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_05.blob"))


if __name__ == '__main__':
    # unittest.main()
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output="./python_unittests_xml"))