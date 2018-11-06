# TODO Improve this temporary solution
import sys
sys.path.append('../api/')

import os.path
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
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        self.assertFalse(ca.isValid(om))

    def test_SWIG_connection(self):
        self.assertEqual(1, ca.testSWIG())

    def test_minimal_om(self):
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        ca.output(om, in_)

        self.assertTrue(ca.isValid(om))

    def test_convolution(self):
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)

        shape = ca.getShape(32, 32, 3)

        weightData = ca.getData(np.arange(3 * 3 * 3 * 3).astype(np.float64))
        weights = ca.constant(om, weightData, ca.getShape(3, 3, 3, 3))

        in_ = ca.input(om, shape)

        c_ = ca.conv2D(om, in_, weights, 4, 4, 1, 1)
        cOp_ = ca.getSourceOp(om, c_)
        ca.output(om, c_)

        self.assertTrue(ca.isValid(om))
        self.assertEqual(ca.testConv(cOp_, 4, 4, 1, 1), 0)

    def test_maxpool(self):
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        mx_ = ca.maxpool2D(om, in_, 2, 2, 2, 2, 0, 0)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    def test_concat(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(256, 256, 1)
        in_ = ca.input(om, shape)

        k1data = [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ]
        k1blurData = ca.getData(np.array(k1data).astype(np.float64))
        bweights = ca.constant(om, k1blurData, ca.getShape(3, 3, 1, 1))


        k2data = [ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 ]
        k2edgeData = ca.getData(np.array(k2data).astype(np.float64))
        eweights = ca.constant(om, k2edgeData, ca.getShape(3, 3, 1, 1))

        c1_ = ca.conv2D(om, in_, bweights, 1, 1, 0, 0)
        c2_ = ca.conv2D(om, in_, eweights, 1, 1, 0, 0)
        
        vec = ca.pushVector(None, c1_)
        vec = ca.pushVector(vec, c2_)

        
        cc_ = ca.concat(om, vec)
        ca.output(om, cc_)

        self.assertTrue(ca.isValid(om))

    def test_avgpool(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        kernels = ca.get2DVector(7, 7)
        stride = ca.get2DVector(1, 1)
        padding = ca.get4DVector(0, 0, 0, 0)
        mx_ = ca.avgpool2D(om, in_, kernels, stride, padding)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    def test_matMul(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(4, 100)

        in_ = ca.input(om, shape)
        weightData = ca.getData(np.arange(100 * 100).astype(np.float64))
        weights_ = ca.constant(om, weightData, ca.getShape(100, 100))
        mx_ = ca.matMul(om, in_, weights_)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    def test_fullyConnected(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(1, 1, 100)

        in_ = ca.input(om, shape)
        weightData = ca.getData(np.arange(100 * 100).astype(np.float64))
        weights_ = ca.constant(om, weightData, ca.getShape(100, 100))
        mx_ = ca.fullyConnected(om, in_, weights_)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    def test_batchNorm(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(28, 28, 4)

        in_ = ca.input(om, shape)
        meanData = ca.getData(np.arange(28 * 28 * 4).astype(np.float64))
        mean_ = ca.constant(om, meanData, ca.getShape(28, 28, 4))

        varData = ca.getData(np.arange(28 * 28 * 4).astype(np.float64))
        variance_ = ca.constant(om, varData, ca.getShape(28, 28, 4))

        offsetData = ca.getData(np.arange(28 * 28 * 4).astype(np.float64))
        offset_ = ca.constant(om, offsetData, ca.getShape(28, 28, 4))

        scaleData = ca.getData(np.arange(28 * 28 * 4).astype(np.float64))
        scale_ = ca.constant(om, scaleData, ca.getShape(28, 28, 4))

        eps_ = 2

        mx_ = ca.batchNorm(om, in_, mean_, variance_, offset_, scale_, eps_)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    def test_scale(self):
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        weightData = ca.getData(np.arange(32 * 32 * 3).astype(np.float64))

        scaleMatrix_ = ca.constant(om, weightData, ca.getShape(32, 32, 3))
        mx_ = ca.scale(om, in_, scaleMatrix_)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    def test_relu(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        mx_ = ca.relu(om, in_)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    def test_prelu(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)

        data = ca.getData(np.arange(3).astype(np.float64))
        slope_ = ca.constant(om, data, ca.getShape(3))

        pr_ = ca.prelu(om, in_, slope_)
        ca.output(om, pr_)

        self.assertTrue(ca.isValid(om))

    def test_softmax(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        mx_ = ca.softmax(om, in_)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    def test_add(self):
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)

        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        mx_ = ca.maxpool2D(om, in_, 1, 1, 1, 1, 0, 0)
        ad_ = ca.add(om, in_, mx_)
        ca.output(om, ad_)

        self.assertTrue(ca.isValid(om))

    def test_bias(self):
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)

        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)

        data = ca.getData(np.arange(3).astype(np.float64))
        bias = ca.constant(om, data, ca.getShape(3))
        bi_ = ca.bias(om, in_, bias)
        ca.output(om, bi_)

        self.assertTrue(ca.isValid(om))

    def test_subtract(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        mx_ = ca.maxpool2D(om, in_, 1, 1, 1, 1, 0, 0)

        sb_ = ca.subtract(om, in_, mx_)
        ca.output(om, sb_)

        self.assertTrue(ca.isValid(om))

    def test_multiply(self):
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)

        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        mx_ = ca.maxpool2D(om, in_, 1, 1, 1, 1, 0, 0)

        mu_ = ca.multiply(om, in_, mx_)
        ca.output(om, mu_)

        self.assertTrue(ca.isValid(om))

    def test_divide(self):

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        mx_ = ca.maxpool2D(om, in_, 1, 1, 1, 1, 0, 0)
        mu_ = ca.divide(om, in_, mx_)

        ca.output(om, mu_)

        self.assertTrue(ca.isValid(om))

    def test_reshape(self):
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 3)

        in_ = ca.input(om, shape)
        new_shape = ca.getShape(32, 32, 3)
        mx_ = ca.reshape(om, in_, new_shape)
        ca.output(om, mx_)

        self.assertTrue(ca.isValid(om))

    @unittest.skip("temporary skipped while developing higher priority items")
    def testDOT(self):

        cm = ca.getCompilationUnit()
        om = ca.getModel(cm)
        shape = ca.getShape(256, 256, 1)
        in_ = ca.input(om, shape)

        k1data = [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ]
        k1blurData = ca.getData(np.array(k1data).astype(np.float64))
        bweights = ca.constant(om, k1blurData, ca.getShape(3, 3, 1, 1))


        k2data = [ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 ]
        k2edgeData = ca.getData(np.array(k2data).astype(np.float64))
        eweights = ca.constant(om, k2edgeData, ca.getShape(3, 3, 1, 1))

        c1_ = ca.conv2D(om, in_, bweights, 1, 1, 0, 0)
        c2_ = ca.conv2D(om, in_, eweights, 1, 1, 0, 0)
        cc_ = ca.concat(om, c1_, c2_)
        ca.output(om, cc_)

        # ca.produceDOT(om)
        ca.compile(cm)
        self.assertTrue(os.path.isfile("pycm.dot"))
        os.system("dot -Tsvg pycm.dot -o pycm.svg")
        self.assertTrue(os.path.isfile("pycm.svg"))

    @unittest.skip("temporary skipped while developing higher priority items")
    def test_compile_convolution_01(self):
        """
            32x32x1
                |
               < >  (3x3x1x1)
                |
            32x32x1

        """

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(32, 32, 1)

        arr = [
            0.1111, 0.1121, 0.1131, 0.1141, 0.1151, 0.1161, 0.1171, 0.1181, 0.1191
        ]
        weightData = ca.getData(np.array(arr).astype(np.float64))

        weights = ca.constant(om, weightData, ca.getShape(3, 3, 1, 1))

        in_ = ca.input(om, shape)

        c_ = ca.conv2D(om, in_, weights, 4, 4, 0, 0)
        cOp_ = ca.getSourceOp(om, c_)
        ca.output(om, c_)

        self.assertTrue(ca.isValid(om))
        self.assertEqual(ca.testConv(cOp_, 4, 4, 0, 0), 0)

        fs = ca.compile(cu)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_01.blob"))

    @unittest.skip("temporary skipped while developing higher priority items")
    def test_compile_convolution_02(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """
        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)

        shape = ca.getShape(32, 32, 3)

        weightData = ca.getData(np.linspace(0.101, 0.101 + (3 * 3 * 3 * 3 - 1) * 0.001, 3 * 3 * 3 * 3).astype(np.float64))

        weights = ca.constant(om, weightData, ca.getShape(3, 3, 3, 3))

        in_ = ca.input(om, shape)

        c_ = ca.conv2D(om, in_, weights, 4, 4, 0, 0)
        cOp_ = ca.getSourceOp(om, c_)
        out_ = ca.output(om, c_)

        self.assertTrue(ca.isValid(om))
        self.assertEqual(ca.testConv(cOp_, 4, 4, 0, 0), 0)

        fs = ca.compile(cu)
        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_02.blob"))

    @unittest.skip("temporary skipped while developing higher priority items")
    def test_compile_convolution_03(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)

        shape = ca.getShape(256, 256, 3)

        weightData = ca.getData(np.linspace(0.101, 0.101 + (3 * 3 * 3 * 3 - 1) * 0.001, 3 * 3 * 3 * 3).astype(np.float64))

        weights = ca.constant(om, weightData, ca.getShape(3, 3, 3, 3))

        in_ = ca.input(om, shape)

        c_ = ca.conv2D(om, in_, weights, 2, 2, 0, 0)
        cOp_ = ca.getSourceOp(om, c_)
        out_ = ca.output(om, c_)

        self.assertTrue(ca.isValid(om))
        self.assertEqual(ca.testConv(cOp_, 2, 2, 0, 0), 0)

        fs = ca.compile(cu)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_03.blob"))

    @unittest.skip("temporary skipped while developing higher priority items")
    def test_compile_convolution_04(self):
        """
            32x32x3
                |
               < >  (3x3x1x3)
                |
            32x32x3

        """

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(256, 256, 3)

        weightData = ca.getData(np.linspace(0.101, 0.101 + (5 * 5 * 3 * 3 - 1) * 0.001, 5 * 5 * 3 * 3).astype(np.float64))

        weights = ca.constant(om, weightData, ca.getShape(5, 5, 3, 3))

        in_ = ca.input(om, shape)

        c_ = ca.conv2D(om, in_, weights, 2, 2, 0, 0)
        cOp_ = ca.getSourceOp(om, c_)
        ca.output(om, c_)

        self.assertTrue(ca.isValid(om))
        self.assertEqual(ca.testConv(cOp_, 2, 2, 0, 0), 0)

        fs = ca.compile(cu)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_04.blob"))


    @unittest.skip("temporary skipped while developing higher priority items")
    def test_compile_convolution_05(self):
        """
            256x256x1
                |
               < >  (3x3x1x1)
                |
               < >  (3x3x1x1)
                |
            256x256x1

        """

        cu = ca.getCompilationUnit()
        om = ca.getModel(cu)
        shape = ca.getShape(256, 256, 1)
        in_ = ca.input(om, shape)

        k1data = [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ]
        k1blurData = ca.getData(np.array(k1data).astype(np.float64))
        bweights = ca.constant(om, k1blurData, ca.getShape(3, 3, 1, 1))


        k2data = [ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 ]
        k2edgeData = ca.getData(np.array(k2data).astype(np.float64))
        eweights = ca.constant(om, k2edgeData, ca.getShape(3, 3, 1, 1))

        c1_ = ca.conv2D(om, in_, bweights, 1, 1, 0, 0)
        c2_ = ca.conv2D(om, c1_, eweights, 1, 1, 0, 0)
        ca.output(om, c2_)

        self.assertTrue(ca.isValid(om))

        ca.compile(cu)

        self.assertTrue(filecmp.cmp("cpp.blob", "../../tests/data/gold_05.blob"))


if __name__ == '__main__':
    # unittest.main()
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output="./python_unittests_xml"))
