import numpy as np
import unittest
from ctypes import *
import composition_api as ca

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
        # self.assertEqual(out_.getOutputShape(), ca.getShape(1, 8, 8, 3))
        # self.assertEqual(c_.attrsCount(), 10)

        # self.assertEqual(ca.getAttrByte(c_.getAttr("weights")), 10)
        # self.assertEqual(ca.getAttrByte(c_.getAttr("strideX")), 4)
        # self.assertEqual(ca.getAttrByte(c_.getAttr("strideY")), 4)
        # self.assertEqual(ca.getAttrByte(c_.getAttr("padX")), 1)
        # self.assertEqual(ca.getAttrByte(c_.getAttr("padY")), 1)



if __name__ == '__main__':
    unittest.main()