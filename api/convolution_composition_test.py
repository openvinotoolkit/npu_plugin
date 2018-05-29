import numpy as np
import unittest

import composition_api as ca

NWHC = (0, 3, 2, 1)


class Counter():
    class __Counter():
        def __init__(self):
            self.val = 0

        def count(self):
            self.val += 1
            return self.val

    instance = None

    def __init__(self):
        if not Counter.instance:
            Counter.instance = Counter.__Counter()

    def __getattr__(self, arg):
        return getattr(self.instance, arg)

    def count(self):
        return self.instance.count()


class Link():
    def __init__(self, shape, data=None):
        self.ref = Counter.count()
        self.shape = shape
        self.format = np.float16
        self.layout = NHWC
        if data is None:
            self.data = np.zeros(shape)
        else:
            self.data = data


class Layer():
    def __init__(self):
        self.ref = Counter.count()
        pass


class Empty(Layer):
    def __init__(self, i, o):
        super().__init__()
        self.i = i
        self.o = o


class Output(Layer):
    def __init__(self, out):
        super().__init__()
        pass


class Conv(Layer):
    def __init__(self, inP, weP, rx, ry, px, py):
        super().__init__()
        self.inP = inP
        self.weP = weP
        self.rx = rx
        self.ry = ry
        self.px = px
        self.py = py
        self.out = None


class TestComposition(unittest.TestCase):

    def setUp(self):

        self.g = ca.getOM()      # mv::OpModel om;

        print("Graph:", self.g)

        self.assertFalse(self.g.isValid())

    def testMinimal(self):
        shape = ca.getShape(1, 32, 32, 3)
        dtype = ca.getDtype()
        order = ca.getOrder()

        in_ = self.g.input(shape, dtype, order)
        out_ = self.g.output(in_)

        self.assertTrue(self.g.isValid())

    def testConvolution(self):

        shape = ca.getShape(1, 32, 32, 3)
        dtype = ca.getDtype()
        order = ca.getOrder()

        weights = ca.getData(np.array(
            [   1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27]
            )
        )
        weights = ca.getConstantTensor(ca.getShape(3, 3, 1, 3), dtype, order, weights)

        in_ = self.g.input(shape, dtype, order)
        c_ = self.conv(in_, weights, 4, 4, 1, 1)
        out_ = self.g.output(c_)

        self.assertTrue(self.g.isValid())
        self.assertEqual(out_.getOutputShape(), ca.getShape(1, 8, 8, 3))
        self.assertEqual(c_.attrsCount(), 10)

        self.assertEqual(ca.getAttrByte(c_.getAttr("weights")), 10)
        self.assertEqual(ca.getAttrByte(c_.getAttr("strideX")), 4)
        self.assertEqual(ca.getAttrByte(c_.getAttr("strideY")), 4)
        self.assertEqual(ca.getAttrByte(c_.getAttr("padX")), 1)
        self.assertEqual(ca.getAttrByte(c_.getAttr("padY")), 1)


    def test_simple(self):
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()