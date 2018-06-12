#ifndef MAXPOOL2D_HPP_
#define MAXPOOL2D_HPP_

#include "include/fathom/computation/op/pool2d_op.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)
    class MaxPool2D : public Pool2DOp
    {

    public:

        MaxPool2D(UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string &name) :
        ComputationOp(OpType::MaxPool2D, name),
        Pool2DOp(OpType::MaxPool2D, kernelSize, stride, padding, name)
        {
            addAttr("executable", AttrType::BoolType, true);
        }

    };

}

#endif // MAXPOOL2D_HPP_