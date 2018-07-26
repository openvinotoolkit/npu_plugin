#ifndef MAXPOOL2D_HPP_
#define MAXPOOL2D_HPP_

#include "include/mcm/computation/op/pool2d_op.hpp"

namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)
        class MaxPool2D : public Pool2DOp
        {

        public:

            MaxPool2D(UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string &name);
            MaxPool2D(mv::json::Value &obj);

        };

    }

}

#endif // MAXPOOL2D_HPP_
