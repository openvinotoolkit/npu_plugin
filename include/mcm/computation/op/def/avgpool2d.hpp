#ifndef AVGPOOL2D_HPP_
#define AVGPOOL2D_HPP_

#include "include/mcm/computation/op/pool2d_op.hpp"

namespace mv
{   
    
    namespace Op
    {

        /// \todo Add assertions (dimensions)
        class AvgPool2D : public Pool2DOp
        {

        public:

            AvgPool2D(UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string &name) :
            ComputationOp(OpType::AvgPool2D, name),
            Pool2DOp(OpType::AvgPool2D, kernelSize, stride, padding, name)
            {
                addAttr("executable", AttrType::BoolType, true);
            }

        };
        
    }

}

#endif // AVGPOOL2D_HPP_