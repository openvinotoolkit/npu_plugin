#ifndef AVGPOOL2D_HPP_
#define AVGPOOL2D_HPP_

#include "include/mcm/computation/op/pool2d_op.hpp"

namespace mv
{   
    
    namespace op
    {

        /// \todo Add assertions (dimensions)
        class AvgPool2D : public Pool2DOp
        {

        public:

            AvgPool2D(UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const std::string &name);
            AvgPool2D(mv::json::Value &obj);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };
        
    }

}

#endif // AVGPOOL2D_HPP_
