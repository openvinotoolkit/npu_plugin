#ifndef AVGPOOL2D_HPP_
#define AVGPOOL2D_HPP_

#include "include/mcm/computation/op/pool2d_op.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"

namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)
        class AvgPool2D : public Pool2DOp
        {

        public:

            AvgPool2D(std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string &name);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

            void gatherSerialFields() override;
        };

    }

}

#endif // AVGPOOL2D_HPP_
