#ifndef FUSE_SCALE_HPP_
#define FUSE_SCALE_HPP_

#include "include/mcm/pass/transform_pass.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"

namespace mv
{

    namespace pass
    {

        class FuseScale : public TransformPass
        {

            bool run_(ComputationModel &model);

        public:

            FuseScale();

        };

    }

}

#endif // FUSE_SCALE_HPP_
