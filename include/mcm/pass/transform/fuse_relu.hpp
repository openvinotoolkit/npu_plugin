#ifndef FUSE_RELU_HPP_
#define FUSE_RELU_HPP_

#include "include/mcm/pass/transform_pass.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"

namespace mv
{

    namespace pass
    {

        class FuseReLU : public TransformPass
        {

            bool run_(ComputationModel &model);

        public:

            FuseReLU();

        };

    }

}

#endif // DEPLOY_PASS_HPP_