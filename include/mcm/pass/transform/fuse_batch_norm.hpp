#ifndef FUSE_BATCH_NORM_HPP_
#define FUSE_BATCH_NORM_HPP_

#include "include/mcm/pass/transform_pass.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"

namespace mv
{

    namespace pass
    {

        class FuseBatchNorm : public TransformPass
        {

            bool run_(ComputationModel &model);

        public:

            FuseBatchNorm();

        };

    }

}

#endif // DEPLOY_PASS_HPP_