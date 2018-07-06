#ifndef FUSE_BIAS_HPP_
#define FUSE_BIAS_HPP_

#include "include/mcm/pass/transform_pass.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"

namespace mv
{

    namespace pass
    {

        class FuseBias : public TransformPass
        {

            bool run_(ComputationModel &model);

        public:

            FuseBias();

        };

    }

}

#endif // FUSE_BIAS_HPP_
