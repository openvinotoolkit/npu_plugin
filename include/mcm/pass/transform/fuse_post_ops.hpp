#ifndef FUSE_POST_OPS_HPP_
#define FUSE_POST_OPS_HPP_

#include "include/mcm/pass/transform_pass.hpp"

namespace mv
{

    namespace pass
    {

        class FusePostOps : public TransformPass
        {

            bool run_(ComputationModel &model)
            {

            }

        public:

            FusePostOps()
            {
                
            }

        };

    }

}

#endif // DEPLOY_PASS_HPP_