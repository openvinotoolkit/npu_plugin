#ifndef MV_PASS_FUSE_PASSES_HPP_
#define MV_PASS_FUSE_PASSES_HPP_

#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"
#include "include/mcm/pass/pass_registry.hpp"

namespace mv
{

    namespace pass
    {

        namespace __fuse_pass_detail_
        {
        
            void fuseBatchNormFcn(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);
            void fuseBiasFcn(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);
            void fuseReluFcn(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);
            void fuseScaleFcn(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);

        }

    }

}

#endif // MV_PASS_FUSE_PASSES_HPP_
