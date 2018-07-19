#ifndef MV_PASS_GENERATE_BLOB_HPP_
#define MV_PASS_GENERATE_BLOB_HPP_

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"

namespace mv
{

    namespace pass
    {

        namespace __generate_blob_detail_
        {

            void generateBlobFcn(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);
            uint64_t __debug_generateBlobFcn_(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);

        }

    }

}

#endif // MV_PASS_GENERATE_BLOB_HPP_