#ifndef GENERATE_JSON_HPP_
#define GENERATE_JSON_HPP_

#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/utils/parser/exception/argument_error.hpp"
#include <fstream>


namespace mv
{

    namespace pass 
    {

        namespace __generate_json_detail_
        {
            void generateJsonFcn(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);
            uint64_t __debug_generateJsonFcn_(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);
        }

    }

}

#endif // JSON_PASS_HPP_
