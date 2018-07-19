#ifndef GENERATE_DOT_HPP_
#define GENERATE_DOT_HPP_

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

        namespace __generate_dot_detail_
        {

            void generateDotFcn(ComputationModel& model, TargetDescriptor& targetDesc, json::Object& compDesc);

        }

    }

}

#endif // DOT_PASS_HPP_