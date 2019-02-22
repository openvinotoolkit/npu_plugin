#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include <math.h>

static void setSparsityFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);


namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(Sparsity)
        .setFunc(setSparsityFnc)
        .setDescription(
            "Add sparsity map for layers/tensor that qualify"
        );
    }
}

void setSparsityFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

}
