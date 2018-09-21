#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/model/types.hpp"

static void splitsOverH(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(SplitsOverH)
        .setFunc(splitsOverH)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            "This pass handles splits over H for each HW CONV"
        );
    }
}



//ASSUMPTION: This pass must be executed after the Mark Hardware Convolution pass.
//REASON: There is no need to pad tensors not involved in HW operations at all.
void splitsOverH(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& pobj, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::Nce1 nce;

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
    {
        if(operationIt->getOpType() != mv::OpType::Conv2D)
            continue;
        if(!operationIt->hasAttr("NCE1_Compatible"))
            continue;
        if(!operationIt->getAttr("NCE1_Compatible").getContent<int>())
            continue;


    }

}
