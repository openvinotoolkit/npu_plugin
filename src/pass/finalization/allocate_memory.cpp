#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"

static void allocatePopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void allocateUnpopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AllocatePopulatedTensors)
        .setFunc(allocatePopulatedTensorsFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            ""
        );

        MV_REGISTER_PASS(AllocateUnpopulatedTensors)
        .setFunc(allocateUnpopulatedTensorsFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            ""
        );

    }

}

void allocatePopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    ControlModel cm(model);
    DataModel dm(model);

    if (!dm.hasAllocator("ConstantMemory"))
        throw ArgumentError("allocator", "ConstantMemory", "Computation model does not have ConstantMemory specified");

    if (cm.stageSize() == 0)
        throw ArgumentError("stages count", "0", "Computation model does not have stages specified");

    for (auto tIt = dm.tensorBegin(); tIt != dm.tensorEnd(); ++tIt)
    {
        if (tIt->isPopulated())
        {
            auto stageIt = cm.getStage(0);
            dm.allocateTensor("ConstantMemory", stageIt, tIt);
        }
    }

}

void allocateUnpopulatedTensorsFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{


    
}