#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"

static void arrangeLinearExecutionFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ArrangeLinearExecution)
        .setFunc(arrangeLinearExecutionFcn)
        .setGenre(PassGenre::Finalization)
        .setDescription(
            ""
        );

    }

}

void arrangeLinearExecutionFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;
    OpModel om(model);
    ControlModel cm(model);

    auto currentOp = om.getInput();

    while (currentOp != om.getOutput())
    {
        Data::OpListIterator nextOp;

        if (currentOp.childrenSize() == 1)
        {
            nextOp = currentOp.leftmostChild();
            cm.defineFlow(currentOp, nextOp);
            if (nextOp->getOpType() != OpType::Output)
            {
                auto stage = cm.addStage();
                cm.addToStage(stage, nextOp);
            }
            currentOp = nextOp;
        }
        else
        {
            for (auto nextChildOp = currentOp.leftmostChild(); nextChildOp != om.opEnd(); ++nextChildOp)
            {
                nextOp = nextChildOp;

                auto executableParents = [&om, &nextOp]()
                {
                    std::size_t result = 0;
                    for (auto parent = nextOp.leftmostParent(); parent != om.opEnd(); ++parent)
                        if (parent->isExecutable() || parent->getOpType() == OpType::Input)
                            ++result;
                    return result;
                };

                while (nextOp.parentsSize() == 1 || executableParents() == 1)
                {
                    cm.defineFlow(currentOp, nextOp);
                    if (nextOp->getOpType() != OpType::Output)
                    {
                        auto stage = cm.addStage();
                        cm.addToStage(stage, nextOp);
                    }
                    currentOp = nextOp;
                    nextOp = currentOp.leftmostChild();
                }
            }
        }
    }
} 
