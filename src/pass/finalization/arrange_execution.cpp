#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void arrangeLinearExecutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void arrangeKeembayExecutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);


namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ArrangeLinearExecution)
        .setFunc(arrangeLinearExecutionFcn)
        .setDescription(
            ""
        );

        MV_REGISTER_PASS(ArrangeKeembayExecution)
        .setFunc(arrangeKeembayExecutionFcn)
        .setDescription(
            ""
        );

    }

}

// NOTE: Probably this function is not needed anymore as we are adding ControlFlows in a more cautious way in two separate passes
bool checkControlFlowExistence(mv::Control::OpListIterator source, mv::Control::OpListIterator sink, mv::ControlModel cm)
{
    bool found = false;
    for(auto childOp = source.leftmostChild(); childOp != cm.opEnd(); ++childOp)
    {
        if(childOp == sink)
        {
            found = true;
            break;
        }
    }

    return found;
}

// This pass has one main duty
// 1) Put the stages
// Point 1) is trivial for now (just 1 stage), but will be probably updated when Pat completes his analysis

void arrangeKeembayExecutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    pass.log(mv::Logger::MessageType::Debug, "Starting arrange Keembay execution");

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // Point 2)
    auto stage = cm.addStage();
    cm.addToStage(stage, om.getOutput());

    pass.log(mv::Logger::MessageType::Debug, "Exiting arrange Keembay execution");

}

void arrangeLinearExecutionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    std::cout << "Arrange execution" << std::endl;

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto currentOp = om.getInput();

    while (currentOp != om.getOutput())
    {
        mv::Data::OpListIterator nextOp;

        if (currentOp.childrenSize() == 1)
        {
            nextOp = currentOp.leftmostChild();
            cm.defineFlow(currentOp, nextOp);
            if (nextOp->getOpType() != "Output")
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
                        if (parent->hasTypeTrait("executable") || parent->getOpType() == "Input")
                            ++result;
                    return result;
                };

                while (nextOp.parentsSize() == 1 || executableParents() == 1)
                {
                    cm.defineFlow(currentOp, nextOp);
                    if (nextOp->getOpType() != "Output")
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

    std::cout << "Exiting arrange execution" << std::endl;

}
