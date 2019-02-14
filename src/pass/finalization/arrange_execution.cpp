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

// This pass has two main duties
// 1) Put all the control flows needed in the graph
// 2) Put the stages
// Point 2) is trivial for now (just 1 stage), but will be probably updated when Pat completes his analysis

// For point 1, we know that non trivial control flows have already been added during the conversion pass
// We have to add extra layers of control flows
// 1) A layer made up of ControlFlows that start from input and go to every DMA DDR to CMX operation
// Rationale: Until we know more about memory addresses, DMA potentially have no impediments after input operation
// 2) A layer made up of ControlFlows that are coincident with DataFlows
// Rationale: Wherever there is a data dependency, there is also a execution dependency

// WARNING: This two layers of control flow can generate unnecessary control flows.
// At the end those need to be eliminated by Transitive reduction on the ControlFlow pass.
void arrangeKeembayExecutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    std::cout << "Arrange Keembay execution" << std::endl;

    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    // Point 1)

    // Adding layer 1
    auto inputOp = om.getInput();
    auto dmaOps = om.getOps("DMATask");

    for(auto op : dmaOps)
        if(op->get<mv::DmaDirection>("direction") == mv::DDR2CMX)
            cm.defineFlow(inputOp, op);

    // Adding layer 2
    for(auto dataFlow = dm.flowBegin(); dataFlow != dm.flowEnd(); ++dataFlow)
    {
        auto source = cm.switchContext(dataFlow.source());
        auto sink = cm.switchContext(dataFlow.sink());

        // No control flow shall be added between constant operation and their
        // DMA op. Constant Ops are used only to carry data.

        // NOTE: Possibly this check can be eliminated for better readibility
        // Since Transitive Reduction will take care of it
        if(source->getOpType() == "Constant" && sink->getOpType() == "DMATask")
            continue;

        // Must check if a control flow exists already
        bool found = false;
        for(auto childOp = source.leftmostChild(); childOp != cm.opEnd(); ++childOp)
        {
            if(childOp == sink)
            {
                found = true;
                break;
            }
        }
        if(!found)
            cm.defineFlow(dataFlow.source(), dataFlow.sink());
    }

    // Cleaning unnecessary edges.
    cm.transitiveReduction();

    // Point 2)
    auto stage = cm.addStage();
    cm.addToStage(stage, om.getOutput());

    std::cout << "Exiting arrange Keembay execution" << std::endl;

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
