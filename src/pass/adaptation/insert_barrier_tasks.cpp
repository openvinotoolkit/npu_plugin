#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/barrier_definition.hpp"
#include "include/mcm/target/keembay/barrier_deps.hpp"

static void insertBarrierTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(InsertBarrierTasks)
        .setFunc(insertBarrierTasksFcn)
        .setDescription(
            "This pass inserts barrier tasks into the compute graph"
        );
    }
}

void insertBarrierTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    int numBarriers = 0 ;
    int barrierIndex = 0;
    int barrierGroup = 0;

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();

        if (opType == "DPUTask" || (opType == "DMATask" && opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::CMX2DDR))
        {
            std::string barrierName("BarrierTask_" + std::to_string(numBarriers));

            int numProducers = opIt->inputSlots();
            int numConsumers = opIt->outputSlots();
            int wait = -1;

            numBarriers++ ;
            barrierGroup = numBarriers / 8;
            barrierIndex = numBarriers % 8;

            struct mv::Barrier b(barrierGroup, barrierIndex, numProducers, numConsumers);
            struct mv::BarrierDependencies bdep;
            bdep.setWaitBarrier(wait);

            om.barrierTask(b, bdep, barrierName);

            // add control flows to insert this barrier to the control flow graph
            auto barrierOp = om.getOp(barrierName);
            auto inputTensors = opIt->getInputTensor();

            // Input flow
            for (auto tensorIn = inputTensors.begin(); tensorIn != inputTensors.end(); tensorIn++)
            {
                auto sourceOp = om.getSourceOp(*tensorIn);
                cm.defineFlow(sourceOp, barrierOp);
            }

            // Output flow
            auto outputTensors = opIt->getOutputTensor();
            for (auto tensorOut = outputTensors.begin(); tensorOut != outputTensors.end(); tensorOut++)
            {
                auto destOp = om.getSourceOp(*tensorOut);
                cm.defineFlow(barrierOp, destOp);
            }
        }
    }
}
