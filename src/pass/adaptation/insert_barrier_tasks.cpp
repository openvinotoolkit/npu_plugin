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

static void setBarrierGroupAndIndex(std::vector<mv::Barrier>& barriers)
{
    // TODO: Update barrier group and index based on graph coloring algorithm

    int numBarriers = 0 ;
    int barrierIndex = 0;
    int barrierGroup = 0;

    for (auto& barrier: barriers)
    {
        // TODO: Update barrier group and index based on graph coloring algorithm
        barrierGroup = numBarriers / 8;
        barrierIndex = numBarriers % 8;

        barrier.setGroup(barrierGroup);
        barrier.setIndex(barrierIndex);

        numBarriers++;
    }
}

static void insertBarriersIntoControlFlowGraph(mv::OpModel& om, mv::ControlModel& cm, const std::vector<mv::Barrier>& barriers)
{
    for (auto& barrier: barriers)
    {
        int group = barrier.getGroup();
        int index = barrier.getIndex();
        int barrierNum = group * 8 + index;

        int wait = -1;
        struct mv::BarrierDependencies bdep;
        bdep.setWaitBarrier(wait);

        std::string barrierName("BarrierTask_" + std::to_string(barrierNum));
        om.barrierTask(barrier, bdep, barrierName);

        // Add control flows to insert this barrier to the control flow graph
        auto barrierOp = om.getOp(barrierName);

        // Input flow
        for (auto producer: barrier.getProducers())
        {
            auto sourceOp = om.getOp(producer);
            cm.defineFlow(sourceOp, barrierOp);
        }

        // Output flow
        for (auto consumer: barrier.getConsumers())
        {
            auto destOp = om.getOp(consumer);
            cm.defineFlow(barrierOp, destOp);
        }
    }
}

void insertBarrierTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    std::vector<mv::Barrier> barriers;

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        bool isDPUTask = opType == "DPUTask";
        bool isDMAToCMXTask = (opType == "DMATask" && opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::CMX2DDR);
        // TODO: may need to add dealloc tasks

        if (isDPUTask || isDMAToCMXTask)
        {
            std::unordered_set<std::string> producers;
            std::unordered_set<std::string> consumers;

            // Input flow
            auto inputTensors = opIt->getInputTensor();
            for (auto tensorIn = inputTensors.begin(); tensorIn != inputTensors.end(); tensorIn++)
            {
                auto sourceOp = om.getSourceOp(*tensorIn);
                producers.insert(sourceOp->getName());
            }

            auto outputTensors = opIt->getOutputTensor();
            for (auto tensorOut = outputTensors.begin(); tensorOut != outputTensors.end(); tensorOut++)
            {
                auto destOp = om.getSourceOp(*tensorOut);
                consumers.insert(destOp->getName());
            }

            // Do not insert this barrier into the barrierTask list if a barrier
            // with the same producers already exists.
            for (auto& b: barriers)
            {
                if (b.getProducers() == producers)
                {
                    for (auto consumer: consumers)
                        b.addConsumer(consumer);

                    break;
                }
            }

            struct mv::Barrier new_barrier(producers, consumers);
            barriers.push_back(new_barrier);
        }
    }

    setBarrierGroupAndIndex(barriers);

    insertBarriersIntoControlFlowGraph(om, cm, barriers);
}
