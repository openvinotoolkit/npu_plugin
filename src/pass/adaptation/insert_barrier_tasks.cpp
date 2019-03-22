#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/barrier_definition.hpp"
#include "include/mcm/target/keembay/barrier_deps.hpp"
#include "include/mcm/target/keembay/workloads.hpp"

static void insertBarrierTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void updateCountsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(InsertBarrierTasks)
        .setFunc(insertBarrierTasksFcn)
        .setDescription(
            "This pass inserts barrier tasks into the compute graph"
        );

        MV_REGISTER_PASS(UpdateBarrierProducerConsumerCounts)
        .setFunc(updateCountsFcn)
        .setDescription(
            "This pass updates producer and consumer counts in barriers based on workloads in producer and consumer \
            DxxTasks in the compute graph"
        );

    }

}

static void setBarrierGroupAndIndex(std::vector<mv::Barrier>& barriers, const mv::Element& passDesc)
{
    // TODO: Update barrier group and index based on graph coloring algorithm

    int numBarriers = 0 ;
    int barrierIndex = 0;
    int barrierGroup = 0;
    std::string indexAssignment = passDesc.get<std::string>("barrier_index_assignment");

    for (auto& barrier: barriers)
    {
        if (indexAssignment == "Static")
        {
            barrierGroup = numBarriers / 8;
            barrierIndex = numBarriers % 8;
        }
        else
        {
            barrierGroup = 0;
            barrierIndex = numBarriers;
        }

        // TODO: Update barrier group and index based on graph coloring algorithm
        barrier.setGroup(barrierGroup);
        barrier.setIndex(barrierIndex);

        numBarriers++;
    }
}

static void insertBarriersIntoControlFlowGraph(mv::ComputationModel& model, const mv::Element& passDesc, const std::vector<mv::Barrier>& barriers)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    std::string indexAssignment = passDesc.get<std::string>("barrier_index_assignment");

    for (auto& barrier: barriers)
    {
        int group = barrier.getGroup();
        int index = barrier.getIndex();
        int barrierNum;

        if (indexAssignment == "Static")
            barrierNum = group * 8 + index;
        else
            barrierNum = index;

        std::string barrierName("BarrierTask_" + std::to_string(barrierNum));
        om.barrierTask(barrier, barrierName);

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

void insertBarrierTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
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
            bool addBarrier = true;
            for (auto& b: barriers)
            {
                if (b.getProducers() == producers)
                {
                    addBarrier = false;

                    for (auto consumer: consumers)
                        b.addConsumer(consumer);

                    break;
                }
            }

            if (addBarrier)
            {
                struct mv::Barrier new_barrier(producers, consumers);
                barriers.push_back(new_barrier);
            }
        }
    }

    setBarrierGroupAndIndex(barriers, passDesc);

    insertBarriersIntoControlFlowGraph(model, passDesc, barriers);
}

static void updateCountsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto barrierTasks = om.getOps("BarrierTask");

    for (auto bt: barrierTasks)
    {
        auto& barrier = bt->get<mv::Barrier>("Barrier");

        for (auto producer: barrier.getProducers())
        {
            auto producerOp = om.getOp(producer);
            if (producerOp->hasAttr("Workloads"))
            {
                auto& workloads = producerOp->get<mv::Workloads>("Workloads");
                int count = barrier.getNumProducers();
                count = count - 1 + workloads.nWorkloads();
                barrier.setNumProducers(count);
            }
        }

        for (auto consumer: barrier.getConsumers())
        {
            auto consumerOp = om.getOp(consumer);

            if (consumerOp->hasAttr("Workloads"))
            {
                auto& workloads = consumerOp->get<mv::Workloads>("Workloads");
                int count = barrier.getNumConsumers();
                count = count - 1 + workloads.nWorkloads();
                barrier.setNumConsumers(count);
            }
        }
    }
}
