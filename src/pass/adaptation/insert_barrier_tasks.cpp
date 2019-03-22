#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/barrier_definition.hpp"
#include "include/mcm/target/keembay/barrier_deps.hpp"
#include <algorithm>

static void insertBarrierTasksFcn(const mv::pass::PassEntry &pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::Element &, mv::json::Object &);

namespace mv
{

namespace pass
{

MV_REGISTER_PASS(InsertBarrierTasks)
    .setFunc(insertBarrierTasksFcn)
    .setDescription(
        "This pass inserts barrier tasks into the compute graph");
}
} // namespace mv

static void setBarrierGroupAndIndex(std::vector<mv::Barrier> &barriers)
{
    // TODO: Update barrier group and index based on graph coloring algorithm

    int numBarriers = 0;
    int barrierIndex = 0;
    int barrierGroup = 0;

    for (auto &barrier : barriers)
    {
        // TODO: Update barrier group and index based on graph coloring algorithm
        barrierGroup = numBarriers / 8;
        barrierIndex = numBarriers % 8;

        barrier.setGroup(barrierGroup);
        barrier.setIndex(barrierIndex);

        numBarriers++;
    }
}

static void insertBarriersIntoControlFlowGraph(mv::OpModel &om, mv::ControlModel &cm, const std::vector<mv::Barrier> &barriers)
{
    for (auto &barrier : barriers)
    {
        int group = barrier.getGroup();
        int index = barrier.getIndex();
        int barrierNum = group * 8 + index;

        std::string barrierName("BarrierTask_" + std::to_string(barrierNum));
        om.barrierTask(barrier, barrierName);

        // Add control flows to insert this barrier to the control flow graph
        auto barrierOp = om.getOp(barrierName);

        // Input flow
        for (auto producer : barrier.getProducers())
        {
            auto sourceOp = om.getOp(producer);
            cm.defineFlow(sourceOp, barrierOp);
        }

        // Output flow
        for (auto consumer : barrier.getConsumers())
        {
            auto destOp = om.getOp(consumer);
            cm.defineFlow(barrierOp, destOp);
        }
    }
}

static bool opHasBarrier(const std::string &opName , std::vector<mv::Barrier> &barriers)
{
    for (auto b : barriers)
    {
        auto bConsumers = b.getConsumers() ;
        if ( std::find(bConsumers.begin() , bConsumers.end(), opName ) != bConsumers.end() )
        {
            return true;
        }
    }
    return false;
}

void insertBarrierTasksFcn(const mv::pass::PassEntry &, mv::ComputationModel &model, mv::TargetDescriptor &, mv::Element &, mv::json::Object &)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    std::vector<mv::Barrier> barriers;
/*
    // for testing, add edge from partial serialization
    auto inbounddmaOp = om.getOp("DMATask_3");
    auto aconvOp = om.getOp("DPU_Conv_0");
    auto bconvOp = om.getOp("DMATask_2");
    cm.defineFlow(aconvOp, inbounddmaOp);
    cm.defineFlow(bconvOp, inbounddmaOp);
*/
    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        bool isDPUTask = opType == "DPUTask";
        bool isDMAToCMXTask = (opType == "DMATask" && opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::CMX2DDR);

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
            for (auto &b : barriers)
            {
                if (b.getProducers() == producers)
                {
                    addBarrier = false;

                    for (auto consumer : consumers)
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

    // add/update barriers for control flows added by partial serialization (no tensor on edge)
    for (auto ctlFlow = cm.getFirst(); ctlFlow != cm.getLast(); ++ctlFlow)
    {
        auto ctlFlowOpType = ctlFlow->getOpType();
        std::cout << "Op, type = "<< ctlFlow->getName()<<" , " << ctlFlow->getOpType() <<std::endl;
        if ((ctlFlowOpType=="DMATask") | (ctlFlowOpType == "DPUTask"))
        {
            for (auto parentOp = ctlFlow.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
            {
                auto parentOpType = parentOp->getOpType();
                std::cout << "    parentOp, type = "<< parentOp->getName()<<" , " << parentOp->getOpType() <<std::endl;
                if ((parentOpType == "DPUTask")|(parentOpType == "DMATask" ))
                {
                    std::cout << "    found ctl edge" <<std::endl;
                    auto sinkOpName = ctlFlow->getName();
                    auto sourceOpName = parentOp->getName();

                    // add dependency to existing barrier if this op already preceded by a barrier
                    if (opHasBarrier( sinkOpName, barriers ))
                    {
                        std::cout << "    exsiting preceeding barrier"<< std::endl ;
                        for (mv::Barrier& b : barriers)
                        {
                            auto bConsumers = b.getConsumers() ;
                            if ( std::find(bConsumers.begin() , bConsumers.end(), sinkOpName ) != bConsumers.end() )
                            {
                                b.addProducer(sourceOpName);
                                auto updatedList = b.getProducers();
                                for (auto x : updatedList) std::cout <<      x << std::endl ;
                            }
                        }
                    }

                    // create new barrier if this op had no existing barrier preceeding it
                    else
                    {
                        std::cout << "    NO exsiting preceeding barrier" <<std::endl;
                        std::unordered_set<std::string> producers;
                        std::unordered_set<std::string> consumers;
                        producers.insert(sourceOpName);
                        consumers.insert(sinkOpName);
                        struct mv::Barrier new_barrier(producers, consumers);
                        barriers.push_back(new_barrier);
                    }
                }
            }
        }
    }

    setBarrierGroupAndIndex(barriers);

    insertBarriersIntoControlFlowGraph(om, cm, barriers);
}
