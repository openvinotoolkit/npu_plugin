#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/barrier_definition.hpp"
#include "include/mcm/target/keembay/barrier_deps.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/path_exists.hpp"
#include "include/mcm/algorithms/edge_exists.hpp"
#include <algorithm>

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

using BarrierInterferenceGraph = mv::graph<mv::Barrier, int>;

static void drawBIG(BarrierInterferenceGraph& g, std::string outputFile)
{
    std::ofstream ostream;

    ostream.open(outputFile, std::ios::trunc | std::ios::out);
    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        std::string vIdx = std::to_string((*it).getID());
        std::string nodeDef = "\t\"" + vIdx + "\" [shape=box,";
        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + vIdx + "</B></FONT></TD></TR>";
        nodeDef += "</TABLE>>";
        ostream << nodeDef << "];\n";
    }

    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        std::string vIdxSrc = std::to_string((*(it->source())).getID());
        std::string vIdxSnk = std::to_string((*(it->sink())).getID());
        std::string edgeDef = "\t\"" + vIdxSrc + "\" -> \"" +  vIdxSnk + "\"";
        ostream << edgeDef << "\n";
    }
    ostream << "}\n";
    ostream.close();

    std::string pngFile = outputFile.substr(0, outputFile.find(".dot")) + ".png";

    std::string systemCmd = "dot -Tpng " + outputFile + " -o " + pngFile;
    std::cout << systemCmd;
    system(systemCmd.c_str());
}

static void addEdge(const mv::Barrier& b1, const mv::Barrier& b2, BarrierInterferenceGraph& big)
{
    auto n1 = big.node_find(b1);
    auto n2 = big.node_find(b2);
    big.edge_insert(n1, n2, b1.getID());
    big.edge_insert(n2, n1, b2.getID());
}

static void generateBarrierInterferenceGraph(mv::OpModel& om, std::vector<mv::Barrier>& barriers)
{
    BarrierInterferenceGraph big;

    // Insert all barriers into interference graph
    for (auto& b: barriers)
    {
        big.node_insert(b);
    }

    // Draw edges
    // Case 1: 2 barriers share the same op in their producer and consumer list,
    // like so: b1->op->b2. i.e. op is in b1's consumer list and b2's producer list.
    // In this case b1 and b2 are concurrent, so an edge exists between b1 and b2.

    for (auto& b1: barriers)
    {
        for (auto& c: b1.getConsumers())
        {
            for (auto& b2: barriers)
            {
                auto concurrentBarrier = std::find(b2.getProducers().begin(), b2.getProducers().end(), c);
                if (concurrentBarrier != b2.getProducers().end())
                {
                    // b1's consumer is the same as b2's producer, so add an edge
                    addEdge(b1, b2, big);
                }
            }
        }
    }

    // Case 2: b1 and b2 are on parallel paths.
    // Try all possible combinations of operations & check whether they're an ancestor
    // or a descendent of this op.
    // Subsequently check whether an edge already exists between barriers associated with
    // the two ops. If not, add an edge between the two barriers in the BIG.

    for (auto& b1: barriers)
    {
        for (auto& c1: b1.getConsumers())
        {
            for (auto& b2: barriers)
            {
                if (b1 != b2)
                {
                    for (auto& c2: b2.getConsumers())
                    {
                        auto b1It = big.node_find(b1);
                        auto b2It = big.node_find(b2);
                        // BIG is an undirected graph, so nodes in the opModel are considered to be
                        // disconnected only if c2 is neither an ancestor, nor a descendent of c1.
                        // pathExists checks only from source to sink on a directed graph, hence
                        // this check needs to be performed both from c1 to c2, and c2 to c1.
                        if (!om.pathExists(om.getOp(c1), om.getOp(c2)) && !om.pathExists(om.getOp(c2), om.getOp(c1)))
                        {
                            if (!mv::edgeExists(big, b1It, b2It))
                                addEdge(b1, b2, big);
                        }
                    }
                }
            }
        }
    }

    drawBIG(big, "big.dot");
}

static void setBarrierGroupAndIndex(mv::OpModel& om, std::vector<mv::Barrier>& barriers, mv::Element& passDesc)
{
    int numBarriers = 0 ;
    int barrierIndex = 0;
    int barrierGroup = 0;
    std::string indexAssignment = passDesc.get<std::string>("barrier_index_assignment");

    // TODO: Update barrier group and index based on graph coloring algorithm
    if (indexAssignment == "Static")
    {
        // 1) generate BIG
        // 2) Apply coloring algorithm to assign indices below

        generateBarrierInterferenceGraph(om, barriers);

        for (auto& barrier: barriers)
        {
            barrierGroup = barrier.getID() / 8;
            barrierIndex = barrier.getID() % 8;

            barrier.setGroup(barrierGroup);
            barrier.setIndex(barrierIndex);
        }
    }
    else
    {
        for (auto& barrier: barriers)
            barrier.setIndex(barrier.getID());
    }
}

static void insertBarriersIntoControlFlowGraph(mv::ComputationModel& model, const mv::Element& passDesc, const std::vector<mv::Barrier>& barriers)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    std::string indexAssignment = passDesc.get<std::string>("barrier_index_assignment");

    for (auto& barrier: barriers)
    {
        std::string barrierName("BarrierTask_" + std::to_string(barrier.getID()));
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

static bool opHasBarrier(const std::string& opName , std::vector<mv::Barrier>& barriers)
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

void insertBarrierTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    std::vector<mv::Barrier> barriers;

    //add barriers due to data flow dependencies
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        bool isDPUTask = opType == "DPUTask";
        bool isDMAToCMXTask = (opType == "DMATask" && opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::CMX2DDR);

        if (isDPUTask || isDMAToCMXTask)
        {
            std::unordered_set<std::string> producers;
            std::unordered_set<std::string> consumers;

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

            struct mv::Barrier new_barrier(producers, consumers);
            barriers.push_back(new_barrier);
        }
    }

    // add/update barriers for control flows added by partial serialization (no tensor on edge)
    for (auto ctlFlow = cm.getFirst(); ctlFlow != cm.getLast(); ++ctlFlow)
    {
        auto ctlFlowOpType = ctlFlow->getOpType();
        if ((ctlFlowOpType == "DMATask") || (ctlFlowOpType == "DPUTask"))
        {
            for (auto parentOp = ctlFlow.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
            {
                auto parentOpType = parentOp->getOpType();
                if ((parentOpType == "DPUTask") || (parentOpType == "DMATask" ))
                {
                    auto sinkOpName = ctlFlow->getName();
                    auto sourceOpName = parentOp->getName();

                    // add dependency to existing barrier if this op already preceded by a barrier
                    if (opHasBarrier( sinkOpName, barriers ))
                    {
                        for (mv::Barrier& b : barriers)
                        {
                            auto bConsumers = b.getConsumers() ;
                            if ( std::find(bConsumers.begin() , bConsumers.end(), sinkOpName ) != bConsumers.end() )
                            {
                                b.addProducer(sourceOpName);
                                auto updatedList = b.getProducers();
                            }
                        }
                    }

                    // create new barrier if this op had no existing barrier preceeding it
                    else
                    {
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

    // combine redundant barriers (same producers) into 1 barrier
    for (auto b = barriers.begin(); b != barriers.end(); b++ )
    {
        for (auto c = std::next(b); c!= barriers.end(); c++ )
        {
            if ((b->getProducers() == c->getProducers()) && (c->hasConsumers()) && (b->hasConsumers()))
            {
                // move c consumers to b
                for (auto consumer : c->getConsumers())
                {
                    b->addConsumer(consumer);
                    c->removeConsumer(consumer);
                }
            }
        }
    }

    // remove redundant barriers
    auto newEnd = std::remove_if(barriers.begin(), barriers.end(), [](mv::Barrier& x)
        { return !(x.hasConsumers()); } );
    barriers.erase(newEnd, barriers.end());

    setBarrierGroupAndIndex(om, barriers, passDesc);

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
