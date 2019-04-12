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
#include "../contrib/koala/graph/graph.h"
#include "../contrib/koala/io/text.h"
#include "../contrib/koala/io/parsetgml.h"
#include "../contrib/koala/coloring/vertex.h"
#include "../contrib/koala/io/graphml.h"
#include "include/mcm/logger/logger.hpp"


#define MAX_AVAILABLE_BARRIERS 8

using namespace Koala;

struct BigKVertexInfo
{
    int barrierId;

    BigKVertexInfo(int id) : barrierId(id) {};

    friend std::ostream& operator<<(std::ostream& os, const BigKVertexInfo& arg)
    {
        return os << arg.barrierId;
    }
};

struct BigKEdgeInfo
{
    int srcId;
    int snkId;

    BigKEdgeInfo(int src, int snk) : srcId(src), snkId(snk) {};

    friend std::ostream& operator<<(std::ostream& os, const BigKEdgeInfo& arg)
    {
        return os << "src = " << arg.srcId << " | snk = " << arg.snkId;
    }
};

using BIGKoala = Koala::Graph <BigKVertexInfo, BigKEdgeInfo>;

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
    system(systemCmd.c_str());
}

static void addEdge(const mv::Barrier& b1, const mv::Barrier& b2, BarrierInterferenceGraph& big)
{
    auto n1 = big.node_find(b1);
    auto n2 = big.node_find(b2);
    big.edge_insert(n1, n2, b1.getID());
    big.edge_insert(n2, n1, b2.getID());
}


static int colorKoalaGraph(BIGKoala& bigK, std::vector<BIGKoala::PVertex> verts, AssocArray<BIGKoala::PVertex, int>& colors)
{
    return SeqVertColoring::greedy(bigK, colors);
}

static void convertToKoalaGraph(const mv::pass::PassEntry& pass, const BarrierInterferenceGraph& big, BIGKoala& bigK, std::vector<BIGKoala::PVertex>& koalaVerts)
{
    for (auto it = big.node_begin(); it != big.node_end(); ++it)
    {
        BIGKoala::PVertex v = bigK.addVert(BigKVertexInfo((*it).getID()));
        koalaVerts.push_back(v);
    }

    for (auto it = big.edge_begin(); it != big.edge_end(); ++it)
    {
        auto& src = *it->source();
        auto& snk = *it->sink();

        auto srcVtx = std::find_if(
                    koalaVerts.begin(),
                    koalaVerts.end(), 
                    [src](BIGKoala::PVertex& v) { return src.getID() == v->info.barrierId; }
        );

        auto snkVtx = std::find_if(
                    koalaVerts.begin(),
                    koalaVerts.end(), 
                    [snk](BIGKoala::PVertex& v) { return snk.getID() == v->info.barrierId; }
        );

        if (*srcVtx && *snkVtx)
        {
            pass.log(mv::Logger::MessageType::Info,
                "Trying to add edge between: k_src = " + std::to_string((*srcVtx)->info.barrierId)
                + ", k_snk = " + std::to_string((*snkVtx)->info.barrierId));

            if (!bigK.getEdgeNo(*srcVtx, *snkVtx, Koala::Undirected))
            {
                bigK.addEdge(*srcVtx, *snkVtx, BigKEdgeInfo(src.getID(), snk.getID()), Koala::Undirected);
            }
            else
            {
                pass.log(mv::Logger::MessageType::Info,
                    "edge exists between k_src = " + std::to_string((*srcVtx)->info.barrierId)
                    + ", k_snk = " + std::to_string((*snkVtx)->info.barrierId));
            }
        }
    }
}

static void drawBigK(const BIGKoala& bigK)
{
    Koala::IO::GraphML gml;
    Koala::IO::GraphMLGraph *gmlg;

    //Koala::IO::writeGraphText(bigK, std::cout, Koala::IO::RG_VertexLists);
    
    gmlg = gml.createGraph("BIGKoala");
    gmlg->writeGraph(bigK, Koala::IO::gmlIntField(&BigKVertexInfo::barrierId, "barrierId"),
                            Koala::IO::gmlIntField(&BigKEdgeInfo::srcId, "srcId")
                            & Koala::IO::gmlIntField(&BigKEdgeInfo::snkId, "snkId"));
    gml.writeFile("bigK.graphml");
}

static BarrierInterferenceGraph generateBarrierInterferenceGraph(mv::OpModel& om, std::vector<mv::Barrier>& barriers)
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
                auto b1It = big.node_find(b1);
                auto b2It = big.node_find(b2);
                auto concurrentBarrier = std::find(b2.getProducers().begin(), b2.getProducers().end(), c);
                if (concurrentBarrier != b2.getProducers().end())
                {
                    // b1's consumer is the same as b2's producer, so add an edge
                    if (!mv::edgeExists(big, b1It, b2It) && !mv::edgeExists(big, b2It, b1It))
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
                            if (!mv::edgeExists(big, b1It, b2It) && !mv::edgeExists(big, b2It, b1It))
                                addEdge(b1, b2, big);
                        }
                    }
                }
            }
        }
    }

    return big;
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

static void combineRedundantBarriers(std::vector<mv::Barrier>& barriers)
{
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

    auto newEnd = std::remove_if(barriers.begin(), barriers.end(), [](mv::Barrier& x)
        { return !(x.hasConsumers()); } );
    barriers.erase(newEnd, barriers.end());
}

static void addDataFlowBarriers(mv::OpModel& om, std::vector<mv::Barrier>& barriers)
{
    auto sortedOps = om.topologicalSort();

    for (auto opIt: sortedOps)
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
}

static void addControlFlowBarriers(mv::ControlModel& cm, std::vector<mv::Barrier>& barriers)
{
    // add/update barriers for control flows added by partial serialization (no tensor on edge)

    auto sortedControlFlow = cm.topologicalSort();

    for (auto ctlFlow: sortedControlFlow)
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
}

static void setBarrierGroupAndIndex(const mv::pass::PassEntry& pass, mv::OpModel& om, std::vector<mv::Barrier>& barriers, mv::Element& passDesc)
{
    std::string indexAssignment = passDesc.get<std::string>("barrier_index_assignment");
    
    BarrierInterferenceGraph big = generateBarrierInterferenceGraph(om, barriers);
    drawBIG(big, "big.dot");

    BIGKoala bigK;
    std::vector<BIGKoala::PVertex> koalaVertices;
    convertToKoalaGraph(pass, big, bigK, koalaVertices);
    drawBigK(bigK);

    AssocArray<BIGKoala::PVertex, int> colors;
    int numColors = colorKoalaGraph(bigK, koalaVertices, colors);
    if (numColors > MAX_AVAILABLE_BARRIERS)
        throw mv::RuntimeError(om,
                "Cannot execute graph with " +
                std::to_string(MAX_AVAILABLE_BARRIERS) +
                " barriers; more graph serialization required.");

    for (int i = 0; i < bigK.getVertNo(); i++)
    {
        pass.log(mv::Logger::MessageType::Info,
            "bId = " + std::to_string(koalaVertices[i]->info.barrierId)
            + " : color = " + std::to_string(colors[koalaVertices[i]]));
    }

    if (indexAssignment == "Static")
    {
        // assign colors to indices
        for (auto& b: barriers)
        {
            auto koalaVertex = std::find_if(koalaVertices.begin(), koalaVertices.end(),
                            [b](BIGKoala::PVertex v){ return v->info.barrierId == b.getID(); } );

            b.setIndex(colors[*koalaVertex]);
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

void insertBarrierTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    std::vector<mv::Barrier> barriers;

    addDataFlowBarriers(om, barriers);

    addControlFlowBarriers(cm, barriers);

    combineRedundantBarriers(barriers);

    setBarrierGroupAndIndex(pass, om, barriers, passDesc);

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
