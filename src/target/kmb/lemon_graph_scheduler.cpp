#include "include/mcm/target/kmb/lemon_graph_scheduler.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

#include "lemon/dijkstra.h"
#include <lemon/preflow.h>
#include <lemon/connectivity.h>

mv::LemonGraphScheduler::LemonGraphScheduler(): 
nodes_(graph_), 
edges_(graph_), 
edgesMemory_(graph_), 
edgesLength_(graph_), 
graphSourceNode_(lemon::INVALID), 
graphSinkNode_(lemon::INVALID)
{  }

mv::LemonGraphScheduler::~LemonGraphScheduler()
{
    //delete graph_;
}

lemon::ListDigraph& mv::LemonGraphScheduler::getGraph()
{
    return graph_;
}

/**
 * @brief Convert McM graph (control model view) to Lemon graph and store the data required to perform the max topoloigcal cut algorithm on the Lemon graph edges
 * @param pass  - pass object
 * @param model - McM computation model
 */
void  mv::LemonGraphScheduler::convertMcMGraphToLemonGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model) 
{
    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    /* For each task in the ControlModel view of the MCM graph create a corresponding node (task) in the Lemon graph. */
    for (auto opIt = cm.getFirst(); opIt != cm.opEnd(); ++opIt)
    {
        /*Add node to Lemon graph*/
        bool nodeAdded = false;
        /*Check if node is a DMA task "CMX to DDR" (this is the sink node in Lemon graph and we need to keep track of it) */
        //if (opIt->hasAttr("lastDMAOp") && opIt->get<bool>("lastDMAOp"))
        if (opIt->hasAttr("MaxCutSinkNode") && opIt->get<bool>("MaxCutSinkNode"))
        {
            this->log(mv::Logger::MessageType::Debug, "Adding vertex to Lemon graph: " + opIt->getName());

            lemon::ListDigraph::Node currentNode = this->graph_.addNode();
            this->nodes_[currentNode] = nodeDescription(this->graph_.id(currentNode), opIt->getName(), opIt->getOpType(), 0, false, true);
            this->graphSinkNode_ = currentNode;
            nodeAdded = true;
        }
        /*Keep track of the source node i.e. input*/
        if (opIt->getOpType() == "Input") 
        { 
            pass.log(mv::Logger::MessageType::Debug, "Adding vertex to Lemon graph: " + opIt->getName());
            lemon::ListDigraph::Node currentNode = this->graph_.addNode();
            this->nodes_[currentNode] = nodeDescription(this->graph_.id(currentNode), opIt->getName(), opIt->getOpType(), 0, true, false);
            this->graphSourceNode_ = currentNode;
            nodeAdded = true;
        }
        else if (!nodeAdded) 
        {   /*All other nodes between source and sink node*/
            pass.log(mv::Logger::MessageType::Debug, "Adding vertex to Lemon graph: " + opIt->getName());
            lemon::ListDigraph::Node currentNode = this->graph_.addNode();
            this->nodes_[currentNode] = nodeDescription(this->graph_.id(currentNode), opIt->getName(), opIt->getOpType(), 0, false, false);
            nodeAdded = true;
        }
    }

    /* Add the edges to the Lemon graph store attributes on the edges to perform the max topoloigcal cut algorithm.
     * Iterate over the the control flow edges in the MCMgraph.  */
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) 
    {
        /* 1. Don't add the edge going to Ouput in the MCM graph to the Lemon graph
         * 2. Don't add edge coming from a ConstantInt operation (Sparsity Map and Weights Table) */
        // if (flowIt.sink()->getOpType() != "Output" &&
        //     flowIt.source()->getOpType() != "ConstantInt" &&
        //     flowIt.source()->getOpType() != "ConstantDataElement" &&
        //     flowIt.source()->getOpType() != "Constant")
        // {
            // if (flowIt.sink()->getOpType() == "Deallocate" && flowIt.source()->getOpType() == "Deallocate")
            // {
            //     pass.log(mv::Logger::MessageType::Debug, "Not adding dealloc-dealloc edge: " + flowIt->getName());
            //     continue;
            // }

            auto sourceName = flowIt.source()->getName();
            auto sinkName  = flowIt.sink()->getName();
            if ((sinkName.substr(0,6) == "Output") && ( !flowIt.sink()->hasAttr("MaxCutSinkNode") )) 
                continue;   // keep graphs small. Only add edge to "output" if its the MaxcutSinkNode

            /*If the control flow has a memoryRequirment attribute add it to edges*/
            uint64_t memReq = 0;
            if(flowIt->hasAttr("MemoryRequirement"))
                memReq = flowIt->get<int>("MemoryRequirement");
            
            pass.log(mv::Logger::MessageType::Debug, "Adding edge to Lemon graph from: " + sourceName + " --> " + sinkName + " with memory requirement " + std::to_string(memReq));
            lemon::ListDigraph::Node sourceNode, sinkNode;
            for (lemon::ListDigraph::NodeIt n(this->graph_); n != lemon::INVALID; ++n)
            {
                mv::nodeDescription desc = this->nodes_[n];
                if (sourceName == desc.name) 
                    sourceNode = n;
                else if (sinkName == desc.name)
                    sinkNode = n;
            }
            lemon::ListDigraph::Arc tmpEdge = this->graph_.addArc(sourceNode, sinkNode);
            this->edges_[tmpEdge] = edgeDescription(this->graph_.id(tmpEdge), memReq, flowIt->getName());
            this->edgesMemory_[tmpEdge] = memReq;
            this->edgesLength_[tmpEdge] = 1;
        // }
    }
    pass.log(mv::Logger::MessageType::Debug, "Lemon graph has " + std::to_string(lemon::countNodes(this->graph_)) + " nodes and " + std::to_string(lemon::countArcs(this->graph_)) + " edges");
    pass.log(mv::Logger::MessageType::Debug, "Source: " + this->nodes_[this->graphSourceNode_].name + " | Sink: " + this->nodes_[this->graphSinkNode_].name);
}

uint64_t mv::LemonGraphScheduler::calculateFMax(mv::ComputationModel& model) 
{
    mv::ControlModel cm(model);
    /*Compute Fmax - (defined as sum of memory requirments + 1)*/
    uint64_t Fmax = 0;
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) {
        if(flowIt->hasAttr("MemoryRequirement")) {
            Fmax += flowIt->get<int>("MemoryRequirement");
        }
    }
    Fmax += 1; /*add 1 to Fmax as per algorithm*/
    return Fmax;
}

std::pair<int, std::vector<mv::edgeDescription>> mv::LemonGraphScheduler::calculateMaxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model) 
{
    mv::ControlModel cm(model);

    /* Calculate Fmax - Defined as sum of memory requirments + 1)*/
    auto Fmax = this->calculateFMax(model); 
    pass.log(mv::Logger::MessageType::Debug, "FMax: " + std::to_string(Fmax));
    
    /* Construct the graph demand: circle over the edge and add a flow equal to Fmax on a shorest path containing that node */

    /*For each edge
     * Find the shortest path from source node (Input) to the edge's source node and
     * Find the shortest path from the edge's sink node to the graph sink node (DMA task CMX to DDR) */
    for (lemon::ListDigraph::ArcIt thisArc(this->graph_); thisArc != lemon::INVALID; ++thisArc)
    {   
        /*get the source and sink node of this edge*/    
        lemon::ListDigraph::Node northNode = this->graph_.source(thisArc);
        lemon::ListDigraph::Node southNode = this->graph_.target(thisArc);
        pass.log(mv::Logger::MessageType::Debug, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        pass.log(mv::Logger::MessageType::Debug, "North Node: " + this->nodes_[northNode].name);
        pass.log(mv::Logger::MessageType::Debug, "South Node: " + this->nodes_[southNode].name);

        // >>>>>>>>>>>>>>>>>>>>> Source -> this Node <<<<<<<<<<<<<<<<<<<<<<<<<<
        /*Find the shortest path from the "input" node of graph to the source node of this edge*/
        lemon::Dijkstra<lemon::ListDigraph, lemon::ListDigraph::ArcMap<int>> dijkstraSourceToNode(this->graph_, this->edgesLength_); 
        dijkstraSourceToNode.run(this->graphSourceNode_, northNode);

        pass.log(mv::Logger::MessageType::Debug, "<<< Dystra: Source (" + this->nodes_[this->graphSourceNode_].name + ") -> Node (" + this->nodes_[northNode].name + ") >>> ");
        for (lemon::ListDigraph::Node currentNode = northNode; currentNode != this->graphSourceNode_; currentNode = dijkstraSourceToNode.predNode(currentNode))
        {   //walk the shortest path route
            if (currentNode != lemon::INVALID && dijkstraSourceToNode.reached(currentNode))
            {
                lemon::ListDigraph::Arc predarc = dijkstraSourceToNode.predArc(currentNode);
                this->edges_[predarc].flow += Fmax;
                pass.log(mv::Logger::MessageType::Debug, "currentEdge: " + this->edges_[predarc].name + " : " + std::to_string(this->edges_[predarc].flow));
            }
        }
        pass.log(mv::Logger::MessageType::Debug, "FINISHED SOURCE TO NODE");
        // >>>>>>>>>>>>>>>>>>>>> this Node -> Sink <<<<<<<<<<<<<<<<<<<<<<<<<<
        /*Find the shortest path from the sink node of this edge to the sink node of graph (DMA task CMX to DDR)*/
        //lemon::Dijkstra<lemon::ListDigraph, lemon::ListDigraph::ArcMap<uint64_t>> dijkstraNodeToSink(this->graph_, this->edgesMemory_);
        lemon::Dijkstra<lemon::ListDigraph, lemon::ListDigraph::ArcMap<int>> dijkstraNodeToSink(this->graph_, this->edgesLength_);
        dijkstraNodeToSink.run(southNode, this->graphSinkNode_);

        pass.log(mv::Logger::MessageType::Debug, "<<<Dystra: Node -> Sink>>> ");
        pass.log(mv::Logger::MessageType::Debug, "<<< Dystra: Node (" + this->nodes_[southNode].name + ") -> Sink (" + this->nodes_[this->graphSinkNode_].name + ") >>> ");
        for (lemon::ListDigraph::Node currentNode = this->graphSinkNode_; currentNode != southNode; currentNode = dijkstraNodeToSink.predNode(currentNode))
        {   //walk the shortest path route
            if (currentNode != lemon::INVALID && dijkstraNodeToSink.reached(currentNode))
            {
                lemon::ListDigraph::Arc predarc = dijkstraNodeToSink.predArc(currentNode);
                this->edges_[predarc].flow += Fmax;
                pass.log(mv::Logger::MessageType::Debug, "currentEdge: " + this->edges_[predarc].name + " : " + std::to_string(this->edges_[predarc].flow));
            }
        }
        /*Above calculation stops at source node of the edge so doesn't include this edge - add Fmax to this edge */
        this->edges_[thisArc].flow += Fmax;
    }
    //pass.log(mv::Logger::MessageType::Debug, "Printing all flow values: ");
    /*Subtract Memory attribute of edge from the Flow attribute of the edge*/
    lemon::ListDigraph::ArcMap<uint64_t> edgesFlow(this->graph_);
    for (lemon::ListDigraph::ArcIt thisArc(this->graph_); thisArc != lemon::INVALID; ++thisArc)
    {
        this->edges_[thisArc].flow -= this->edges_[thisArc].memoryRequirement;
        edgesFlow[thisArc] = this->edges_[thisArc].flow;
        //pass.log(mv::Logger::MessageType::Debug, this->edges_[thisArc].name + ": flow: " + std::to_string(this->edges_[thisArc].flow) + " mem: " + std::to_string(this->edges_[thisArc].memoryRequirement));
    }
   
    // Perform Min cut on the graph, example: https://gist.github.com/huanyud/45f98d8bf8d6df66d3e7ab3e9a85af90
    // Edge capacities = flow attribute of the edge
    // cutEdges contains all edges, but marked True/False if actually part of cut
    lemon::ListDigraph::NodeMap<bool> cutEdges(this->graph_);
    lemon::Preflow<lemon::ListDigraph, lemon::ListDigraph::ArcMap<uint64_t>> preflow(this->graph_, edgesFlow, this->graphSourceNode_, this->graphSinkNode_);
    preflow.run();
    preflow.minCutMap(cutEdges);

    uint64_t maxTopologicalCutValue = 0;
    std::vector<mv::edgeDescription> cutEdgesOnly;
    for(lemon::ListDigraph::ArcIt e(this->graph_); e!=lemon::INVALID; ++e)
    {
        if (cutEdges[this->graph_.source(e)] && !(cutEdges[this->graph_.target(e)])) 
        {
            int id = this->graph_.id(e);
            maxTopologicalCutValue += this->edgesMemory_[e];
            this->edges_[e].id = id;
            cutEdgesOnly.push_back(this->edges_[e]);
        }
    }
    
    /*Add Max topological cut value as attribute to output node*/
    pass.log(mv::Logger::MessageType::Debug, "The maximum peak memory of the graph is " + std::to_string(maxTopologicalCutValue) + " bytes");
    auto output = cm.getOutput();
    output->set<uint64_t>("MaxTopologicalCutValue", maxTopologicalCutValue);
    return std::make_pair(maxTopologicalCutValue, cutEdgesOnly);
}

void mv::LemonGraphScheduler::insertpartialSerialisationEdgesInMcmGraph(mv::ComputationModel& model)
{
    std::set<std::pair<std::string, std::string>> addedEdges;
    for (const auto& edge : partialSerialisationEdgesAdded_) 
    {
        lemon::ListDigraph::Arc tmpArc = this->graph_.arcFromId(edge.id);
        lemon::ListDigraph::Node edgeSourceNode = this->graph_.source(tmpArc);
        lemon::ListDigraph::Node edgeSinkNode = this->graph_.target(tmpArc);

        std::string edgeSourceName = this->nodes_[edgeSourceNode].name;
        std::string edgeSinkName = this->nodes_[edgeSinkNode].name;

        mv::ControlModel cm(model);
        mv::Control::OpListIterator mcmSourceNodeIterator;
        mv::Control::OpListIterator mcmSinkNodeIterator;

        /*Find the McM iterator for the source and sink node*/
        for (auto opItSource = cm.getFirst(); opItSource != cm.opEnd(); ++opItSource) 
        {    
            if(opItSource->getName() == edgeSourceName) 
                mcmSourceNodeIterator = opItSource;
            if(opItSource->getName() == edgeSinkName) 
                mcmSinkNodeIterator = opItSource;
        }

        auto inserted = addedEdges.insert(std::make_pair(edgeSourceName, edgeSinkName));
        if (inserted.second)
        {   
            /*Add the edge to graph*/

            auto partialSerialisationEdge = cm.defineFlow(mcmSourceNodeIterator, mcmSinkNodeIterator);
            partialSerialisationEdge->set<bool>("PartialSerialisationEdge", true);
        }
    }
}

// /**
//  * @brief Perform partial serilisation of KOALA graph to reduce maximum peak memory
//  * @param cutValue  - Maximum peak memory of the graph
//  * @param cutEdges - Vector of cut edges from the max topological cut
//  * @param graphSource - Source node of KOALA graph
//  * @param graphSink - Sink node of KOALA graph
//  * @param vertices - Vector of KOALA vertices iterators
//  * @param edges - Vector of KOALA edge iterators
//  * @return - 0 success
//  */
// void mv::KoalaGraphScheduler::performPartialSerialisation(const mv::pass::PassEntry& pass, std::vector<koalaGraph::PEdge> cutEdges) {
bool mv::LemonGraphScheduler::performPartialSerialisation(const mv::pass::PassEntry& pass, std::vector<mv::edgeDescription> cutEdges, mv::ComputationModel& model)
{    
    /* Partial serialisation works by getting the source and sink nodes of the cutEdges returned from max topoloigcal cut
     * It then creates a pool of all possible edges that it can add to the graph using these source and sink nodes.
     * Do not include the original cut edges in this pool as they are already in the graph.
     * The direction of the new edge is however in the opposite direction, sink --> source. Take care to ensure the correct direction.  */

    mv::ControlModel cm(model);
    /*Sources and sinks of cut edges*/
    //std::vector<mv::nodeDescription> sources;
    //std::vector<mv::nodeDescription> sinks;
    std::map<std::string, mv::nodeDescription> sourcesMap;
    std::map<std::string, mv::nodeDescription> sinksMap;

    /*Pool of possible edges to add to the graph expressed as souce and sink nodes*/
    std::vector<std::pair<mv::nodeDescription, mv::nodeDescription>> possibleEdges;

    /*Cut edges source and sink vectors*/
    std::vector<std::pair<mv::nodeDescription, mv::nodeDescription>> cutEdgesSourceSink;

    // for (const auto& edge : cutEdges) 
    // {
    //     lemon::ListDigraph::Arc thisArc = this->graph_.arcFromId(edge.id);
    //     lemon::ListDigraph::Node sourceNode = this->graph_.source(thisArc);
    //     lemon::ListDigraph::Node targetNode = this->graph_.target(thisArc);

    //     pass.log(mv::Logger::MessageType::Debug, "Cut edges: " + nodes_[sourceNode].name + " --> " + nodes_[targetNode].name );
    // }

    // initialize this for checking connectivity between nodes
    lemon::Dfs<lemon::ListDigraph> dfs(graph_);
    dfs.run();

    /*Get the source and sink of each cut edge*/
    for (const auto& edge : cutEdges) 
    {
        lemon::ListDigraph::Arc thisArc = this->graph_.arcFromId(edge.id);
        lemon::ListDigraph::Node sourceNode = this->graph_.source(thisArc);
        lemon::ListDigraph::Node targetNode = this->graph_.target(thisArc);


        std::string edgeSourceName = this->nodes_[sourceNode].name;
        std::string edgeSinkName = this->nodes_[targetNode].name;

        if (nodes_[sourceNode].opType == "Deallocate" && nodes_[targetNode].opType == "Deallocate")
        {
            pass.log(mv::Logger::MessageType::Debug, "Not adding dealloc-dealloc edge to Lemon graph from: " + edgeSourceName + " --> " + edgeSinkName);
            continue;
        }

        // sources.push_back(this->nodes_[sourceNode]);
        // sinks.push_back(this->nodes_[targetNode]);

        // Don't add the same node multiple times (ie, map container)
        sourcesMap[this->nodes_[sourceNode].name] = this->nodes_[sourceNode];
        sinksMap[this->nodes_[targetNode].name] = this->nodes_[targetNode];
        cutEdgesSourceSink.push_back(std::make_pair(this->nodes_[sourceNode], this->nodes_[targetNode]));
    }
    
    /*Create pool of possible partial serialisation edges but not including original cutset edges*/
    for (const auto& sinknode : sinksMap) 
    {
        for (const auto& sourcenode : sourcesMap) 
        {
            bool found = false;
            //for(int i = 0; i < cutEdgesSourceSink.size(); i++) 
            for(auto e : cutEdgesSourceSink) 
            {   /* Check if new potential partial serialisation edge is an original cut set edge, if it is then do not add it to the pool */
                if((e.first.name == sourcenode.first) && (e.second.name == sinknode.first)) 
                {
                    found = true;
                    break;
                }
            }

            bool do_not_add = sourcenode.second.opType == "Deallocate" && sinknode.second.opType == "Deallocate";
            if (do_not_add)
            {
                pass.log(mv::Logger::MessageType::Debug, "Not adding dealloc-dealloc edge to Lemon graph from: " + sourcenode.first + " --> " + sinknode.first);
                continue;
            }

            if (!found && !do_not_add) /*Edge not found in original cut set therefore add it to the pool*/
                possibleEdges.push_back(std::make_pair(sourcenode.second, sinknode.second));
        }
    }

    /* Attempt to add each edge to edge in the graph and check if it is still a DAG*/
    /* It is still a DAG then recalculate max topological cut*/
    /* Note in future here is where the optimal edge should be selected such that it minimises the increase in the critical path of the graph*/
    for (const auto& possibleEdge : possibleEdges)
    {
        auto sourceName = possibleEdge.second.name;
        auto sinkName  = possibleEdge.first.name;

        lemon::ListDigraph::Node tmpSourceNode = this->graph_.nodeFromId(possibleEdge.second.id);
        lemon::ListDigraph::Node tmpTargetNode = this->graph_.nodeFromId(possibleEdge.first.id);

        // TODO: Add logic here to add additional edges if source is a dealloc node emanating from a DPUTask
        auto srcNode = possibleEdge.second;
        auto snkNode = possibleEdge.first;
        if (srcNode.opType == "Deallocate")
        {
            for (lemon::ListDigraph::InArcIt in(graph_, tmpSourceNode); in != lemon::INVALID; ++in)
            {
                lemon::ListDigraph::Node parent = graph_.source(in);
                if (nodes_[parent].opType == "DPUTask")
                {
                    // Found the DPUTask that is the parent of the Dealloc task
                    for (lemon::ListDigraph::OutArcIt out(graph_, parent); out != lemon::INVALID; ++out)
                    {
                        lemon::ListDigraph::Node childOpOfDpuTask = graph_.target(out);

                        if (nodes_[childOpOfDpuTask].opType == "Deallocate")
                        {
                            // i.e. no path exists between childOfDpuTask and the targetNode
                            if (dfs.run(childOpOfDpuTask, tmpTargetNode) == false)
                            {

                                pass.log(mv::Logger::MessageType::Debug,
                                        "Adding TIG partial serialisation edge to Lemon graph from: " + sourceName + " --> " + sinkName );
                            
                                lemon::ListDigraph::Arc newArc = this->graph_.addArc(tmpSourceNode, tmpTargetNode);
                                mv::edgeDescription newDesc = edgeDescription(this->graph_.id(newArc), 0, "PS_edge_"+sinkName+sourceName);

                                lemon::ListDigraph::NodeMap<mv::nodeDescription> order(this->graph_);
                                lemon::topologicalSort(this->graph_, order);

                                /*Check if graph still direccted DAG*/
                                if (lemon::dag(this->graph_))
                                {   
                                    /*add edge iterator to the vector of LEMON edge iterators*/
                                    this->edges_[newArc] = newDesc;

                                    /*keep track of the edges added as these edges will be added to mcmGraph*/
                                    pass.log(mv::Logger::MessageType::Debug,
                                            "Graph still DAG after adding TIG partial serialisation edge, recalulating max topological cut value...");
                                    partialSerialisationEdgesAdded_.push_back(newDesc);
                                    return true;
                                }
                                else
                                {
                                    pass.log(mv::Logger::MessageType::Debug,
                                            "Removing TIG partial serialisation edge as graph is no longer a DAG, from: " + sourceName + " --> " + sinkName );
                                    this->graph_.erase(newArc);
                                }
                            }
                        }

                    }
                }
            }
        }
        else
        {

            pass.log(mv::Logger::MessageType::Debug, "Adding partial serialisation edge to Lemon graph from: " + sourceName + " --> " + sinkName );
        
            lemon::ListDigraph::Arc newArc = this->graph_.addArc(tmpSourceNode, tmpTargetNode);
            mv::edgeDescription newDesc = edgeDescription(this->graph_.id(newArc), 0, "PS_edge_"+sinkName+sourceName);

            lemon::ListDigraph::NodeMap<mv::nodeDescription> order(this->graph_);
            lemon::topologicalSort(this->graph_, order);

            /*Check if graph still direccted DAG*/
            if (lemon::dag(this->graph_))
            {   
                /*add edge iterator to the vector of LEMON edge iterators*/
                this->edges_[newArc] = newDesc;

                /*keep track of the edges added as these edges will be added to mcmGraph*/
                pass.log(mv::Logger::MessageType::Debug, "Graph still DAG after adding partial serialisation edge, recalulating max topological cut value...");
                partialSerialisationEdgesAdded_.push_back(newDesc);
                return true;
            }
            else {
                pass.log(mv::Logger::MessageType::Debug, "Removing partial serialisation edge as graph is no longer a DAG, from: " + sourceName + " --> " + sinkName );
                this->graph_.erase(newArc);
            }
        }
    }
    return false;
    //throw std::runtime_error("The maximum peak memory requirment of the graph exceeds CMX and the partial serialisation algorithm is unable to reduce parallelism, exiting now, this is normal behaviour");
}


std::string mv::LemonGraphScheduler::getLogID() const
{
    return "Lemon";
}
