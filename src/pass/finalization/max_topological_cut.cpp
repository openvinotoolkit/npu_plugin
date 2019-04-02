#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "../../../contrib/koala/graph/graph.h"
#include "../../../contrib/koala/algorithm/conflow.h"
#include "../../../contrib/koala/algorithm/weights.h"
#include "../../../contrib/koala/io/graphml.h"
#include "../../../contrib/koala/io/text.h"
#include "../../../contrib/koala/io/parsetgml.h"
#include "../../../contrib/koala/classes/create.h"
#include <iostream>

//static void addMemoryRequirmentAttributeOnControlFlows(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void maxTopologicalCutAndPartialSerialisationPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MaxTopologicalCutAndPartialSerialisation)
        .setFunc(maxTopologicalCutAndPartialSerialisationPass)
        .setDescription(
            "Perform the max topological cut algorithm and partial serialisation (if required) to schedule the DAG."
        );
    }

    // namespace pass
    // {

    //     MV_REGISTER_PASS(AddMemoryRequirmentAttributeOnControlFlows)
    //     .setFunc(addMemoryRequirmentAttributeOnControlFlows)
    //     .setDescription(
    //         "Add memory requirment as attrbute to control flow edges. "
    //     );
    // }
}

/*KOALA Node Description*/
struct nodeDescription {

	std::string name;
    int cost; /*Required for partial serialisation*/
    bool sourceNode;
    bool sinkNode;
    

    nodeDescription(std::string aname = "", int cost = 0, bool sourcenode = false, bool sinknode = false) :name(aname), cost(cost), sourceNode(sourcenode), sinkNode(sinknode){}
    
};

/*KOALA Edge Description*/
struct edgeDescription {

	int memoryRequirement;
	std::string name;
    int flow;
    int length;

    edgeDescription(int m = 0, std::string aname = "", int f = 0, int l = 1) : memoryRequirement(m), name(aname), flow(f), length(l) {}
    
};

/*Define KOALA graph's node and edge type*/
using koalaGraph = Koala::Graph <nodeDescription, edgeDescription>;

/**
 * @brief Returns a KOALA vertex iterator corresonding to the name of the iterator 
 * @param vertexName - the name of the KOALA vertex you are searching for
 * @param koalaVertices - vector of KOALA vertices iterators
 * @return The KOALA vertex iterator 
 * 
 */
koalaGraph::PVertex lookUpKoalaVertexbyName(const std::string& vertexName, const std::vector<koalaGraph::PVertex>& koalaVertices) {

    for (size_t i = 0; i < koalaVertices.size(); i++) {

        if(koalaVertices[i]->info.name == vertexName) 
            return koalaVertices[i];
    }
}

/**
 * @brief Returns a KOALA vertex iterator corresonding to the sink node of the KOALA graph 
 * @param sinkNode - attribute of the KOALA node indicating if it is the sink node (true) 
 * @param koalaVertices - vector of KOALA vertices iterators
 * @return The KOALA vertex iterator 
 * 
 */
koalaGraph::PVertex lookUpKoalaSinkNode(bool sinknode, const std::vector<koalaGraph::PVertex>& koalaVertices) {

    for (size_t i = 0; i < koalaVertices.size(); i++) {

        if(koalaVertices[i]->info.sinkNode == sinknode) 
            return koalaVertices[i];
    }
}

/**
 * @brief Returns a KOALA vertex iterator corresonding to the source node of the KOALA graph 
 * @param sinkNode - attribute of the KOALA node indicating if it is the source node (true) 
 * @param koalaVertices - vector of KOALA vertices iterators
 * @return The KOALA vertex iterator 
 * 
 */
koalaGraph::PVertex lookUpKoalaSourceNode(bool sourcenode, const std::vector<koalaGraph::PVertex>& koalaVertices) {

    for (size_t i = 0; i < koalaVertices.size(); i++) {

        if(koalaVertices[i]->info.sourceNode == sourcenode) 
            return koalaVertices[i];
    }
}

/**
 * @brief Returns a KOALA edge iterator corresonding to the name of the iterator 
 * @param edgeName - the name of the KOALA vertex you are searching for
 * @param koalaEdges - vector of KOALA edges iterators
 * @return The KOALA edge iterator 
 * 
 */
koalaGraph::PEdge lookUpKoalaEdgebyName(std::string edgeName, const std::vector<koalaGraph::PEdge>& koalaEdges) {

    for (size_t i = 0; i < koalaEdges.size(); i++) {

        if(koalaEdges[i]->info.name == edgeName) 
            return koalaEdges[i];
    }
}

/**
 * TODO: Moce this to Control flow pass
 * 
 * @brief Encode the mememory requirements of each task by adding a "MemoryRequirment" attribute to the flow in the MCM task graph.
 *        The memory requirment is defined as the output tensor (N*W*H*C) * dataType.
 *        
 *        Memory requirments are added to these flows only.
 *        1. DMA -> Dealloc
 *        2. DPU task -> Corresponding DPU Task Dealloc
 *        3. DPU task -> DMA CMX2DDR direction (last DPU task)
 */

// void encodeMemoryRequirmentsOnEdges(mv::ComputationModel& model) {

//     mv::OpModel om(model);
//     mv::ControlModel cm(model);

//     for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) {

//         if(((flowIt.source()->getOpType() == "DMATask") && (flowIt.sink()->getOpType() == "Deallocate")) || ((flowIt.source()->getOpType() == "DPUTask") && (flowIt.sink()->getName().find("Deallocate"+flowIt.source()->getName()) != std::string::npos)) || ((flowIt.source()->getOpType() == "DPUTask") && (flowIt.sink()->getOpType() == "DMATask")))
//         {
//             if(flowIt.sink()->hasAttr("direction")) {

//                 if(flowIt.sink()->get<mv::DmaDirection>("direction") == mv::CMX2DDR) {
                    
//                     int memoryRequirement = 1;
//                     auto dType = flowIt.source()->getInputTensor()[0]->get<mv::DType>("dType").getSizeInBits();
             
//                     for (unsigned int i = 0; i < flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
//                         memoryRequirement = flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

//                     memoryRequirement = memoryRequirement * dType/8;
//                     flowIt->set<int>("MemoryRequirement", memoryRequirement);
//                     flowIt->set<bool>("PositiveMemory", true); //Required for transitive reduction filter
//                 }
//             }
//             else {
//             int memoryRequirement = 1;
//             auto dType = flowIt.source()->getInputTensor()[0]->get<mv::DType>("dType").getSizeInBits();
             
//             for (unsigned int i = 0; i < flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
//                 memoryRequirement = flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

//             memoryRequirement = memoryRequirement * dType/8;
//             flowIt->set<int>("MemoryRequirement", memoryRequirement);
//             flowIt->set<bool>("PositiveMemory", true);  
//             }
         
//          }
//          else {
//             flowIt->set<int>("MemoryRequirement", 0);
//          }
//     }
// }


/**
 * @brief Convert McM graph (control model view) to KOALA graph and store the data required to perform the max topoloigcal cut algorithm on the KOALA graph edges
 * @param pass  - pass object
 * @param model - MCM computation model
 * @param flowGraph - An instance of KOALA graph
 * @param V - Vector to store iterators to KOALA vertices 
 * @param E - Vector to store iterators to KOALA edges
 */
void convertMcMGraphToKoalaGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model, koalaGraph& flowGraph, std::vector<koalaGraph::PVertex>& vertices, std::vector<koalaGraph::PEdge>& edges) {

    mv::ControlModel cm(model);
    mv::DataModel dm(model);
    mv::OpModel om(model);

    /* For each task in the ControlModel view of the MCM graph
     * create a corresponding node (task) in the KOALA graph.
     * Add all the nodes to the KOALA graph first and then add the edges.
    */
    for (auto opIt = cm.getFirst(); opIt != cm.opEnd(); ++opIt)
    {
       
       /* We do not require MCM constant operations and MCM ouput operation in the KOALA graph. The sink node in the KOALA graph is the DMATask CMX2DDR.
        * For all other tasks in the ControlModel view of the MCM graph create a corresponding node in the KOALA graph.
       */
       if (opIt->getOpType() != "ConstantDataElement" && opIt->getOpType() != "Output" && opIt->getOpType() != "ConstantInt") {
           
           /*Add node to KOALA graph*/
           /*Check if the node is a DMA task CMX to DDR (this is the sink node in KOALA graph and we need to keep track of it)*/
           if ((opIt->getOpType() == "DMATask") && (opIt->get<mv::DmaDirection>("direction") == mv::CMX2DDR)) {
               pass.log(mv::Logger::MessageType::Debug, "Adding vertex to KOALA graph: " + opIt->getName());
               vertices.push_back(flowGraph.addVert(nodeDescription(opIt->getName(),0, false, true)));
           }
           if (opIt->getOpType() == "Input") {
               pass.log(mv::Logger::MessageType::Debug, "Adding vertex to KOALA graph: " + opIt->getName());
               vertices.push_back(flowGraph.addVert(nodeDescription(opIt->getName(),0, true, false)));
           }
           else {
               pass.log(mv::Logger::MessageType::Debug, "Adding vertex to KOALA graph: " + opIt->getName());
               vertices.push_back(flowGraph.addVert(nodeDescription(opIt->getName(),0, false,false)));
           }
       }
    }
    
    /* Add the edges to the KOALA graph store attributes on the edges to perform the max topoloigcal cut algorithm.
     * Iterate over the the control flow edges in the MCMgraph.  
    */
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) {
        
        /* 1. Don't add the edge going to Ouput in the MCM graph to the KOALA graph
         * 2. Don't add edge coming from a ConstantInt operation (Sparsity Map and Weights Table)
        */ 
       
        if (flowIt.sink()->getOpType() != "Output" && flowIt.source()->getOpType() != "ConstantInt") { 

            auto sourceName = flowIt.source()->getName();
            auto sinkName  = flowIt.sink()->getName();

            if(flowIt->hasAttr("MemoryRequirement"))
                pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + sourceName + " --> " + sinkName + " with memory requirement " + std::to_string(flowIt->get<int>("MemoryRequirement")));
            else
                pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + sourceName + " --> " + sinkName + " with memory requirement " + std::to_string(0));

            /*If the control flow has a memoryRequirment attribute add it to the KOALA edge*/
            if(flowIt->hasAttr("MemoryRequirement"))
                edges.push_back(flowGraph.addEdge(*std::find_if(vertices.begin(), vertices.end(), [&sourceName](koalaGraph::PVertex const& vertices) {return sourceName == vertices->info.name;}), 
                                             *std::find_if(vertices.begin(), vertices.end(), [&sinkName](koalaGraph::PVertex const& vertices) {return sinkName == vertices->info.name;}), 
                                             edgeDescription(flowIt->get<int>("MemoryRequirement"),flowIt->getName()), 
                                             Koala::Directed));
            else
                edges.push_back(flowGraph.addEdge(*std::find_if(vertices.begin(), vertices.end(), [&sourceName](koalaGraph::PVertex const& vertices) {return sourceName == vertices->info.name;}), 
                                             *std::find_if(vertices.begin(), vertices.end(), [&sinkName](koalaGraph::PVertex const& vertices) {return sinkName == vertices->info.name;}), 
                                             edgeDescription(0,flowIt->getName()), 
                                             Koala::Directed));
        }
    }
    pass.log(mv::Logger::MessageType::Debug, "KOALA graph has " + std::to_string(flowGraph.getVertNo()) + " vertices and " + std::to_string(flowGraph.getEdgeNo()) + " edges");
}

/**
 * @brief Set the lengths of the edges of the KOALA graph to be 1. This is required for shorest path algorithm which is a step in the max topoloigcal cut algorithm
 * @param egdeMap  - Container to store the edge length lengths  
 * @param E - The KOALA edge iterators
 * 
 * Example: http://koala.os.niwa.gda.pl/api/examples/weights/dijkstra_h/dijkstra_h.html
 */

void setEdgeLengths(const Koala::AssocArray <koalaGraph::PEdge, Koala::DijkstraHeap::EdgeLabs<int >> &edgeMap, const std::vector<koalaGraph::PEdge>& E) 
{
   for (const auto& e : E) {
        edgeMap[e].length = 1;
    }
}


/*
 * See Max topological cut algorithm description in this paper:
 * 
 * L. Marchal, H. Nagy, B. Simon and F. Vivien, "Parallel Scheduling of DAGs under Memory Constraints," 
 * 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS), Vancouver, BC, 2018, pp. 204-213.
 * doi: 10.1109/IPDPS.2018.00030 
*/ 

int calculateFMax(mv::ComputationModel& model) {

    mv::ControlModel cm(model);

    /*Compute Fmax - (defined as sum of memory requirments + 1)*/
    int Fmax = 0;
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) {
        if(flowIt->hasAttr("MemoryRequirement")) {
            std::cout << "memory is " << flowIt->get<int>("MemoryRequirement") << std::endl;
            Fmax += flowIt->get<int>("MemoryRequirement");
        }
    }
    
    Fmax += 1;

    return Fmax;
}

int criticalPathCost(koalaGraph::PVertex source, koalaGraph::PVertex target) {

}

/**
 * @brief Perform partial serilisation of KOALA graph to reduce maximum peak memory
 * @param cutValue  - Maximum peak memory of the graph
 * @param cutEdges - Vector of cut edges from the max topological cut
 * @param graphSource - Source node of KOALA graph
 * @param graphSource - Sink node of KOALA graph
 */
void performPartialSerialisation(const mv::pass::PassEntry& pass, koalaGraph& flowGraph, int cutValue, std::vector<koalaGraph::PEdge> cutEdges, koalaGraph::PVertex graphSource, koalaGraph::PVertex graphSink, std::vector<koalaGraph::PVertex>& vertices, std::vector<koalaGraph::PEdge>& edges) {

    /* Partial serialisation works by getting the source and sink nodes of the cutEdges returned from max topoloigcal cut
     * It then creates a pool of all possible edges to that it can add to the graph using these source and sink nodes.
     * This does not include the original cut edges as they are already in the graph.
     * The direction of the new edge is however in the opposite direction, sink --> source. 
    */  

    /*Sources and sinks of cut edges*/
    std::vector<koalaGraph::PVertex> sources;
    std::vector<koalaGraph::PVertex> sinks;

    std::vector<std::pair<koalaGraph::PVertex,koalaGraph::PVertex>> possibleEdges;
    std::vector<std::pair<koalaGraph::PVertex,koalaGraph::PVertex>> cutEdgesSourceSink;

    /*Get the source and sink of each cut edge*/
    for (const auto& edge : cutEdges)
        cutEdgesSourceSink.push_back(std::make_pair(edge->getEnd1(), edge->getEnd2()));
    
    /*Get cut edges sources*/
    for (const auto& edge : cutEdges) {
        if(std::find(sources.begin(), sources.end(), edge->getEnd1()) != sources.end()) {   
            /* sources already contains the edge source node */
        } else {
            /* add edge source node to sources */
            sources.push_back(edge->getEnd1());
        }   
    }

    /*Get cut edges sinks*/
    for (const auto& edge : cutEdges) {
        if(std::find(sinks.begin(), sinks.end(), edge->getEnd2()) != sinks.end()) {   
            /* sources already contains the edge sink node */
        } else {
            /* add edge sink node to sources */
            sinks.push_back(edge->getEnd2());
        }   
    }

    /*Create pool of possible partial serialisation edges but not including original cutset edges*/
    for (const auto& sinknode : sinks) {
        for (const auto& sourcenode : sources) {
            bool found = false;

            for(int i = 0; i < cutEdgesSourceSink.size(); i++) {
                
                /*Check if new potential partial serialisation edge is an original cut set edge, if it then do not add it to the pool*/
                if((cutEdgesSourceSink[i].first->info.name == sourcenode->info.name) && (cutEdgesSourceSink[i].second->info.name == sinknode->info.name)) {
                     found = true;
                     break;
                }
    
            }
            if (!found) /*Edge not found in original cut set therefore add it to the pool*/
                possibleEdges.push_back(std::make_pair(sourcenode, sinknode));
        }
    }

    for(int i = 0; i < possibleEdges.size(); i++) {

        auto sourceName = possibleEdges[i].second->info.name;
        auto sinkName  = possibleEdges[i].first->info.name;

        pass.log(mv::Logger::MessageType::Debug, "Adding partial serialisation edge to KOALA graph from: " + sourceName + " --> " + sinkName );

        auto newEdge = flowGraph.addEdge(*std::find_if(vertices.begin(), vertices.end(), [&sourceName](koalaGraph::PVertex const& vertices) {return sourceName == vertices->info.name;}), 
                                             *std::find_if(vertices.begin(), vertices.end(), [&sinkName](koalaGraph::PVertex const& vertices) {return sinkName == vertices->info.name;}), 
                                             edgeDescription(0,"PS_edge_"+sinkName+sourceName), 
                                             Koala::Directed);
        
        int n = flowGraph.getVertNo();
		koalaGraph::PVertex LOCALARRAY(tabV, n);
		
		Koala::DAGAlgs::topOrd(flowGraph, tabV); /* Get topological order*/
		
		bool isDag = Koala::DAGAlgs::isDAG(flowGraph, tabV, tabV + n);

        if(isDag) {
            pass.log(mv::Logger::MessageType::Debug, "The graph is still a DAG after adding partial serialisation edge, recalulating max topological cut value");
            edges.push_back(newEdge); /*add edge to the vector*/
            break;
        }
        else {
            pass.log(mv::Logger::MessageType::Debug, "Removing partial serialisation edge as graph is no longer a DAG, from: " + sourceName + " --> " + sinkName );
            flowGraph.delEdge(newEdge);
        }

    }
}


void addMemoryRequirmentAttributeOnControlFlows(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    /*Add the memory requirement of a task as an attribute on the MCM graph*/
    //encodeMemoryRequirmentsOnEdges(model);

}

std::pair<int,std::vector<koalaGraph::PEdge>> calcualteMaxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, koalaGraph& flowGraph, std::vector<koalaGraph::PVertex>& Vertices, std::vector<koalaGraph::PEdge>& Edges) {

    mv::ControlModel cm(model);
    mv::OpModel om(model);

    /* Calculate Fmax - Defined as sum of memory requirments + 1)*/
    int Fmax = calculateFMax(model); 
    
    /*See the shortest path KOALA example here: http://koala.os.niwa.gda.pl/api/examples/weights/dijkstra_h/dijkstra_h.html*/

    Koala::AssocArray <koalaGraph::PEdge, Koala::DijkstraHeap::EdgeLabs<int>> edgeMap; /*input container*/
    Koala::AssocArray <koalaGraph::PVertex, Koala::DijkstraHeap::VertLabs<int,koalaGraph>> vertMap; /*output container*/
     
    /*Set edge lengths to 1*/
    setEdgeLengths(edgeMap, Edges);

    /* Construct the graph demand: cicle over the edge and add
     * a flow equal to Fmax on a shorest path containing that node
    */

    /*containter to store the edges on shorest paths*/
    std::vector <koalaGraph::PEdge> shortestPathEdges;
    koalaGraph::PEdge LOCALARRAY(edges, flowGraph.getEdgeNo());
    int numberofEdges = flowGraph.getEdges(edges);
 
    /*For each edge
     *
     * Find the shortest path from source node (Input) to the edges source node and
     * Find the shortest path from the edges sink node to the sink node (DMA task CMX to DDR) 
    */
    for (int i = 0; i < numberofEdges; i++) {

        /*get the source and sink node of the edge*/
        pass.log(mv::Logger::MessageType::Debug, "Source Node " + flowGraph.getEdgeEnds(Edges[i]).first->info.name);
        pass.log(mv::Logger::MessageType::Debug, "Sink Node " + flowGraph.getEdgeEnds(Edges[i]).second->info.name);

        /*Find the shortest path from the input node to the source node of the edge*/
        Koala::DijkstraHeap::PathLengths <int> resInputToSource = Koala::DijkstraHeap::findPath(flowGraph, edgeMap, lookUpKoalaSourceNode(true, Vertices),flowGraph.getEdgeEnds(Edges[i]).first, Koala::DijkstraHeap::outPath(blackHole, back_inserter(shortestPathEdges)));

        pass.log(mv::Logger::MessageType::Debug, "Number of edges on the path is " + std::to_string(resInputToSource.edgeNo));

	    for (int i = 0; i < resInputToSource.edgeNo; i++) {

            pass.log(mv::Logger::MessageType::Debug, shortestPathEdges[i]->info.name);

            /*Add Fmax to the flow attribute of the edge*/
            auto edge = lookUpKoalaEdgebyName(shortestPathEdges[i]->info.name, Edges);
            edge->info.flow +=Fmax;
	    }

        /*The above calculation stops at source node of the edge so doesn't include the edge in question - add Fmax to this edge*/
        Edges[i]->info.flow +=Fmax;

        /*Clear the container used to store the the edges on shorest paths*/
        shortestPathEdges.clear(); 

        /*Find the shortest path from the sink node of the edge to the sink node (DMA task CMX to DDR)*/

        Koala::DijkstraHeap::PathLengths <int> resSinkToOuput = Koala::DijkstraHeap::findPath(flowGraph, edgeMap, flowGraph.getEdgeEnds(Edges[i]).second, lookUpKoalaSinkNode(true, Vertices), Koala::DijkstraHeap::outPath(blackHole, back_inserter(shortestPathEdges)));

        pass.log(mv::Logger::MessageType::Debug, "Number of edges on the path is " + std::to_string(resSinkToOuput.edgeNo));

	    for (int i = 0; i < resSinkToOuput.edgeNo; i++) {
		    
            pass.log(mv::Logger::MessageType::Debug, shortestPathEdges[i]->info.name);

            /*Add Fmax to the flow attribute of the edge*/
            auto edge = lookUpKoalaEdgebyName(shortestPathEdges[i]->info.name, Edges);
            edge->info.flow +=Fmax;
	    }
        /*Clear the container used to store the the edges on shorest paths*/
        shortestPathEdges.clear();
    }

    /*Subtract Memory attribute of edge from the Flow attribute of the edge*/
    for (int i = 0; i < numberofEdges; i++)
		Edges[i]->info.flow = Edges[i]->info.flow - Edges[i]->info.memoryRequirement;
    

    /* Perform Min cut on the graph, see this example: http://koala.os.niwa.gda.pl/api/examples/flow/example_Flow.html*/

    /* Set edge capacities (flow attribute of the edge ) and costs (=1)*/
	Koala::AssocArray< koalaGraph::PEdge, Koala::Flow::EdgeLabs<int,int>> cap;

    for (int i = 0; i < numberofEdges; i++) {
        cap[Edges[i]].capac = Edges[i]->info.flow; 
        cap[Edges[i]].cost = 1;
    }

    /*store the cut edges*/
    std::vector<koalaGraph::PEdge> cutEdges;
    int maxTopologicalCutValue = 0;

    /*compute minimal cut*/
    Koala::Flow::minEdgeCut(flowGraph, cap, lookUpKoalaSourceNode(true, Vertices), lookUpKoalaSinkNode(true, Vertices), Koala::Flow::outCut(blackHole, std::back_inserter(cutEdges)));
    
    for (size_t i = 0; i < cutEdges.size(); i++)
        maxTopologicalCutValue += cutEdges[i]->info.memoryRequirement;

    /*Add Max topological cut value as attribute to output node*/
    auto output = cm.getOutput();
    output->set<int>("MaxTopologicalCutValue", maxTopologicalCutValue); 

    pass.log(mv::Logger::MessageType::Debug, "The maximum peak memory of the graph is " + std::to_string(maxTopologicalCutValue) + " bytes");

    return std::make_pair(maxTopologicalCutValue, cutEdges);
}

void maxTopologicalCutAndPartialSerialisationPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    /*KOALA graph instance*/
    koalaGraph flowGraph;

    /*Vectors to store KOALA vertices and edges iterators*/
    std::vector<koalaGraph::PVertex> Vertices; 
    std::vector<koalaGraph::PEdge> Edges;

    /*Convert to MCM graph to KOALA graph*/
    convertMcMGraphToKoalaGraph(pass, model, flowGraph, Vertices, Edges);

    /*Calculate max topological cut and get the cut edges*/
    auto maxTopologicalCut = calcualteMaxTopologicalCut(pass, model, flowGraph, Vertices, Edges);
   
    //TODO get CMX memory from target descriptor and multiply by safety factory
    /*Repeat partial serialisation until max topological cut is less thean CMX memory*/
    while (maxTopologicalCut.first > 846028) {
        performPartialSerialisation(pass, flowGraph, maxTopologicalCut.first, maxTopologicalCut.second,lookUpKoalaSourceNode(true, Vertices),lookUpKoalaSinkNode(true, Vertices),Vertices, Edges);
        maxTopologicalCut = calcualteMaxTopologicalCut(pass, model, flowGraph, Vertices, Edges);
    }
}
