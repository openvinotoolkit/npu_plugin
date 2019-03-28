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

static void addMemoryRequirmentAttributeOnControlFlows(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void maxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MaxTopologicalCut)
        .setFunc(maxTopologicalCut)
        .setDescription(
            "Perform the max topological cut algorithm to find the maximum eak memory of a task graph."
        );
    }

    namespace pass
    {

        MV_REGISTER_PASS(AddMemoryRequirmentAttributeOnControlFlows)
        .setFunc(addMemoryRequirmentAttributeOnControlFlows)
        .setDescription(
            "Add memory requirment as attrbute to control flow edges. "
        );
    }
}

/*KOALA Node Description*/
struct nodeDescription {

	std::string name;
    bool sinkNode = false;

    nodeDescription(std::string aname = "", bool sinknode = false) :name(aname), sinkNode(sinknode){}
    
    friend std::ostream& operator<<(std::ostream& os, const nodeDescription& nd) {
        os << "Node name: " << nd.name << std::endl;
        return os;
    }
};

/*KOALA Edge Description*/
struct edgeDescription {

	int memoryRequirement;
	std::string name;
    int flow;
    int length;

    edgeDescription(int m = 0, std::string aname = "", int f = 0, int l = 1) : memoryRequirement(m), name(aname), flow(f), length(l) {}
    
    friend std::ostream& operator<<(std::ostream& os, const edgeDescription& ed) {
        os << "Edge name: " << ed.name << std::endl;
        os << "Memory requirement: " << ed.memoryRequirement << std::endl;  
        os << "Flow: " << ed.flow << std::endl;
        os << "Length: " << ed.length << std::endl;
        return os;
    }
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


// koalaGraph::PEdge lookUpKoalaEdgebyName(std::string edgeName, const std::vector<koalaGraph::PEdge>& koalaEdges) {

//      for (const auto& e : koalaEdges) {

//         if(e->info.name == edgeName) 
//             return e;
//     }
// }

/**
 * @brief Encode the mememory requirements of each task by adding a "MemoryRequirment" attribute to the flow in the MCM task graph.
 *        The memory requirment is defined as the output tensor (N*W*H*C) * dataType.
 */

void encodeMemoryRequirmentsOnEdges(mv::ComputationModel& model) {

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) {

        if(((flowIt.source()->getOpType() == "DMATask") && (flowIt.sink()->getOpType() == "Deallocate")) || ((flowIt.source()->getOpType() == "DPUTask") && (flowIt.sink()->getName().find("Deallocate"+flowIt.source()->getName()) != std::string::npos)) || ((flowIt.source()->getOpType() == "DPUTask") && (flowIt.sink()->getOpType() == "DMATask")))
        {
            if(flowIt.sink()->hasAttr("direction")) {

                if(flowIt.sink()->get<mv::DmaDirection>("direction") == mv::CMX2DDR) {
                    
                    int memoryRequirement = 1;
                    auto dType = flowIt.source()->getInputTensor()[0]->get<mv::DType>("dType").getSizeInBits();
             
                    for (unsigned int i = 0; i < flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
                        memoryRequirement = flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

                    memoryRequirement = memoryRequirement * dType/8;
                    flowIt->set<int>("MemoryRequirement", memoryRequirement);
                    flowIt->set<bool>("PositiveMemory", true); //Required for transitive reduction filter
                }
            }
            else {
            int memoryRequirement = 1;
            auto dType = flowIt.source()->getInputTensor()[0]->get<mv::DType>("dType").getSizeInBits();
             
            for (unsigned int i = 0; i < flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
                memoryRequirement = flowIt.source()->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

            memoryRequirement = memoryRequirement * dType/8;
            flowIt->set<int>("MemoryRequirement", memoryRequirement);
            flowIt->set<bool>("PositiveMemory", true);  
            }
         
         }
         else {
            flowIt->set<int>("MemoryRequirement", 0);
         }
    }
}


/**
 * @brief Convert McM graph (control model view) to KOALA graph and store the data required to perform the max topoloigcal cut algorithm on the KOALA graph edges
 * @param pass  - pass object
 * @param model - MCM computation model
 * @param flowGraph - An instance of KOALA graph
 * @param V - Vector to store iterators to KOALA vertices 
 * @param E - Vector to store iterators to KOALA edges
 */
void convertMcMGraphToKoalaGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model, koalaGraph& flowGraph, std::vector<koalaGraph::PVertex>& V, std::vector<koalaGraph::PEdge>& E) {

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
               V.push_back(flowGraph.addVert(nodeDescription(opIt->getName(),true)));
           }
           else {
               pass.log(mv::Logger::MessageType::Debug, "Adding vertex to KOALA graph: " + opIt->getName());
               V.push_back(flowGraph.addVert(nodeDescription(opIt->getName(),false)));
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

            pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + sourceName + " --> " + sinkName + " with memory requirement " + std::to_string(flowIt->get<int>("MemoryRequirement")));

            E.push_back(flowGraph.addEdge(*std::find_if(V.begin(), V.end(), [&sourceName](koalaGraph::PVertex const& V) {return sourceName == V->info.name;}), 
                                             *std::find_if(V.begin(), V.end(), [&sinkName](koalaGraph::PVertex const& V) {return sinkName == V->info.name;}), 
                                             edgeDescription(flowIt->get<int>("MemoryRequirement"),flowIt->getName()), 
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
        std::cout << "memory is " << flowIt->get<int>("MemoryRequirement") << std::endl;
        Fmax += flowIt->get<int>("MemoryRequirement");
    }
    
    Fmax += 1;

    std::cout << "Fmax is " << Fmax << std::endl;

    return Fmax;
}

// void performPartialSerialisation(int cutValue, std::vector<koalaGraph::PEdge> cutEdges, koalaGraph::PVertex source, koalaGraph::PVertex sink) {


// }

void addMemoryRequirmentAttributeOnControlFlows(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    /*Add the memory requirement of a task as an attribute on the MCM graph*/
    encodeMemoryRequirmentsOnEdges(model);

}

void maxTopologicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::ControlModel cm(model);
    mv::OpModel om(model);

    /*Name of KOALA graph*/
    koalaGraph flowGraph;

    /*Vector to store KOALA vertices and edges iterators*/
    std::vector<koalaGraph::PVertex> V; 
    std::vector<koalaGraph::PEdge> E;

    /*Convert to MCM graph to KOALA graph*/
    convertMcMGraphToKoalaGraph(pass, model, flowGraph, V, E);
   
    /*Calculate Fmax*/
    int Fmax = calculateFMax(model); /*Compute Fmax - (defined as sum of memory requirments + 1)*/
    
    /*See the shortest path KOALA example here: http://koala.os.niwa.gda.pl/api/examples/weights/dijkstra_h/dijkstra_h.html*/

    Koala::AssocArray <koalaGraph::PEdge, Koala::DijkstraHeap::EdgeLabs<int >> edgeMap; /*input container*/
    Koala::AssocArray <koalaGraph::PVertex, Koala::DijkstraHeap::VertLabs<int,koalaGraph>> vertMap; /*output container*/
     
    /*Set edge lengths to 1*/
    setEdgeLengths(edgeMap, E);

    /* Construct the graph demand: cicle over the edge and add
     * a flow equal to Fmax on a shorest path containing that node
    */

    /*containter to store the edges on shorest paths*/
    std::vector <koalaGraph::PEdge> vecE;
    koalaGraph::PEdge LOCALARRAY(edges, flowGraph.getEdgeNo());
    int numberofEdges = flowGraph.getEdges(edges);
 
    /*For each edge
     *
     * Find the shortest path from source node (Input) to the edges source node and
     * Find the shortest path from the edges sink node to the sink node (DMA task CMX to DDR) 
    */
    for (int i = 0; i < numberofEdges; i++) {

        /*get the source and sink node of the edge*/
        pass.log(mv::Logger::MessageType::Debug, "Source Node " + flowGraph.getEdgeEnds(E[i]).first->info.name);
        pass.log(mv::Logger::MessageType::Debug, "Sink Node " + flowGraph.getEdgeEnds(E[i]).second->info.name);

        /*Find the shortest path from the input node to the source node of the edge*/
        /*The source node in the KOALA graph will always be called "Input_0", the same as the MCM graph*/
        Koala::DijkstraHeap::PathLengths <int> res0 = Koala::DijkstraHeap::findPath(flowGraph, edgeMap, lookUpKoalaVertexbyName("Input_0", V),flowGraph.getEdgeEnds(E[i]).first, Koala::DijkstraHeap::outPath(blackHole, back_inserter(vecE)));

        pass.log(mv::Logger::MessageType::Debug, "Number of edges on the path is " + std::to_string(res0.length));

	    for (int i = 0; i < res0.edgeNo; i++) {
		    //std::cout << ' ' << vecE[i]->info.name << std::endl;
            pass.log(mv::Logger::MessageType::Debug, vecE[i]->info.name);

            /*Add Fmax to the flow attribute of the edge*/
            auto edge = lookUpKoalaEdgebyName(vecE[i]->info.name, E);
            edge->info.flow +=Fmax;
	    }

        /*The above calculation stops at source node of the edge so doesn't include the edge in question - add Fmax to this edge*/
        E[i]->info.flow +=Fmax;

        /*Clear the container used to store the the edges on shorest paths*/
        vecE.clear(); 

        /*Find the shortest path from the sink node of the edge to the sink node (DMA task CMX to DDR)*/

        Koala::DijkstraHeap::PathLengths <int> res1 = Koala::DijkstraHeap::findPath(flowGraph, edgeMap, flowGraph.getEdgeEnds(E[i]).second, lookUpKoalaSinkNode(true, V), Koala::DijkstraHeap::outPath(blackHole, back_inserter(vecE)));

        pass.log(mv::Logger::MessageType::Debug, "Number of edges on the path is " + res1.length);

	    for (int i = 0; i < res1.edgeNo; i++) {
		    
            //std::cout << ' ' << vecE[i]->info.name << std::endl;
            pass.log(mv::Logger::MessageType::Debug, vecE[i]->info.name);

            /*Add Fmax to the flow attribute of the edge*/
            auto edge = lookUpKoalaEdgebyName(vecE[i]->info.name, E);
            edge->info.flow +=Fmax;
	    }
        /*Clear the container used to store the the edges on shorest paths*/
        vecE.clear();
    }

    //----------------------------------------------------------------------------------------------------
    /* Write KOALA graph to GraphML file: View the graph with this tool: https://gephi.org/ */
	// Koala::IO::GraphML gml;
	// Koala::IO::GraphMLGraph *gmlg;

    // /*Display all graph info in the console*/
    // //Koala::IO::writeGraphText(flowGraph, std::cout, Koala::IO::RG_VertexLists | Koala::IO::RG_Info);

    // gmlg = gml.createGraph("first");
	// 	gmlg->writeGraph(flowGraph, 
    //     Koala::IO::gmlStringField(&nodeDescription::name, "nodeName"), 
    //     Koala::IO::gmlLongField(&edgeDescription::memoryRequirement, "memory")
    //     & Koala::IO::gmlLongField(&edgeDescription::flow, "flow")
    //     & Koala::IO::gmlLongField(&edgeDescription::length, "length")
    //     & Koala::IO::gmlStringField(&edgeDescription::name, "edgeName")
    //     );
	// /*write GraphML to a file*/
	// gml.writeFile("max_topological_cut.graphml");
    //----------------------------------------------------------------------------------------------------

    /*Subtract Memory attribute of edge from the Flow attribute of the edge*/
    for (int i = 0; i < numberofEdges; i++)
		E[i]->info.flow = E[i]->info.flow - E[i]->info.memoryRequirement;
    

    /* Perform Min cut on the graph, see this example: http://koala.os.niwa.gda.pl/api/examples/flow/example_Flow.html*/

    /* Set edge capacities (flow attribute of the edge ) and costs (=1)*/
	Koala::AssocArray< koalaGraph::PEdge, Koala::Flow::EdgeLabs<int,int>> cap;

    for (int i = 0; i < numberofEdges; i++) {
        cap[E[i]].capac = E[i]->info.flow; 
        cap[E[i]].cost = 1;
    }

    /*store the cut edges*/
    std::vector<koalaGraph::PEdge> cutEdges;
    int memoryRequirement = 0;

    /*compute minimal cut*/
    Koala::Flow::minEdgeCut(flowGraph, cap, lookUpKoalaVertexbyName("Input_0", V), lookUpKoalaSinkNode(true, V), Koala::Flow::outCut(blackHole, std::back_inserter(cutEdges)));
    
    for (size_t i = 0; i < cutEdges.size(); i++)
        memoryRequirement += cutEdges[i]->info.memoryRequirement;

    /*Add Max topological cut value as attribute to output node*/
    auto output = cm.getOutput();
    output->set<int>("MaxTopologicalCutValue", memoryRequirement); 

    pass.log(mv::Logger::MessageType::Debug, "The maximum peak memory of the graph is " + std::to_string(memoryRequirement) + " bytes");

    // if (memoryRequirement > 943718.4) {
    //     performPartialSerialisation(memoryRequirment, cutEdges,lookUpKoalaVertexbyName("Input_0", V),lookUpKoalaSinkNode(true, V));

    // }

}
