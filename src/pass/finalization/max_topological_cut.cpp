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

static void maxTopogicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MaxTopogicalCut)
        .setFunc(maxTopogicalCut)
        .setDescription(
            "Calculate max topological cut."
        );
    }
}

/*KOALA Node Description*/
struct nodeDescription {
	std::string name;

    nodeDescription(std::string aname = "") :name(aname){}
    
    friend std::ostream& operator<<(std::ostream& os, const nodeDescription& nd) {
        os << "node name: " << nd.name << std::endl;
        return os;
    }

};

/*KOALA Edge Description*/
struct edgeDescription {
	int64_t memoryRequirement;
	std::string name;
    int64_t flow;
    int64_t length;

    edgeDescription(int m = 0, std::string aname = "", int f = 0, int l = 1) : memoryRequirement(m), name(aname), flow(f), length(l) {}
    
    friend std::ostream& operator<<(std::ostream& os, const edgeDescription& ed) {
        os << "Edge name: " << ed.name << std::endl;
        os << "Memory requirement: " << "xxx" << ed.memoryRequirement << "yyy" << std::endl;
        os << "Flow: " << ed.flow << std::endl;
        os << "Length: " << ed.length << std::endl;
        return os;
    }

    int getMemoryRequirment() {
        return this->memoryRequirement;
    }

};

/*Define KOALA graph's node and edge content type i.e. information to be stored on nodes and edges*/
using koalaGraph = Koala::Graph <nodeDescription, edgeDescription>;

/**
 * @brief Returns a KOALA vertex iterator corresonding to the name of the iterator 
 * @param vertexName - the name of the KOALA vertex you are searching for
 * @param koalaVertices - an array of KOALA vertices iterators
 * @param numberKoalaVertices - numberOfKoala vertices 
 * @return The KOALA vertex iterator 
 * 
 */
koalaGraph::PVertex lookUpKoalaVertexbyName(std::string vertexName, koalaGraph::PVertex koalaVertices[], int numberKoalaVertices) {

    for (int i = 0; i < numberKoalaVertices; i++) {

        if(koalaVertices[i]->info.name == vertexName) 
            return koalaVertices[i];
    }
}

/**
 * @brief Returns a KOALA edge iterator corresonding to the name of the iterator 
 * @param edgeName - the name of the KOALA vertex you are searching for
 * @param koalaEdges - an array of KOALA edges iterators
 * @param numberKoalaEdges - numberOfKoala edges
 * @return The KOALA edge iterator 
 * 
 */
koalaGraph::PEdge lookUpKoalaEdgebyName(std::string edgeName, koalaGraph::PEdge koalaEdges[], int numberKoalaEdges) {

    for (int i = 0; i < numberKoalaEdges; i++) {

        if(koalaEdges[i]->info.name == edgeName) 
            return koalaEdges[i];
    }
}

/**
 * @brief Encode the mememory requirements of each tash by adding a "MemoryRequirment" attribute to the task.
 *        The memory requirment is defined as the output tensor N*W*H*C.
 */
void encodeMemoryRequirmentsofTask(mv::ComputationModel& model) {

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    /* Store memory equirement as an attribute of a task. 
     * This is required also for Ops with no output tensor (i.e. Dealloc tasks).
    */
    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt) {
        
        std::cout << "Op is " << opIt->getName() << std::endl; 
        
        if (opIt->getOpType() == "Constant"){// || opIt->getOpType() == "Input" ) {

            mv::Shape shape = opIt->get<mv::Shape>("shape");
            std::cout << "number dmins " << shape.ndims() << std::endl;
            int memoryRequirement = 1;

            for (unsigned int i = 0; i < shape.ndims(); i++) 
                memoryRequirement = opIt->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

            opIt->set<int>("MemoryRequirement", memoryRequirement);
            std::cout << "Memory size " << memoryRequirement << std::endl;  
        }

          if (opIt->getOpType() == "Input") {

            int memoryRequirement = 0;
            opIt->set<int>("MemoryRequirement", memoryRequirement);
        }
        
        if (opIt->getOpType() == "DPUTask") { 

            int memoryRequirement = 1;
            std::cout << opIt->getOutputTensor()[0]->getShape().toString() << std::endl;
            for (unsigned int i = 0; i < opIt->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
                memoryRequirement = opIt->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

            std::cout << "Memory size " << memoryRequirement << std::endl;
            opIt->set<int>("MemoryRequirement", memoryRequirement);    
        }
        
        if ((opIt->getOpType() == "DMATask") && (opIt->get<mv::DmaDirection>("direction") == mv::DDR2CMX)) { 

            int memoryRequirement = 1;
            for (unsigned int i = 0; i < opIt->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
                memoryRequirement = opIt->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

            std::cout << "Memory size " << memoryRequirement << std::endl;
            opIt->set<int>("MemoryRequirement", memoryRequirement);    
        }

        if (opIt->getOpType() == "Deallocate") {
            int memoryRequirement = 0;
            opIt->set<int>("MemoryRequirement", memoryRequirement);
            std::cout << "Memory size " << memoryRequirement << std::endl;
        }
    }

    /*REMOVE THIS WHEN SPARISTY AND WEIGHT TABLE FIXED - HARD CODING MEMORY REQUIRMENT FOR NOW*/
    auto op = cm.getOp("DMATask_3");
    op->set<int>("MemoryRequirement", 4096);

    auto op1 = cm.getOp("DMATask_4");
    op1->set<int>("MemoryRequirement", 1024);
}


/**
 * @brief Convert McM graph (control model view) to KOALA graph
 * @param pass - 
 * @param model - MCM computation model
 * @param flowGraph - An instance of KOALA graph
 * @param V - Array to store iterators to KOALA vertices 
 * @param E - Array to store iterators to KOALA edges
 */
void convertMcMGraphToKoalaGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model, koalaGraph& flowGraph, koalaGraph::PVertex V[], koalaGraph::PEdge E[]) {

    mv::ControlModel cm(model);
    mv::DataModel dm(model);
    mv::OpModel om(model);

    int vertexIndex = 0;
    int edgeIndex = 0;
    
    /* For each task in the ControlModel view of the MCM graph
     * create a corresponding node (task) in the KOALA graph.
     * Add all the nodes to the KOALA graph first and then add the edges.
    */
    for (auto opIt = cm.getFirst(); opIt != cm.opEnd(); ++opIt)
    {
       
       /* We do not require MCM constant operations and MCM ouput operation in the KOALA graph. The sink node is the DMATask CMX2DDR.
        * For all other tasks in the ControlModel view of the MCM graph create a corresponding node in the KOALA graph.
       */
       if (opIt->getOpType() != "Constant" && opIt->getOpType() != "Output") {
           
           /*Add node to KOALA graph*/
           pass.log(mv::Logger::MessageType::Debug, "Adding vertex to KOALA graph: " + opIt->getName());
           V[vertexIndex] = flowGraph.addVert(nodeDescription(opIt->getName()));
           vertexIndex++;
       }
    }
    
    /*Add the edges to the KOALA graph*/
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt) {
        
        /*Don't add the edge going to Ouput in the MCM graph to the KOALA graph*/
        if (flowIt.sink()->getOpType() != "Output") { 

            auto sourceName = flowIt.source()->getName();
            auto sinkName  = flowIt.sink()->getName();

            /* (1) If the sink node of the flow iterator is a Deallocate Op or
             * (2) or the source node of the flow iterator is a DPU task then
             *     encode the memory requirment (size of the output tensor) on the edge.
             *     If not then the memory requirement in 0.
             */ 
            if(((flowIt.sink()->getOpType() == "Deallocate") && (flowIt.source()->getOpType() != "DPUTask"))  || ((flowIt.source()->getOpType() == "DPUTask") && (flowIt.sink()->getOpType() != "Deallocate"))) {
                
                pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + sourceName + " --> " + sinkName + " with memory requirement " + std::to_string(flowIt.source()->get<int>("MemoryRequirement")));
        
                /*Get KOALA Vertix iterator*/
                auto parentVertex = lookUpKoalaVertexbyName(sourceName, V, vertexIndex);
                auto thisVertex = lookUpKoalaVertexbyName(sinkName, V, vertexIndex);

                E[edgeIndex] = flowGraph.addEdge(parentVertex, thisVertex, edgeDescription(flowIt.source()->get<int>("MemoryRequirement"),flowIt->getName()), Koala::Directed);
                edgeIndex++;
            }
            else {
                
                pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + sourceName + " --> " + sinkName + " with memory requirement 0");
        
                /*Get KOALA Vertix iterator*/
                auto parentVertex = lookUpKoalaVertexbyName(sourceName, V, vertexIndex);
                auto thisVertex = lookUpKoalaVertexbyName(sinkName, V, vertexIndex);

                E[edgeIndex] = flowGraph.addEdge(parentVertex, thisVertex, edgeDescription(0,flowIt->getName()), Koala::Directed);
                edgeIndex++;
            }
        }
    }
    
    pass.log(mv::Logger::MessageType::Debug, "KOALA graph has " + std::to_string(flowGraph.getVertNo()) + " vertices and " + std::to_string(flowGraph.getEdgeNo()) + " edges");

}

void setEdgeLengths(Koala::AssocArray <koalaGraph::PEdge, Koala::DijkstraHeap::EdgeLabs<int >> &edgeMap, koalaGraph::PEdge E[] ,int numberOfKoalaEdges) // see graph
{
    for (int i = 0; i < numberOfKoalaEdges; i++) {
	edgeMap[E[i]].length = 1;
    }
}


/*
 * See Max topological cut algorithm description:
 * 
 * L. Marchal, H. Nagy, B. Simon and F. Vivien, "Parallel Scheduling of DAGs under Memory Constraints," 
 * 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS), Vancouver, BC, 2018, pp. 204-213.
 * doi: 10.1109/IPDPS.2018.00030 
*/ 

void maxTopogicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    mv::ControlModel cm(model);
    mv::OpModel om(model);

    /*Add the memory requirement of a task as an attribute on the MCM graph*/
    encodeMemoryRequirmentsofTask(model);

    /*Name of KOALA graph*/
    koalaGraph flowGraph;
    int numberOfKoalaVertices = 0;
    int numberOfKoalaEdges = 0;

    /*Count number of vertices required for KOALA graph*/
    for (auto opIt = cm.getFirst(); opIt != cm.opEnd(); ++opIt) {
        if (opIt->getOpType() != "Constant" && opIt->getOpType() != "Output") 
            numberOfKoalaVertices++;
    }
    
    /*Count number of edges required for KOALA graph*/
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt)
        numberOfKoalaEdges++;

    /*Array to store KOALA vertices and edges iterators*/
    koalaGraph::PVertex V[numberOfKoalaVertices]; 
    koalaGraph::PEdge E[numberOfKoalaEdges -1];   /* subtract 1 as we do not need last edge to ouput node in MCM graph*/

    /*Convert to MCM graph to KOALA graph*/
    convertMcMGraphToKoalaGraph(pass, model, flowGraph, V, E);
   
   //----------------------------------------------------------------

    /*Compute Fmax - (defined as sum of memory requirments + 1)*/
    int Fmax = 0;
    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt) {
        if (opIt->getOpType() != "Constant" && opIt->getOpType() != "Output") {
            if (opIt->hasAttr("MemoryRequirement")) {
                Fmax += opIt->get<int>("MemoryRequirement");
            }
        }  
    }
    Fmax += 1;
    
//----------------------------------------------------------------
    
    //----------------------------------------------------------------
    /*Print out KOALA edge information for debug*/
    koalaGraph::PEdge LOCALARRAY(edges, flowGraph.getEdgeNo());

    int numberofEdges = flowGraph.getEdges(edges);
	for (int i = 0; i < numberofEdges; i++) 
        std::cout << E[i]->info << std::endl;

    //----------------------------------------------------------------

    Koala::AssocArray <koalaGraph::PEdge, Koala::DijkstraHeap::EdgeLabs<int >> edgeMap; // input container
    Koala::AssocArray < koalaGraph::PVertex, Koala::DijkstraHeap::VertLabs < int, koalaGraph > > vertMap; // output container
     
    /*Clear distance info*/
    vertMap.clear();

    /*Set edge lengths to 1 - http://koala.os.niwa.gda.pl/api/examples/weights/dijkstra_h/dijkstra_h.html*/
    setEdgeLengths(edgeMap, E, (numberOfKoalaEdges-1));

    //----------------------------------------------------------------

    /* Construct the graph demand: cicle over the edge and add
     * a flow equal to c_max on a simple path containing that node
     * Note that this method succeeds because the graph is acyclic,
     * so every edge is part of a simple  path  (without  cycle)
    */

    // containters for vertices and edges on paths
    std::vector < koalaGraph::PEdge > vecE;
 
    /*For each edge*/
    for (int i = 0; i < numberofEdges; i++) {


        /*get the source and sink of the edge*/
		std::cout << E[i]->info << "Source: " << flowGraph.getEdgeEnds(E[i]).first->info << std::endl;
        std::cout << "Sink: " << flowGraph.getEdgeEnds(E[i]).second->info << std::endl;

        Koala::DijkstraHeap::PathLengths < int> res = Koala::DijkstraHeap::findPath(flowGraph, edgeMap, lookUpKoalaVertexbyName("Input_0", V, numberOfKoalaVertices),flowGraph.getEdgeEnds(E[i]).first, Koala::DijkstraHeap::outPath(blackHole, back_inserter(vecE)));

        std::cout <<  res.length << " Edges on the path" << std::endl;;
	    for (int i = 0; i < res.edgeNo; i++) {
		    std::cout << ' ' << vecE[i]->info.name << std::endl;

            /*Update flow attribute of Edge*/
            auto edge = lookUpKoalaEdgebyName(vecE[i]->info.name, E, numberOfKoalaEdges);
            std::cout << "Edge is " <<  edge->info.name << std::endl;

            edge->info.flow +=Fmax;
	    }

        /*also dont forget to +=Fmax to this edge*/
        E[i]->info.flow +=Fmax;

        vecE.clear();

        Koala::DijkstraHeap::PathLengths < int> res1 = Koala::DijkstraHeap::findPath(flowGraph, edgeMap, flowGraph.getEdgeEnds(E[i]).second, lookUpKoalaVertexbyName("DMATask_2", V, numberOfKoalaVertices), Koala::DijkstraHeap::outPath(blackHole, back_inserter(vecE)));

    
        std::cout <<  res1.length << " Edges on the path" << std::endl;
	    for (int i = 0; i < res1.edgeNo; i++) {
		    std::cout << ' ' << vecE[i]->info.name << std::endl;

            /*Update flow attribute of Edge*/
            auto edge = lookUpKoalaEdgebyName(vecE[i]->info.name, E, numberOfKoalaEdges);
            std::cout << "Edge is " <<  edge->info.name << std::endl;;
            edge->info.flow +=Fmax;
	    }
        vecE.clear();
    }
    //----------------------------------------------------------------------------------------------------
    /* Write KOALA graph to GraphML file*/
	Koala::IO::GraphML gml;
	Koala::IO::GraphMLGraph *gmlg;

    //show it
    Koala::IO::writeGraphText(flowGraph, std::cout, Koala::IO::RG_VertexLists | Koala::IO::RG_Info);

    gmlg = gml.createGraph("first");
		gmlg->writeGraph(flowGraph, 
        Koala::IO::gmlStringField(&nodeDescription::name, "nodeName"), 
        Koala::IO::gmlLongField(&edgeDescription::memoryRequirement, "memory")
        & Koala::IO::gmlLongField(&edgeDescription::flow, "flow")
        & Koala::IO::gmlLongField(&edgeDescription::length, "length")
        & Koala::IO::gmlStringField(&edgeDescription::name, "edgeName")
        );
	//write GraphML to a file
	gml.writeFile("john.graphml");
    //----------------------------------------------------------------------------------------------------

    /*Subtract Memory from Flow*/
    for (int i = 0; i < numberofEdges; i++)
		E[i]->info.flow = E[i]->info.flow - E[i]->info.memoryRequirement;
    

    /*Min cut*/

    
    struct edgeIter {
	void operator=(koalaGraph::PEdge e) { 
        std::cout << e->info.name << std::endl; 
    }
	void operator++() {std::cout << "calling ++" << std::endl; }
	edgeIter &operator*() {
        std::cout << "calling *" << std::endl;
         return *this; 
        }
    };

    edgeIter ed;

    
    
    // set edge capacities and costs
	Koala::AssocArray< koalaGraph::PEdge, Koala::Flow::EdgeLabs<int,int > > cap;

    for (int i = 0; i < numberofEdges; i++) {

        cap[E[i]].capac = E[i]->info.flow; 
        cap[E[i]].cost = 1;
    }


	// compute minimal cut
	std::cout << "The cut-set between vertices s and t consists of edges: ";
    auto res = Koala::Flow::minEdgeCut(flowGraph, cap, lookUpKoalaVertexbyName("Input_0", V, numberOfKoalaVertices), lookUpKoalaVertexbyName("DMATask_2", V, numberOfKoalaVertices), Koala::Flow::outCut(blackHole, ed));
	std::cout << "." << std::endl;

    std::cout << "cut value of memory " << res.cutValue << std::endl;
	
}
