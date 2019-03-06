#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "../../../contrib/koala/graph/graph.h"
#include "../../../contrib/koala/algorithm/conflow.h"

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

/*KOALA Edge Description*/
struct edgeDescription {
	int memoryRequirement;
	std::string name;

    edgeDescription(int m = 0, std::string aname = "") : memoryRequirement(m), name(aname) {}
};

/*Create KOALA graph*/
using koalaGraph = Koala::Graph <std::string, edgeDescription>;

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

        if(koalaVertices[i]->info == vertexName) {

            return koalaVertices[i];
        }
    }
}

/**
 * @brief Encode the mememory requirements of each tash by adding a "MemoryRequirment" attribute to the task.
 *        Note it should be added to the output tensor but deallocate currently doesn't have an output tensor. 
 *        The memory requirment is defined as the output tensor N*W*H*C.
 */
void encodeMemoryRequirmentsofTask(mv::ComputationModel& model) {

    mv::OpModel om(model);

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
}

void maxTopogicalCut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    mv::ControlModel cm(model);
    mv::DataModel dm(model);

    encodeMemoryRequirmentsofTask(model);
   
    /*Name of KOALA graph*/
    koalaGraph flowGraph;
    
    /*Values to store on KoalaGraph edges*/
    //Koala::AssocArray< koalaGraph::PEdge, Koala::ed<int, int>> memoryRequirement;

    int numberOfKoalaVertices = 0;
    int numberOfKoalaEdges = 0;
    int vertexIndex = 0;
    int edgeIndex = 0;

    /*Count number of vertices required for KOALA graph*/
    for (auto opIt = cm.getFirst(); opIt != cm.opEnd(); ++opIt) 
        if (opIt->getOpType() != "Constant" && opIt->getOpType() != "Output") 
            numberOfKoalaVertices++;
    
    /*Count number of edges required for KOALA graph*/
    for (auto flowIt = cm.flowBegin(); flowIt != cm.flowEnd(); ++flowIt)
        numberOfKoalaEdges++;

    /*Array to store KOALA vertices and edges iterators*/
    koalaGraph::PVertex V[numberOfKoalaVertices]; 
    koalaGraph::PEdge E[numberOfKoalaEdges -1];   /* subtract 1 as we do not need last edge to ouput node in MCM graph*/
    
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
           V[vertexIndex] = flowGraph.addVert(opIt->getName());
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

	// for(koalaGraph::PEdge e = flowGraph.getEdge(); e; e = flowGraph.getEdgeNext(e)) {
	//  	std::cout << flowGraph.getVertInfo(flowGraph.getEdgeEnd1(e)) << flowGraph.getVertInfo(flowGraph.getEdgeEnd2(e)) /*<< "(" << flowGraph.getEdgeInfo(e) <<") "*/;
	//  std::cout << std::endl << std::endl;
    //  }
}
