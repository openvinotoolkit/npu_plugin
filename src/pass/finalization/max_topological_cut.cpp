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

/*Create KOALA graph*/
    using koalaGraph = Koala::Graph <std::string, std::string>;

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
 *        Note it should be added to the output tensor but deallocate currently doesn' have an output tensor. 
 *        The memory requirment is defined as the output tensor N*W*H*C.
 */
void encodeMemoryRequirmentsofTask(mv::ComputationModel& model) {

    mv::OpModel om(model);

    /* Store memory equirement as an attribute of a task. 
     * This is required also for Ops with no output tensor (i.e. Dealloc tasks).
    */
    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt) {
        
        std::cout << "Op is " << opIt->getName() << std::endl; 
        
        if (opIt->getOpType() == "Constant" || opIt->getOpType() == "Input" ) {

            mv::Shape shape = opIt->get<mv::Shape>("shape");
            std::cout << "number dmins " << shape.ndims() << std::endl;
            int memoryRequirement = 1;

            for (unsigned int i = 0; i < shape.ndims(); i++) 
                memoryRequirement = opIt->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

            opIt->set<int>("MemoryRequirement", memoryRequirement);
            std::cout << "Memory size " << memoryRequirement << std::endl;  
        }
        
        if (opIt->getOpType() == "DPUTask") { 

            int memoryRequirement = 1;
            for (unsigned int i = 0; i < opIt->getOutputTensor()[0]->get<mv::Shape>("shape").ndims(); i++) 
                memoryRequirement = opIt->getOutputTensor()[0]->get<mv::Shape>("shape")[i] * memoryRequirement;

            std::cout << "Memory size " << memoryRequirement << std::endl;
            opIt->set<int>("MemoryRequirement", memoryRequirement);    
        }

        if (opIt->getOpType() == "DMATask") { 

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

    encodeMemoryRequirmentsofTask(model);
   
    /*Name of KOALA graph*/
    koalaGraph flowGraph; 

    koalaGraph::PVertex V[12]; /*Number of vertices - need to dynamically allocate*/
    koalaGraph::PEdge E[16];   /*Number of vertices - need to dynamically allocate*/

    int vertexIndex = 0;
    int edgeIndex = 0;

    /* For each task in the ControlModel view of the MCM graph
     * create a corresponding node (task) in the KOALA graph.
     * 
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
    for (auto opIt = cm.getFirst(); opIt != cm.opEnd(); ++opIt) {

        /*This operations are not required in the KOALA graph, therefore skip them*/
        if (opIt->getOpType() != "Input" && opIt->getOpType() != "Constant" && opIt->getOpType() != "Output") { 
            
            /*If more than 1 parent i.e DPU task*/ 
            if(opIt.parentsSize() > 1) {
                
                for (auto parentIt = opIt.leftmostParent(); parentIt != opIt.rightmostParent(); ++parentIt) {
                    
                    /*Need to look up KOALA vertex iteraor by name (KOALA vertex name is same as MCM vertex name)*/
                    auto parentVertex = lookUpKoalaVertexbyName(parentIt->getName(), V, vertexIndex);
                    auto thisVertex = lookUpKoalaVertexbyName(opIt->getName(), V, vertexIndex);

                    pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + parentVertex->info + " --> " + thisVertex->info);
                    E[edgeIndex] = flowGraph.addEdge(parentVertex, thisVertex, opIt.leftmostInput()->getName(), Koala::Directed);
                    edgeIndex++;
                }
            
                /*Deal with the rightmost parent which is not added in the previous loop*/
                auto parentVertex = lookUpKoalaVertexbyName(opIt.rightmostParent()->getName(), V, vertexIndex);
                auto thisVertex = lookUpKoalaVertexbyName(opIt->getName(), V, vertexIndex);

                pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + parentVertex->info + " --> " + thisVertex->info);
                E[edgeIndex] = flowGraph.addEdge(parentVertex, thisVertex, opIt.leftmostInput()->getName(), Koala::Directed);
                edgeIndex++;
            }
        
            /*Only one parent i.e. DMA task, Deallocate task*/
            if(opIt.parentsSize() == 1) {
                
                auto parentIt = opIt.leftmostParent();

                /*Need to look up KOALA vertex iteraor by name (KOALA vertex name is same as mcmcompiler vertex name)*/
                auto parentVertex = lookUpKoalaVertexbyName(parentIt->getName(), V, vertexIndex);
                auto thisVertex = lookUpKoalaVertexbyName(opIt->getName(), V, vertexIndex);

                pass.log(mv::Logger::MessageType::Debug, "Adding edge to KOALA graph from: " + parentVertex->info + " --> " + thisVertex->info);
                E[edgeIndex] = flowGraph.addEdge(parentVertex, thisVertex, opIt.leftmostInput()->getName(), Koala::Directed);
                edgeIndex++;
            }
        }
    }
    
    pass.log(mv::Logger::MessageType::Debug, "KOALA graph has " + std::to_string(flowGraph.getVertNo()) + " vertices and " + std::to_string(flowGraph.getEdgeNo()) + " edges");

	for(koalaGraph::PEdge e = flowGraph.getEdge(); e; e = flowGraph.getEdgeNext(e)) {
		std::cout << flowGraph.getVertInfo(flowGraph.getEdgeEnd1(e)) << flowGraph.getVertInfo(flowGraph.getEdgeEnd2(e)) << "(" << flowGraph.getEdgeInfo(e) <<") ";
	std::cout << std::endl << std::endl;
    }
}
