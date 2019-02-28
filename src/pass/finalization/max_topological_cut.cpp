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


void maxTopogicalCut(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{

    mv::ControlModel cm(model);

    
	
    /*Name of KOALA graph*/
    koalaGraph flowGraph; 

    koalaGraph::PVertex V[12]; /*Number of vertices - need to dynamically allocate*/
    koalaGraph::PEdge E[16];   /*Number of vertices - need to dynamically allocate*/

   /* For each task in the ControlModel view of the MCM graph
    * create a corresponding node in the KOALA graph 
   */

    int vertexIndex = 0;
    int edgeIndex = 0;
    for (auto opIt = cm.getFirst(); opIt != cm.opEnd(); ++opIt)
    {
       
       /* We do not required McM Constant operations and MCM ouput operation in the KOALA graph.
        * For all other tasks in the ControlModel view of the MCM graph
        * create a corresponding node in the KOALA graph.
       */
       if (opIt->getOpType() != "Constant" && opIt->getOpType() != "Output") {
           
           /*Add node*/
           std::cout << "Adding vertex to KOALA graph: " << opIt->getName() << std::endl;
           V[vertexIndex] = flowGraph.addVert(opIt->getName());
           vertexIndex++;
           

           /*Add edges*/
           if (opIt->getOpType() != "Input") { /*There is no input edge to input*/
               
               /*if more than 1 parent*/
               if(opIt.parentsSize() > 1) { 
                   
                   for (auto parentIt = opIt.leftmostParent(); parentIt != opIt.rightmostParent(); ++parentIt) {
                       //TODO
                    }
               }
                else {
                    auto parentIt = opIt.leftmostParent();

                    /*Get parent in mcm graph*/
                    std::cout << "Parent name: " << parentIt->getName() << std::endl;

                    std::cout << "Input flow edge name is " << opIt.leftmostInput()->getName() << std::endl;

                    /*Need to look up KOALA vertex iteraor by name (KOALA vertex name is same as mcmcompiler vertex name) - how to do this?*/
                    auto vertex = lookUpKoalaVertexbyName(parentIt->getName(), V, vertexIndex);

                    std::cout << "Adding edge to KOALA graph from: " << vertex->info << " -- " << V[vertexIndex-1]->info << std::endl;
                    E[edgeIndex] = flowGraph.addEdge(vertex, V[vertexIndex-1], opIt.leftmostInput()->getName());
                    edgeIndex++;
                }

               

           }

       }
    }

       
    
       
       
       
       
       
       
        // if (opIt->getOpType() != "Input" && opIt->getOpType() != "Constant" ) {
        //  std::cout << "Name: " << opIt->getName() << std::endl;
        //  std::cout << "Parent size: " << opIt.parentsSize() << std::endl;
        //  std::cout << "Name: " << opIt.leftmostInput()->getName() << std::endl;
        //  //std::cout << "Left parent output edge: " << opIt.leftmostParent()->getOutputTensor()[0]->getName() << std::endl;
        //  std::cout << "Right parent: " << opIt.rightmostParent()->getName() << std::endl;
        //  std::cout << "Input slots: " << opIt->inputSlots() << std::endl;
        //}
    

    
    

    std::cout << "KOALA graph has " << flowGraph.getVertNo() << " vertices and " << flowGraph.getEdgeNo() << " edges" << std::endl;
	for(koalaGraph::PEdge e = flowGraph.getEdge(); e; e = flowGraph.getEdgeNext(e))
		std::cout << flowGraph.getVertInfo(flowGraph.getEdgeEnd1(e)) << flowGraph.getVertInfo(flowGraph.getEdgeEnd2(e)) << "(" << flowGraph.getEdgeInfo(e) <<") ";
	std::cout << std::endl << std::endl;

  
}
