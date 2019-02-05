#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/graph/graph.hpp"

#include <math.h>
#include <metis.h>

static void generateWorkloadsFcn(const mv::pass::PassEntry &, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(GenerateWorkloads)
            .setFunc(generateWorkloadsFcn)
            .setGenre(PassGenre::Finalization)
            .setDescription(
                "This pass generates workloads");
    }
}


struct MetisGraphStructure
{
    idx_t* xadj; // Indexes of starting points in adjacent array
    idx_t* adjncy; // Adjacent vertices in consecutive index order

    int m_nVertices;
    int m_nEdges;
    int m_xDim;
    int m_yDim;

    MetisGraphStructure(int nVertices, int nEdges, int xDim, int yDim) : m_nVertices(nVertices), m_nEdges(nEdges), m_xDim(xDim), m_yDim(yDim)  {
        
        xadj = new idx_t[nVertices + 1];
        adjncy = new idx_t[2*nEdges];
    }
    
    ~MetisGraphStructure() {
        delete[] xadj;
        delete[] adjncy;
    } 
};
 


int gen_partion_graph(MetisGraphStructure& metisGraph) {
    
    std::vector<int> nodeNumbers  = mv::utils::generateSequence<int>(metisGraph.m_nVertices -1);
    
    // for (auto it = nodeNumbers.begin(); it != nodeNumbers.end(); it++) 
    //     std::cout << *it << " " << std::endl; 
    
    int index = 0;
    for (auto it = nodeNumbers.begin(); it != nodeNumbers.end(); it++) {

        //Node 0 (top left)
        if((*it%metisGraph.m_xDim == 0) && (*it == 0)) {
            std::cout << "Left side Node - Node " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it + 1;
            index++;
            metisGraph.adjncy[index] = *it + (metisGraph.m_xDim); 
            index++;
        }
  
        //Intermediate left side node
        if((*it%metisGraph.m_xDim == 0) && ((*it + metisGraph.m_xDim) < ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {
            std::cout << "Left side Node - Intermediate " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it - metisGraph.m_xDim;
            index++;
            metisGraph.adjncy[index] = *it + 1;
            index++;
            metisGraph.adjncy[index] = *it + (metisGraph.m_xDim); 
            index++;
        }

        //Bottom left node
        if((*it%metisGraph.m_xDim == 0) && ((*it + metisGraph.m_xDim) > ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {
            std::cout << "Bottom left Node - Intermediate " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it - metisGraph.m_xDim;
            index++;
            metisGraph.adjncy[index] = *it + 1;
            index++;
           
        }

        //node top right
        if(((*it - (metisGraph.m_xDim-1)%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1) && ((*it-(metisGraph.m_xDim-1) == 0)))) {
            std::cout << "Top right side Node " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it - 1;
            index++;
            metisGraph.adjncy[index] = *it + (metisGraph.m_xDim); 
            index++;
        }

       //Intermediate right side node
        if(((*it - (metisGraph.m_xDim-1))%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1)  && ((*it-(metisGraph.m_xDim-1) != 0))  && (*it %(nodeNumbers.size()-1) != 0)) {
            std::cout << "Intermediate right side Node " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it - metisGraph.m_xDim;
            index++;
            metisGraph.adjncy[index] = *it - 1;
            index++;
            metisGraph.adjncy[index] = *it + (metisGraph.m_xDim); 
            index++;
        }

        //node bottm right
        if(((*it - (metisGraph.m_xDim-1))%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1) && (*it %(nodeNumbers.size()-1) == 0)) {
            std::cout << "Bottom right side Node " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it - (metisGraph.m_xDim); 
            index++;
            metisGraph.adjncy[index] = *it - 1;
            index++;
        }
        
        //Middle nodes (1-3)
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) < (metisGraph.m_xDim - 1))) {
            std::cout << "Middle node top row " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it - 1;
            index++;
            metisGraph.adjncy[index] = *it + 1;
            index++;
            metisGraph.adjncy[index] = *it + (metisGraph.m_xDim); 
            index++;
        }

         //Middle nodes (11-13)
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) > ((int)nodeNumbers.size()-1) - metisGraph.m_xDim) && ((*it) != ((int)nodeNumbers.size()-1))) {
            std::cout << "Middle node bottom row " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it - (metisGraph.m_xDim); 
            index++;
            metisGraph.adjncy[index] = *it - 1;
            index++;
            metisGraph.adjncy[index] = *it + 1;
            index++;
            
        }

        //Middle nodes (6-18)
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) < ((int)nodeNumbers.size()-1) - metisGraph.m_xDim) && ((*it) > (metisGraph.m_xDim-1))) {
            std::cout << "Middle node " << *it << " " << std::endl; 

            metisGraph.adjncy[index] = *it - (metisGraph.m_xDim); 
            index++;
            metisGraph.adjncy[index] = *it - 1;
            index++;
            metisGraph.adjncy[index] = *it + 1;
            index++;
            metisGraph.adjncy[index] = *it + (metisGraph.m_xDim); 
            index++;
        }

for(int e = 0; e < index; e++) {
     
     std::cout << metisGraph.adjncy[e] << " ";
}
std::cout << std::endl;
}
return 0;        
} 






    
//Returns a number divided by 2 repeatly. Example maxSplitRange = 16 -> returns 16,8,4,2
std::set<int> getSplitsFromRange(int maxSplitRange, int maxSplits = 50)
{
    std::set<int> splits;

    if((maxSplitRange < maxSplits) && (maxSplitRange >1)) 
    {
        splits.insert(maxSplitRange);
        do 
        {
            maxSplitRange = maxSplitRange >> 1;
            splits.insert(maxSplitRange);
        } 
        while ((maxSplitRange >> 1) > 1);
    }
    return splits;
}

std::set<int> getNWorkloads(std::vector<mv::Data::TensorIterator> tensor, int nDPUxClusterS)
{
    std::cout << "Test getNWorkloads" << std::endl;
    std::cout << "Tensor shape is " << tensor[0]->getShape().toString() << std::endl;

    //maxSplitsXY
    auto xDim = tensor[0]->get<mv::Shape>("shape")[0];
    auto yDim = tensor[0]->get<mv::Shape>("shape")[1];
    auto maxSplitsXY = ceil(xDim/4) * ceil(yDim/4);

    std::cout << "maxSplitsXY is " << maxSplitsXY << std::endl;

    //maxSplitsZ
    auto maxSplitsZ = ceil(tensor[0]->get<mv::Shape>("shape")[2]/16);

    std::cout << "maxSplitsZ is " << maxSplitsZ << std::endl;

    //Pool of possible splits
    std::set<int> XYTileSplits;
    std::set<int> ZTileSplits;
    std::set<int> splitPool;

    XYTileSplits = getSplitsFromRange(maxSplitsXY);
    ZTileSplits = getSplitsFromRange(maxSplitsZ);

    std::set_union(std::begin(XYTileSplits), std::end(XYTileSplits),
               std::begin(ZTileSplits), std::end(ZTileSplits),                  
               std::inserter(splitPool, std::begin(splitPool)));
    
    return splitPool;
}

void generateWorkloadsFcn(const mv::pass::PassEntry &, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &)
{
    using namespace mv;

    mv::OpModel om(model);

    int nDPU = 3;
    int nClusters = 1;
    int nDPUxCluster = nDPU/nClusters;
    std::set<int> workloadsList;
    std::pair <int,int> MPEMode (4, 4);


    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {
            std::cout << "Found DPUTask" << std::endl;
 
            //Get output tensor
            auto outputTensor = opIt->getOutputTensor();

            std::cout << "Tensor shape is " << outputTensor[0]->getShape().toString() << std::endl;

            int tensorXDim = outputTensor[0]->get<mv::Shape>("shape")[0]; // why is dimension not full tensor size?
            int tensorYDim = outputTensor[0]->get<mv::Shape>("shape")[1];

            int numberTensorVertices = (tensorXDim  * tensorYDim) + 1;
            int numberTensorEdges = (2 * tensorXDim * tensorYDim) - tensorXDim - tensorYDim + 2;

            std::cout << "numberTensorVertices: " << numberTensorVertices << " " << "numberTensorEdges: " << numberTensorEdges << std::endl;

            //Create Metis struct
            MetisGraphStructure metisGraph(numberTensorVertices, numberTensorEdges, tensorXDim, tensorYDim);

            gen_partion_graph(metisGraph); 


            // workloadsList = getNWorkloads(outputTensor, nDPUxCluster);

            // //Already began to calculate a pool of workloads as per PoC compiler but for ww09 
            // //Forcing number of workloads to be nDPU/nCluster round to nearest even number
            // auto nWorkloads = round(nDPUxCluster/2)*2; 

            // std::cout << "Number of workloads is " << nWorkloads << std::endl;

            // //Workload class
            // Workloads workloads(opIt->getName());

            // mv::Workload w1;   
            // w1.MinX = 0;
            // w1.MaxX = 15;
            // w1.MinY = 0;
            // w1.MaxY = 3;
            // w1.MinZ = 0;
            // w1.MaxZ = 15;

            // mv::Workload w2;   
            // w2.MinX = 0;
            // w2.MaxX = 15;
            // w2.MinY = 0;
            // w2.MaxY = 3;
            // w2.MinZ = 0;
            // w2.MaxZ = 15;

            // workloads.getWorkloads().push_back(w1);
            // workloads.getWorkloads().push_back(w2);
            
            // opIt->set<mv::Workloads>("Workloads", workloads);

            // std::cout << opIt->toString();

        }
    }
    std::cout << "Exiting Workload Generation Pass " << std::endl;
}
