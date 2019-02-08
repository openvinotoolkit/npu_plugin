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

/*
 * Struct containing the parameters for METIS.
*/

struct MetisGraphStructure
{
    idx_t* xadj;                      /*Indexes of starting points in adjacent array*/
    idx_t* adjncy;                    /*Adjacent vertices in consecutive index order*/
    idx_t* vwgt;
    idx_t* part; 
    idx_t objval;
    idx_t nWeights  = 1;              /*Each vertex stores 1 weight (which is number_nodes_x * number_nodes_y)*/
    idx_t options[METIS_NOPTIONS];

    idx_t m_numberTensorVertices;
    idx_t m_numberTensorEdges;
    int m_xDim;
    int m_yDim;

    MetisGraphStructure(mv::Shape outputTensor, std::pair <int,int> MPEMode){

        /*Shape of output tensor x-y*/
        double tensorXDim = outputTensor[0]; 
        double tensorYDim = outputTensor[1];

        /*METIS lattic graph of tensor*/
        m_numberTensorVertices = ceil(tensorXDim / MPEMode.first)  * ceil(tensorYDim / MPEMode.second);    
        m_numberTensorEdges = (2 * ceil(tensorXDim / MPEMode.first) * ceil(tensorYDim / MPEMode.second)) - ceil(tensorXDim / MPEMode.first) - ceil(tensorYDim / MPEMode.second);
        
        m_xDim = ceil((tensorXDim / MPEMode.first));
        m_yDim = ceil((tensorYDim / MPEMode.second));

        /*Description page 23 Metis manual*/
        xadj = new idx_t[m_numberTensorVertices + 1]; 
        adjncy = new idx_t[2*m_numberTensorEdges];
        part = new idx_t[m_numberTensorVertices];
        vwgt = new idx_t[m_numberTensorVertices* nWeights];
    
        
        int n_elem_y;
        int n_elem_x;
        int nodeIndex = 0;
        for(int j=0; j < m_yDim; j++) {
            
            if ((j+1 < m_yDim) || (!(int)tensorYDim%MPEMode.first)) 
                    n_elem_y = MPEMode.first;
                else 
                    n_elem_y = (int)tensorYDim%MPEMode.first;
                            
            for(int k=0; k < m_xDim; k++) {
                
                if ((k+1 < m_xDim) || (!(int)tensorXDim%MPEMode.first)) 
                    n_elem_x = MPEMode.first;
                else 
                    n_elem_x = (int)tensorXDim%MPEMode.first;
            
                vwgt[nodeIndex] = n_elem_x * n_elem_y;
                std::cout << "Node " << nodeIndex << "weight is " << n_elem_x * n_elem_y << std::endl;
                nodeIndex++;
            }
            
        }
    }
    
    ~MetisGraphStructure() {
        delete[] xadj;
        delete[] adjncy;
        delete[] part;
    } 
};
 
/**
 * @brief Creates a METIS adjacency structure of a graph as per 23/45 METIS manual. 
 * @brief Representing the lattic structure of the tensor shape (in the X-Y corrdinate) 
 * @param metisGraph - a struct containing necessary parameters to pass to METIS
 * @return None
 * 
 * ***NOTE - this will only work for tensor (x,y) sizes that are a factor of 4 i.e. 4,16,20,32
 */
void generateMetisGraph(MetisGraphStructure& metisGraph) {

    /*Nodes in the graph*/
    std::vector<int> nodeNumbers  = mv::utils::generateSequence<int>(metisGraph.m_numberTensorVertices);
    int adjncyIndex = 0;
    int xadjIndex = 0;

    /* ncon is the number of weights associated with each vertex, the array vwgt contains n ∗ ncon 
     * elements (recall that n is the number of vertices)
     *
     * The weight of each vertex is the same and is the number of vertices in the METIS graph 
    */

    /*Populate the weight of each vertex*/
    for (auto i : nodeNumbers)
        metisGraph.vwgt[i] = nodeNumbers.size();
    
    /* A Sample Graph
     * 0---1---2---3---4
     * |   |   |   |   |
     * 5---6---7---8---9
     * |   |   |   |   |
     * 10--11---12-13--14
     */

    for (auto it = nodeNumbers.begin(); it != nodeNumbers.end(); it++) {

        /*Top left node*/ 
        if((*it%metisGraph.m_xDim == 0) && (*it == 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }
  
        /*Intermediate node left side*/ 
        if((*it%metisGraph.m_xDim == 0) && ((*it + metisGraph.m_xDim) < ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - metisGraph.m_xDim;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }

        /*Bottom left node*/
        if((*it%metisGraph.m_xDim == 0) && ((*it + metisGraph.m_xDim) > ((int)nodeNumbers.size() -1)) && ((*it) != 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - metisGraph.m_xDim;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
        }

        /*Top right node*/
        if(((*it - (metisGraph.m_xDim-1)%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1) && ((*it-(metisGraph.m_xDim-1) == 0)))) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }

       /*Intermediate right side node*/
        if(((*it - (metisGraph.m_xDim-1))%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1)  && ((*it-(metisGraph.m_xDim-1) != 0))  && (*it %(nodeNumbers.size()-1) != 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - metisGraph.m_xDim;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }

        /*Bottm right node*/
        if(((*it - (metisGraph.m_xDim-1))%metisGraph.m_xDim == 0) && ((*it - (*it -(metisGraph.m_xDim-1))) == metisGraph.m_xDim -1) && (*it %(nodeNumbers.size()-1) == 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - (metisGraph.m_xDim); 
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.xadj[xadjIndex] = adjncyIndex;
        }
        
        /*Middle nodes top row*/
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) < (metisGraph.m_xDim - 1))) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }

        /*Middle nodes bottom row*/
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) > ((int)nodeNumbers.size()-1) - metisGraph.m_xDim) && ((*it) != ((int)nodeNumbers.size()-1))) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - (metisGraph.m_xDim); 
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
        }

        /*Middle nodes not on bottom or top rows or the side columns*/
        if(((*it)%metisGraph.m_xDim != 0) && ((*it) < ((int)nodeNumbers.size()-1) - metisGraph.m_xDim) && ((*it) > (metisGraph.m_xDim-1)) && ((*it+1)%metisGraph.m_xDim != 0)) {

            metisGraph.xadj[xadjIndex] = adjncyIndex;
            xadjIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - (metisGraph.m_xDim); 
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it - 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + 1;
            adjncyIndex++;
            metisGraph.adjncy[adjncyIndex] = *it + (metisGraph.m_xDim); 
            adjncyIndex++;
        }
    }   
} 

/**
 * @brief Partition a tensor using the METIS library
 * @param metisGraph - a struct containing necessary parameters to pass to METIS
 * @param nWorkloads - the number of partitions (workloads) to partition the tensor into 
 * @return return code from METIS
 */

int partitionTensorMETIS(MetisGraphStructure& metisGraph, idx_t nWorkloads) 
{
    /*METIS call*/
    int res = METIS_PartGraphRecursive(&metisGraph.m_numberTensorVertices,&metisGraph.nWeights, metisGraph.xadj, metisGraph.adjncy,
                    metisGraph.vwgt, NULL, NULL, &nWorkloads, NULL,
				    NULL, metisGraph.options, &metisGraph.objval, metisGraph.part);
                    
    return res;
}

/**
 * @brief Divides a number by 2 repeatedly. Example if maxSplitRange = 16 -> returns 16,8,4,2
 * @param Number to be divided by 2 repeatedly
 * @param A maximum value to divide  by 2 repeatedly
 * @return A set of the number divided by two repeatedly
 */
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

/**
 * @brief Generate a pool of possible splits (workloads) of a tensor 
 * @param A tensor 
 * @param The numnber of DPUs per cluster
 * @return A pool of possible splits (possible workloads)
 */
std::set<int> getNWorkloads(std::vector<mv::Data::TensorIterator> tensor, int nDPUxClusterS)
{
    std::cout << "Test getNWorkloads" << std::endl;
    std::cout << "Tensor shape is " << tensor[0]->getShape().toString() << std::endl;

    /*maxSplitsXY*/
    auto xDim = tensor[0]->get<mv::Shape>("shape")[0];
    auto yDim = tensor[0]->get<mv::Shape>("shape")[1];
    auto maxSplitsXY = ceil(xDim/4) * ceil(yDim/4);

    std::cout << "maxSplitsXY is " << maxSplitsXY << std::endl;

    /*maxSplitsZ*/
    auto maxSplitsZ = ceil(tensor[0]->get<mv::Shape>("shape")[2]/16);

    std::cout << "maxSplitsZ is " << maxSplitsZ << std::endl;

    /*Pool of possible splits*/
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

    int nDPU = 4;                       /*Number of DPUs*/
    int nClusters = 1;                  /*Number of clusters*/
    int nDPUxCluster = nDPU/nClusters;  /*Number of DPUs per cluster*/
    std::set<int> workloadsList;
    std::pair <int,int> MPEMode (4, 4); /*MPE mode*/

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask") /*Should check for taskOP value here, it should be convolution*/
        {
            std::cout << "Found DPUTask of type convolution" << std::endl;
 
            /*Get output tensor*/
            auto outputTensor = opIt->getOutputTensor();

            /*Metis struct*/
            MetisGraphStructure metisGraph(outputTensor[0]->getShape(), MPEMode);
            
            /* Populate Metis adjacency structure*/ 
            generateMetisGraph(metisGraph); 
            METIS_SetDefaultOptions(metisGraph.options);

            /*Partition tensor into workloads*/            
            /*Should calculate a pool of workloads as per PoC compiler here*/ 

            //workloadsList = getNWorkloads(outputTensor, nDPUxCluster);
        
            /*Forcing number of workloads to be nDPU/nCluster round to nearest even number*/
            //idx_t nWorkloads = round(nDPUxCluster/2)*2; 

            
            idx_t nWorkloads    = 4;
            std::cout << "Number of workloads is " << nWorkloads << std::endl;

            /*Partition tensor into workloads with METIS*/
            auto res = partitionTensorMETIS(metisGraph,nWorkloads);

            for(int i =0; i < 36; i++) {

                std::cout << metisGraph.xadj[i] << std::endl;
            }

            for(int i =0; i < 120; i++) {

                std::cout << metisGraph.adjncy[i] << std::endl;
            }
            
            if( res != 1 ) {
                throw "Error occured during tensor partitioning into workloads using METIS, ensure number of workloads is even!";
            }

            std::cout << "Value of the objective function that was minimized (should be same as PoC compiler) is: " << metisGraph.objval << std::endl;
           
            /*Print partition*/
            for(int part_i = 0; part_i < metisGraph.m_numberTensorVertices; part_i++) { 
                
                std::cout << "Node " << part_i << " " << "is in partition " << metisGraph.part[part_i] << std::endl;
            }

            /*Workloads class instance*/
            Workloads workloads(opIt->getName());

            /*Populate each workload*/
            for(int workload = 0; workload < nWorkloads; workload++) { 

                /*In some cases METIS might return less than the number or partitions (workloads) than you expect*/
                /*This needs to be handled*/

                workloads.getWorkloads().push_back(Workload()); /*Add each workload (struct) to vector of workloads*/
                
                workloads.getWorkloads()[workload].workloadID = workload;
                workloads.getWorkloads()[workload].clusterID = 0; /*Need to configure this*/
                workloads.getWorkloads()[workload].MinZ = 0; /*Need to configure this*/
                workloads.getWorkloads()[workload].MinZ = 15; /*Need to configure this*/
                workloads.getWorkloads()[workload].padTop = 0; /*Need to configure this*/
                workloads.getWorkloads()[workload].padBottom = 0; /*Need to configure this*/
                workloads.getWorkloads()[workload].padLeft = 0; /*Need to configure this*/
                workloads.getWorkloads()[workload].padRight = 0; /*Need to configure this*/


            }

               for(unsigned part_i = 0; part_i < metisGraph.m_numberTensorVertices; part_i++) {
                   
                   std::cout << part_i << " " << metisGraph.part[part_i] << std::endl;
                }


            



            
       



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
