#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/graph/graph.hpp"
#include <algorithm>
#include <climits>
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
                "Generate workloads using the METIS graph partitioning library");
    }
}

/* Coordinates of METIS graph node:
 * use it to restore coordinates
 * after leveraging METIS for
 * partitioning of workloads.
*/ 

struct NodeCoords
{
    int16_t min_x, n_elem_x;
    int16_t min_y, n_elem_y;
};

/* METIS parameters*/
struct MetisGraphStructure
{
    idx_t* xadj;                      /*Indexes of starting points in adjacent array*/
    idx_t* adjncy;                    /*Adjacent vertices in consecutive index order*/
    idx_t* vwgt;
    idx_t* part; 
    idx_t objval;
    idx_t nWeights  = 1;              /*Each vertex stores 1 weight*/
    idx_t options[METIS_NOPTIONS];

    idx_t m_numberTensorVertices;
    idx_t m_numberTensorEdges;
    int m_xDim;
    int m_yDim;

    NodeCoords* node_coords;

    MetisGraphStructure(mv::Shape outputTensor, std::pair <int,int> MPEMode){

        /*Shape of output tensor x-y*/
        double tensorXDim = outputTensor[0]; 
        double tensorYDim = outputTensor[1];

        /*Number of vertices and edges in METIS lattic graph of tensor*/
        m_numberTensorVertices = ceil(tensorXDim / MPEMode.first)  * ceil(tensorYDim / MPEMode.second);    
        m_numberTensorEdges = (2 * ceil(tensorXDim / MPEMode.first) * ceil(tensorYDim / MPEMode.second)) - ceil(tensorXDim / MPEMode.first) - ceil(tensorYDim / MPEMode.second);
        
        /*X-Y dimension of METIS lattic graph*/
        m_xDim = ceil((tensorXDim / MPEMode.first));
        m_yDim = ceil((tensorYDim / MPEMode.second));

        /*METIS parameters - description page 23 Metis manual*/
        xadj = new idx_t[m_numberTensorVertices + 1]; 
        adjncy = new idx_t[2*m_numberTensorEdges];
        part = new idx_t[m_numberTensorVertices];
        vwgt = new idx_t[m_numberTensorVertices* nWeights];

        node_coords = new NodeCoords [ m_numberTensorVertices ];
        
        /* Weights of METIS vertices
         * Description page 23 Metis manual
         * 
         * Required when tensor size is not a multiple of 4 for MPE mode (4,4) which is only supported for WW09
         * When tensor size is not a multiple of 4 then not all DPUs will be fully utilised (i.e. < 256 multiplication operations)
         * Therefore we assign nodes different weights when partitioning
        */
        int n_elem_y;
        int n_elem_x;
        int nodeIndex = 0;
        for(int j=0; j < m_yDim; j++) {
            
            if ((j+1 < m_yDim) || (!fmod(tensorYDim,MPEMode.first)))
                    n_elem_y = MPEMode.first;                 
                else 
                    n_elem_y = (int)tensorYDim%MPEMode.first; 
                            
            for(int k=0; k < m_xDim; k++) {
                
                if ((k+1 < m_xDim) || (!fmod(tensorXDim,MPEMode.second)))
                    n_elem_x = MPEMode.second;
                else 
                    n_elem_x = (int)tensorXDim%MPEMode.second;
            
                vwgt[nodeIndex] = n_elem_x * n_elem_y;

                node_coords[nodeIndex].min_x = k * MPEMode.first;
                node_coords[nodeIndex].min_y = j * MPEMode.second;
                node_coords[nodeIndex].n_elem_x = n_elem_x;
                node_coords[nodeIndex].n_elem_y = n_elem_y;
                nodeIndex++;
            }
            
        }
    }
    
    ~MetisGraphStructure() {
        delete[] xadj;
        delete[] adjncy;
        delete[] part;
        delete[] vwgt;
        delete[] node_coords;
    }
};
 
/**
 * @brief Creates a METIS adjacency structure of a graph as per 23/45 METIS manual. 
 * @brief Representing the lattic structure of the tensor shape (in the X-Y corrdinate) 
 * @param metisGraph - a struct containing necessary parameters to pass to METIS
 * @return None
 * 
 */
void generateMetisGraph(MetisGraphStructure& metisGraph) {

    /*Nodes in the graph*/
    std::vector<int> nodeNumbers  = mv::utils::generateSequence<int>(metisGraph.m_numberTensorVertices);
    int adjncyIndex = 0;
    int xadjIndex = 0;
    
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
 * @param metisGraph A struct containing necessary parameters to pass to METIS
 * @param nWorkloads The number of partitions (workloads) to partition the tensor into 
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

void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &)
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
        if ((opIt->getOpType() == "DPUTask") && (opIt->get<std::string>("taskOp") == "Conv")) 
        {

            pass.log(Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + "of type " + opIt->get<std::string>("taskOp"));
 
            /*Get output tensor*/
            auto outputTensor = opIt->getOutputTensor();

            /*Metis struct*/
            MetisGraphStructure metisGraph(outputTensor[0]->getShape(), MPEMode);
            
            /* Populate Metis adjacency structure*/ 
            generateMetisGraph(metisGraph); 
            METIS_SetDefaultOptions(metisGraph.options);

            /*Partition tensor into workloads*/            
            /*Should calculate a pool of workloads as per PoC compiler here and find the best one based on the cost functions*/ 
            //workloadsList = getNWorkloads(outputTensor, nDPUxCluster);
        
            /*Forcing number of workloads to be nDPU/nCluster (round to nearest even number) for WW09 deliverbale*/
            idx_t nWorkloads = round(nDPUxCluster/2)*2; 

            pass.log(Logger::MessageType::Debug, "Number of workloads is:" + std::to_string(nWorkloads));

            /*Partition tensor into workloads with METIS*/
            auto res = partitionTensorMETIS(metisGraph,nWorkloads);
            
            if( res != 1 ) {
                throw "Error occured during tensor partitioning into workloads using METIS, ensure number of workloads is even!";
            }

            pass.log(Logger::MessageType::Debug, "Value of the objective function that was minimized (should be same as PoC compiler) is:" + std::to_string(metisGraph.objval));
           
            /*Print node partition*/
            for(int part_i = 0; part_i < metisGraph.m_numberTensorVertices; part_i++) { 
                
                pass.log(Logger::MessageType::Debug, "Node " + std::to_string(part_i) + "of type " + "is in partition " + std::to_string(metisGraph.part[part_i]));
            }

            /*Workloads class instance*/
            Workloads workloads(opIt->getName());

            /*Populate each workload*/
            /*In some cases METIS might return a number or partitions (workloads) less than you specified (i.e. small tensor and large number of partitions*/
            /*This needs to be handled here for now assuming number of partitions is the number or workloads*/
            for(int workload = 0; workload < nWorkloads; workload++) { 

                workloads.getWorkloads().push_back(Workload()); /*Add each workload (struct) to vector of workloads*/
                
                workloads.getWorkloads()[workload].workloadID = workload;
                workloads.getWorkloads()[workload].clusterID = 0;           /*WW09 deliverbale is 1 cluster*/
                workloads.getWorkloads()[workload].MinZ = 0;                /*WW09 deliverbale is less than 16 channels*/
                workloads.getWorkloads()[workload].MaxZ = 15;               /*WW09 deliverbale is less than 16 channels*/
                workloads.getWorkloads()[workload].padTop = 0;              /*These are zero in PoC compiler - relevant after WW09*/
                workloads.getWorkloads()[workload].padBottom = 0;           /*These are zero in PoC compiler - relevant after WW09*/
                workloads.getWorkloads()[workload].padLeft = 0;             /*These are zero in PoC compiler - relevant after WW09*/
                workloads.getWorkloads()[workload].padRight = 0;            /*These are zero in PoC compiler - relevant after WW09*/
                
                workloads.getWorkloads()[workload].MPEMode = Matrix;        /*Matrix is MPE Mode (4,4)*/
                
                /* Converting the paritions returned by METIS 
                 * into tensor coordinates and populating these fields of workload 
                */

                /*NB: references (just shorter aliases for WL coordinates)*/
                int16_t& wl_min_x = workloads.getWorkloads()[workload].MinX;
                int16_t& wl_min_y = workloads.getWorkloads()[workload].MinY;
                int16_t& wl_max_x = workloads.getWorkloads()[workload].MaxX;
                int16_t& wl_max_y = workloads.getWorkloads()[workload].MaxY;

                wl_min_x = SHRT_MAX;
                wl_min_y = SHRT_MAX;
                wl_max_x = -1;
                wl_max_y = -1;

                for (int i=0; i < metisGraph.m_numberTensorVertices; i++)
                {
                    if (metisGraph.part[i] == workload)
                    {
                        int16_t min_x = metisGraph.node_coords[i].min_x;
                        int16_t min_y = metisGraph.node_coords[i].min_y;
                        int16_t max_x = min_x + metisGraph.node_coords[i].n_elem_x - 1;
                        int16_t max_y = min_y + metisGraph.node_coords[i].n_elem_y - 1;

                        /* Guard calling to std::min/max with parentheses,
                         * as they may mess with same-named macro on Windows
                        */ 
                        wl_min_x = (std::min)(wl_min_x, min_x);
                        wl_min_y = (std::min)(wl_min_y, min_y);
                        wl_max_x = (std::max)(wl_max_x, max_x);
                        wl_max_y = (std::max)(wl_max_y, max_y);
                    }
                }
            }
           
            /*Add workloads as Attribute*/
            opIt->set<mv::Workloads>("Workloads", workloads);
        }
    }
    std::cout << "Exiting Workload Generation Pass " << std::endl;
}
