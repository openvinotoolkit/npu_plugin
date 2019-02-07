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
    idx_t* xadj; /*Indexes of starting points in adjacent array*/
    idx_t* adjncy; /*Adjacent vertices in consecutive index order*/
    idx_t* vwgt;
    idx_t* part; 
    idx_t objval;
    idx_t nWeights  = 1; /*Each vertex stores 1 weight (which is n_elem_x * n_elem_y)*/
    idx_t options[METIS_NOPTIONS];

    idx_t m_nVertices;
    int m_xDim;
    int m_yDim;

    MetisGraphStructure(int nVertices, int nEdges, int xDim, int yDim) : m_nVertices(nVertices), m_xDim(xDim), m_yDim(yDim)  {
        
        xadj = new idx_t[nVertices + 1]; /*Description page 23 Metis manual*/
        adjncy = new idx_t[2*nEdges];
        part = new idx_t[nVertices];
        vwgt = new idx_t[nVertices * nWeights];
    }
    
    ~MetisGraphStructure() {
        //delete[] xadj;
        //delete[] adjncy;
        //delete[] part;
    } 
};
 
/**
 * @brief Creates a METIS adjacency structure of a graph as per 23/45 METIS manual
 * @brief representing the lattic structure of the shape (in the X-Y corrdinate) of a tensor
 * @param metisGraph 
 * @return None
 */
void generateMetisGraph(MetisGraphStructure& metisGraph) {

    /*Nodes in the graph*/
    std::vector<int> nodeNumbers  = mv::utils::generateSequence<int>(metisGraph.m_nVertices);
    int adjncyIndex = 0;
    int xadjIndex = 0;

    /*ncon is the number of weights associated with each vertex, the array vwgt contains n ∗ ncon 
     *elements (recall that n is the number of vertices)*/

    /*Popoulate vwgt*/
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
 * @return A pool of possible splits (workloads)
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

            int tensorXDim = outputTensor[0]->get<mv::Shape>("shape")[0]; 
            int tensorYDim = outputTensor[0]->get<mv::Shape>("shape")[1];

            int numberTensorVertices = (tensorXDim/4  * tensorYDim/4); /*MPE mode (4,4)*/
            int numberTensorEdges = (2 * tensorXDim/4 * tensorYDim/4) - tensorXDim/4 - tensorYDim/4;

            std::cout << "numberTensorVertices: " << numberTensorVertices << " " << "numberTensorEdges: " << numberTensorEdges << std::endl;

            /*Metis struct*/
            MetisGraphStructure metisGraph(numberTensorVertices, numberTensorEdges, tensorXDim/4, tensorYDim/4);
            
            /* Populate Metis adjacency structure structures*/ 
            generateMetisGraph(metisGraph); 
            METIS_SetDefaultOptions(metisGraph.options);
            metisGraph.options[METIS_OPTION_NUMBERING] = 0; /*nubering starts from 0*/

            for(int i = 0; i < 17; i++) {

                 std::cout << metisGraph.xadj[i] << std::endl;
            }

             for(int i = 0; i < 48; i++) {

                 std::cout << metisGraph.adjncy[i] << std::endl;
             }
            
            /*Popoulate vwgt*/
            // for (int i = 0; i < 16; i++)
            //     metisGraph.vwgt[i] = 16;

            // metisGraph.xadj[0] = 0;
            // metisGraph.xadj[1] = 2;
            // metisGraph.xadj[2] = 5;
            // metisGraph.xadj[3] = 8;
            // metisGraph.xadj[4] = 12;
            // metisGraph.xadj[5] = 15;
            // metisGraph.xadj[6] = 19;
            // metisGraph.xadj[7] = 21;
            // metisGraph.xadj[8] = 24;
            // metisGraph.xadj[9] = 27;
            // metisGraph.xadj[10] = 31;
            // metisGraph.xadj[11] = 35;
            // metisGraph.xadj[12] = 38;
            // metisGraph.xadj[13] = 40;
            // metisGraph.xadj[14] = 43;
            // metisGraph.xadj[15] = 46;
            // metisGraph.xadj[16] = 48;

            // metisGraph.adjncy[0] = 1;
            // metisGraph.adjncy[1] = 2;
            // metisGraph.adjncy[2] = 0;
            // metisGraph.adjncy[3] = 8;
            // metisGraph.adjncy[4] = 3;
            // metisGraph.adjncy[5] = 0;
            // metisGraph.adjncy[6] = 3;
            // metisGraph.adjncy[7] = 4;
            // metisGraph.adjncy[8] = 1;
            // metisGraph.adjncy[9] = 2;
            // metisGraph.adjncy[10] = 9;
            // metisGraph.adjncy[11] = 5;
            // metisGraph.adjncy[12] = 2;
            // metisGraph.adjncy[13] = 5;
            // metisGraph.adjncy[14] = 6;
            // metisGraph.adjncy[15] = 3;
            // metisGraph.adjncy[16] = 4;
            // metisGraph.adjncy[17] = 10;
            // metisGraph.adjncy[18] = 7;
            // metisGraph.adjncy[19] = 4;
            // metisGraph.adjncy[20] = 7;
            // metisGraph.adjncy[21] = 5;
            // metisGraph.adjncy[22] = 6;
            // metisGraph.adjncy[23] = 11;
            // metisGraph.adjncy[24] = 1;
            // metisGraph.adjncy[25] = 12;
            // metisGraph.adjncy[26] = 9;
            // metisGraph.adjncy[27] = 3;
            // metisGraph.adjncy[28] = 8;
            // metisGraph.adjncy[29] = 13;
            // metisGraph.adjncy[30] = 10;
            // metisGraph.adjncy[31] = 5;
            // metisGraph.adjncy[32] = 9;
            // metisGraph.adjncy[33] = 14;
            // metisGraph.adjncy[34] = 11;
            // metisGraph.adjncy[35] = 7;
            // metisGraph.adjncy[36] = 10;
            // metisGraph.adjncy[37] = 15;
            // metisGraph.adjncy[38] = 8;
            // metisGraph.adjncy[39] = 13;
            // metisGraph.adjncy[40] = 9;
            // metisGraph.adjncy[41] = 12;
            // metisGraph.adjncy[42] = 14;
            // metisGraph.adjncy[43] = 10;
            // metisGraph.adjncy[44] = 13;
            // metisGraph.adjncy[45] = 15;
            // metisGraph.adjncy[46] = 11;
            // metisGraph.adjncy[47] = 14;
            
            idx_t nWorkloads    = 4;

            int ret = METIS_PartGraphRecursive(&metisGraph.m_nVertices,&metisGraph.nWeights, metisGraph.xadj, metisGraph.adjncy,
				       metisGraph.vwgt, NULL, NULL, &nWorkloads, NULL,
				       NULL, metisGraph.options, &metisGraph.objval, metisGraph.part);

    
            for(unsigned part_i = 0; part_i < metisGraph.m_nVertices; part_i++){
                std::cout << part_i << " " << metisGraph.part[part_i] << std::endl;
            }

            std::cout << "objval " << metisGraph.objval << std::endl;

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
