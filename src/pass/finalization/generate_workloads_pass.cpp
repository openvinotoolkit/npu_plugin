#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/target/keembay/rectangle.hpp"
#include "include/mcm/graph/graph.hpp"
#include <algorithm>
#include <climits>
#include <math.h>
#include <metis.h>

/** 
 * @brief Cost Function types to be used when evaluating execution cycles of a workload 
 */ 
enum class CostFunctions
{
    Balanced,
    CriticalPath,
    Greedy,
    MinMaxWorkloads
};

// method declarations
static void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static bool validateWorkloads(std::vector<mv::Data::TensorIterator>& Tensor, mv::Workloads& workloads);
static std::vector<float> getExecutionCycles(std::vector<mv::Data::TensorIterator>& outputTensor, mv::Workloads& workloads, int nDPUxCluster, std::pair <int,int> MPEMode, CostFunctions costFunction);
float greedyTaskAssignment(int nProcessors, std::vector<float>& workloadCosts);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(GenerateWorkloads)
            .setFunc(generateWorkloadsFcn)
            .setDescription(
                "Generate workloads using the METIS graph partitioning library");
    }
}

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

    mv::Rectangle* node_coords;

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

        node_coords = new mv::Rectangle [ m_numberTensorVertices ];
        
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

                int min_x = k * MPEMode.first;
                int min_y = j * MPEMode.second;
        
                node_coords[nodeIndex] = mv::Rectangle(min_x, min_y, n_elem_x, n_elem_y);
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


void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object &)
{

    pass.log(mv::Logger::MessageType::Debug, "Starting workload generation pass");

    mv::OpModel om(model);

    int nDPU = 4;                       /*Number of DPUs*/
    int nClusters = 1;                  /*Number of clusters*/
    int nDPUxCluster = nDPU/nClusters;  /*Number of DPUs per cluster*/
    std::set<int> workloadsList;
    std::pair <int,int> MPEMode (4, 4); /*MPE mode*/

    //parse CostFunction from Comp Descriptor
    CostFunctions costFunction = CostFunctions::Balanced; //default
    std::string sCostFunction = std::string(); 
    if (passDesc.hasAttr("costfunction")) {
        sCostFunction = passDesc.get<std::string>("costfunction");
        if (sCostFunction == "balanced")
            costFunction = CostFunctions::Balanced;
        else if (sCostFunction == "criticalpath")
            costFunction = CostFunctions::CriticalPath;
        else if (sCostFunction == "minmax")
            costFunction = CostFunctions::MinMaxWorkloads;
        else if (sCostFunction == "greedy")
            costFunction = CostFunctions::Greedy;
        else 
            pass.log(mv::Logger::MessageType::Warning, "Could not parse the Cost Function type (only \"balanced | criticalpath | minmax | greedy\" currently supported). Using \"Balanced\"...");
    }
    else
        pass.log(mv::Logger::MessageType::Info, "No Cost Function specified in descriptor, using \"Balanced\"...");


    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {

            pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));
 
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

            pass.log(mv::Logger::MessageType::Debug, "Number of workloads is:" + std::to_string(nWorkloads));

            /*Partition tensor into workloads with METIS*/
            auto res = partitionTensorMETIS(metisGraph,nWorkloads);
            
            if( res != 1 ) {
                throw "Error occured during tensor partitioning into workloads using METIS, ensure number of workloads is even!";
            }

            pass.log(mv::Logger::MessageType::Debug, "Value of the objective function that was minimized (should be same as PoC compiler) is:" + std::to_string(metisGraph.objval));
           
            /*Print node partition*/
            for(int part_i = 0; part_i < metisGraph.m_numberTensorVertices; part_i++) { 
                
                pass.log(mv::Logger::MessageType::Debug, "Node " + std::to_string(part_i) + " is in partition " + std::to_string(metisGraph.part[part_i]));
            }

            /*Workloads class instance*/
            mv::Workloads workloads(opIt->getName());

            /*Populate each workload*/
            /*In some cases METIS might return a number or partitions (workloads) less than you specified (i.e. small tensor and large number of partitions*/
            /*This needs to be handled here for now assuming number of partitions is the number or workloads*/
            for(int workload = 0; workload < nWorkloads; workload++) { 

                workloads.getWorkloads().push_back(mv::Workload()); /*Add each workload (struct) to vector of workloads*/
                
                workloads.getWorkloads()[workload].workloadID = workload;
                workloads.getWorkloads()[workload].clusterID = 0;           /*WW09 deliverbale is 1 cluster*/
                workloads.getWorkloads()[workload].MinZ = 0;                /*WW09 deliverbale is less than 16 channels*/
                workloads.getWorkloads()[workload].MaxZ = outputTensor[0]->getShape()[2] -1;  //output channels
                workloads.getWorkloads()[workload].padTop = 0;              /*These are zero in PoC compiler - relevant after WW09*/
                workloads.getWorkloads()[workload].padBottom = 0;           /*These are zero in PoC compiler - relevant after WW09*/
                workloads.getWorkloads()[workload].padLeft = 0;             /*These are zero in PoC compiler - relevant after WW09*/
                workloads.getWorkloads()[workload].padRight = 0;            /*These are zero in PoC compiler - relevant after WW09*/
                
                workloads.getWorkloads()[workload].MPEMode = mv::Matrix;        /*Matrix is MPE Mode (4,4)*/
                
                /* Converting the paritions returned by METIS 
                 * into tensor coordinates and populating these fields of workload 
                */

                using xyz_type = decltype(mv::Workload::MinX);

                // NB: references (just shorter aliases for WL coordinates)
                xyz_type& wl_min_x = workloads.getWorkloads()[workload].MinX;
                xyz_type& wl_min_y = workloads.getWorkloads()[workload].MinY;
                xyz_type& wl_max_x = workloads.getWorkloads()[workload].MaxX;
                xyz_type& wl_max_y = workloads.getWorkloads()[workload].MaxY;

                wl_min_x = std::numeric_limits<xyz_type>::max();
                wl_min_y = std::numeric_limits<xyz_type>::max();
                wl_max_x = -1;
                wl_max_y = -1;

                for (int i=0; i < metisGraph.m_numberTensorVertices; i++)
                {
                    if (metisGraph.part[i] == workload)
                    {
                        int min_x = metisGraph.node_coords[i].min_x();
                        int max_x = metisGraph.node_coords[i].max_x();
                        int min_y = metisGraph.node_coords[i].min_y();
                        int max_y = metisGraph.node_coords[i].max_y();

                        // NB: guard calling to std::min/max with parentheses,
                        //     as they may mess with same-named macro on Windows
                        wl_min_x = (std::min)(wl_min_x, static_cast<xyz_type>(min_x));
                        wl_max_x = (std::max)(wl_max_x, static_cast<xyz_type>(max_x));
                        wl_min_y = (std::min)(wl_min_y, static_cast<xyz_type>(min_y));
                        wl_max_y = (std::max)(wl_max_y, static_cast<xyz_type>(max_y));
                    }
                }
                pass.log(mv::Logger::MessageType::Debug, "\nworkload: " + std::to_string(workload));
                pass.log(mv::Logger::MessageType::Debug, " min_x: " + std::to_string(workloads.getWorkloads()[workload].MinX));
                pass.log(mv::Logger::MessageType::Debug, " max_x: " + std::to_string(workloads.getWorkloads()[workload].MaxX));
                pass.log(mv::Logger::MessageType::Debug, " min_y: " + std::to_string(workloads.getWorkloads()[workload].MinY));
                pass.log(mv::Logger::MessageType::Debug, " max_y: " + std::to_string(workloads.getWorkloads()[workload].MaxY));
                pass.log(mv::Logger::MessageType::Debug, " min_z: " + std::to_string(workloads.getWorkloads()[workload].MinZ));
                pass.log(mv::Logger::MessageType::Debug, " max_z: " + std::to_string(workloads.getWorkloads()[workload].MaxZ));
            }
           
            /*Add workloads as Attribute*/
            opIt->set<mv::Workloads>("Workloads", workloads);

            std::vector<float> exeCycle = getExecutionCycles(outputTensor, workloads, nDPUxCluster, MPEMode, costFunction);
            // TODO: process the execution cycles
        }
    }
    pass.log(mv::Logger::MessageType::Debug, "Exiting workload generation pass");
}

static bool validateWorkloads(std::vector<mv::Data::TensorIterator>& inputTensor, mv::Workloads& workloads)
{
    //    Check if the generated workloads are valid
    //    Check 1: the union of the workload have to make the whole tensor
    //          - Same Volume
    //          - Same vertices
    //    Check 2: no workload intersection

    // Check 0: empty workloads are not valid
    // Using size_t variable (nWorkloads) below, you may see a warning. Casting to double or int is unnecessary
    if ((workloads.nWorkloads()) == 0)
    {
        workloads.log(mv::Logger::MessageType::Debug, "METIS partition failed because of total number of the partitions <=0");
        return false;
    }

    // Check 1: Volume of the tensor = sum of volumes of the individual workloads
    double vol = workloads.getAllWorkloadsVolume();
    std::size_t totalVol = inputTensor[0]->getShape().totalSize();
    if (inputTensor[0]->getShape().totalSize() != workloads.getAllWorkloadsVolume())
    {
        workloads.log(mv::Logger::MessageType::Warning, "METIS partition failed because of volume differences. Original Tensor: " + 
                    std::to_string(inputTensor[0]->getShape().totalSize()) + " Partitioned Tensor: " + std::to_string(workloads.getAllWorkloadsVolume()));
        return false;
    }

    // Check for same vertices for each of the X, Y and X dimensions. This is done by comparing the shape of the inputTensor and min max of (all) workloads
    if (workloads.getShapefromMinMax() != inputTensor[0]->getShape())
    {
        workloads.log(mv::Logger::MessageType::Warning, "METIS partition failed because vertices/bounds different between Original Tensor " + 
                                     inputTensor[0]->getShape().toString() + " and Partitioned Tensor " + workloads.getShapefromMinMax().toString());
        return false;
    }

    // Check 2: No intersection between workloads.
    if (!workloads.noOverlap())
    {
        workloads.log(mv::Logger::MessageType::Debug, "METIS partition failed because of overlap of paritions");
        return false;
    }

    return true;
}

static std::vector<float> getExecutionCycles(std::vector<mv::Data::TensorIterator>& outputTensor, mv::Workloads& workloads, int nDPUxCluster, std::pair <int,int> MPEMode, CostFunctions costFunction)
{
    // notes from POC compiler:  Execution time is bounded by
    //      sum(WL)/DPU <= T <= max(WL_max)*(P-1)/P
    if (nDPUxCluster < 1)
        throw mv::ArgumentError("Generate Workloads Pass", "nDPUxCluster", std::to_string(nDPUxCluster), "Invalid number of DPUs");

    std::vector<float> workloads_execution_cycles;
    if (validateWorkloads(outputTensor, workloads))
    {   
        for(std::vector<mv::Workload>::iterator itWL = workloads.getWorkloads().begin(); itWL != workloads.getWorkloads().end(); ++itWL) 
        {
            float height = itWL->MaxY - itWL->MinY + MPEMode.first;
            float width = itWL->MaxX - itWL->MinX + MPEMode.second;

            float sumExeCycles = ceil(outputTensor[0]->getShape()[2]/16.0) * ceil(height / MPEMode.first) * ceil(width / MPEMode.second);
            workloads_execution_cycles.push_back(sumExeCycles);
        }
    }
    else
    {   //workload not schedulable
        workloads_execution_cycles = {INFINITY};
    }
    
    float critical_wl = *std::max_element(workloads_execution_cycles.begin(), workloads_execution_cycles.end());
    //float lower_wl = *std::min_element(workloads_execution_cycles.begin(), workloads_execution_cycles.end());

    float wl_sum = float(0);
    for (auto& cycles : workloads_execution_cycles)
        wl_sum += cycles;

    float min_range = wl_sum/nDPUxCluster;
    float max_range = wl_sum/nDPUxCluster + critical_wl;
    
    if (costFunction == CostFunctions::Balanced)
    {
        float balancing = float(0.0);
        if (!isinf(wl_sum))
            balancing = wl_sum/(ceil(wl_sum/nDPUxCluster) * nDPUxCluster);

        return {-balancing, -balancing};
    }
    else if(costFunction == CostFunctions::MinMaxWorkloads)
         return {min_range, max_range};

    else if(costFunction == CostFunctions::CriticalPath)
    {
        if (nDPUxCluster == 1)
            return {min_range, min_range};
        else
            return {max_range, max_range};
    }
    
    else if(costFunction == CostFunctions::Greedy)
    {
        if (isinf(wl_sum))
            return {INFINITY, INFINITY};
        else
        {
            float greedy = greedyTaskAssignment(nDPUxCluster, workloads_execution_cycles);
            return {greedy, greedy};
        }
    }

    else
        throw mv::ArgumentError("Generate Workloads Pass", "costFunction", "unknown", "Unsupported cost function");
}

/**
 * @brief
 * @param nProcessors is the number of computing resources
 * @param workloadCosts vector of workload costs
 */
float greedyTaskAssignment(int nProcessors, std::vector<float>& workloadCosts)
{
    std::priority_queue<int, std::vector<int>, std::greater<int> > exeCycles; //ascending sizes
    for (int i=0; i<nProcessors; ++i)
        exeCycles.push(0);
    
    for (size_t idxWorkload=0; idxWorkload<workloadCosts.size(); ++idxWorkload)
    {
        int smallestTime = exeCycles.top();
        exeCycles.pop();
        exeCycles.push(smallestTime + workloadCosts[idxWorkload]);
    }
    
    //return max value (ie, last value) in queue
    for (int i=0; i<nProcessors-1; ++i)
        exeCycles.pop();
    return exeCycles.top();
}
