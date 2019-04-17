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
//void splitTensorsOverClusterFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object &);

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

void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object &)
{

    pass.log(mv::Logger::MessageType::Debug, "Starting workload generation pass");

    /*Get the number of clusters that the VPU supports*/
    auto nceDefs = target.nceDefs();
    //auto nClusters = nceDefs.find("Clusters")->second.totalNumber;

    /*Get all tensors*/


    int nDPU = 4;                       /*Number of DPUs*/
    int nClusters = 1;                  /*Number of clusters*/
    int nDPUxCluster = nDPU/nClusters;  /*Number of DPUs per cluster*/
    std::set<int> workloadsList;
    std::pair <int,int> MPEMode (4, 4); /*MPE mode*/

    //parse CostFunction from Comp Descriptor
    CostFunctions costFunction = CostFunctions::Balanced; //default
    std::string sCostFunction = std::string(); 
    if (passDesc.hasAttr("costfunction")) 
    {
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


    mv::OpModel om(model);
    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {

            pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));
 
            /*Get output tensor*/
            auto outputTensor = opIt->getOutputTensor();

            /*Workload's instance, name and tensorShape, MPE mode*/
            mv::Workloads workloads(opIt->getName(),outputTensor[0]->getShape(), MPEMode);

            /* Partition tensor into workloads  
             * Calculate a pool of possible workloads select the best one based on the cost functions
            */ 
            
            /*Metis algorithm for workloads*/

            /*Get Metis adjacency structure object*/
            auto metisGraph = workloads.getMetisGraph();

            /* Populate Metis adjacency structure*/ 
            workloads.generateMetisGraph(metisGraph); 

            /*Forcing number of workloads to be nDPU/nCluster (round to nearest even number)*/
            idx_t nWorkloads  = workloads.getNWorkloads(outputTensor[0]->getShape(), nDPUxCluster);
        
            pass.log(mv::Logger::MessageType::Debug, "Number of workloads is:" + std::to_string(nWorkloads));

            /*Partition tensor into workloads with METIS*/
            auto res = workloads.partitionTensorMETIS(metisGraph, nWorkloads);
            
            if( res != 1 ) 
                std::runtime_error("Error occured during tensor partitioning into workloads using METIS, ensure number of workloads is even!");
            
            pass.log(mv::Logger::MessageType::Debug, "Value of the objective function that was minimized by METIS (should be same as PoC compiler) is: " + std::to_string(metisGraph.objval));
           
            /*Print node partition*/
            for(int part_i = 0; part_i < metisGraph.m_numberTensorVertices; part_i++) 
                pass.log(mv::Logger::MessageType::Debug, "Node " + std::to_string(part_i) + " is in partition " + std::to_string(metisGraph.part[part_i]));
            

            /*Populate each workload*/
            workloads.populateWorkloadsFromPartitions(metisGraph, nWorkloads, pass);

            /*Add workloads as Attribute*/
            opIt->set<mv::Workloads>("Workloads", workloads);
            

            //std::vector<float> exeCycle = getExecutionCycles(outputTensor, workloads, nDPUxCluster, MPEMode, costFunction);
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

    std::vector<float> workloadsExecutionCycles;
    if (validateWorkloads(outputTensor, workloads))
    {   
        for(std::vector<mv::Workload>::iterator itWL = workloads.getWorkloads().begin(); itWL != workloads.getWorkloads().end(); ++itWL) 
        {
            float height = itWL->MaxY - itWL->MinY + MPEMode.first;
            float width = itWL->MaxX - itWL->MinX + MPEMode.second;

            float sumExeCycles = ceil(outputTensor[0]->getShape()[2]/16.0) * ceil(height / MPEMode.first) * ceil(width / MPEMode.second);
            workloadsExecutionCycles.push_back(sumExeCycles);
        }
    }
    else
    {   //workload not schedulable
        workloadsExecutionCycles = {INFINITY};
    }
    
    float critical_wl = *std::max_element(workloadsExecutionCycles.begin(), workloadsExecutionCycles.end());
    //float lower_wl = *std::min_element(workloadsExecutionCycles.begin(), workloads_execution_cycles.end());

    float wl_sum = float(0);
    for (auto& cycles : workloadsExecutionCycles)
        wl_sum += cycles;

    float min_range = wl_sum/nDPUxCluster;
    float max_range = wl_sum/nDPUxCluster + critical_wl;

    if (costFunction == CostFunctions::Balanced)
    {
        float balancing = float(0.0);
        if (!std::isinf(wl_sum))
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
        if (std::isinf(wl_sum))
            return {INFINITY, INFINITY};
        else
        {
            float greedy = greedyTaskAssignment(nDPUxCluster, workloadsExecutionCycles);
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



// void splitTensorsOverClusterFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object &)
// {

//     pass.log(mv::Logger::MessageType::Debug, "Starting split tensors over clusters pass");

//     /*Get the number of clusters that the VPU supports*/
//     auto nceDefs = target.nceDefs();
//     auto nClusters = nceDefs.find("Clusters")->second.totalNumber;

//     /*Get all tensors*/
//     mv::OpModel om(model);
//     for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
//     {
//         /*number outputs for operation*/
//         for(std::size_t i = 0; i < opIterator.outputsSize(); i++) {

//         }
        
//     }


    
//     pass.log(mv::Logger::MessageType::Debug, "Exiting split tensors over clusters pass");
// }