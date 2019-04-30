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



static void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

    //Get number of Clusters and DPU's
    int nDPU = 20;                      //Default number of DPUs
    int nClusters = 4;                  //Default number of Clusters
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("Number_of_DPUs")) 
        nDPU = globalParams->get<int>("Number_of_DPUs");
    if (globalParams->hasAttr("Number_of_Clusters")) 
        nClusters = globalParams->get<int>("Number_of_Clusters");
    
    int nDPUxCluster = nDPU/nClusters;  /*Number of DPUs per cluster*/
    std::set<int> workloadsList;

    mv::OpModel om(model);
    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {
            pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));
 
            /*Get output tensor*/
            auto outputTensor = opIt->getOutputTensor();
            std::vector<mv::Workloads> solutions;

            /*Workload's instance, name and tensorShape, MPE mode*/
            std::pair <int,int> MPEMode (1, 16); /*MPE mode*/
            mv::Workloads workloads(opIt->getName(),outputTensor[0]->getShape(), MPEMode);
            std::vector<std::string> algorithms = workloads.getTensorSplitAlgorithms(passDesc, pass);

            /*Forcing number of workloads to be nDPU/nCluster (round to nearest even number)*/
            idx_t nWorkloads  = workloads.getNWorkloads(outputTensor[0]->getShape(), nDPUxCluster);
            pass.log(mv::Logger::MessageType::Debug, "Number of workloads is:" + std::to_string(nWorkloads));

            // Partition tensor into workloads
            for (std::string algorithm : algorithms)
            {
                if (algorithm == "Metis")
                {
                    // Populate Metis adjacency structure and partition tensor
                    workloads.generateMetisGraph(); 
                    auto res = workloads.partitionTensorWithMETIS(nWorkloads, pass);
                    if( res==1)
                        workloads.populateWorkloadsFromPartitions(nWorkloads, pass);
                    else
                        pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using METIS, ensure number of workloads is even!");
                }
                else if (algorithm == "Rectangle")
                {
                    //Partition tensor into workloads with Rectangle
                    auto res = workloads.partitionTensorWithRectangleHeuristic(nWorkloads, pass);
                    if(res==1)
                        workloads.populateWorkloadsFromPartitions(nWorkloads, pass);
                    else
                        pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using Rectangle!");
                }
                else if (algorithm == "Z-Tiling")
                {
                    //Partition tensor into workloads with Rectangle

                }
                
                // Calculate execution cycles for these workloads
                auto costFunction = workloads.getCostFunction(passDesc, pass);
                workloads.generateExecutionCycles(outputTensor, nDPUxCluster, costFunction);
                solutions.push_back(workloads);
            }
            
            // TODO: return workload with lowest mean execution time
            //mv::Workloads optimal_workload = min(solutions, key= lambda x: (np.mean(x[0]), len(x[1])))



             opIt->set<mv::Workloads>("Workloads", workloads);
        }
    }
    pass.log(mv::Logger::MessageType::Debug, "Exiting workload generation pass");
}
