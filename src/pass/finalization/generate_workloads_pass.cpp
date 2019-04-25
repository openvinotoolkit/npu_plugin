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
            std::pair <int,int> MPEMode (4, 4); /*MPE mode*/
            mv::Workloads workloads(opIt->getName(),outputTensor[0]->getShape(), MPEMode);
            std::vector<std::string> algorithms = workloads.getTensorSplitAlgorithms(passDesc, pass);

            // Partition tensor into workloads
            for (std::string algorithm : algorithms)
            {
                if (algorithm == "Metis")
                {
                    /* Populate Metis adjacency structure*/ 
                    workloads.generateMetisGraph(); 

                    /*Forcing number of workloads to be nDPU/nCluster (round to nearest even number)*/
                    idx_t nWorkloads  = workloads.getNWorkloads(outputTensor[0]->getShape(), nDPUxCluster);
                    pass.log(mv::Logger::MessageType::Debug, "Number of workloads is:" + std::to_string(nWorkloads));

                    /*Partition tensor into workloads with METIS*/
                    auto res = workloads.partitionTensorWithMETIS(nWorkloads, pass);
                    if( res != 1 ) 
                        std::runtime_error("Error occured during tensor partitioning into workloads using METIS, ensure number of workloads is even!");
                    /*Populate each workload*/
                    workloads.populateWorkloadsFromPartitions(nWorkloads, pass);
                    solutions.push_back(workloads);
                }
                else if (algorithm == "Rectangle")
                {

                }
                else if (algorithm == "Z-Tiling")
                {

                }
            }
            /*Add workloads as Attribute*/
            opIt->set<mv::Workloads>("Workloads", workloads);
            /* Calculate a pool of possible workloads select the best one based on the cost functions */ 
            auto costFunction = workloads.getCostFunction(passDesc, pass);

            std::vector<float> exeCycle = workloads.getExecutionCycles(outputTensor, nDPUxCluster, costFunction);
            // TODO: process the execution cycles





        }
    }
    pass.log(mv::Logger::MessageType::Debug, "Exiting workload generation pass");
}
