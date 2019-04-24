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

    /*Get the number of clusters that the VPU supports*/
    auto nceDefs = target.nceDefs();
    //auto nClusters = nceDefs.find("Clusters")->second.totalNumber;

    /*Get all tensors*/


    int nDPU = 4;                       /*Number of DPUs*/
    int nClusters = 5;                  /*Number of clusters*/
    int nDPUxCluster = nDPU/nClusters;  /*Number of DPUs per cluster*/
    std::set<int> workloadsList;
    std::pair <int,int> MPEMode (4, 4); /*MPE mode*/

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

            /*Add workloads as Attribute*/
            opIt->set<mv::Workloads>("Workloads", workloads);


            auto costFunction = workloads.getCostFunction(passDesc, pass);

            std::vector<float> exeCycle = workloads.getExecutionCycles(outputTensor, nDPUxCluster, costFunction);
            // TODO: process the execution cycles





        }
    }
    pass.log(mv::Logger::MessageType::Debug, "Exiting workload generation pass");
}
