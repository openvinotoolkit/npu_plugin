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

    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("Number_of_DPUs")) 
        int nDPU = globalParams->get<int>("Number_of_DPUs");
    if (globalParams->hasAttr("Number_of_Clusters")) 
        int nClusters = globalParams->get<int>("Number_of_Clusters");
    
    /*Get nWorkloads and mpe_mode from compilation descriptor*/
    int nWorkloads = globalParams->get<int>("nWorkloads");

    std::pair <int,int> MPEMode;
    std::string mpeMode  = globalParams->get<std::string>("MPE_mode");
    
    /*MPE mode*/
    if(mpeMode == "Matrix") { 
        MPEMode.first = 4;
        MPEMode.second = 4; 
    }
    else if (mpeMode == "Vector")
    {
        MPEMode.first = 1;
        MPEMode.second = 16; 
    }
    
    mv::OpModel om(model);
    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {
            pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));
 
            /*Get output tensor*/
            auto outputTensor = opIt->getOutputTensor();
            std::vector<mv::Workloads> solutions;

<<<<<<< HEAD
=======
            /*Workload's instance, name and tensorShape, MPE mode*/
            std::pair <idx_t,idx_t> MPEMode (4, 4); /*MPE mode*/
>>>>>>> VPUNND-1019_polygon_shape_metis
            mv::Workloads workloads(opIt->getName(),outputTensor[0]->getShape(), MPEMode);
            std::vector<std::string> algorithms = workloads.getTensorSplitAlgorithms(passDesc);

            pass.log(mv::Logger::MessageType::Debug, "Number of workloads is:" + std::to_string(nWorkloads));

            /*Partition tensor into workloads*/
            for (std::string algorithm : algorithms)
            {
                if (algorithm == "Metis")
                {
                    /*Populate Metis adjacency structure and partition tensor*/
                    workloads.generateMetisGraph(); 
                    auto res = workloads.partitionTensorWithMETIS(nWorkloads, pass);
                    if( res==1)
                        workloads.populateWorkloadsFromPartitions(nWorkloads, pass, MPEMode);
                    else
                        pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using METIS, ensure number of workloads is even!");
                }
                else if (algorithm == "Rectangle")
                {
                    /*Partition tensor into workloads with Rectangle*/
                    auto res = workloads.partitionTensorWithRectangleHeuristic(nWorkloads, pass);
                    if(res==1)
                        workloads.populateWorkloadsFromPartitions(nWorkloads, pass, MPEMode);
                    else
                        pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using Rectangle!");
                }
                else if (algorithm == "Z-Tiling")
                {
                    //Partition tensor into workloads with Rectangle

                }
                
            }
            
            opIt->set<mv::Workloads>("Workloads", workloads);
        }
    }
    pass.log(mv::Logger::MessageType::Debug, "Exiting workload generation pass");
}
