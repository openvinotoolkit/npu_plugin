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

/* Note making the call to the METIS library a non-member function of the workloads class.
 * 
 * This facilitates linking METIS to the pass CMAKE group and not to the base CMAKE group.
 * 
 * The current plan is to port METIS to MCMcompiler in the future. For that reason other METIS related utility fucntions are members of the workloads class,
 * such as (1) generateMetisGraphNodeNumbers() and (2) generateMetisGraph(), which generate the adjacency structure of the graph, a set of arrays, 
 * which represents a tensor to be partitioned. These will be required when we port METIS to to MCMcompiler.   
 * 
*/ 
int partitionTensorWithMETIS(const std::shared_ptr<mv::MetisGraphStructure>& metisGraph, idx_t nWorkloads, const mv::pass::PassEntry& pass) 
{
    METIS_SetDefaultOptions(metisGraph->options);

    pass.log(mv::Logger::MessageType::Debug, "The adjancy data for METIS is ");
    for(int i =0; i < 2*metisGraph->m_numberTensorEdges; i++) 
        pass.log(mv::Logger::MessageType::Debug, std::to_string(metisGraph->adjncy[i]));
    
    pass.log(mv::Logger::MessageType::Debug, "The xadj data for METIS is ");
    for(int i =0; i < (metisGraph->m_numberTensorVertices + 1); i++) 
        pass.log(mv::Logger::MessageType::Debug, std::to_string(metisGraph->xadj[i]));
    
    pass.log(mv::Logger::MessageType::Debug, "The vwgt data for METIS is ");
    for(int i =0; i < (metisGraph->m_numberTensorVertices); i++) 
        pass.log(mv::Logger::MessageType::Debug, std::to_string(metisGraph->vwgt[i]));
    
    /*METIS call*/
    int res = METIS_PartGraphRecursive(&metisGraph->m_numberTensorVertices,&metisGraph->nWeights, metisGraph->xadj.get(), metisGraph->adjncy.get(),
                    metisGraph->vwgt.get(), NULL, NULL, &nWorkloads, NULL,
				    NULL, metisGraph->options, &metisGraph->objval, metisGraph->part.get());

    pass.log(mv::Logger::MessageType::Debug, "Value of the objective function that was minimized by METIS (should be same as PoC compiler) is: " + std::to_string(metisGraph->objval));

    /*Print node partition*/
    for(int part_i = 0; part_i < metisGraph->m_numberTensorVertices; part_i++) 
            pass.log(mv::Logger::MessageType::Debug, "Node " + std::to_string(part_i) + " is in partition " + std::to_string(metisGraph->part[part_i]));  
    
    return res;
}


void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object &)
{

    pass.log(mv::Logger::MessageType::Debug, "Starting workload generation pass");

    //Get number of Clusters and DPU's
    int nDPU = 20;                      //Default number of DPUs
    int nClusters = 4;                  //Default number of Clusters
    static const mv::DPUModeList dpu_mode_poc = {{4, 4}, {16, 1}};

    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("Number_of_DPUs")) 
        int nDPU = globalParams->get<int>("Number_of_DPUs");
    if (globalParams->hasAttr("Number_of_Clusters")) 
        int nClusters = globalParams->get<int>("Number_of_Clusters");
    
    int nWorkloads;
    std::pair <int,int> MPEMode;
    std::string mpeMode;
    mv::OpModel om(model);

    /*The global mpe mode and number of workloads must be set for unit tests that don't have layer names matching resnet50 layer names*/
    /*Get nWorkloads and mpe_mode from compilation descriptor*/	
    if (globalParams->hasAttr("nWorkloads")) 	
        nWorkloads = globalParams->get<int>("nWorkloads");	
    else	
        std::runtime_error("Exiting, set the number of workloads and MPE mode in the compilation descriptor");	

     if (globalParams->hasAttr("MPE_mode")) 	
        mpeMode  = globalParams->get<std::string>("MPE_mode");	
    else	
        std::runtime_error("Exiting, set the MPE mode in the compilation descriptor");	

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

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {
            if (opIt->hasAttr("WorkloadStrategy_MPE_mode"))
            {
                pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));
                MPEMode.first = std::stoi(std::string(1, opIt->get<std::string>("WorkloadStrategy_MPE_mode")[1]));
                MPEMode.second = std::stoi(std::string(1, opIt->get<std::string>("WorkloadStrategy_MPE_mode")[3]));
                nWorkloads = opIt->get<int>("WorkloadStrategy_nWorkloads");
            }

            /*Create workload*/
            mv::Workloads workloads(opIt->getName(), opIt->getOutputTensor()[0]->getShape(), MPEMode);

            std::vector<std::string> algorithms = workloads.getTensorSplitAlgorithms(passDesc);

            pass.log(mv::Logger::MessageType::Debug, "Number of workloads is:" + std::to_string(nWorkloads));

            /*Partition tensor into workloads*/
            for (std::string algorithm : algorithms)
            {
                if (algorithm == "Metis")
                {
                    /*Populate Metis adjacency structure and partition tensor*/
                    workloads.generateMetisGraph();
                    auto metisGraph = workloads.getMetisGraph();
                    auto res = partitionTensorWithMETIS(metisGraph, nWorkloads, pass);
                    
                    if (res == 1)
                        workloads.populateWorkloadsFromPartitions(nWorkloads, pass, MPEMode);
                    else
                        pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using METIS");

                    if(!workloads.validateWorkloads(opIt->getOutputTensor()[0]->getShape()))
                        std::runtime_error("Invalid workloads have been generated, the individual workloads do not sum the output tensor size");

                }
                else if (algorithm == "Rectangle")
                {
                    // /*Partition tensor into workloads with Rectangle*/
                    // auto res = workloads.partitionTensorWithRectangleHeuristic(dpu_mode_poc, nWorkloads, pass);
                    // if (res == 1)
                    //     workloads.populateWorkloadsFromPartitions(nWorkloads, pass, MPEMode);
                    // else
                    //     pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using Rectangle!");
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
