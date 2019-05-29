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
    
    /* The METIS numbering convention for partitions starts at 0, for number of partitions > 1. i.e. nodes are assinged to partions labelled 0 -> (number partitions -1) 
     * However, the METIS numbering convention for partitions starts at 1, for number of partitions exactly = 1. i.e. nodes are assinged to partions labelled 1
     * Need to test for this condition here, as it impacts the idexing of logic that calculate the workload coordinates MinX, MaxX, MinY, MaxY
     * Here, if nWorkloads is 1, then we change the METIS numbering convention to start at 0 so that it is the same as the 'normal' convention.
     */
    if(nWorkloads == 1) 
        for (int i=0; i < metisGraph->m_numberTensorVertices; i++) 
            metisGraph->part[i] = 0; 

    return res;
}


void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object &)
{
    pass.log(mv::Logger::MessageType::Debug, "Starting workload generation pass");

    int nWorkloads;
    int nDPU;
    int nClusters;
    std::pair <int,int> MPEMode;
    std::string mpeMode;
    mv::OpModel om(model);
    
    /*Get nDPUs and nClsuters from gloabl compilation descriptor*/ 
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("Number_of_DPUs")) 
        nDPU = globalParams->get<int>("Number_of_DPUs");
    if (globalParams->hasAttr("Number_of_Clusters")) 
        nClusters = globalParams->get<int>("Number_of_Clusters");

    /*DPUs per cluster*/    
    int nDPUxCluster = nDPU / nClusters;
    
    /*The global mpe mode must be set*/
    if (globalParams->hasAttr("MPE_mode")) 	
        mpeMode  = globalParams->get<std::string>("MPE_mode");	
    else	
        std::runtime_error("Exiting, set the MPE mode in the compilation descriptor");	

     /*MPE mode*/	
    if(mpeMode == "Matrix")
    {
        MPEMode.first = 4;	
        MPEMode.second = 4; 
        pass.log(mv::Logger::MessageType::Debug, "MPE mode is Matrix");
	
    }	
    else if (mpeMode == "Vector")	
    {	
        MPEMode.first = 1;	
        MPEMode.second = 16; 	
        pass.log(mv::Logger::MessageType::Debug, "MPE mode is Vector");
    }	

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {
            pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));
                 
            /*Create workload instance*/
            mv::Workloads workloads(opIt->getName(), opIt->getOutputTensor()[0]->getShape(), MPEMode);

            /*Get the worklaods algorithm*/
            std::vector<std::string> algorithms = workloads.getTensorSplitAlgorithms(passDesc);
            
            /*Generate the split pool and select the first one*/
            auto nWorkloadsSplitPool = workloads.getWorkloadSplitPool(opIt->getOutputTensor()[0], nDPUxCluster);
            nWorkloads = nWorkloadsSplitPool[0];
            nWorkloads = 1;

            pass.log(mv::Logger::MessageType::Debug, "Number of workloads is:" + std::to_string(nWorkloads));
            pass.log(mv::Logger::MessageType::Debug, "Output size is: " + opIt->getOutputTensor()[0]->getShape().toString());

            /*Partition tensor into workloads*/
            for (std::string algorithm : algorithms)
            {
                if (algorithm == "Metis")
                {
                    /*Populate Metis adjacency structure and partition tensor*/
                    workloads.generateMetisGraph();
                    auto metisGraph = workloads.getMetisGraph();
                    auto res = partitionTensorWithMETIS(metisGraph, nWorkloads, pass);

                    /*Store METIS optimization value as attrbute for unit testing*/
                    opIt->set<int>("Metis_edge_cut", metisGraph->objval);

                    if (res == 1)
                        workloads.populateWorkloadsFromPartitions(nWorkloads, pass, MPEMode);
                    else
                        pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using METIS");

                    if(!workloads.validateWorkloads(opIt->getOutputTensor()[0]->getShape())) {

                        pass.log(mv::Logger::MessageType::Warning, "A volume mismatch error occcured during METIS workload validation \
                                                                    METIS likely retured a U-Shaped partiton, which we do not currently support \
                                                                    Retrying to generate workloads using METIS with a different number of workloads \
                                                                    This is probably not an optimal number of workloads");
                        
                        /*Create new workload instance*/
                        mv::Workloads workloads(opIt->getName(), opIt->getOutputTensor()[0]->getShape(), MPEMode);
                        
                        /*Take the next workload number in the pool*/
                        nWorkloads = nWorkloadsSplitPool[1]; 

                        /*Populate Metis adjacency structure and partition tensor*/
                        workloads.generateMetisGraph();
                        auto metisGraph = workloads.getMetisGraph();
                        auto res = partitionTensorWithMETIS(metisGraph, nWorkloads, pass);

                        /*Store METIS optimization value as attrbute for unit testing*/
                        opIt->set<int>("Metis_edge_cut", metisGraph->objval);

                        if (res == 1)
                            workloads.populateWorkloadsFromPartitions(nWorkloads, pass, MPEMode);
                        else
                            pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using METIS");

                        if(!workloads.validateWorkloads(opIt->getOutputTensor()[0]->getShape())) 
                            throw std::runtime_error("Invalid workloads have been generated from the METIS partition after two attempts, \ 
                                                        this is proboably due to U-shaped partions, \
                                                        in future if this happens we will switch to using Rectangle Heurisitc algorithm");
                    }

                    /*Set valid workload attribute to true*/
                    opIt->set<bool>("Valid_workload", true);
                }
                else if (algorithm == "Rectangle")
                {
                    /*Partition tensor into workloads with Rectangle*/
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
