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

/** 
 * @brief Returns the supported Tensor Split Algorithms to be used
 */
std::vector<std::string> getTensorSplitAlgorithms(mv::Element& passDesc, const mv::pass::PassEntry& pass)
{
    /*parse TensorSplitAlgorithms from Compilation Descriptor*/
    std::vector<std::string> algorithms = {"Metis", "Rectangle", "Z-Tiling"}; //default
    if (passDesc.hasAttr("TensorSplitAlgorithms")) 
    {
        algorithms.clear();
        std::string sAlgorithms = passDesc.get<std::string>("TensorSplitAlgorithms");
        std::stringstream ss(sAlgorithms);
        while( ss.good() )
        {
            std::string tempStr;
            std::getline(ss, tempStr, ',');
            if (tempStr=="Metis" || tempStr=="Rectangle" || tempStr=="Z-Tiling")
                algorithms.push_back(tempStr);
            else
                pass.log(mv::Logger::MessageType::Warning, "Could not parse the TensorSplitAlgorithms type (only \"Metis, Rectangle, Z-Tiling\" currently supported).");
        }
    }
    else 
        pass.log(mv::Logger::MessageType::Info, "No TensorSplitAlgorithms specified in descriptor, using  \"Metis, Rectangle, Z-Tiling\"...");
    
    //if parsing problem, return all 3
    if (algorithms.size() == 0)
        algorithms = {"Metis", "Rectangle", "Z-Tiling"};
    return algorithms;
}

void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object &)
{
    pass.log(mv::Logger::MessageType::Debug, "Starting workload generation pass");

    int nWorkloads;
    int nDPU;
    int nClusters;
    bool metisValidWorkload = false;
    int attempsforValidateWorkloads = 0;
    std::pair <int,int> MPEMode;
    std::string mpeMode;
    std::vector<mv::Workloads> workloadsVector;
    int workloadsVectorIndex = 0;
    int metisResult;
    std::shared_ptr<mv::MetisGraphStructure> metisGraph;

    mv::OpModel om(model);
    
    /*Get nDPUs and nClsuters from gloabl compilation descriptor*/ 
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("Number_of_DPUs")) 
        nDPU = globalParams->get<int>("Number_of_DPUs");
    if (globalParams->hasAttr("Number_of_Clusters")) 
        nClusters = globalParams->get<int>("Number_of_Clusters");

    /*DPUs per cluster*/    
    int nDPUxCluster = nDPU / nClusters;
    
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

     /*Set MPE mode from global*/	
    if(mpeMode == "Matrix")
    {
        MPEMode.first = 4;	
        MPEMode.second = 4; 
        pass.log(mv::Logger::MessageType::Debug, "Global MPE mode is Matrix");
	
    }	
    else if (mpeMode == "Vector")	
    {	
        MPEMode.first = 1;	
        MPEMode.second = 16; 	
        pass.log(mv::Logger::MessageType::Debug, "Global MPE mode is Vector");
    }	
    pass.log(mv::Logger::MessageType::Debug, "The global number of workloads is: " + std::to_string(nWorkloads));

    /*Get the worklaods algorithm*/
    std::vector<std::string> algorithms = getTensorSplitAlgorithms(passDesc, pass);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {
            pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));

            /*Partition tensor into workloads*/
            for (std::string algorithm : algorithms)
            {
                if (algorithm == "Metis")
                {
                    if(opIt->hasAttr("WorkloadStrategy_MPE_mode")) {
                        if(opIt->get<std::string>("WorkloadStrategy_MPE_mode") == "Matrix") {
                            MPEMode.first = 4;	
                            MPEMode.second = 4;
                            pass.log(mv::Logger::MessageType::Debug, "This layer has a workload strategy, using MPE mode Matrix"); 	
                        }
                        if(opIt->get<std::string>("WorkloadStrategy_MPE_mode") == "Vector") {
                        MPEMode.first = 1;	
                        MPEMode.second = 16;
                        pass.log(mv::Logger::MessageType::Debug, "This layer has a workload srategy, using MPE mode Vector"); 	
                        }
                    }

                    pass.log(mv::Logger::MessageType::Debug, "Output size is: " + opIt->getOutputTensor()[0]->getShape().toString());

                    /*Create workload instance*/
                    workloadsVector.push_back(mv::Workloads(opIt->getName(), opIt->getOutputTensor()[0]->getShape(), MPEMode));

                    if(opIt->hasAttr("WorkloadStrategy_nWorkloads")) {
                        nWorkloads = opIt->get<int>("WorkloadStrategy_nWorkloads");
                        pass.log(mv::Logger::MessageType::Debug, "This layer has number of workloads in the workload strategy using : "+ std::to_string(nWorkloads));
                    }
                    else {
                        /*Generate the split pool and select the first one*/
                        auto nWorkloadsSplitPool = workloadsVector.at(workloadsVectorIndex).getWorkloadSplitPool(opIt->getOutputTensor()[0], nDPUxCluster);
                        nWorkloads = nWorkloadsSplitPool[0];
                        pass.log(mv::Logger::MessageType::Debug, "This layer does not have number of workloads in the workload strategy using from the split pool is: " + std::to_string(nWorkloads));
                    }

                    /*Populate Metis adjacency structure and partition tensor*/
                    workloadsVector.at(workloadsVectorIndex).generateMetisGraph();
                    metisGraph = workloadsVector.at(workloadsVectorIndex).getMetisGraph();
                    metisResult = partitionTensorWithMETIS(metisGraph, nWorkloads, pass);

                    /*Store METIS optimization value as attrbute for unit testing*/
                    opIt->set<int>("Metis_edge_cut", metisGraph->objval);

                    if (metisResult == 1)
                        workloadsVector.at(workloadsVectorIndex).populateWorkloadsFromPartitions(nWorkloads, pass, MPEMode);
                    else
                        pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using METIS");

                    if(!workloadsVector.at(workloadsVectorIndex).validateWorkloads(opIt->getOutputTensor()[0]->getShape())) {

                        pass.log(mv::Logger::MessageType::Warning,"A volume mismatch error occcured during METIS workload validation.");
                        pass.log(mv::Logger::MessageType::Warning,"METIS likely retured a U-Shaped partiton, which we do not currently support.");
                        pass.log(mv::Logger::MessageType::Warning,"Retrying to generate workloads using METIS, increasing the number of workloads by 1.");
                        pass.log(mv::Logger::MessageType::Warning,"This is probably not an optimal number of workloads.");
                        pass.log(mv::Logger::MessageType::Warning,"In future, if this happens we will switch to using Rectangle Heuristic algorithm.");
                        
                        while((!metisValidWorkload) &&(attempsforValidateWorkloads < 5)) {
                        
                            /*Create a new workload instance*/
                            workloadsVector.push_back(mv::Workloads(opIt->getName(), opIt->getOutputTensor()[0]->getShape(), MPEMode));
                            workloadsVectorIndex++;
                        
                            /*Increase nWorkloads by 1*/
                            nWorkloads++;
                            attempsforValidateWorkloads++; 

                            /*Populate Metis adjacency structure and partition tensor*/
                            workloadsVector.at(workloadsVectorIndex).generateMetisGraph();
                            metisGraph = workloadsVector.at(workloadsVectorIndex).getMetisGraph();
                            metisResult = partitionTensorWithMETIS(metisGraph, nWorkloads, pass);

                            /*Store METIS optimization value as attrbute for unit testing*/
                            opIt->set<int>("Metis_edge_cut", metisGraph->objval);

                            if (metisResult == 1)
                                workloadsVector.at(workloadsVectorIndex).populateWorkloadsFromPartitions(nWorkloads, pass, MPEMode);
                            else
                                pass.log(mv::Logger::MessageType::Warning, "Error partitioning tensor into workloads using METIS");

                            if(workloadsVector.at(workloadsVectorIndex).validateWorkloads(opIt->getOutputTensor()[0]->getShape())) 
                                metisValidWorkload = true;
                        }

                        if(!workloadsVector.at(workloadsVectorIndex).validateWorkloads(opIt->getOutputTensor()[0]->getShape())) 
                            throw std::runtime_error("Invalid workloads have been generated from METIS partitions after five attempts, exiting.");
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

            opIt->set<mv::Workloads>("Workloads", workloadsVector.at(workloadsVectorIndex));
        }
    }
    pass.log(mv::Logger::MessageType::Debug, "Exiting workload generation pass");
}
