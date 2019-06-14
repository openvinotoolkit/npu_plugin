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

int countNumberOfWorkloadsMetisReturned(int arr[], int n) 
{ 
    /*First sort the array so that all occurrences become consecutive*/
    std::sort(arr, arr + n); 
  
    /*Traverse the sorted array*/ 
    int res = 0; 
    for (int i = 0; i < n; i++) { 
  
        /*Move the index ahead while there are duplicates */
        while (i < n - 1 && arr[i] == arr[i + 1]) 
            i++; 
  
        res++; 
    } 
  
    return res; 
} 


std::pair<int,int> partitionTensorWithMETIS(const std::shared_ptr<mv::MetisGraphStructure>& metisGraph, idx_t nWorkloads, const mv::pass::PassEntry& pass) 
{
     
    /*METIS call*/
    METIS_SetDefaultOptions(metisGraph->options);
    int res = METIS_PartGraphRecursive(&metisGraph->m_numberTensorVertices,&metisGraph->nWeights, metisGraph->xadj.get(), metisGraph->adjncy.get(),
                    metisGraph->vwgt.get(), NULL, NULL, &nWorkloads, NULL,
				    NULL, metisGraph->options, &metisGraph->objval, metisGraph->part.get());

    pass.log(mv::Logger::MessageType::Debug, "Value of the objective function that was minimized by METIS (should be same as PoC compiler) is: " + std::to_string(metisGraph->objval));
   
    /* The METIS numbering convention for partitions starts at 0, for number of partitions > 1. i.e. nodes are assinged to partions labelled 0 -> (number partitions -1) 
     * However, the METIS numbering convention for partitions starts at 1, for number of partitions exactly = 1. i.e. nodes are assinged to partions labelled 1
     * Need to test for this condition here, as it impacts the idexing of logic that calculate the workload coordinates MinX, MaxX, MinY, MaxY
     * Here, if nWorkloads is 1, then we change the METIS numbering convention to start at 0 so that it is the same as the 'normal' convention.
     */
    if(nWorkloads == 1) 
        for (int i=0; i < metisGraph->m_numberTensorVertices; i++) 
            metisGraph->part[i] = 0; 
    
     /*Check if METIS returned fewer than the requested number of partitions - this is an error condition*/
    int numberOfMetisPartitions = countNumberOfWorkloadsMetisReturned(metisGraph->part.get(), metisGraph->m_numberTensorVertices);

    return {res, numberOfMetisPartitions};
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

std::pair<int,int> getGlobalCompilationDescriptorConf(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {

    int nDPU = 1;
    int nClusters = 1;
    int nWorkloads = 0;
   
    /*Get nDPUs and nClsuters from gloabl compilation descriptor*/ 
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("Number_of_DPUs")) 
        nDPU = globalParams->get<int>("Number_of_DPUs");
    if (globalParams->hasAttr("Number_of_Clusters")) 
        nClusters = globalParams->get<int>("Number_of_Clusters");
    if (globalParams->hasAttr("nWorkloads")) 
        nWorkloads= globalParams->get<int>("nWorkloads");
    
    /*DPUs per cluster*/  
    auto nDPUxCluster =  nDPU / nClusters;

    pass.log(mv::Logger::MessageType::Debug, "Number of DPUs per cluster is: " + std::to_string(nDPUxCluster));

    return {nDPUxCluster, nWorkloads};
}

void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object &)
{
    pass.log(mv::Logger::MessageType::Debug, "Starting workload generation pass");
    mv::OpModel om(model);

    mv::DPUModeList dpuMode; /*One DPU mode*/
    mv::DPUModeLists dpuModeLists; /*Both DPU modes*/
    
    std::shared_ptr<mv::MetisGraphStructure> metisGraph;
    std::vector<mv::Workloads> workloadsVector;

    int workloadsVectorIndex = 0;
    int optimalWorkloadIndex = 0;
    bool metisFail = false;
    bool rectangleFail = false;
    std::pair<int,int> metisResult = {0,0};
    
    /*Get the worklaods algorithm*/
    std::vector<std::string> algorithms = getTensorSplitAlgorithms(passDesc, pass);

    /*get cost function*/
    auto costFuntion = mv::Workloads::getCostFunction(passDesc, pass);
    
    /*get number of DPUs per cluster*/
    auto compilationConfigs = getGlobalCompilationDescriptorConf(pass, model);
    auto nDPUxCluster = compilationConfigs.first;
    auto nWorkloadsCompilationDescriptor = compilationConfigs.second;

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt) {

        if (opIt->getOpType() == "DPUTask") {

            pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));

            pass.log(mv::Logger::MessageType::Debug, "Output tensor size is: " + opIt->getOutputTensor()[0]->getShape().toString());
        
            /*For Deptwise convolution, Max pooling and CM convolution MPE mode must be (1,16)*/
            if((opIt->get<std::string>("taskOp") == "DepthwiseConv") || (opIt->get<std::string>("taskOp") == "MaxPool") || (opIt->get<std::string>("taskOp") == "ChannelMajorConvolution")) {
                dpuModeLists = {{{1, 16}}};
            }
            else
                dpuModeLists = {{{4,4},{1, 16}}};

            /*Generate the number of workloads split pool*/
            auto nWorkloadsSplitPool = mv::Workloads::getWorkloadSplitPool(opIt->getOutputTensor()[0], nDPUxCluster, dpuModeLists, 50);


            /* For each dpu mode and for each workload, attempt to partition with METIS and/or Rectangle and create worklaods*/ 
            for (auto dpuMode : dpuModeLists) {

                for(auto nWorkloads : nWorkloadsSplitPool) {

                    pass.log(mv::Logger::MessageType::Debug, "The number of workloads is: " + std::to_string(nWorkloads));
       
                    /*Fore each of the lagorithms specified*/
                    for (std::string algorithm : algorithms) {

                        if (algorithm == "Metis") {
                            
                            /*Create workload instance*/
                            workloadsVector.emplace_back(mv::Workloads(opIt->getName(), opIt->getOutputTensor()[0]->getShape(), dpuMode[0]));

                            /*Populate Metis adjacency structure and partition tensor*/
                            workloadsVector.at(workloadsVectorIndex).generateMetisGraph();
                            metisGraph = workloadsVector.at(workloadsVectorIndex).getMetisGraph();

                            if(nWorkloadsCompilationDescriptor)
                                metisResult = partitionTensorWithMETIS(metisGraph, nWorkloadsCompilationDescriptor, pass);
                            else
                                metisResult = partitionTensorWithMETIS(metisGraph, nWorkloads, pass);

                            /*Store METIS optimization value as attrbute for unit testing and for comparison with POC compiler*/
                            opIt->set<int>("Metis_edge_cut", metisGraph->objval);

                            /*Check that Metis returned the exact number of partitions requested. Sometimes it will return less*/
                            if ((metisResult.first == 1) && (metisResult.second == nWorkloads))
                                workloadsVector.at(workloadsVectorIndex).populateWorkloadsFromPartitions(nWorkloads, pass, dpuMode[0]);
                            else {
                                
                                pass.log(mv::Logger::MessageType::Debug,"Error partitioning tensor into workloads using METIS");
                                
                                /*Remove the original workload created with metis*/
                                workloadsVector.erase(workloadsVector.begin() + workloadsVectorIndex);
                                metisFail = true;
                                
                                /*Now try with Rectangle Heuristic*/
                                goto Rectangle;
                            }

                            if(!workloadsVector.at(workloadsVectorIndex).validateWorkloads(opIt->getOutputTensor()[0]->getShape())) {
                                
                                pass.log(mv::Logger::MessageType::Debug, "Unable to produce valid workloads using METIS for this layer switching to Rectangle Heuristic");

                                /*Remove the original workload created with metis*/
                                workloadsVector.erase(workloadsVector.begin() + workloadsVectorIndex);
                                metisFail = true;

                                /*Now try with Rectangle Heuristic*/
                                goto Rectangle;
                            }
                            else {
                                workloadsVectorIndex++;
                                metisFail = false;
                            }
                        }
                
                        if (algorithm == "Rectangle") {
                        
                            /*Go here if METIS fails*/
                            Rectangle:

                            /*Create workload instance*/
                            workloadsVector.emplace_back(mv::Workloads(opIt->getName(), opIt->getOutputTensor()[0]->getShape()));

                            /*Partition tensor into workloads with Rectangle*/
                            rectangleFail = false;
                            bool split_over_h = true;
                            bool split_over_w = true;
                            bool split_symmetric = false;
                            int rectangleResult = 0;

                            if(nWorkloadsCompilationDescriptor)
                                rectangleResult = workloadsVector.at(workloadsVectorIndex).partitionTensorWithRectangleHeuristic(dpuMode, nWorkloadsCompilationDescriptor,
                                                            split_over_h, split_over_w, split_symmetric,
                                                            mv::WorkloadSplitMode::HW, pass);
                            else
                                rectangleResult = workloadsVector.at(workloadsVectorIndex).partitionTensorWithRectangleHeuristic(dpuMode, nWorkloads,
                                                            split_over_h, split_over_w, split_symmetric,
                                                            mv::WorkloadSplitMode::HW, pass);

                            if (rectangleResult != 1) {
                                pass.log(mv::Logger::MessageType::Debug, "Error using Rectangle to tile the output tensor, erasing this workload instance");
                                workloadsVector.erase(workloadsVector.begin() + (workloadsVectorIndex));
                                rectangleFail = true;
                            }

                            if(!rectangleFail) {

                                if((!workloadsVector.at(workloadsVectorIndex).validateWorkloads(opIt->getOutputTensor()[0]->getShape()))) {
                                    pass.log(mv::Logger::MessageType::Debug, "Error producing valid workloads from Rectangle partitions,erasing this workload instance ");
                                    workloadsVector.erase(workloadsVector.begin() + workloadsVectorIndex);
                                    rectangleFail = true;
                                }

                                if(!rectangleFail) {
                                    rectangleFail = false;
                                    pass.log(mv::Logger::MessageType::Debug, "Valid workload created using Rectangle");
                                    workloadsVectorIndex++;
                                }
                            }               
                    }
                
                    else if (algorithm == "Z-Tiling")
                    {
                    //Partition tensor into workloads with Z-tiling
                    }

                   
                }

            }
        }

            /*Calculate execution cycles for each valid workload*/
            mv::Workloads::generateExecutionCycles(workloadsVector, nDPUxCluster, costFuntion);

            /*Print the execution cycles*/
            int index = 0;
            for (auto wl : workloadsVector) {
                pass.log(mv::Logger::MessageType::Debug, "Index " + std::to_string(index) + " (Min) " + std::to_string(wl.getExecutionCycles()[0]) + " Cost (Max)" + std::to_string(wl.getExecutionCycles()[1]));
                index++;
            }

            /*Pick the workload with minimum (should be mean) execution time*/
            auto optimalWorkload = std::min_element(workloadsVector.begin(), workloadsVector.end(),
                [] (mv::Workloads const& lhs, mv::Workloads const& rhs) 
                {
                    return lhs.getMeanExecutionCycles() < rhs.getMeanExecutionCycles();
                });
        
            /*Get the index of the most optimial workload*/
            optimalWorkloadIndex = std::distance(workloadsVector.begin(), optimalWorkload);

            pass.log(mv::Logger::MessageType::Debug, "Selecting workload at index " + std::to_string(optimalWorkloadIndex) + " as the most optimal one");

            /*Set the most optimal workload as attribute of the op*/
            opIt->set<mv::Workloads>("Workloads", workloadsVector.at(optimalWorkloadIndex));

            /*Reset workloads vector and indices for the next layer*/
            workloadsVector.clear();
            workloadsVectorIndex = 0;
            optimalWorkloadIndex = 0;
    
        }
    }
    pass.log(mv::Logger::MessageType::Debug, "Exiting workload generation pass");
}  

