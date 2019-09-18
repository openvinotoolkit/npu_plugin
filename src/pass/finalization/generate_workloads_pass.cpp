#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/target/keembay/rectangle.hpp"
#include "include/mcm/target/keembay/workload_struct.hpp"
#include "include/mcm/graph/graph.hpp"
#include <algorithm>
#include <climits>
#include <math.h>
#include <metis.h>



static void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

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

float workloadPixelCost(mv::Data::OpListIterator opIt)
{
 auto kh = opIt->get<std::array<unsigned short, 2>>("kSize")[0];
 auto kw = opIt->get<std::array<unsigned short, 2>>("kSize")[1];
 auto ic = opIt->get<std::string>("taskOp") == "ChannelMajorConvolution" || opIt->get<std::string>("taskOp") == "DepthwiseConv" || opIt->get<std::string>("taskOp") == "Conv" ? opIt->getInputTensor()[0]->getShape()[2] : 1;

 return kh*kw*ic;
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
        pass.log(mv::Logger::MessageType::Debug, "No TensorSplitAlgorithms specified in descriptor, using  \"Metis, Rectangle, Z-Tiling\"...");

    //if parsing problem, return all 3
    if (algorithms.size() == 0)
        algorithms = {"Metis", "Rectangle", "Z-Tiling"};
    return algorithms;
}

std::tuple<int,int, int> getGlobalCompilationDescriptorConf(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {

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

    if(nDPU < nClusters)
        throw std::runtime_error("The number of DPUs cannot be less than the number of clusters!, exiting");

    /*DPUs per cluster*/
    int nDPUxCluster =  ceil(nDPU / nClusters);

    pass.log(mv::Logger::MessageType::Debug, "Number of DPUs per cluster is: " + std::to_string(nDPUxCluster));

    return std::make_tuple(nDPUxCluster, nWorkloads, nClusters);
}

void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc, mv::Element&)
{
    pass.log(mv::Logger::MessageType::Debug, "Starting workload generation pass");
    mv::OpModel om(model);

    mv::DPUModeList dpuModes;
    std::shared_ptr<mv::MetisGraphStructure> metisGraph;
    std::vector<mv::Workloads> workloadsVector;
    std::vector<int> nWorkloadsSplitPool;

    int workloadsVectorIndex = 0;
    int optimalWorkloadIndex = 0;
    bool rectangleFail = false;
    uint8_t clusterNumber = 0;
    bool depthWiseSOHA0Workaround = false;

    /*Get the worklaods algorithm*/
    std::vector<std::string> algorithms = getTensorSplitAlgorithms(passDesc, pass);

    /*get cost function*/
    auto costFuntion = mv::Workloads::getCostFunction(passDesc, pass);

    /*get number of DPUs per cluster*/
    auto compilationConfigs = getGlobalCompilationDescriptorConf(pass, model);
    auto nDPUxCluster = std::get<0>(compilationConfigs);
    auto nWorkloadsCompilationDescriptor = std::get<1>(compilationConfigs);
    auto nClusters = std::get<2>(compilationConfigs);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "DPUTask")
        {
            pass.log(mv::Logger::MessageType::Debug, "Found DPU task " + opIt->getName() + " of type " + opIt->get<std::string>("taskOp"));

            depthWiseSOHA0Workaround = false;
            /*Get number of clusters*/
            auto opStrategy = opIt->get<std::string>("splitStrategy");
            if(opStrategy == "Clustering")
                nClusters = 1;
            else
                nClusters = std::get<2>(compilationConfigs);

            /* For Deptwise convolution, Max pooling and CM convolution MPE mode must be (1,16)*/
            /* This should be moved to a target descriptor*/

            if((opIt->get<std::string>("taskOp") == "DepthwiseConv") || (opIt->get<std::string>("taskOp") == "MaxPool") || (opIt->get<std::string>("taskOp") == "ChannelMajorConvolution"))
                dpuModes = {{1, 16}};
            else
                dpuModes = {{4,4},{1, 16}};

            if (opIt->getOutputTensor()[0]->getDType() == mv::DType("Float16"))
                dpuModes = {{1,4}};

            /*Depthwise cov SOH A0 workaround*/
            if((opIt->get<std::string>("taskOp") == "DepthwiseConv") && (opIt->get<std::string>("splitStrategy") == "SplitOverH")) {
                depthWiseSOHA0Workaround = true;
                opIt->set<std::string>("Depthwise_SOH_A0_bug", "True");
            }

            /*For multi-clustering we work on subtensors*/
            for(clusterNumber = 0; clusterNumber < nClusters; clusterNumber++)
            {
                /*get the subtensor*/
                auto subTensor = opIt->getOutputTensor()[0]->getSubTensor(clusterNumber);

                /*Sparse tensors don't use z-tiling*/
                /* This should be moved to a target descriptor*/
                if(subTensor.isSparse())
                    algorithms = {"Rectangle"};
                else
                    algorithms = {"Rectangle", "Z-Tiling"};

                pass.log(mv::Logger::MessageType::Debug, "The shape of subtensor for cluster " + std::to_string(clusterNumber) + "is: " + subTensor.getShape().toString());

                /*Generate the number of workloads split pool*/
                if(nWorkloadsCompilationDescriptor)
                    nWorkloadsSplitPool.push_back(nWorkloadsCompilationDescriptor);
                else
                    nWorkloadsSplitPool = mv::Workloads::getWorkloadSplitPool(subTensor, nDPUxCluster, dpuModes, 50);

                /*if Deptwise operation and SOH trategy, for A0 bug then add these number of worklaods to workload split pool*/
                if((opIt->get<std::string>("taskOp") == "DepthwiseConv") && (!nWorkloadsCompilationDescriptor))
                {
                    int deptwiseSOHworkloadNumbers[5] = {2, 4, 6, 8, 10};
                    nWorkloadsSplitPool.insert(nWorkloadsSplitPool.end(), deptwiseSOHworkloadNumbers, deptwiseSOHworkloadNumbers+5);

                    /*Erase duplicate workload numbers from the split pool*/
                    nWorkloadsSplitPool.erase(std::unique(nWorkloadsSplitPool.begin(), nWorkloadsSplitPool.end()), nWorkloadsSplitPool.end());
                    std::sort(nWorkloadsSplitPool.begin(), nWorkloadsSplitPool.end());
                }

                for(auto nWorkloads : nWorkloadsSplitPool)
                {
                    pass.log(mv::Logger::MessageType::Debug, "The number of workloads is: " + std::to_string(nWorkloads));

                    /*For each of the algorithms specified*/
                    for (std::string algorithm : algorithms)
                    {
                        /*Disabled METIS due to licence*/

                        // if (algorithm == "Metis")
                        // {
                        //     /* For each dpu mode and for each workload, attempt to partition with METIS and/or Rectangle and create worklaods*/
                        //     for (auto dpuMode : dpuModes)
                        //     {
                        //         /*Create workload instance*/
                        //         workloadsVector.emplace_back(mv::Workloads(opIt->getName(), opIt->getOutputTensor()[0]->getShape(), dpuMode));

                        //         /*Populate Metis adjacency structure and partition tensor*/
                        //         workloadsVector.at(workloadsVectorIndex).generateMetisGraph();
                        //         metisGraph = workloadsVector.at(workloadsVectorIndex).getMetisGraph();

                        //         metisResult = partitionTensorWithMETIS(metisGraph, nWorkloads, pass);

                        //         /*Store METIS optimization value as attrbute for unit testing and for comparison with POC compiler*/
                        //         opIt->set<int>("Metis_edge_cut", metisGraph->objval);

                        //         /*Check that Metis returned the exact number of partitions requested. Sometimes it will return less*/
                        //         if ((metisResult.first == 1) && (metisResult.second == nWorkloads))
                        //         {
                        //             workloadsVector.at(workloadsVectorIndex).populateWorkloadsFromPartitions(nWorkloads, pass, dpuMode);

                        //             if(!workloadsVector.at(workloadsVectorIndex).validateWorkloads(opIt->getOutputTensor()[0]->getShape()))
                        //             {
                        //                 pass.log(mv::Logger::MessageType::Debug, "Unable to produce valid workloads using METIS for this layer switching to Rectangle Heuristic");

                        //                 /*Remove the original workload created with metis*/
                        //                 workloadsVector.erase(workloadsVector.begin() + workloadsVectorIndex);
                        //                 metisFail = true;
                        //             }
                        //             else
                        //             {
                        //                 workloadsVectorIndex++;
                        //                 metisFail = false;
                        //             }
                        //         }
                        //         else
                        //         {
                        //             pass.log(mv::Logger::MessageType::Debug,"Error partitioning tensor into workloads using METIS");

                        //             /*Remove the original workload created with metis*/
                        //             workloadsVector.erase(workloadsVector.begin() + workloadsVectorIndex);
                        //             metisFail = true;
                        //         }
                        //     }
                        // }

                        // Rectangle Reuristic performs the same function as METIS
                        if ((algorithm == "Rectangle") && ((!depthWiseSOHA0Workaround) || nWorkloads > 1))
                        {
                            /*Create workload instance*/
                            workloadsVector.emplace_back(mv::Workloads(opIt->getName(), subTensor.getShape()));

                            /*Partition tensor into workloads with Rectangle*/
                            rectangleFail = false;
                            bool split_over_h = true;
                            bool split_over_w = true;
                            int rectangleResult = 0;
                            bool split_symmetric = false;

                            /*If nWorkloads specified in compilation descriptor, then force symmetric splits*/
                            if(nWorkloadsCompilationDescriptor)
                                split_symmetric = false;
                            else
                                split_symmetric = false;

                            /*If the operation is deptwise convolution, then do not split over H due to AO hardware bug*/
                            if(depthWiseSOHA0Workaround)
                                split_over_h = false;

                            rectangleResult = workloadsVector.at(workloadsVectorIndex).partitionTensorWithRectangleHeuristic(dpuModes, nWorkloads,
                                                        split_over_h, split_over_w, split_symmetric,
                                                        mv::WorkloadSplitMode::HW, pass);

                            if (rectangleResult != 1)
                            {
                                pass.log(mv::Logger::MessageType::Debug, "Error using Rectangle to tile the output tensor, erasing this workload instance");
                                workloadsVector.erase(workloadsVector.begin() + (workloadsVectorIndex));
                                rectangleFail = true;
                            }

                            if(!rectangleFail)
                            {

                                /*Check that workloads sum to the orignal output tensor volume*/
                                if((!workloadsVector.at(workloadsVectorIndex).validateWorkloads(subTensor.getShape())))
                                {
                                    pass.log(mv::Logger::MessageType::Debug, "Error producing valid workloads from Rectangle heuristic, the individual workloads do not sum to the original volume or they overlap, erasing this workload instance ");
                                    workloadsVector.erase(workloadsVector.begin() + workloadsVectorIndex);
                                    rectangleFail = true;
                                }

                                if(!rectangleFail)
                                {
                                    rectangleFail = false;
                                    pass.log(mv::Logger::MessageType::Debug, "Valid workload created using Rectangle");
                                    workloadsVectorIndex++;
                                }
                            }
                        }
                        /*Eltwise ops are performed by the PPE, which does not support Z-Tiling*/
                        if (algorithm == "Z-Tiling" && opIt->get<std::string>("taskOp") != "Add" &&
                            opIt->get<std::string>("taskOp") != "Subtract" && opIt->get<std::string>("taskOp") != "Divide"
                            && opIt->get<std::string>("taskOp") != "Multiply" && !depthWiseSOHA0Workaround)
                        {
                            /*Create workload instance*/
                            workloadsVector.emplace_back(mv::Workloads(opIt->getName(), subTensor.getShape()));

                            bool ztilingFail = false;
                            int ztilingResult = 0;

                            ztilingResult = workloadsVector.at(workloadsVectorIndex).partitionTensorWithZsplit(dpuModes, nWorkloads, pass);

                            if (ztilingResult != 1)
                            {
                                pass.log(mv::Logger::MessageType::Debug, "Error using Ztiling to tile the output tensor, erasing this workload instance");
                                workloadsVector.erase(workloadsVector.begin() + (workloadsVectorIndex));
                                ztilingFail = true;
                            }

                            if(!ztilingFail)
                            {

                                if((!workloadsVector.at(workloadsVectorIndex).validateWorkloads(subTensor.getShape())))
                                {
                                    pass.log(mv::Logger::MessageType::Debug, "Error producing valid workloads from Ztiling partitions,erasing this workload instance ");
                                    workloadsVector.erase(workloadsVector.begin() + workloadsVectorIndex);
                                    ztilingFail = true;
                                }
                                if(!ztilingFail)
                                {
                                    ztilingFail = false;
                                    pass.log(mv::Logger::MessageType::Debug, "Valid workload created using Z-Tiling");
                                    workloadsVectorIndex++;
                                }
                            }
                        }
                    }
                }

                float pixelCost = workloadPixelCost(opIt);

                /*Calculate execution cycles for each valid workload for this particular subtensor*/
                mv::Workloads::generateExecutionCycles(workloadsVector, nDPUxCluster, costFuntion, pixelCost);

                /*Sort on number of workloads */
                std::sort(workloadsVector.begin(), workloadsVector.end(),
                    [] (mv::Workloads const& lhs, mv::Workloads const& rhs)
                    {
                        return lhs.nWorkloads() < rhs.nWorkloads();
                    });

                /*Print the execution cycles*/
                int index = 0;
                for (auto wl : workloadsVector)
                {
                    pass.log(mv::Logger::MessageType::Debug, "Index " + std::to_string(index) + " (Min) " + std::to_string(wl.getExecutionCycles()[0]) + " Cost (Max)" + std::to_string(wl.getExecutionCycles()[1]));
                    index++;
                }

                int index1 = 0;
                for (auto wl : workloadsVector)
                {
                    pass.log(mv::Logger::MessageType::Debug, "Index " + std::to_string(index1) + " (Mean cost) " + std::to_string((wl.getExecutionCycles()[0]+wl.getExecutionCycles()[1])/2) + "  Workload number" + std::to_string(wl.nWorkloads()));
                    index1++;
                }

                /*Pick the workload with mean execution time*/
                auto optimalWorkload = std::min_element(workloadsVector.begin(), workloadsVector.end(),
                    [] (mv::Workloads const& lhs, mv::Workloads const& rhs)
                    {
                        return lhs.getMeanExecutionCycles() < rhs.getMeanExecutionCycles();
                    });

                /*Get the index of the most optimial workload*/
                optimalWorkloadIndex = std::distance(workloadsVector.begin(), optimalWorkload);

                pass.log(mv::Logger::MessageType::Debug, "Selecting workload at index " + std::to_string(optimalWorkloadIndex) + " as the most optimal workload for subtensor number " + std::to_string(clusterNumber));

                /*set the clusterID field of the most optimial workload*/
                workloadsVector.at(optimalWorkloadIndex).populateClusterID(clusterNumber);

                pass.log(mv::Logger::MessageType::Debug, "The subtensor for cluster 0 shape is: " + subTensor.getShape().toString());

                /*Apply the SOH offset to the most optimial workload*/
                if((opIt->getOutputTensor()[0]->hasAttr("splitStrategy")) && (nClusters > 1))
                {
                    pass.log(mv::Logger::MessageType::Debug, " op strategy " + opStrategy);
                    if (opStrategy != "Clustering")
                    {
                        auto subTensorOffset = subTensor.get<std::vector<std::size_t>>("offset");
                        workloadsVector.at(optimalWorkloadIndex).add_xy_offset(subTensorOffset);
                        workloadsVector.at(optimalWorkloadIndex).apply_z_offset(subTensorOffset);
                    }
                }

                /*Set the most optimal workload as attribute of the op*/
                opIt->set<mv::Workloads>("Workloads" + std::to_string(clusterNumber), workloadsVector.at(optimalWorkloadIndex));

                /*Reset workloads vector, splitpool and indices for the next sub tensor layer*/
                workloadsVector.clear();
                nWorkloadsSplitPool.clear();
                workloadsVectorIndex = 0;
                optimalWorkloadIndex = 0;
            }

            /*Reset workloads vector, splitpool and indices for the next layer*/
            workloadsVector.clear();
            nWorkloadsSplitPool.clear();
            workloadsVectorIndex = 0;
            optimalWorkloadIndex = 0;

        }
    }
    pass.log(mv::Logger::MessageType::Debug, "Exiting workload generation pass");
}


