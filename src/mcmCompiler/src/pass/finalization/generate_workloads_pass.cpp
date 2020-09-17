#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/target/kmb/workloads.hpp"
#include "include/mcm/target/kmb/rectangle.hpp"
#include "include/mcm/target/kmb/workload_struct.hpp"
#include "include/mcm/graph/graph.hpp"
#include <algorithm>
#include <climits>
#include <math.h>


static void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(GenerateWorkloads)
            .setFunc(generateWorkloadsFcn)
            .setDescription(
                "Generate workloads");
    }
}

float workloadPixelCost(mv::Data::OpListIterator opIt)
{
 auto kh = opIt->get<std::array<unsigned short, 2>>("kSize")[0];
 auto kw = opIt->get<std::array<unsigned short, 2>>("kSize")[1];
 auto ic = opIt->get<std::string>("taskOp") == "ChannelMajorConvolution" || opIt->get<std::string>("taskOp") == "DepthwiseConv" || opIt->get<std::string>("taskOp") == "Conv" ? opIt->getInputTensor()[0]->getShape()[2] : 1;

 return kh*kw*ic;
}


/**
 * @brief Returns the supported Tensor Split Algorithms to be used
 */
std::vector<std::string> getTensorSplitAlgorithms(mv::Element& passDesc, const mv::pass::PassEntry& pass)
{
    /*parse TensorSplitAlgorithms from Compilation Descriptor*/
    std::vector<std::string> algorithms = {"Rectangle", "Z-Tiling"}; //default
    if (passDesc.hasAttr("TensorSplitAlgorithms"))
    {
        algorithms.clear();
        std::string sAlgorithms = passDesc.get<std::string>("TensorSplitAlgorithms");
        std::stringstream ss(sAlgorithms);
        while( ss.good() )
        {
            std::string tempStr;
            std::getline(ss, tempStr, ',');
            if (tempStr=="Rectangle" || tempStr=="Z-Tiling")
                algorithms.push_back(tempStr);
            else
                pass.log(mv::Logger::MessageType::Warning, "Could not parse the TensorSplitAlgorithms type (only \"Rectangle, Z-Tiling\" currently supported).");
        }
    }
    else
        pass.log(mv::Logger::MessageType::Debug, "No TensorSplitAlgorithms specified in descriptor, using  \"Rectangle, Z-Tiling\"...");

    //if parsing problem, return all 3
    if (algorithms.size() == 0)
        algorithms = {"Rectangle", "Z-Tiling"};
    return algorithms;
}

std::tuple<int,int, int, int, int> getGlobalCompilationDescriptorConf(const mv::pass::PassEntry& pass, mv::ComputationModel& model) {

    int nDPU = 1;
    int nClusters = 1;
    int nWorkloads = 0;
    int workloadCost = 0;
    int pad = 16;

    /*Get nDPUs and nClsuters from gloabl compilation descriptor*/
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("Number_of_DPUs"))
        nDPU = globalParams->get<int>("Number_of_DPUs");
    if (globalParams->hasAttr("Number_of_Clusters"))
        nClusters = globalParams->get<int>("Number_of_Clusters");
    if (globalParams->hasAttr("nWorkloads"))
        nWorkloads= globalParams->get<int>("nWorkloads");
    if (globalParams->hasAttr("WorkloadCost"))
        workloadCost = globalParams->get<int>("WorkloadCost");
    if (globalParams->hasAttr("VPU2ChannelPadding"))
        pad = globalParams->get<int>("VPU2ChannelPadding");

    if(nDPU < nClusters)
        throw std::runtime_error("The number of DPUs cannot be less than the number of clusters!, exiting");

    /*DPUs per cluster*/
    int nDPUxCluster =  ceil(nDPU / nClusters);

    pass.log(mv::Logger::MessageType::Debug, "Number of DPUs per cluster is: " + std::to_string(nDPUxCluster));

    return std::make_tuple(nDPUxCluster, nWorkloads, nClusters, pad, workloadCost);
}

void generateWorkloadsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc, mv::Element&)
{
    pass.log(mv::Logger::MessageType::Debug, "Starting workload generation pass");
    mv::OpModel om(model);

    mv::DPUModeList dpuModes;
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
    auto pad = std::get<3>(compilationConfigs);
    auto workloadCost = std::get<4>(compilationConfigs);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    auto referenceDevice = globalParams->get<std::string>("referenceDevice");

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
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

            if((opIt->get<std::string>("taskOp") == "DepthwiseConv") || (opIt->get<std::string>("taskOp") == "MaxPool") 
                || (opIt->get<std::string>("taskOp") == "ChannelMajorConvolution"))
                dpuModes = {{1, 16}};
            else
                dpuModes = {{4,4},{1, 16}};

            if (opIt->getOutputTensor()[0]->getDType() == mv::DType("Float16"))
                dpuModes = {{1,4}};

            /*Depthwise cov SOH A0 workaround*/
            if(((opIt->get<std::string>("taskOp") == "DepthwiseConv") ||
                        (opIt->get<std::string>("taskOp") == "MaxPool")) &&
                    (opIt->get<std::string>("splitStrategy") == "SplitOverH") && referenceDevice == "A0") {
                depthWiseSOHA0Workaround = true;
                opIt->set<std::string>("Depthwise_SOH_A0_bug", "True");
            }

            // Mixed precision A0/B0 workaround
            auto inputDType = opIt->getInputTensor(0)->getDType();
            auto outputDType = opIt->getOutputTensor(0)->getDType();
            bool mixedPrecisionA0B0WorkAround = false;

            if((inputDType != outputDType) && outputDType != mv::DType("Int32") && opIt->get<std::string>("taskOp") == "Conv")
                mixedPrecisionA0B0WorkAround = true;

            /*For multi-clustering we work on subtensors*/
            for(clusterNumber = 0; clusterNumber < nClusters; clusterNumber++)
            {
                /*get the subtensor*/
                auto subTensor = opIt->getOutputTensor(0)->getSubTensor(clusterNumber);

                /* Check if subtensor needs to be aligned to 16 channels*/
                auto subTensorShape = subTensor.getShape();
                auto subTensorChannels = subTensorShape[mv::IO_CHANNEL_DIMENSION];
                if (subTensorChannels % pad != 0)
                {
                    auto outputChannelsPadded = mv::round_up(subTensorShape[mv::IO_CHANNEL_DIMENSION], pad);
                    subTensorShape[mv::IO_CHANNEL_DIMENSION] = outputChannelsPadded;
                }

                /* Sparse tensors don't use z-tiling*/
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

                if(mixedPrecisionA0B0WorkAround)
                {
                    nWorkloadsSplitPool.clear();
                    nWorkloadsSplitPool.push_back(subTensorShape[0]*subTensorShape[1]);
                    algorithms = {"MixedPrecisionA0B0WorkAround"};
                }

                /*if Deptwise operation and SOH trategy, for A0 bug then add these number of worklaods to workload split pool*/
                if(depthWiseSOHA0Workaround &&
                    !nWorkloadsCompilationDescriptor)
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
                        if(algorithm == "MixedPrecisionA0B0WorkAround")
                        {
                            workloadsVector.emplace_back(mv::Workloads(opIt->getName(), subTensorShape));
                            auto& workloads = workloadsVector[workloadsVectorIndex];
                            for(unsigned w = 0; w < subTensorShape[0]; ++w)
                            {
                                for(unsigned h = 0; h < subTensorShape[1]; ++h)
                                {
                                    mv::Workload toAdd;
                                    toAdd.MinX = w;
                                    toAdd.MaxX = w;
                                    toAdd.MinY = h;
                                    toAdd.MaxY = h;
                                    toAdd.MinZ = 0;
                                    toAdd.MaxZ = subTensorShape[2]-1;
                                    toAdd.MPEMode = mv::MPE_Mode::Vector_FP16;
                                    workloads.addWorkload(toAdd);
                                }
                            }
                            workloadsVectorIndex++;

                        }
                        if ((algorithm == "Rectangle") && ((!depthWiseSOHA0Workaround) || nWorkloads > 1))
                        {
                            /*Create workload instance*/
                            workloadsVector.emplace_back(mv::Workloads(opIt->getName(), subTensorShape));

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
                                if((!workloadsVector.at(workloadsVectorIndex).validateWorkloads(subTensorShape)))
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
                        if (algorithm == "Z-Tiling" && opIt->get<std::string>("taskOp") != "Eltwise"
                            && !depthWiseSOHA0Workaround)
                        {
                            /*Create workload instance*/
                            workloadsVector.emplace_back(mv::Workloads(opIt->getName(), subTensorShape));

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

                                if((!workloadsVector.at(workloadsVectorIndex).validateWorkloads(subTensorShape)))
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
                mv::Workloads::generateExecutionCycles(workloadsVector, nDPUxCluster, costFuntion, pixelCost, workloadCost);

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

                if((opIt->getOutputTensor()[0]->hasAttr("asymmetricKernel")))
                {
                    //Check if the optimalWorkload is good
                    auto dim = opIt->getOutputTensor()[0]->get<unsigned>("asymmetricKernel");

                    bool optimalIsGood = true;
                    auto optimalWL = workloadsVector[optimalWorkloadIndex];
                    for (size_t i=0; i< optimalWL.nWorkloads(); i++)
                    {
                        if ((dim == mv::KERNEL_HEIGHT && optimalWL[i].MinY != optimalWL[i].MaxY)
                            || (dim == mv::KERNEL_WIDTH && optimalWL[i].MinX != optimalWL[i].MaxX))
                        {
                            optimalIsGood = false;
                            break;
                        }
                    }
                    if (!optimalIsGood)
                    {
                        //It;s not let's pick another one or add one if none are found
                        size_t wl_index = 0;

                        //find the first optimal workload where the dim needed is divided for one line or row
                        for (auto wl : workloadsVector)
                        {
                            bool found = true;
                            for (size_t i=0; i< wl.nWorkloads(); i++)
                            {
                                if (dim == mv::KERNEL_HEIGHT && wl[i].MinY != wl[i].MaxY)
                                {
                                    found = false;
                                    break;
                                }
                                if (dim == mv::KERNEL_WIDTH && wl[i].MinX != wl[i].MaxX)
                                {
                                    found = false;
                                    break;
                                }

                            }
                            if (found)
                            {
                                break;
                            }
                            wl_index++;
                        }
                        if (wl_index == workloadsVector.size())
                        {
                            pass.log(mv::Logger::MessageType::Debug, "Couldnt find a workload for AsymmetricKernel case, adding one");

                            //Add workload that we need (each row in a workload to avoid striding on both dim)
                            workloadsVector.emplace_back(mv::Workloads(opIt->getName(), subTensorShape));
                            auto& workloads = workloadsVector[workloadsVectorIndex];

                            for(unsigned a = 0; a < subTensorShape[dim]; ++a)
                            {
                                mv::Workload toAdd;
                                if (dim == mv::KERNEL_HEIGHT)
                                {
                                    toAdd.MinX = 0;
                                    toAdd.MaxX = subTensorShape[mv::KERNEL_WIDTH]-1;
                                    toAdd.MinY = a;
                                    toAdd.MaxY = a;
                                }
                                else
                                {
                                    toAdd.MinY = 0;
                                    toAdd.MaxY = subTensorShape[mv::KERNEL_HEIGHT]-1;
                                    toAdd.MinX = a;
                                    toAdd.MaxX = a;
                                }

                                toAdd.MinZ = 0;
                                toAdd.MaxZ = subTensorShape[2]-1;
                                toAdd.MPEMode = mv::MPE_Mode::Vector;
                                workloads.addWorkload(toAdd);

                            }
                            optimalWorkloadIndex = workloadsVectorIndex;
                            workloadsVectorIndex++;
                        }
                        else
                        {
                            //std::cout << " index found " << std::to_string(wl_index) << " Optimal was " << std::to_string(optimalWorkloadIndex) << std::endl;
                            optimalWorkloadIndex = wl_index;
                        }
                    }
                }

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

