#include "include/mcm/pass/graphOptimizations/streaming_performace.hpp"

mv::StreamingPerformance::StreamingPerformance(mv::ComputationModel& model, mv::OpModel& omodel): model_(model), omodel_(omodel), pipelineChains_(omodel_)
{
    globalParams_ = model.getGlobalConfigParams();
    streamingStrategyList_ = globalParams_->get<std::vector<mv::Element>>("streaming_strategy");
    multiClusterStrategyList_ = globalParams_->get<std::vector<mv::Element>>("split_strategy");
    nClusters_ = globalParams_->get<int>("Number_of_Clusters");
    enableChannelMajorConv_ = globalParams_->get<bool>("enable_channel_major_conv");
}

void mv::StreamingPerformance::evaluateStreamingOverKStrategies()
{
    chainSubgraphs_ = pipelineChains_.get_chain_subgraphs(2UL);

    minWeightsPerClusterPerChain_ = calculateMininumWeightsSizePerClusterPerChain();

}

size_t mv::StreamingPerformance::calculateperClusterWeightsSize(mv::Op& op, const mv::Attribute& clustering,
                                                                bool weightsSparsity, const mv::Shape& streamConfig) {
    auto div = [](unsigned x, unsigned y) -> unsigned {
        return (x + y - 1) / y;
    };

    size_t inputSize = 0;
    size_t outputSize = 0;
    size_t weightSize = 0;
    size_t weightTableSize = 0;
    std::string opType = op.getOpType();
    bool isCMConv = false;
    auto clusterStrategy = clustering.get<std::string>();

    if (enableChannelMajorConv_ && op.supportsCMConv())
        isCMConv = true;
    auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

    size_t outChannels = op.outputSlots() ? op.getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] : 0;
    size_t alignedFullChannels = mv::round_up(outChannels, 16);
    size_t alignedSplittedChannels = mv::round_up(alignedFullChannels / streamConfig["K"], 16);

    if (clusterStrategy == "SplitOverK") {
        alignedSplittedChannels = mv::round_up(alignedSplittedChannels / nClusters_, 16);
    }

    if (opType == "Conv" || opType == "DepthwiseConv") {
        weightTableSize = 16 * alignedSplittedChannels;
        if (opType == "Conv") {
            weightSize += alignedWeightsSize(op.getInputTensor(1), {1, 1, 1, streamConfig["K"], 1}, clusterStrategy,
                                             nClusters_);

        } else {
            weightSize += realTensorSize(op.getInputTensor(1), {1, 1, streamConfig["C"], 1, 1}, isCMConv);
            if (clusterStrategy == "SplitOverK")
                weightSize = div(weightSize, nClusters_);
        }

    } else if (opType == "MaxPool") {
        weightTableSize = 16 * alignedSplittedChannels;
        weightSize = 0;
    } else if (opType == "Eltwise" && !software) {
        weightTableSize = 0;
        weightSize = 0;
    }

    if (weightsSparsity) {
        auto tensorSize = op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] *
                          op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] *
                          mv::round_up(op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS], 16) *
                          alignedSplittedChannels;

        auto sparseWeightSize = std::ceil((double)tensorSize / 8);
        sparseWeightSize = mv::round_up(sparseWeightSize, 16);
        weightSize += sparseWeightSize;
    }

    weightSize += weightTableSize;

    return weightSize;
}

std::map<size_t, size_t> mv::StreamingPerformance::calculateMininumWeightsSizePerClusterPerChain() {

    unsigned chainID = 0;
    size_t weightsPerCluster = 0;
    std::string clustering;
    std::vector<size_t> streamsSizes;
    bool weightsSparsity = false;
    bool isHStreaming = false;
    bool isKStreaming = false;
    std::map<size_t, size_t> minWeightsPerClusterPerChain;

    // Header for the network analysis report
    // fprintf(fptr, "%s :  %s :  %s :  %s :  %s :  %s :  %s :  %s", "chainId", "OpName", "kStreaming", "Hstreaming",
    //         "MultiCluster", "TotalSize(Inc WT)", "OutputChannels", "WeightsPerCluster(Inc WT)");
    // fprintf(fptr, "\n");

    for (subgraph_t chain_subgraph : chainSubgraphs_) {
        streamsSizes.clear(); 

        for (auto& op : chain_subgraph.dpu_chain_) {
            mv::Data::OpListIterator opIt = omodel_.getOp(op->getName());

            // Only Conv's are consider for additional K streaming
            if (opIt->getOpType() == "Conv") {
                // Get the strategy for this conv
                for (auto layerNameStrategy : streamingStrategyList_) {
                    std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

                    if (nodeName == op->getName()) {
                        // Get the streaming strategy from graph optimizer
                        auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
                        isKStreaming = streaming_strategy[3].get<int>("K") > 1 ? true : false;
                        isHStreaming = streaming_strategy[1].get<int>("H") > 1 ? true : false;

                        // Get the MC strategy from graph optimizer
                        std::string mcStrategy;
                        for (auto s : multiClusterStrategyList_) {
                            std::string& name_filter = s.get<std::string>("name_filter");
                            std::regex exp(name_filter);
                            if (std::regex_match(opIt->getName(), exp))
                                mcStrategy = s.get<std::string>("strategy");
                        }

                        // The operation must be already assigned stream over K and SOK and not be sream over H to be
                        // considered for a new K stream strategy
                        if (isKStreaming && mcStrategy == "SplitOverK" && !isHStreaming) {
                            if (op->hasAttr("splitStrategy"))
                                clustering = op->get<std::string>("splitStrategy");

                            weightsSparsity = false;
                            if (op->hasAttr("weightsSparsity"))
                                weightsSparsity = op->get<bool>("weightsSparsity");

                            // get the memory size of the streams weights
                            weightsPerCluster = 0;
                            mv::Data::OpListIterator oitr = omodel_.getOp(op->getName());

                            weightsPerCluster =
                                    calculateperClusterWeightsSize(*oitr, clustering, weightsSparsity,
                                                          {1, (unsigned int)streaming_strategy[1].get<int>("H"),
                                                           (unsigned int)streaming_strategy[2].get<int>("C"),
                                                           (unsigned int)streaming_strategy[3].get<int>("K"),
                                                           (unsigned int)streaming_strategy[4].get<int>("N")});

                            streamsSizes.push_back(weightsPerCluster);

                            weightsPerClusterPerOp_.insert({opIt->getName(), weightsPerCluster});

                            size_t alignedFullOutputChannels =
                                    mv::round_up(opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);

                            // fprintf(fptr, "%zu : %s :  %zu : %zu  : %s : %zu : %zu : %zu ", chainID,
                            //         (opIt->getName()).c_str(), streaming_strategy[3].get<int>("K"),
                            //         streaming_strategy[1].get<int>("H"), mcStrategy.c_str(),
                            //         weightsPerCluster * nClusters * streaming_strategy[3].get<int>("K"),
                            //         alignedFullOutputChannels, weightsPerCluster);
                            // fprintf(fptr, "\n");
                        }
                    }
                }
            }
        }

        // Store the minimum weights per cluster for the chain
        std::sort(streamsSizes.begin(), streamsSizes.end());
        if (!streamsSizes.empty())
            minWeightsPerClusterPerChain.insert({chainID, streamsSizes[0]});

        chainID++;
    }
    // fprintf(fptr, "End of network analysis\n");
    // fprintf(fptr, "\n");

    return minWeightsPerClusterPerChain;
}