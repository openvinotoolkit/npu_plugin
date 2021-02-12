#include "include/mcm/pass/graphOptimizations/streaming_performace.hpp"

mv::StreamingPerformance::StreamingPerformance(mv::ComputationModel& model, mv::OpModel& omodel)
        : model_(model),
          omodel_(omodel),
          pipelineChains_(omodel_),
          nClusters_(model.getGlobalConfigParams()->get<int>("Number_of_Clusters")),
          enableChannelMajorConv_(model.getGlobalConfigParams()->get<bool>("enable_channel_major_conv")) {
    

    globalParams_ = model.getGlobalConfigParams();
    streamingStrategyList_ = globalParams_->get<std::vector<mv::Element>>("streaming_strategy");
    multiClusterStrategyList_ = globalParams_->get<std::vector<mv::Element>>("split_strategy");
    tensorMemoryLocation_ = globalParams_->get<std::vector<mv::Element>>("tensor_placement_override");
   
    if (mv::isDebugFilesEnabled()) {
        fptr_ = fopen("./weight_streaming_network_analysis_report.txt", "w");
        if (nullptr == fptr_) {
            throw mv::RuntimeError("StreamingPerformance", "Cannot open file for writing");
        }
    }
}

mv::StreamingPerformance::~StreamingPerformance()
{
    if (mv::isDebugFilesEnabled())
        fclose(fptr_);
}

void mv::StreamingPerformance::increaseStreamingOverKforPerformance()
{
    // Step 1: Get the subgraph chains
    chainSubgraphs_ = pipelineChains_.get_chain_subgraphs(2UL);
    // Step 2: Get the minimum weights per cluster in a chain
    minWeightsPerClusterPerChain_ = calculateMininumWeightsSizePerClusterPerChain();
    // Step 3: Calculate more optimal streaming over K strategies
    if (!minWeightsPerClusterPerChain_.empty()) {

        evaluateGraphOptimizerAssignedKStreamingStrategies();

        // Step4: Assign the new strategies
        assignNewSrategies();

        if (mv::isDebugFilesEnabled()) 
            if(mv::saveNewStreamingStrategiesToJson(newStrategies_))
                throw mv::RuntimeError("StreamingPerformance", "Cannot open file for writing");                   
    }

}

size_t mv::StreamingPerformance::calculateperClusterWeightsSize(mv::Op& op, const mv::Attribute& clustering,
                                                                const bool weightsSparsity,
                                                                const mv::Shape& streamConfig) {
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
    if (mv::isDebugFilesEnabled()) {
        fprintf(fptr_, "%s :  %s :  %s :  %s :  %s :  %s :  %s :  %s", "chainId", "OpName", "kStreaming", "Hstreaming",
                "MultiCluster", "TotalSize(Inc WT)", "OutputChannels", "WeightsPerCluster(Inc WT)");
        fprintf(fptr_, "\n");
    }

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

                        // The operation must be already assigned stream over K and SOK and not be stream over H to be
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

                            if (mv::isDebugFilesEnabled()) {
                            fprintf(fptr_, "%zu : %s :  %zu : %zu  : %s : %zu : %zu : %zu ", chainID,
                                    (opIt->getName()).c_str(), streaming_strategy[3].get<int>("K"),
                                    streaming_strategy[1].get<int>("H"), mcStrategy.c_str(),
                                    weightsPerCluster * nClusters_ * streaming_strategy[3].get<int>("K"),
                                    alignedFullOutputChannels, weightsPerCluster);
                            fprintf(fptr_, "\n");
                            }
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
    if (mv::isDebugFilesEnabled()) {
        fprintf(fptr_, "End of network analysis\n");
        fprintf(fptr_, "\n");
    }

    return minWeightsPerClusterPerChain;
}

// Get the strategy assigned by GO
std::tuple<std::vector<mv::Element>, mv::Attribute, bool> mv::StreamingPerformance::getGraphOptimizerAssignedStategies(const std::string opName) {
    mv::Data::OpListIterator opIt;
    std::vector<mv::Element> streaming_strategy;
    std::string mcStrategy;
    std::string memoryLocation;
    mv::Attribute multiClusterStrategy;
    bool spilling = false;

    for (auto layerNameStrategy : streamingStrategyList_) {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

        if (nodeName == opName) {
            opIt = omodel_.getOp(nodeName);

            // Get the streaming strategy assigned by graph optimizer
            streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");

            // Get the MC strategy assigned by graph optimizer
            for (auto s : multiClusterStrategyList_) {
                std::string& name_filter = s.get<std::string>("name_filter");
                std::regex exp(name_filter);
                if (std::regex_match(opIt->getName(), exp)) {
                    mcStrategy = s.get<std::string>("strategy");
                    multiClusterStrategy = mcStrategy;
                }
            }

            // Get the memory location predicted by graph optimizer
            for (auto s : tensorMemoryLocation_) {
                std::string& name_filter = s.get<std::string>("name_filter");
                std::regex exp(name_filter);
                if (std::regex_match(opIt->getName(), exp))
                    memoryLocation = s.get<std::string>("mem_location");

                if (memoryLocation == "CMX")
                    spilling = false;
                else if (memoryLocation == "DDR")
                    spilling = true;
            }
            break;
        }
    }
    return std::tuple<std::vector<mv::Element>, mv::Attribute, bool>(streaming_strategy, multiClusterStrategy,
                                                                     spilling);
}

void mv::StreamingPerformance::assignNewSrategies() {
    
    for (auto layerNameStrategy : streamingStrategyList_) {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        if (nodeName != "Example") {
            auto opIt = omodel_.getOp(nodeName);

            auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");

            auto newElement = streamingStrategyList_[0];
            auto newName = newElement.get<std::string>("name_filter");
            auto newSplits = newElement.get<std::vector<mv::Element>>("splits");
            for (int i = newSplits.size(); i < 5; i++)
                newSplits.push_back(newSplits[0]);

            if (opIt->hasAttr("optimalNumberOfKStreams")) {
                newElement.set("name_filter", opIt->getName());
                newSplits[0].set<int>("W", streaming_strategy[0].get<int>("W"));
                newSplits[1].set<int>("H", streaming_strategy[1].get<int>("H"));
                newSplits[2].set<int>("C", streaming_strategy[2].get<int>("C"));
                newSplits[3].set<int>("K", opIt->get<unsigned>("optimalNumberOfKStreams"));
                newSplits[4].set<int>("N", streaming_strategy[4].get<int>("N"));
                newElement.set("splits", newSplits);
                newStrategies_.push_back(newElement);
            } else {
                newElement.set("name_filter", opIt->getName());
                newSplits[0].set<int>("W", streaming_strategy[0].get<int>("W"));
                newSplits[1].set<int>("H", streaming_strategy[1].get<int>("H"));
                newSplits[2].set<int>("C", streaming_strategy[2].get<int>("C"));
                newSplits[3].set<int>("K", streaming_strategy[3].get<int>("K"));
                newSplits[4].set<int>("N", streaming_strategy[4].get<int>("N"));
                newElement.set("splits", newSplits);
                newStrategies_.push_back(newElement);
            }
        }
    }

    // Step5: Save the new strategies
    std::shared_ptr<mv::Element> globalParams = model_.getGlobalConfigParams();
    globalParams->set("streaming_strategy", newStrategies_);
}

std::pair<size_t, double> mv::StreamingPerformance::calculatefullWeightsSizeForOpandOptimalKStreaming(
        const std::string multiclusterStrategy, const size_t weightsPerClusterforOp,
        size_t minWeightsPerClusterPerChain, const bool isKStreaming, const int numberOfkStreams) {

    size_t fullWeightsSize = 0;
    size_t optimalNumberOfKStreams = 0;
    std::pair<size_t, double> toReturn;

    // Calculate the optimal number of K streams
    // First calculate the full weight size
    // Then divide by the minStreamSize * nclusters to get the optimal K streams
    if (isKStreaming && multiclusterStrategy == "SplitOverK") {
        fullWeightsSize = weightsPerClusterforOp * nClusters_ * numberOfkStreams;

        if (minWeightsPerClusterPerChain <= minWeightsPerClusterPerChainConstant_)
            minWeightsPerClusterPerChain = minWeightsPerClusterPerChainConstant_;

        optimalNumberOfKStreams = std::round(fullWeightsSize / (minWeightsPerClusterPerChain * nClusters_));

    } else if (isKStreaming && multiclusterStrategy == "Clustering") {
        fullWeightsSize = weightsPerClusterforOp * numberOfkStreams;

        if (minWeightsPerClusterPerChain <= minWeightsPerClusterPerChainConstant_)
            minWeightsPerClusterPerChain = minWeightsPerClusterPerChainConstant_;

        optimalNumberOfKStreams = std::round(fullWeightsSize / minWeightsPerClusterPerChain);

    } else if (multiclusterStrategy == "SplitOverK" && !isKStreaming) {
        fullWeightsSize = weightsPerClusterforOp * nClusters_;

        if (minWeightsPerClusterPerChain <= minWeightsPerClusterPerChainConstant_)
            minWeightsPerClusterPerChain = minWeightsPerClusterPerChainConstant_;

        optimalNumberOfKStreams = std::round(fullWeightsSize / (minWeightsPerClusterPerChain * nClusters_));
    }

    // Ensure K streams is never 0
    if (optimalNumberOfKStreams < 1)
        optimalNumberOfKStreams = 1;

    toReturn.first = fullWeightsSize;
    toReturn.second = optimalNumberOfKStreams;

    return toReturn;
}

void mv::StreamingPerformance::writeStatsToFile(const unsigned chainID, const std::string opName,const int kStreaming,const int hStreaming,
                      const std::string multiclusterStrategy,const size_t fullweightsSize,const size_t alignedFullOutputChannels,
                      const size_t weightsPerClusterPerOp,const size_t minWeightsPerClusterPerChain,
                      const double optimalNumberOfKStreams,const double maxpossibleStreams,const double newKStreams) {
    fprintf(fptr_,
            "%zu : %s :  %zu : %zu  : %s : %zu : %zu : %zu : %zu : %.1f : %.1f : "
            "%.1f ",
            chainID, opName.c_str(), kStreaming, hStreaming, multiclusterStrategy.c_str(), fullweightsSize,
            alignedFullOutputChannels, weightsPerClusterPerOp, minWeightsPerClusterPerChain, optimalNumberOfKStreams,
            maxpossibleStreams, newKStreams);
    fprintf(fptr_, "\n");
}

void mv::StreamingPerformance::evaluateGraphOptimizerAssignedKStreamingStrategies() {
    
    unsigned chainID = 0;
    size_t fullweightsSize = 0;
    double maxpossibleStreams = 0.0;
    double optimalNumberOfKStreams = 0;
    std::vector<mv::Element> graphOptimizerStreamingStrategy;
    std::vector<mv::Element> overWrittenStreamingStrategies;
    std::vector<mv::Element> allStreamingStrategies;
    std::pair<size_t, double> fullWeightsSizeOptimalKStreaming = {};
    std::size_t minStreamSize = 0;
    size_t alignedFullOutputChannels = 0;
    size_t weightsPerCluster = 0;
    size_t fullWeightsSize = 0;
    mv::Attribute graphOptimizerMultiClusterStrategy;
    bool graphOptimizerTensorLocationSpilling;

    // create header for network analysis report file
    if (mv::isDebugFilesEnabled()) {
        fprintf(fptr_,  "%s :  %s :  %s :  %s :  %s :  %s :  %s :  %s :  %s : %s :  %s :  %s", "chainId", "OpName", "Default kStreaming", "Default Hstreaming", "MultiCluster", "TotalSize(Inc WT)", "OutputChannels", "WeightsPerCluster(Inc WT)", "MinWeightsPerClusterInChain", "optimalNumberOfKStreams","maxNumberKStreams", "NewKStreams");
        fprintf(fptr_, "\n");
    }
    
    for (subgraph_t chain_subgraph : chainSubgraphs_) {
        for (auto& op : chain_subgraph.dpu_chain_) {
            mv::Data::OpListIterator opIt = omodel_.getOp(op->getName());

            // Only convs are considered to be streamed more
            if (opIt->getOpType() == "Conv") {
                optimalNumberOfKStreams = 0;

                // Get the strategy assigned by GO for this operation
                auto graphOptimizerAssignedStategies = getGraphOptimizerAssignedStategies(opIt->getName());

                // Get the streaming and multicluster strategy assigned by GO for this operation
                graphOptimizerStreamingStrategy = std::get<0>(graphOptimizerAssignedStategies);
                graphOptimizerMultiClusterStrategy = std::get<1>(graphOptimizerAssignedStategies);
                graphOptimizerTensorLocationSpilling = std::get<2>(graphOptimizerAssignedStategies);

                bool isKStreaming = graphOptimizerStreamingStrategy[3].get<int>("K") > 1 ? true : false;
                bool isHStreaming = graphOptimizerStreamingStrategy[1].get<int>("H") > 1 ? true : false;

                // Get the output channels to determine the max possible K streams so we know the limit
                alignedFullOutputChannels =
                        mv::round_up(opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);

                // Calculate the max possible K streams based on the multi-cluster strategy
                maxpossibleStreams = floor(alignedFullOutputChannels /
                                           minOutputChannels_.at(graphOptimizerMultiClusterStrategy.get<std::string>()));

                // Get the weights per cluster for this op
                weightsPerCluster = weightsPerClusterPerOp_.find(opIt->getName())->second;

                // The operation must be already assigned stream over K and SOK and not be sream over H to be considered
                // for a new K stream strategy
                if (isKStreaming && graphOptimizerMultiClusterStrategy.get<std::string>() == "SplitOverK" &&
                    !isHStreaming) {
                    fullWeightsSizeOptimalKStreaming = {0, 0};
                    if (minWeightsPerClusterPerChain_[chainID] > 0)
                        fullWeightsSizeOptimalKStreaming = calculatefullWeightsSizeForOpandOptimalKStreaming(
                                graphOptimizerMultiClusterStrategy.get<std::string>(), weightsPerCluster,
                                minWeightsPerClusterPerChain_[chainID], isKStreaming,
                                graphOptimizerStreamingStrategy[3].get<int>("K"));

                    fullWeightsSize = fullWeightsSizeOptimalKStreaming.first;
                    optimalNumberOfKStreams = fullWeightsSizeOptimalKStreaming.second;

                    // Validate that the optimal number of k streams doesn't introduce a crop layer which cannot be CMX
                    // concated by LP Scheduler
                    if (!validateKStream(*opIt, graphOptimizerMultiClusterStrategy.get<std::string>(),
                                         fullWeightsSizeOptimalKStreaming.second, graphOptimizerTensorLocationSpilling,
                                         nClusters_))
                        continue;

                    // Assign the new streaming strategies
                    // The optimalNumberOfKStreams must be > 0, less than the max possible K streams and must not
                    // decrease the K streams assinged from the GO
                    if ((optimalNumberOfKStreams > 0) && (optimalNumberOfKStreams <= maxpossibleStreams) &&
                        (optimalNumberOfKStreams > graphOptimizerStreamingStrategy[3].get<int>("K"))) {
                        if (minWeightsPerClusterPerChain_[chainID] < minWeightsPerClusterPerChainConstant_)
                            minWeightsPerClusterPerChain_[chainID] = minWeightsPerClusterPerChainConstant_;

                        if (mv::isDebugFilesEnabled()) {
                            writeStatsToFile(chainID, (opIt->getName()).c_str(),
                                             graphOptimizerStreamingStrategy[3].get<int>("K"),
                                             graphOptimizerStreamingStrategy[1].get<int>("H"),
                                             graphOptimizerMultiClusterStrategy.get<std::string>().c_str(),
                                             fullWeightsSize, alignedFullOutputChannels,
                                             weightsPerClusterPerOp_.find(opIt->getName())->second,
                                             minWeightsPerClusterPerChain_[chainID], optimalNumberOfKStreams,
                                             maxpossibleStreams, optimalNumberOfKStreams);
                        }
                        opIt->set<unsigned>("optimalNumberOfKStreams", optimalNumberOfKStreams);

                    }
                    // Else assign the max possible K streams for the layer
                    else if (optimalNumberOfKStreams > maxpossibleStreams) {
                        if (minWeightsPerClusterPerChain_[chainID] < minWeightsPerClusterPerChainConstant_)
                            minWeightsPerClusterPerChain_[chainID] = minWeightsPerClusterPerChainConstant_;
                        
                        if (mv::isDebugFilesEnabled()) {
                            writeStatsToFile(chainID, (opIt->getName()).c_str(),
                                             graphOptimizerStreamingStrategy[3].get<int>("K"),
                                             graphOptimizerStreamingStrategy[1].get<int>("H"),
                                             graphOptimizerMultiClusterStrategy.get<std::string>().c_str(),
                                             fullWeightsSize, alignedFullOutputChannels,
                                             weightsPerClusterPerOp_.find(opIt->getName())->second,
                                             minWeightsPerClusterPerChain_[chainID], optimalNumberOfKStreams,
                                             maxpossibleStreams, maxpossibleStreams);
                        }

                        opIt->set<unsigned>("optimalNumberOfKStreams", maxpossibleStreams);
                    }
                }
            }
        }
        chainID++;
    }
}