#include "include/mcm/pass/graphOptimizations/streaming_performace.hpp"

mv::StreamingPerformance::StreamingPerformance(mv::OpModel& omodel, const int maxHStreams)
        : omodel_(omodel),
          pipelineChains_(omodel_),
          nClusters_(omodel.getGlobalConfigParams()->get<int>("Number_of_Clusters")),
          enableChannelMajorConv_(omodel.getGlobalConfigParams()->get<bool>("enable_channel_major_conv")),
          clusterMemory_(omodel.getGlobalConfigParams()->get<int>("cmx")),
          totalDpus_(omodel.getGlobalConfigParams()->get<int>("Number_of_DPUs")),
          maxHStreams_(maxHStreams) {
    
    globalParams_ = omodel.getGlobalConfigParams();
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
    std::string taskOp;
    if(op.hasAttr("taskOp"))
        taskOp = op.get<std::string>("taskOp");
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

    if (opType == "Conv" || opType == "DepthwiseConv" || taskOp == "Conv" || taskOp == "DepthwiseConv") {
        weightTableSize = 16 * alignedSplittedChannels;
        if (opType == "Conv" || taskOp == "Conv") {
            weightSize += alignedWeightsSize(op.getInputTensor(1), {1, 1, 1, streamConfig["K"], 1}, clusterStrategy,
                                             nClusters_);

        } else {
            weightSize += realTensorSize(op.getInputTensor(1), {1, 1, streamConfig["C"], 1, 1}, isCMConv);
            if (clusterStrategy == "SplitOverK")
                weightSize = div(weightSize, nClusters_);
        }

    } else if (opType == "MaxPool" || taskOp == "MaxPool") {
        weightTableSize = 16 * alignedSplittedChannels;
        weightSize = 0;
    } else if ((opType == "Eltwise" || taskOp == "Eltwise") && !software) {
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
    std::shared_ptr<mv::Element> globalParams = omodel_.getGlobalConfigParams();
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

bool mv::StreamingPerformance::requiresFakeActivationSparsity(mv::Data::OpListIterator opIt)
{
    if(enableChannelMajorConv_ && opIt->supportsCMConv())
        return true;
    if(opIt->getOpType() == "MaxPool")
        return true;
    if(opIt->getOpType() == "DepthwiseConv")
        return true;

    return false;
}

std::tuple<size_t, size_t, size_t> mv::StreamingPerformance::getMemorySize(mv::Data::OpListIterator opIt, const mv::Shape& streamConfig)
{
    auto clustering = opIt->get<std::string>("splitStrategy");
    auto inputSparse = opIt->get<bool>("inputActivationSparsity");
    auto outputSparse = opIt->get<bool>("outputActivationSparsity");
    auto weightsSparse = opIt->get<bool>("weightsSparsity");
    auto fakeSparse = requiresFakeActivationSparsity(opIt);
    bool spilling = opIt->get<bool>("goPredictsSpill");
    auto prevOp = omodel_.getSourceOp(opIt->getInputTensor(0));
    bool parentSpilling = prevOp->get<bool>("goPredictsSpill");

    return mv::memorySize(*opIt, nClusters_, enableChannelMajorConv_, clustering, inputSparse, outputSparse, weightsSparse, streamConfig,
                        fakeSparse, spilling, parentSpilling);
}

bool mv::StreamingPerformance::validateHStream(mv::Data::OpListIterator opIt, std::string clustering, std::size_t splits)
{
    if( opIt->getOpType() == "Conv" || opIt->getOpType() == "DepthwiseConv")
    {
        if(clustering == "SplitOverH")
        {
            auto weightsShape = opIt->getInputTensor(1)->getShape();
            //Try to guess subtensor height, and avoid situations where kernel is bigger than last workload dimension
            auto outputHeight = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
            auto workloadHeight = ceil((double)outputHeight / (double)(nClusters_ * splits));
            if(nClusters_ > 1) //last
                workloadHeight = outputHeight - (workloadHeight * (nClusters_-1)); //get remaining height
            if(workloadHeight < weightsShape[mv::KERNEL_HEIGHT])
                return false;
        }
    }

    //check that the inputSize will not be smaller than kernel size
    if (opIt->getOpType() == "MaxPool" ||
        opIt->getOpType() == "Conv" || opIt->getOpType() == "DepthwiseConv")
    {
        uint16_t kernelH;
        std::array<unsigned short, 4> padding;

        size_t originalH = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
        std::vector<std::size_t> newOutputSizes = mv::tileSpatialOutputSize(originalH, splits);

        //Reject H streams were the last stream isn't equal or smaller than the rest
        //Reject H streams were the last stream is 1, unless they are all 1
        if(newOutputSizes.back() > newOutputSizes.front() ||
            (newOutputSizes.back() == 1 && newOutputSizes.front() != 1)) 
                return false;

        unsigned short kernelStride;
        if (opIt->hasAttr("stride"))
            kernelStride = opIt->get<std::array<unsigned short, 2>>("stride")[1];
        else
            kernelStride = 1;//fake stride

        if (opIt->hasAttr("padding"))
            padding = opIt->get<std::array<unsigned short, 4>>("padding");
        else
            padding = {0, 0, 0, 0};

        int padStart = 0;
        int padEnd = padding[3];

        if (opIt->hasAttr("kSize"))
        {
            auto kernelShape = opIt->get<std::array<unsigned short, 2>>("kSize");
            kernelH = kernelShape[1];
        }
        else
        {
            auto weightsShape = opIt->getInputTensor(1)->getShape();
            kernelH = weightsShape[mv::KERNEL_HEIGHT];
        }
        int inputSizeForLastSplit = ((newOutputSizes.back() -1) * kernelStride)  -padStart - padEnd + kernelH;
        if ((inputSizeForLastSplit + padEnd) < kernelH)
            return false;
    }

    return true;
}

// Gives the minimum number of streams over H to fit this layer, or if no number of streams enable streaming
// (for example, weights don't fit) then return 0
unsigned mv::StreamingPerformance::getMinStreamOverH(mv::Data::OpListIterator opIt)
{
    auto clusterStrategy = opIt->get<std::string>("splitStrategy");

    size_t input, output, weights;
    // in case initialization in memorySize fails
    input = output = weights = 0;
    std::tie(input, output, weights) = getMemorySize(opIt, {1,1,1,1,1});
    auto activationsSize = input + output;
    auto weightsSize = weights;
    double availableMemory = (double) clusterMemory_ - (double) weightsSize;

    if (availableMemory <= 0) // Weights don't fit, can't stream over H
        return 0;

    // Keep increasing H until we find one big enough to fit, or we run out of H dimension to stream
    auto outputHeight = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];

    // Every output slice must have at least one line to compute
    unsigned upperBoundH = outputHeight;
    if(clusterStrategy == "SplitOverH")
    {
        upperBoundH = upperBoundH/nClusters_;
    }

    // Start searching for min stream at naive requirement for splits to fit, rather than 1
    for(unsigned splits = ceil((double)activationsSize/availableMemory); splits <= upperBoundH; splits++)
    {
        auto memFitCheck = getMemorySize(opIt, {1,splits,1,1,1});

        if((std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory_) &&
                validateHStream(opIt, clusterStrategy, splits))
        {
            return splits;
        }
    }

    return 0;
}

// Note: Validate a stream so that its largest slice fits in CMX and no workload issues
unsigned mv::StreamingPerformance::findOptimalValidStream(mv::Data::OpListIterator opIt, size_t startStream)
{
    auto clusterStrategy = opIt->get<std::string>("splitStrategy");

    for(unsigned splits = startStream; splits >= 1; splits--)
    {
        auto memFitCheck = getMemorySize(opIt,{1,splits,1,1,1});
        if( (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory_) &&
            validateHStream(opIt, clusterStrategy, splits))
                return splits;
    }

    return 1;
}

bool mv::StreamingPerformance::isStreamOptimizable(mv::Data::OpListIterator opIt, std::vector<mv::Element> streaming_strategy)
{
    auto opType = opIt->getOpType();
    if(!(opIt->hasTypeTrait("optimizable") && (opType == "Conv" || opType == "MaxPool" || opType == "DepthwiseConv")))
        return false;

    if (opIt->hasAttr("DilatedSubConv") && (opIt->get<bool>("DilatedSubConv")))
        return false;

    
    auto prevOp = omodel_.getSourceOp(opIt->getInputTensor(0));
    auto prevOpType = prevOp->getOpType();

    // Note: If op is streaming, location is set to DDR even if it can fit. Use the GO decision, 
    // as that will give indication if this will be CMX concatted later
    bool spilling = opIt->get<bool>("goPredictsSpill");
    bool parentSpilling = prevOp->get<bool>("goPredictsSpill");
    
    // Only consider ops that have input tensor in DDR
    // In case of parallel branches, don't trust the GO prediction
    if(!parentSpilling || (prevOpType == "Concat" || prevOpType == "Eltwise"))
        return false;

    auto clusteringStrategy = opIt->get<std::string>("splitStrategy");
    // GO rules disallow SOH with H streaming for convs, accuracy issues
    if(opType == "Conv" && clusteringStrategy == "SplitOverH")
        return false;

    bool isStreamingOtherDim = (streaming_strategy[3].get<int>("K") > 1 ||
                                    streaming_strategy[2].get<int>("C") > 1 ||
                                    streaming_strategy[4].get<int>("N") > 1 ) ? true : false;

    // Note: conservative approach, only consider ops that were already streaming over H
    // This preferences us to only dealing with big tensors - but could still see benefit
    // with tensors that may fit in CMX but are bigger than "overhead size" dmas 
    // (see magic number in K stream pass)
    if(streaming_strategy[1].get<int>("H") == 1)
        isStreamingOtherDim = true; 

    // Note: Special handling here for performance on yolov2, 
    // even if GO said to stream over K, we will still optimize over H if we can
    // Update in GO to give clustering and H stream in this case makese this less relevant for perf
    // but will leave here, as could still help a z-major compilation
    bool firstOp = false;
    if(prevOpType == "Input") firstOp = true;
    if(firstOp && isStreamingOtherDim)
    {
        auto minSplits = getMinStreamOverH(opIt);
        if(minSplits == 0) // stream over H alone won't fit
            return false;

        //See accuracy issues with clustering streamed over h and cmx concat?
        if(!spilling)
            return false;

        // In this case, ignore other streams (they will be erased) and just stream over H
        // (if other tests passed are passed of course)
        isStreamingOtherDim = false;
    }

    if(isStreamingOtherDim)
        return false;

    // Note: for SOH to stream over H, must concat in DDR. 
    // This is an un-solvable limitation of the data format, not a workaround.
    if( clusteringStrategy == "SplitOverH" && spilling )
    {
        // If this guy is newly streamed over H and is SOH, make sure it doesn't cmx concat!
        opIt->set<bool>("avoidCmxConcat", true);
        return true;
    }

    // Note: clustering and SOK can stream over H without regard to tensor placement, 
    if( clusteringStrategy == "SplitOverK" || clusteringStrategy == "Clustering")
        return true;

    return false;
}

//ASSUMPTION: the op must fit in cmx with just streaming over H (or no streaming at all)
std::size_t mv::StreamingPerformance::findOptimalStream(mv::Data::OpListIterator opIt, size_t originalHStream)
{
    double dpuPerCluster = std::floor(totalDpus_/nClusters_);
    auto clusteringStrategy = opIt->get<std::string>("splitStrategy");

    // Step 1. Decide which tensor will be the benchmark for how many streams we should do
    size_t input, output, weights;
    input = output = weights = 0;
    std::tie(input, output, weights) = getMemorySize(opIt, {1,1,1,1,1});

    // Step 2. Calculate a possible number of streams using experimetnally found magic number
    // Idea is, if possible, allow multiple slices to fit in CMX to maximize paths
    // For example, for first layer of YoloV2, (input=416*416*3, output=416*416*32)
    // The GO will choose to stream this over H, just enough to fit or H=7
    //      (416*416*3) + (416*416*32) = 6056960 / 7 = 865280, which is less than CMX (weights are under 1kb)
    // Experimentally, we found highest performance gain when we plan to fit input and output
    // slice, the next input slice, and the full weights in around 60% of CMX. This gives
    // the scheduler plenty of room, and creates small slices that better overlap with compute
    //      ((416*416*3)*2 + (416*416*32)) / (917504-(weights) * 0.6) = ~12.01 -> 13
    // Note that input and output in these equations are the full tensors because the 
    // getMemorySize call above is invoked without any streams {1,1,1,1,1}
    size_t magicStreams = std::ceil((2*input + output)/ ((clusterMemory_-weights)*0.6));
    if(magicStreams < originalHStream)
        magicStreams = originalHStream; //If GO gave carved it up into smaller pieces, must be for a reason
    else if(magicStreams > originalHStream*3)
        magicStreams = originalHStream*3; // Let's not get crazy with the H streams

    // Can't exceed the max, which ensures at least one line of output for each stream to compute
    size_t maxStreams = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
    if(clusteringStrategy == "SplitOverH") maxStreams = std::ceil((double)maxStreams/nClusters_);
    maxStreams = std::ceil((double) maxStreams / dpuPerCluster);

    size_t proposedStreams = std::min(magicStreams, maxStreams); //will be in range [originalHStream, maxStream]

    // Step 3. Find valid stream starting from proposedStreams and decreasing towards originalHStreams
    // Ensures lines are divided in such a way that it still fits in CMX, no workload issues etc
    auto optStream = findOptimalValidStream(opIt, proposedStreams);

    if(optStream < originalHStream)
        return originalHStream; // Never return fewer streams than GO assigned
    
    return optStream;
}

void mv::StreamingPerformance::increaseStreamingOverHforPerformance(const mv::pass::PassEntry& pass)
{

    auto globalParams = omodel_.getGlobalConfigParams();
    if (!globalParams->hasAttr("split_strategy"))
    {
         pass.log(mv::Logger::MessageType::Debug, "No custom splitting strategy provided, exiting..."); 
        return;
    }
    if(!enableChannelMajorConv_) return;

    auto streamingStrategies = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    std::vector<mv::Element> newStreamingStrategies;

    int streamCount=0;
    for(auto streamingStrategy : streamingStrategies)
    {
        std::string nodeName = streamingStrategy.get<std::string>("name_filter");
        // In case of manual strategy
        if (!omodel_.checkOp(nodeName))
            continue;

        auto opIt = omodel_.getOp(nodeName);
        bool updated = false;
        auto streams = streamingStrategy.get<std::vector<mv::Element>>("splits");

        // Step 0. Decide if we can insert activation streaming for this op
        if(isStreamOptimizable(opIt, streams))
        {
            size_t originalHStream = streams[1].get<int>("H");
            
            // Step 1. Choose optimal stream over H number for this op
            auto newHstream = findOptimalStream(opIt, originalHStream);

            // Step 2. Create the new streaming strategy and add to vector
            if(newHstream != originalHStream)
            {
                pass.log(mv::Logger::MessageType::Debug, "Op " + nodeName + " H stream strategy updated");
                mv::Element element(""); 
                element.set("name_filter",nodeName);

                std::vector<mv::Element> copySplits(streams.size(),mv::Element(""));
                copySplits[0].set<int>("W", 1);
                copySplits[1].set<int>("H", newHstream);
                copySplits[2].set<int>("C", 1);
                copySplits[3].set<int>("K", 1);
                copySplits[4].set<int>("N", 1);
                element.set("splits",copySplits);

                newStreamingStrategies.emplace_back(std::move(element));
                updated = true;
            }
        }

        //Step 3. Keep streaming stratgies that don't change too!
        if(!updated)
        {
            newStreamingStrategies.emplace_back(std::move(streamingStrategy));
        }

        // check # streams added
        auto strategySelected = newStreamingStrategies.back();
        auto streamsSelected = strategySelected.get<std::vector<mv::Element>>("splits");
        
        // Streams are mostly 1's, so multiply total streams added
        int streamsAdded =  streamsSelected[0].get<int>("W") * 
                            streamsSelected[1].get<int>("H") * 
                            streamsSelected[2].get<int>("C") *
                            streamsSelected[3].get<int>("K") *
                            streamsSelected[4].get<int>("N");
        
        streamCount += streamsAdded;
    }

    //Step 4. Save the streaming strategies into the compilation descriptor to be read by the streaming pass
    // 1050 streams equals ~ 7000 tasks which causes the runtime load/parse time to increase to >10 seconds.
    // vpuMgr currently times out all calls to the VPU that don't return in < 10secs
    if (streamCount < maxHStreams_)
    {
        globalParams->set<std::vector<mv::Element>>("streaming_strategy", newStreamingStrategies);
    }
}