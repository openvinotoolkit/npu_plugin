#include "include/mcm/base/attribute.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/graphOptimizations/strategy_utils.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "pass/lp_scheduler/pipeline_transform.hpp"
#include "pass/lp_scheduler/pipeline_chains_transform.hpp"
#include "include/mcm/op_model.hpp"
#include <regex>
#include <iterator>
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "mcm/utils/custom_strings.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/tensor/shape.hpp"
#include <unordered_map> 
#include <iostream>
#include <iomanip>
#include "chrono"

using StrategySet = std::unordered_map<std::string,mv::Attribute>;
typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;

void validateStreamingOverKForChainPipeliningFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(ValidateStreamingOverKForChainPipelining)
        .setFunc(validateStreamingOverKForChainPipeliningFnc)
        .setDescription(
            "This pass evaluates and re-calculates streams over K to enable better performance when combined with chained pipelining in the scheduler"
        );

        
    }

   
}

size_t perClusterWeightsSize(mv::Op& op, const mv::Attribute& clustering, bool weightsSparsity, const mv::Shape& streamConfig, size_t nClusters) {
    
    auto div = [](unsigned x, unsigned y) -> unsigned {
        return (x + y - 1) / y;
    };

    bool enableChannelMajorConv = true;
    int totalClusters = 4;
    size_t inputSize = 0;
    size_t outputSize = 0;
    size_t weightSize = 0;
    size_t weightTableSize = 0;
    auto opType = op.getOpType();
    auto isCMConv = false;
    auto clusterStrategy = clustering.get<std::string>();
    if (enableChannelMajorConv && op.supportsCMConv()) isCMConv = true;
    auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

    size_t outChannels = op.outputSlots() ? op.getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] : 0;
    size_t alignedFullChannels = mv::round_up(outChannels, 16);
    size_t alignedSplittedChannels = mv::round_up(alignedFullChannels / streamConfig["K"], 16);

    if (clusterStrategy == "SplitOverK") {
        alignedSplittedChannels = mv::round_up(alignedSplittedChannels / totalClusters, 16);
    }

    if (opType == "Conv" || opType == "DepthwiseConv") {
        weightTableSize = 16 * alignedSplittedChannels;
        if (opType == "Conv") {
            weightSize += alignedWeightsSize(op.getInputTensor(1), {1, 1, 1, streamConfig["K"], 1}, clusterStrategy, nClusters);

        } else {
            weightSize += realTensorSize(op.getInputTensor(1), {1, 1, streamConfig["C"], 1, 1}, isCMConv);
            if (clusterStrategy == "SplitOverK")
                weightSize = div(weightSize, totalClusters);
        }

    } else if (opType == "MaxPool") {
        weightTableSize = 16 * alignedSplittedChannels;
        weightSize = 0;
    } else if (opType == "Eltwise" && !software) {
        weightTableSize = 0;
        weightSize = 0;
    }

    if (weightsSparsity) {
        // Alignment due to output/input channels mult of 16 requirement
        auto tensorSize = op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] *
                          op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] *
                          mv::round_up(op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS], 16) *
                          alignedSplittedChannels;

        // Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseWeightSize = std::ceil((double)tensorSize / 8);
        // Sparse pointers taken into account in weight table ...
        sparseWeightSize = mv::round_up(sparseWeightSize, 16);
        weightSize += sparseWeightSize;
    }

    weightSize += weightTableSize;

    return weightSize;
}

void assignNewSrategies(mv::ComputationModel& model, std::vector<mv::Element>& newStrategies,
                        std::shared_ptr<mv::Element> globalParams, size_t optimalNumberOfKStreams = 0) {
    mv::OpModel om(model);
    mv::DataModel dm(model);

    std::vector<mv::Element> strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    for (auto layerNameStrategy : strategyList) {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        if (nodeName != "Example") {
            auto opIt = om.getOp(nodeName);

            auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
            auto compDesc = model.getGlobalConfigParams();
            auto streamingStrategyList = compDesc->get<std::vector<mv::Element>>("streaming_strategy");

            auto newElement = streamingStrategyList[0];
            auto newName = newElement.get<std::string>("name_filter");
            auto newSplits = newElement.get<std::vector<mv::Element>>("splits");
            for (int i = newSplits.size(); i < 5; i++) newSplits.push_back(newSplits[0]);

            if (opIt->hasAttr("optimalNumberOfKStreams")) {
                newElement.set("name_filter", opIt->getName());
                newSplits[0].set<int>("W", streaming_strategy[0].get<int>("W"));
                newSplits[1].set<int>("H", streaming_strategy[1].get<int>("H"));
                newSplits[2].set<int>("C", streaming_strategy[2].get<int>("C"));
                newSplits[3].set<int>("K", opIt->get<unsigned>("optimalNumberOfKStreams"));
                newSplits[4].set<int>("N", streaming_strategy[4].get<int>("N"));
                newElement.set("splits", newSplits);
                newStrategies.push_back(newElement);
            } else {
                newElement.set("name_filter", opIt->getName());
                newSplits[0].set<int>("W", streaming_strategy[0].get<int>("W"));
                newSplits[1].set<int>("H", streaming_strategy[1].get<int>("H"));
                newSplits[2].set<int>("C", streaming_strategy[2].get<int>("C"));
                newSplits[3].set<int>("K", streaming_strategy[3].get<int>("K"));
                newSplits[4].set<int>("N", streaming_strategy[4].get<int>("N"));
                newElement.set("splits", newSplits);
                newStrategies.push_back(newElement);
            }
        }
    }
}

std::unordered_map<std::string,StrategySet> getStrategies(mv::Element& passDesc)
{
    std::unordered_map<std::string,StrategySet> layerStrategies;
    auto config = passDesc.get<mv::Element>("ValidateStreamingOverKForChainPipelining");
    auto layerStrategySets  = config.get<std::vector<mv::Element>>("layerStrategies");

    for( auto layerStrategySet : layerStrategySets)
    {
        auto layerName = layerStrategySet.getName();
        auto strategySets = layerStrategySet.get<std::vector<mv::Element>>("strategies");

        for(auto strategySet : strategySets)
        {
            auto strategySetName = strategySet.getName();
            auto strategies = strategySet.get<std::vector<std::string>>("value");

            auto strategyValue = strategySet.get("value");
            layerStrategies[layerName][strategySetName] = strategyValue;

        }
    }
    return layerStrategies;
}

std::vector<mv::Attribute> getStrategiesfromCompilationDescriptor(std::unordered_map<std::string,StrategySet>& strategies, mv::Data::OpListIterator opIt, std::string strategy)
{
    std::vector<mv::Attribute> attr;
    auto layerEntry = strategies.find(opIt->getOpType());
    auto& layerCfg = layerEntry->second;
    auto strategyEntry = layerCfg.find(strategy);

    if(strategyEntry == layerCfg.end())
        throw std::runtime_error("No " + strategy + " strategy specified in the compilation descriptor for this pass");
    

    for (auto elem : strategyEntry->second.get<std::vector<std::string>>())
    {
        attr.push_back(elem);

    }

    return attr;
}

std::map<size_t, size_t> getMinWeightsPerClusterSizePerChain(std::list<subgraph_t>& chainSubgraphs, const mv::pass::PassEntry& pass,
                                         mv::ComputationModel& model, std::map<std::string, size_t>& weightsPerClusterPerOp, FILE *fptr=stdout) {
    mv::OpModel om(model);
    typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
    typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    std::vector<mv::Element> streamingStrategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    auto multiClusterStrategyList = globalParams->get<std::vector<mv::Element>>("split_strategy");
    std::map<size_t, size_t> minWeightsPerClusterPerChain;
    unsigned chainID = 0;
    std::string clustering;
    std::vector<size_t> streamsSizes;  // Store streamsSizes
    size_t nClusters = globalParams->get<int>("Number_of_Clusters");
    bool clusteringStrategyFound = false;
    bool streamingStrategyFound = false;
    size_t weightsPerCluster = 0;
    bool weightsSparsity = false;
    bool isHStreaming = false;
    bool isKStreaming = false;
    

    //Header for the excel file
    fprintf(fptr,  "%s :  %s :  %s :  %s :  %s :  %s :  %s :  %s", "chainId", "OpName", "kStreaming", "Hstreaming", "MultiCluster", "TotalSize(Inc WT)", "OutputChannels", "WeightsPerCluster(Inc WT)");
    fprintf(fptr, "\n");

    for (subgraph_t chain_subgraph : chainSubgraphs) {
        streamsSizes.clear();  // clear stream sizes for chain i

        // For each operation in chain[i]
        for (auto& op : chain_subgraph.dpu_chain_) {

            mv::Data::OpListIterator opIt = om.getOp(op->getName());

            // If its a conv
            if (opIt->getOpType() == "Conv") 
            {
                // Get the strategy for this conv
                for (auto layerNameStrategy : streamingStrategyList) {
                    std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

                    if (nodeName == op->getName()) {
                        
                        //Get the streaming strategy from graph optimizer
                        auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
                        isKStreaming = streaming_strategy[3].get<int>("K") > 1 ? true : false;
                        isHStreaming = streaming_strategy[1].get<int>("H") > 1 ? true : false;

                        
                        // Get the MC strategy from graph optimizer
                        std::string mcStrategy;
                        for (auto s : multiClusterStrategyList) {
                            std::string& name_filter = s.get<std::string>("name_filter");
                            std::regex exp(name_filter);
                            if (std::regex_match(opIt->getName(), exp))
                                mcStrategy = s.get<std::string>("strategy");
                        }

                        // The operation must be already assigned stream over K and SOK and not be sream over H to be considered for a new K stream strategy
                        if (isKStreaming && mcStrategy == "SplitOverK" && !isHStreaming) {

                            if (op->hasAttr("splitStrategy"))
                                clustering = op->get<std::string>("splitStrategy");

                            weightsSparsity = false;
                            if (op->hasAttr("weightsSparsity"))
                                weightsSparsity = op->get<bool>("weightsSparsity");

                            // get the memory size of the streams weights
                            weightsPerCluster = 0;
                            mv::Data::OpListIterator oitr = om.getOp(op->getName());

                            weightsPerCluster =
                                    perClusterWeightsSize(*oitr, clustering,
                                               weightsSparsity,
                                               {1, (unsigned int)streaming_strategy[1].get<int>("H"),
                                                (unsigned int)streaming_strategy[2].get<int>("C"),
                                                (unsigned int)streaming_strategy[3].get<int>("K"),
                                                (unsigned int)streaming_strategy[4].get<int>("N")},
                                                nClusters
                                              );
                            
                            streamsSizes.push_back(weightsPerCluster);
                            
                            weightsPerClusterPerOp.insert({opIt->getName(),weightsPerCluster});
                            
                            size_t alignedFullOutputChannels = mv::round_up( opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);

                            fprintf(fptr, "%zu : %s :  %zu : %zu  : %s : %zu : %zu : %zu ", chainID,(opIt->getName()).c_str(),  streaming_strategy[3].get<int>("K"),  
                            streaming_strategy[1].get<int>("H"), mcStrategy.c_str(),weightsPerCluster*nClusters*streaming_strategy[3].get<int>("K"),
                            alignedFullOutputChannels, weightsPerCluster);
                            fprintf(fptr, "\n");
                        }
                    }
                }
            }
        }

        //Store the minimum weights per cluster for the chain
        std::sort(streamsSizes.begin(),streamsSizes.end());
        if(!streamsSizes.empty())
            minWeightsPerClusterPerChain.insert({chainID, streamsSizes[0]});
        

        chainID++;
    }
    fprintf(fptr, "End of network analysis\n");
    fprintf(fptr, "\n");

    return minWeightsPerClusterPerChain;
}

void createHeaderforReportFile(FILE *fptr=stdout)
{
    fprintf(fptr,  "%s :  %s :  %s :  %s :  %s :  %s :  %s :  %s :  %s : %s :  %s :  %s", "chainId", "OpName", "Default kStreaming", "Default Hstreaming", "MultiCluster", "TotalSize(Inc WT)", "OutputChannels", "WeightsPerCluster(Inc WT)", "MinWeightsPerClusterInChain", "optimalNumberOfKStreams","maxNumberKStreams", "NewKStreams");
    fprintf(fptr, "\n");
}

void printInfoToFile(unsigned chainID, std::string opName, int kStreaming, int hStreaming,
                     std::string multiclusterStrategy, size_t fullweightsSize, size_t alignedFullOutputChannels,
                     size_t weightsPerClusterPerOp, size_t minWeightsPerClusterPerChain, double optimalNumberOfKStreams,
                     double maxpossibleStreams, double newKStreams, FILE* fptr = stdout) {
    fprintf(fptr,
            "%zu : %s :  %zu : %zu  : %s : %zu : %zu : %zu : %zu : %.1f : %.1f : "
            "%.1f ",
            chainID, opName.c_str(), kStreaming, hStreaming, multiclusterStrategy.c_str(), fullweightsSize,
            alignedFullOutputChannels, weightsPerClusterPerOp, minWeightsPerClusterPerChain, optimalNumberOfKStreams,
            maxpossibleStreams, newKStreams);
    fprintf(fptr, "\n");
}

 // Get the strategy assigned by GO for this conv
std::tuple<std::vector<mv::Element>, mv::Attribute, bool> getGraphOptimizerAssignedStategies(
        std::vector<mv::Element>& streamingStrategyList, std::vector<mv::Element>& multiClusterStrategyList,
        std::vector<mv::Element>& tensorMemoryLocation, mv::ComputationModel& model, std::string opName) {

    mv::OpModel om(model);
    mv::Data::OpListIterator opIt;
    std::vector<mv::Element> streaming_strategy;
    std::string mcStrategy;
    std::string memoryLocation;
    mv::Attribute multiClusterStrategy;

    std::pair<std::vector<mv::Element>, std::string> graphOptimizerAssignedStategies;
    bool spilling = false;

    for (auto layerNameStrategy : streamingStrategyList) {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

        if (nodeName == opName) {
            opIt = om.getOp(nodeName);

            // Get the streaming strategy assigned by graph optimizer
            streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");

            // Get the MC strategy assigned by graph optimizer
            for (auto s : multiClusterStrategyList) {
                std::string& name_filter = s.get<std::string>("name_filter");
                std::regex exp(name_filter);
                if (std::regex_match(opIt->getName(), exp))
                {
                    mcStrategy = s.get<std::string>("strategy");
                    multiClusterStrategy = mcStrategy;
                }
            }

            for (auto s : tensorMemoryLocation) {
                std::string& name_filter = s.get<std::string>("name_filter");
                std::regex exp(name_filter);
                if (std::regex_match(opIt->getName(), exp))
                    memoryLocation = s.get<std::string>("mem_location");
                
                if(memoryLocation == "CMX")
                    spilling = false;
                else if(memoryLocation == "DDR")
                    spilling = true;
            }
            break;
        }
    }
    return std::tuple<std::vector<mv::Element>, mv::Attribute, bool> (streaming_strategy,multiClusterStrategy,spilling);
}

std::pair<size_t, double> fullWeightsSizeForOpandOptimalKStreaming(std::string multiclusterStrategy, size_t weightsPerClusterforOp, size_t minWeightsPerClusterPerChain, bool isKStreaming, int numberOfkStreams, int nClusters) 
{
    size_t fullWeightsSize = 0;
    size_t optimalNumberOfKStreams =0;
    std::pair<size_t, double> toReturn;
    size_t minWeightsPerClusterPerChainConstant = 66560; // This value was derived from emperical testing 

    // Calculate the optimal number of K streams
    // First calculate the full weight size
    // Then divide by the minStreamSize * nclusters to get the optimal K streams
    if (isKStreaming && multiclusterStrategy == "SplitOverK") {

        fullWeightsSize = weightsPerClusterforOp * nClusters * numberOfkStreams;

        if(minWeightsPerClusterPerChain <= minWeightsPerClusterPerChainConstant)
            minWeightsPerClusterPerChain = minWeightsPerClusterPerChainConstant;

        optimalNumberOfKStreams = std::round(fullWeightsSize / (minWeightsPerClusterPerChain * nClusters));
       
    } 
    else if (isKStreaming && multiclusterStrategy == "Clustering")
     {
        fullWeightsSize = weightsPerClusterforOp * numberOfkStreams;

        if(minWeightsPerClusterPerChain <= minWeightsPerClusterPerChainConstant)
            minWeightsPerClusterPerChain = minWeightsPerClusterPerChainConstant;

        optimalNumberOfKStreams = std::round(fullWeightsSize / minWeightsPerClusterPerChain);
        
    } 
    else if (multiclusterStrategy == "SplitOverK" && !isKStreaming) 
    {
        fullWeightsSize = weightsPerClusterforOp * nClusters;

        if(minWeightsPerClusterPerChain <= minWeightsPerClusterPerChainConstant)
            minWeightsPerClusterPerChain = minWeightsPerClusterPerChainConstant;
            
        optimalNumberOfKStreams = std::round(fullWeightsSize / (minWeightsPerClusterPerChain * nClusters));
        
    }

    // Ensure K streams is never 0
    if (optimalNumberOfKStreams < 1)
        optimalNumberOfKStreams = 1;
    
    toReturn.first = fullWeightsSize;
    toReturn.second = optimalNumberOfKStreams;

    return toReturn;
}

auto checkIfStrategyIsInCompilationDescriptor = [](std::vector<mv::Attribute>& vec, const std::string& str) -> bool {
    for (const auto elem : vec)
        if (str == elem.get<std::string>())
            return true;
    return false;
};

void evaluateAndAssignStrategies(std::list<subgraph_t>& chainSubgraphs, const mv::pass::PassEntry& pass,
                                 mv::ComputationModel& model, std::map<size_t, size_t>& minWeightsPerClusterPerChain,
                                 std::map<std::string, size_t>& weightsPerClusterPerOp, FILE* fptr = stdout) {
    unsigned chainID = 0;
    mv::OpModel om(model);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    size_t nClusters = globalParams->get<int>("Number_of_Clusters");
    std::vector<mv::Element> strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    std::vector<mv::Element> streamingStrategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    std::vector<mv::Element> multiClusterStrategyList = globalParams->get<std::vector<mv::Element>>("split_strategy");
    std::vector<mv::Element> tensorMemoryLocation =
            globalParams->get<std::vector<mv::Element>>("tensor_placement_override");
    std::shared_ptr<mv::Element> compDesc = model.getGlobalConfigParams();
    std::map<std::string, size_t> minOutputChannels = {{"SplitOverK", 64},
                                                       {"Clustering", 16},
                                                       {"SplitOverH", 16},
                                                       {"HKSwitch", 16}};

    std::string clustering;
    size_t fullweightsSize = 0;
    double maxpossibleStreams = 0.0;
    double optimalNumberOfKStreams = 0;
    std::vector<mv::Element> graphOptimizerStreamingStrategy;
    std::vector<mv::Element> overWrittenStreamingStrategies;
    std::vector<mv::Element> allStreamingStrategies;
    bool clusteringStrategyFoundinCompilationDescriptor = false;
    bool streamingStrategyFoundinCompilationDescriptor = false;
    std::pair<size_t, double> fullWeightsSizeOptimalKStreaming = {};
    std::size_t minStreamSize = 0;
    size_t alignedFullOutputChannels = 0;
    size_t weightsPerCluster = 0;
    size_t fullWeightsSize = 0;
    size_t minWeightsPerClusterPerChainConstant = 66560;  // This value was derived from emperical testing
    mv::Attribute graphOptimizerMultiClusterStrategy;
    bool graphOptimizerTensorLocationSpilling;

    // create header for report file
    createHeaderforReportFile(fptr);

    for (subgraph_t chain_subgraph : chainSubgraphs) {
        for (auto& op : chain_subgraph.dpu_chain_) {
            mv::Data::OpListIterator opIt = om.getOp(op->getName());

            // We only stream Convs more
            if (opIt->getOpType() == "Conv") {
                optimalNumberOfKStreams = 0;

                // Get the strategy assigned by GO for this operation
                auto graphOptimizerAssignedStategies = getGraphOptimizerAssignedStategies(
                        streamingStrategyList, multiClusterStrategyList, tensorMemoryLocation, model, opIt->getName());

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
                maxpossibleStreams =
                        floor(alignedFullOutputChannels / minOutputChannels[graphOptimizerMultiClusterStrategy.get<std::string>()]);
                // Get the weights per cluster for this op
                weightsPerCluster = weightsPerClusterPerOp.find(opIt->getName())->second;

                // The operation must be already assigned stream over K and SOK and not be sream over H to be considered
                // for a new K stream strategy
                if (isKStreaming && graphOptimizerMultiClusterStrategy.get<std::string>() == "SplitOverK" && !isHStreaming) {
                    
                    fullWeightsSizeOptimalKStreaming = {0,0};
                    if(minWeightsPerClusterPerChain[chainID] > 0)
                        fullWeightsSizeOptimalKStreaming = fullWeightsSizeForOpandOptimalKStreaming(graphOptimizerMultiClusterStrategy.get<std::string>(), weightsPerCluster, minWeightsPerClusterPerChain[chainID], isKStreaming,graphOptimizerStreamingStrategy[3].get<int>("K"), nClusters);
                    
                    fullWeightsSize = fullWeightsSizeOptimalKStreaming.first;
                    optimalNumberOfKStreams = fullWeightsSizeOptimalKStreaming.second;
                    
                    if(!validateKStream(*opIt,graphOptimizerMultiClusterStrategy.get<std::string>(),fullWeightsSizeOptimalKStreaming.second,graphOptimizerTensorLocationSpilling, nClusters))
                        continue;

                    // Assign the new streaming strategies
                    // The optimalNumberOfKStreams must be > 0, less than the max possible K streams and must not decrease the K streams assinged from the GO
                     if ((optimalNumberOfKStreams > 0) && (optimalNumberOfKStreams <= maxpossibleStreams) && (optimalNumberOfKStreams > graphOptimizerStreamingStrategy[3].get<int>("K"))) {

                        if(minWeightsPerClusterPerChain[chainID] < minWeightsPerClusterPerChainConstant)
                            minWeightsPerClusterPerChain[chainID] = minWeightsPerClusterPerChainConstant;
                        
                         printInfoToFile(chainID, (opIt->getName()).c_str(), graphOptimizerStreamingStrategy[3].get<int>("K"),
                                         graphOptimizerStreamingStrategy[1].get<int>("H"), graphOptimizerMultiClusterStrategy.get<std::string>().c_str(),
                                         fullWeightsSize, alignedFullOutputChannels,
                                         weightsPerClusterPerOp.find(opIt->getName())->second,
                                         minWeightsPerClusterPerChain[chainID], optimalNumberOfKStreams,
                                         maxpossibleStreams, optimalNumberOfKStreams, fptr);

                         opIt->set<unsigned>("optimalNumberOfKStreams", optimalNumberOfKStreams);

                     } 
                     // Else assign the max possible K streams for the layer
                     else if (optimalNumberOfKStreams > maxpossibleStreams) {

                        if(minWeightsPerClusterPerChain[chainID] < minWeightsPerClusterPerChainConstant)
                            minWeightsPerClusterPerChain[chainID] = minWeightsPerClusterPerChainConstant;
                         printInfoToFile(chainID, (opIt->getName()).c_str(), graphOptimizerStreamingStrategy[3].get<int>("K"),
                                         graphOptimizerStreamingStrategy[1].get<int>("H"), graphOptimizerMultiClusterStrategy.get<std::string>().c_str(),
                                         fullWeightsSize, alignedFullOutputChannels,
                                         weightsPerClusterPerOp.find(opIt->getName())->second,
                                          minWeightsPerClusterPerChain[chainID], optimalNumberOfKStreams,
                                         maxpossibleStreams, maxpossibleStreams, fptr);
                         opIt->set<unsigned>("optimalNumberOfKStreams", maxpossibleStreams);
                     }
                 }
             }
         }
         chainID++;
     }
 }

 void validateStreamingOverKForChainPipeliningFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                                             mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&) {
     mv::OpModel om(model);
     typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
     typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;
     std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
     std::vector<mv::Element> strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
     auto multiClusterStrategyList = globalParams->get<std::vector<mv::Element>>("split_strategy");
     std::vector<mv::Element> newStreamingStrategies;
     std::map<std::string, size_t> weightsPerClusterPerOp;
     FILE* network_report_fptr = fopen("validateStreamingOverKForChainPipelining.txt", "w");

    if (!network_report_fptr)
        throw std::runtime_error("Cannot open validateStreamingOverKForChainPipelining.txt for write");
     
     pipeline_chains_t pipeliner(om);
     size_t pipeline_stages = 0UL;
     if (passDesc.hasAttr("select_stages")) {
         pipeline_stages = (size_t)passDesc.get<int>("select_stages");
     }

     // Step 1: Get the subgraph chains
     auto chainSubgraphs = pipeliner.get_chain_subgraphs(pipeline_stages);

     // Step 2: Get the min weights per cluster in a chain
     auto minWeightsPerClusterPerChain = getMinWeightsPerClusterSizePerChain(
             chainSubgraphs, pass, model, weightsPerClusterPerOp, network_report_fptr);

     // Step 3: Calculate more optimal streaming over K strategies
     if(!minWeightsPerClusterPerChain.empty())
     {
        evaluateAndAssignStrategies(chainSubgraphs, pass, model, minWeightsPerClusterPerChain,
                                 weightsPerClusterPerOp, network_report_fptr);

        // Step4: Assign the new strategies
        assignNewSrategies(model, newStreamingStrategies, globalParams);

        // Step5: Save the new strategies
        globalParams->set("streaming_strategy", newStreamingStrategies);
        saveNewStreamingStrategiesToJson(pass, newStreamingStrategies, "Streaming_activations_and_weights_performance_strategies");
     }
 }
