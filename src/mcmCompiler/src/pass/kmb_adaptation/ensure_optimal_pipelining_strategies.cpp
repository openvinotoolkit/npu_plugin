#include "include/mcm/base/attribute.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
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

void addOptimalChainPipeliningStrategiesFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(AddOptimalChainPipeliningStrategies)
        .setFunc(addOptimalChainPipeliningStrategiesFnc)
        .setDescription(
            "This pass re-calculates the optimum number of streams over K for pipelining"
        );

        
    }

   
}

size_t alignedWeightsSize(
    const mv::Data::TensorIterator tensorToSize, const mv::Shape& streamConfig, std::string clustering) {
    int totalClusters = 4;
    auto div = [](unsigned x, unsigned y) -> unsigned {
        return (x + y - 1) / y;
    };
    auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits() / 8.0);
    size_t alignedFullInputChannels = mv::round_up(tensorToSize->getShape()[mv::KERNEL_INPUT_CHANNELS], 16);

    size_t alignedFullOutputChannels = mv::round_up(tensorToSize->getShape()[mv::KERNEL_OUTPUT_CHANNELS], 16);
    size_t alignedStreamedOutputChannels = mv::round_up(alignedFullOutputChannels / streamConfig["K"], 16);

    if (clustering == "SplitOverK") {
        size_t alignedSplittedOutputChannels = div(alignedStreamedOutputChannels, totalClusters);
        alignedSplittedOutputChannels = mv::round_up(alignedSplittedOutputChannels, 16);

        return (alignedFullInputChannels * alignedSplittedOutputChannels * tensorToSize->getShape()[mv::KERNEL_WIDTH] *
                   tensorToSize->getShape()[mv::KERNEL_HEIGHT]) *
               dtypeMultiplier;
    } else {
        return (alignedFullInputChannels * alignedStreamedOutputChannels * tensorToSize->getShape()[mv::KERNEL_WIDTH] *
                   tensorToSize->getShape()[mv::KERNEL_HEIGHT]) *
               dtypeMultiplier;
    }
}

std::vector<std::size_t> tileSpatialOutputSize(std::size_t outputSize, std::size_t numberOfSplits) {
    // aim is to get the splits such that the last split is smallest and rest of the splits are equal
    int newOutputSize = ceil((double)(outputSize) / (double)numberOfSplits);
    int remainderOutputSize = outputSize - (newOutputSize * (numberOfSplits - 1));
    if (remainderOutputSize <= 0) {
        newOutputSize = trunc((double)(outputSize) / (double)numberOfSplits);
        remainderOutputSize = outputSize - (newOutputSize * (numberOfSplits - 1));
    }
    std::vector<std::size_t> outputSizes(numberOfSplits, newOutputSize);

    outputSizes[numberOfSplits - 1] = remainderOutputSize;
    return outputSizes;
}

std::size_t realTensorSize(const mv::Data::TensorIterator tensorToSize, const mv::Shape& streamingPool, bool isCMConv) {
    mv::Shape worstStreamPool = streamingPool;

    // TODO harmonize this, for now only consider worst shape for nested streams
    if (streamingPool["H"] > 1 && streamingPool["K"] > 1) {
        mv::Shape tensorShape = tensorToSize->getShape();
        // update the streamingPool to the worst combination, based on slice sizes
        auto outputSize = tensorShape[mv::IO_HEIGHT_DIMENSION];
        auto numberOfSplits = streamingPool[mv::IO_HEIGHT_DIMENSION];

        auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
        auto newOutputSize = newOutputSizes.front();

        auto worstNumberOfSplits = outputSize / newOutputSize;
        worstStreamPool[mv::IO_HEIGHT_DIMENSION] = worstNumberOfSplits;
    }

    // TODO add handling for weights case if we dont align it to 16 always
    std::size_t streamDivisor = 1;
    for (std::size_t dim = 0; dim < worstStreamPool.ndims(); ++dim) {
        streamDivisor = streamDivisor * worstStreamPool[dim];
    }

    if (isCMConv) return tensorToSize->computeTotalSize(16, false, false, false) / streamDivisor;

    return tensorToSize->computeTotalSize(16, false, false, true) / streamDivisor;
}

size_t memorySize(mv::Op& op, const mv::Attribute& clustering, bool weightsSparsity, const mv::Shape& streamConfig) {
    
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
            weightSize += alignedWeightsSize(op.getInputTensor(1), {1, 1, 1, streamConfig["K"], 1}, clusterStrategy);

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
    std::vector<mv::Element>& overWrittenStreamingStrategies, std::shared_ptr<mv::Element> globalParams,
    size_t optimalNumberOfKStreams = 0) {
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
                std::cout << "Changing " << opIt->getName() << " from " << streaming_strategy[3].get<int>("K") << " to " << opIt->get<unsigned>("optimalNumberOfKStreams") << " streams " << std::endl;
                newElement.set("name_filter", opIt->getName());
                newSplits[0].set<int>("W", streaming_strategy[0].get<int>("W"));
                newSplits[1].set<int>("H", streaming_strategy[1].get<int>("H"));
                newSplits[2].set<int>("C", streaming_strategy[2].get<int>("C"));
                newSplits[3].set<int>("K", opIt->get<unsigned>("optimalNumberOfKStreams"));
                newSplits[4].set<int>("N", streaming_strategy[4].get<int>("N"));
                newElement.set("splits", newSplits);
                newStrategies.push_back(newElement);
                overWrittenStreamingStrategies.push_back(newElement);
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

void saveNewStreamingStrategiesToJson(const mv::pass::PassEntry& pass, const mv::Attribute& streamingStrategyElements) {
    pass.log(mv::Logger::MessageType::Debug, "Saving New Streaming Strategies to JSON file");
    std::ofstream jsonOutputFile;
    std::string jsonOutFileName = "./output/mcmCompiler_new_streaming_strategy_output.json";
    jsonOutputFile.open(jsonOutFileName, std::ios::out);
    if (!(jsonOutputFile.is_open()))
        pass.log(mv::Logger::MessageType::Debug, "AddOptimalChainPipeliningStrategies Could not open output file " + jsonOutFileName);

    auto currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string timeStamp(ctime(&currentTime));
    if (!timeStamp.empty() && timeStamp[timeStamp.length() - 1] == '\n') timeStamp.erase(timeStamp.length() - 1);

    mv::Element SSA("Streaming strategies generated by mcmCompiler " + timeStamp);
    SSA.set("streaming_strategy", streamingStrategyElements);
    auto jsonSStrategy = SSA.toJSON(true);
  
    jsonOutputFile << jsonSStrategy.stringifyPretty() << "," << std::endl;
    jsonOutputFile.close();
}

std::unordered_map<std::string,StrategySet> getStrategies(mv::Element& passDesc)
{
    std::unordered_map<std::string,StrategySet> layerStrategies;
    auto config = passDesc.get<mv::Element>("AddOptimalChainPipeliningStrategiesConfig");
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

    for (auto elem : strategyEntry->second.get<std::vector<std::string>>())
    {
        attr.push_back(elem);
    }

    return attr;
}

std::map<size_t, size_t> getMinWeightsPerClusterSizePerChain(std::list<subgraph_t>& chainSubgraphs, const mv::pass::PassEntry& pass,
                                         mv::ComputationModel& model, std::unordered_map<std::string,StrategySet>& strategies, std::map<std::string, size_t>& weightsPerClusterPerOp, FILE *fptr=stdout) {
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
    

    //Header for the excel file
    fprintf(fptr,  "%s :  %s :  %s :  %s :  %s :  %s :  %s :  %s", "chainId", "OpName", "kStreaming", "Hstreaming", "MultiCluster", "TotalSize(Inc WT)", "OutputChannels", "WeightsPerCluster(Inc WT)");
    fprintf(fptr, "\n");

    for (subgraph_t chain_subgraph : chainSubgraphs) {
        streamsSizes.clear();  // clear stream sizes for chain i

        // For each operation in chain[i]
        for (auto& op : chain_subgraph.dpu_chain_) {

            pass.log(mv::Logger::MessageType::Debug,
                     "Process Op " + op->getName() + " in chain " + std::to_string(chainID));
            mv::Data::OpListIterator opIt = om.getOp(op->getName());

            // If its a conv
            if (opIt->getOpType() == "Conv") 
            {
                auto streamingStrategies = getStrategiesfromCompilationDescriptor(strategies,opIt,"streamingStrategies");
                auto clusteringStrategies = getStrategiesfromCompilationDescriptor(strategies,opIt,"clusteringStrategies");

                // Get the strategy for this conv
                for (auto layerNameStrategy : streamingStrategyList) {
                    std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

                    if (nodeName == op->getName()) {
                        
                        //Get the streaming strategy from graph optimizer
                        auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
                        bool isKStreaming = streaming_strategy[3].get<int>("K") > 1 ? true : false;
                        isHStreaming = streaming_strategy[1].get<int>("H") > 1 ? true : false;

                        
                        // Get the MC strategy from graph optimizer
                        std::string mcStrategy;
                        for (auto s : multiClusterStrategyList) {
                            std::string& name_filter = s.get<std::string>("name_filter");
                            std::regex exp(name_filter);
                            if (std::regex_match(opIt->getName(), exp))
                                mcStrategy = s.get<std::string>("strategy");
                        }

                        clusteringStrategyFound = false;
                        auto findStrategy = [](std::vector<mv::Attribute>& vec,const std::string& str) ->bool { for(const auto elem : vec) if(str==elem.get<std::string>()) return true; return false;};
                        clusteringStrategyFound = findStrategy(clusteringStrategies,mcStrategy);

                        streamingStrategyFound = false;
                        if(isKStreaming)
                            streamingStrategyFound = findStrategy(streamingStrategies,"StreamOverK");
                        if(isHStreaming)
                            streamingStrategyFound = findStrategy(streamingStrategies,"StreamOverH");

                        // If the layer has a strategy that is also in the CD then include it in the analysis
                        // Just SOK 

                        if (clusteringStrategyFound && !isHStreaming) {
                        // SOK and streaming
                        //if (streamingStrategyFound && clusteringStrategyFound) {

                            if (op->hasAttr("splitStrategy"))
                                clustering = op->get<std::string>("splitStrategy");

                            weightsSparsity = false;
                            if (op->hasAttr("weightsSparsity"))
                                weightsSparsity = op->get<bool>("weightsSparsity");

                            // get the memory size of the streams weights
                            weightsPerCluster = 0;
                            mv::Data::OpListIterator oitr = om.getOp(op->getName());

                            weightsPerCluster =
                                    memorySize(*oitr, clustering,
                                               weightsSparsity,
                                               {1, (unsigned int)streaming_strategy[1].get<int>("H"),
                                                (unsigned int)streaming_strategy[2].get<int>("C"),
                                                (unsigned int)streaming_strategy[3].get<int>("K"),
                                                (unsigned int)streaming_strategy[4].get<int>("N")}
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

        //Only store the min stream sizes if layer strategies match those in the CD
        //Just SOK
        if (clusteringStrategyFound && !isHStreaming) {
        // SOK and streaming
        //if (streamingStrategyFound && clusteringStrategyFound) {
            // Store the min weights per chain
            std::sort(streamsSizes.begin(), streamsSizes.end());
            streamsSizes.erase(unique(streamsSizes.begin(), streamsSizes.end()), streamsSizes.end());

            minWeightsPerClusterPerChain.insert({chainID, streamsSizes[0]});
        }
        chainID++;
    }
    fprintf(fptr, "End of network analysis\n");
    fprintf(fptr, "\n");

    if(minWeightsPerClusterPerChain.empty())
        std::runtime_error("No layers with strategies found matching strategies in the CD");
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
std::pair<std::vector<mv::Element>, std::string> getGraphOptimizerAssignedStategies(
        std::vector<mv::Element>& streamingStrategyList, std::vector<mv::Element>& multiClusterStrategyList,
        mv::ComputationModel& model, std::string opName) {

    mv::OpModel om(model);
    mv::Data::OpListIterator opIt;
    std::vector<mv::Element> streaming_strategy;
    std::string mcStrategy;
    std::pair<std::vector<mv::Element>, std::string> graphOptimizerAssignedStategies;

    for (auto layerNameStrategy : streamingStrategyList) {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

        if (nodeName == opName) {
            opIt = om.getOp(nodeName);

            // Get the streaming strategy assigned by graph optimizer
            streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
            graphOptimizerAssignedStategies.first = streaming_strategy;

            // Get the MC strategy assigned by graph optimizer
            for (auto s : multiClusterStrategyList) {
                std::string& name_filter = s.get<std::string>("name_filter");
                std::regex exp(name_filter);
                if (std::regex_match(opIt->getName(), exp))
                    mcStrategy = s.get<std::string>("strategy");
                graphOptimizerAssignedStategies.second = mcStrategy;
            }
            break;
        }
    }
    return graphOptimizerAssignedStategies;
}

std::pair<size_t, double> fullWeightsSizeForOpandOptimalKStreaming(std::string multiclusterStrategy, size_t weightsPerClusterforOp, size_t minWeightsPerClusterPerChain, bool isKStreaming, int numberOfkStreams, int nClusters) 
{
    size_t fullWeightsSize = 0;
    size_t optimalNumberOfKStreams =0;
    std::pair<size_t, double> toReturn;
    size_t minWeightsPerClusterPerChainoverwrite = 34816; //OVERWRITE

    if (isKStreaming && multiclusterStrategy == "SplitOverK") {

        fullWeightsSize = weightsPerClusterforOp * nClusters * numberOfkStreams;
        optimalNumberOfKStreams = std::round(fullWeightsSize / (minWeightsPerClusterPerChainoverwrite * nClusters));
       
    } 
    else if (isKStreaming && multiclusterStrategy == "Clustering")
     {
        fullWeightsSize = weightsPerClusterforOp * numberOfkStreams;
        optimalNumberOfKStreams = std::round(fullWeightsSize / minWeightsPerClusterPerChainoverwrite);
        
    } 
    else if (multiclusterStrategy == "SplitOverK") 
    {
        fullWeightsSize = weightsPerClusterforOp * nClusters;
        optimalNumberOfKStreams = std::round(fullWeightsSize / (minWeightsPerClusterPerChainoverwrite * nClusters));
        
    }

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
                                  mv::ComputationModel& model, std::unordered_map<std::string, StrategySet>& strategies,
                                  std::map<size_t, size_t>& minWeightsPerClusterPerChain,
                                  std::map<std::string, size_t>& weightsPerClusterPerOp, FILE* fptr = stdout) {
     unsigned chainID = 0;
     mv::OpModel om(model);
     std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
     size_t nClusters = globalParams->get<int>("Number_of_Clusters");
     std::vector<mv::Element> strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
     std::vector<mv::Element> streamingStrategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
     std::vector<mv::Element> multiClusterStrategyList = globalParams->get<std::vector<mv::Element>>("split_strategy");
     std::shared_ptr<mv::Element> compDesc = model.getGlobalConfigParams();
     std::map<std::string, size_t> minOutputChannels = {{"SplitOverK", 64},
                                                        {"Clustering", 16},
                                                        {"SplitOverH", 16},
                                                        {"HKSwitch", 16}};

     std::string clustering;
     size_t fullweightsSize = 0;
     double maxpossibleStreams = 0.0;
     double optimalNumberOfKStreams = 0;
     std::vector<mv::Element> streaming_strategy;
     std::vector<mv::Element> overWrittenStreamingStrategies;
     std::vector<mv::Element> allStreamingStrategies;
     bool clusteringStrategyFoundinCompilationDescriptor = false;
     bool streamingStrategyFoundinCompilationDescriptor = false;
     std::size_t minStreamSize = 0;
     size_t alignedFullOutputChannels = 0;
     size_t weightsPerCluster = 0;
     size_t fullWeightsSize = 0;

     // Header for excel file
     createHeaderforReportFile(fptr);

     for (subgraph_t chain_subgraph : chainSubgraphs) {
         for (auto& op : chain_subgraph.dpu_chain_) {
             mv::Data::OpListIterator opIt = om.getOp(op->getName());

             // If its a conv
             if (opIt->getOpType() == "Conv") {
                 optimalNumberOfKStreams = 0;

                 // Get the strategy assigned by GO for this operation
                 auto graphOptimizerAssignedStategies = getGraphOptimizerAssignedStategies(
                         streamingStrategyList, multiClusterStrategyList, model, opIt->getName());

                 auto streaming_strategy = graphOptimizerAssignedStategies.first;
                 auto multiclusterStrategy = graphOptimizerAssignedStategies.second;

                 // Get the streaming and clustering strategies to be included in the analysis from the
                 auto streamingStrategies =
                         getStrategiesfromCompilationDescriptor(strategies, opIt, "streamingStrategies");
                 auto clusteringStrategies =
                         getStrategiesfromCompilationDescriptor(strategies, opIt, "clusteringStrategies");
                 bool isKStreaming = streaming_strategy[3].get<int>("K") > 1 ? true : false;
                 bool isHStreaming = streaming_strategy[1].get<int>("H") > 1 ? true : false;

                 // Check if the Multi-cluster strategy assigned by graph optimizer is included in the
                 // strategies in the CD
                 clusteringStrategyFoundinCompilationDescriptor = false;
                 clusteringStrategyFoundinCompilationDescriptor =
                         checkIfStrategyIsInCompilationDescriptor(clusteringStrategies, multiclusterStrategy);

                 // Check if the streaming strategy assigned by graph optimizer is included in the CD
                 streamingStrategyFoundinCompilationDescriptor = false;
                 if (isKStreaming)
                     streamingStrategyFoundinCompilationDescriptor =
                             checkIfStrategyIsInCompilationDescriptor(streamingStrategies, "StreamOverK");

                 if (isHStreaming)
                     streamingStrategyFoundinCompilationDescriptor =
                             checkIfStrategyIsInCompilationDescriptor(streamingStrategies, "StreamOverH");

                 // Get the output channels
                 alignedFullOutputChannels =
                         mv::round_up(opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);

                 // Calculate the max possible K streams based on the multi-cluster strategy
                 maxpossibleStreams = floor(alignedFullOutputChannels / minOutputChannels[multiclusterStrategy]);
                
                 // Get the weights per cluster for this op
                 weightsPerCluster = weightsPerClusterPerOp.find(opIt->getName())->second;

                 // Calculate the optimal number of K streams
                 // First calculate the full weight size
                 // Then divide by the minStreamSize * nclusters

                  //Just SOK
                  if (clusteringStrategyFoundinCompilationDescriptor && !isHStreaming) {
                  // SOK and streaming
                 //if (streamingStrategyFoundinCompilationDescriptor && clusteringStrategyFoundinCompilationDescriptor) {
                    
                    auto fullWeightsSizeOptimalKStreaming = fullWeightsSizeForOpandOptimalKStreaming(multiclusterStrategy, weightsPerCluster, minWeightsPerClusterPerChain[chainID], isKStreaming, streaming_strategy[3].get<int>("K"), nClusters);
                    fullWeightsSize = fullWeightsSizeOptimalKStreaming.first;
                    optimalNumberOfKStreams = fullWeightsSizeOptimalKStreaming.second;

                    //Assign the new streaming strategies
                     if (optimalNumberOfKStreams <= maxpossibleStreams) {
                         printInfoToFile(chainID, (opIt->getName()).c_str(), streaming_strategy[3].get<int>("K"),
                                         streaming_strategy[1].get<int>("H"), multiclusterStrategy.c_str(),
                                         fullWeightsSize, alignedFullOutputChannels,
                                         weightsPerClusterPerOp.find(opIt->getName())->second,
                                         34816, optimalNumberOfKStreams,
                                         maxpossibleStreams, optimalNumberOfKStreams, fptr);

                         opIt->set<unsigned>("optimalNumberOfKStreams", optimalNumberOfKStreams);

                     } else if (optimalNumberOfKStreams > maxpossibleStreams) {
                         printInfoToFile(chainID, (opIt->getName()).c_str(), streaming_strategy[3].get<int>("K"),
                                         streaming_strategy[1].get<int>("H"), multiclusterStrategy.c_str(),
                                         fullWeightsSize, alignedFullOutputChannels,
                                         weightsPerClusterPerOp.find(opIt->getName())->second,
                                          34816, optimalNumberOfKStreams,
                                         maxpossibleStreams, maxpossibleStreams, fptr);
                         opIt->set<unsigned>("optimalNumberOfKStreams", maxpossibleStreams);
                     }
                 }
             }
         }
         chainID++;
     }
 }

 void addOptimalChainPipeliningStrategiesFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                                             mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&) {
     mv::OpModel om(model);
     typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
     typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;
     std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
     std::vector<mv::Element> strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
     auto multiClusterStrategyList = globalParams->get<std::vector<mv::Element>>("split_strategy");
     auto compDesc = model.getGlobalConfigParams();
     std::vector<mv::Element> overWrittenStreamingStrategies;
     std::vector<mv::Element> allStreamingStrategies;
     std::map<std::string, size_t> weightsPerClusterPerOp;
     FILE* network_report_fptr = fopen("networkAnalysis.txt", "w");
     

     // Get the strategies from compialtion descriptor to be included in pass
     auto strategies = getStrategies(passDesc);

     // instantiate pipeline chains class
     pipeline_chains_t pipeliner(om);
     size_t pipeline_stages = 0UL;
     if (passDesc.hasAttr("select_stages")) {
         pipeline_stages = (size_t)passDesc.get<int>("select_stages");
     }

     // Step 1: Get the subgraph chains
     auto chainSubgraphs = pipeliner.get_chain_subgraphs(pipeline_stages);

     // Step 2: Get the min eeights per cluster in a chain
     auto minWeightsPerClusterPerChain = getMinWeightsPerClusterSizePerChain(
             chainSubgraphs, pass, model, strategies, weightsPerClusterPerOp, network_report_fptr);

     // Step 3:
     evaluateAndAssignStrategies(chainSubgraphs, pass, model, strategies, minWeightsPerClusterPerChain,
                                 weightsPerClusterPerOp, network_report_fptr);

     // Assign the new strategies
     assignNewSrategies(model, allStreamingStrategies, overWrittenStreamingStrategies, globalParams);

     compDesc->set("streaming_strategy", allStreamingStrategies);
     // saveNewStreamingStrategiesToJson(pass, overWrittenStreamingStrategies);
     saveNewStreamingStrategiesToJson(pass, allStreamingStrategies);
 }
