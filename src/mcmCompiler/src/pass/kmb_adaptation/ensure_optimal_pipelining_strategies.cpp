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

std::size_t activationTensorSize(const mv::Data::TensorIterator tensorToSize, std::string clustering,
    const mv::Shape& streamingPool, bool isCMConv, mv::Op& op, bool isInput, bool dilation = false) {
    bool enableChannelMajorConv = true;
    int totalClusters = 4;

    auto div = [](unsigned x, unsigned y) -> unsigned {
        return (x + y - 1) / y;
    };
    auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits() / 8.0);
    auto opType = op.getOpType();
    auto tensorShape = tensorToSize->getShape();
    if (dilation) tensorShape = tensorToSize->get<mv::Shape>("originalShape");
    // Note: For now, all batched operations stream over batch so that N = 1
    size_t streamedBatch = 1;

    size_t fullTensorHeight = tensorShape[mv::IO_HEIGHT_DIMENSION];
    size_t streamedHeight = fullTensorHeight;

    size_t fullTensorChannels = tensorShape[mv::IO_CHANNEL_DIMENSION];
    size_t streamedChannels = fullTensorChannels;

    if (streamingPool["H"] > 1) {
        auto newOutputSizes = tileSpatialOutputSize(fullTensorHeight, streamingPool["H"]);
        streamedHeight = newOutputSizes.front();
        if (streamedHeight < newOutputSizes.back()) streamedHeight = newOutputSizes.back();

        // Kernel and padding will add extra lines to final size of streamed portion
        size_t kHeight = 1;
        std::array<unsigned short, 4> padding;
        if ((op.getOpType() == "Conv") || (op.getOpType() == "DepthwiseConv"))
            kHeight = op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT];
        else if (op.getOpType() == "MaxPool")
            kHeight = op.get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_HEIGHT];
        if (op.hasAttr("padding"))
            padding = op.get<std::array<unsigned short, 4>>("padding");
        else
            padding = {0, 0, 0, 0};

        size_t extraLines = 0;

        if (extraLines < kHeight - 1) {
            extraLines = kHeight - 1;
        }

        if (padding[2] > padding[3]) {
            if (padding[2] > extraLines) extraLines = padding[2];
        } else {
            if (padding[3] > extraLines) extraLines = padding[3];
        }

        streamedHeight += extraLines;
    }
    if (streamingPool["C"] > 1) {
        streamedChannels = div(fullTensorChannels, streamingPool["C"]);
    }
    if (streamingPool["K"] > 1) {
        streamedChannels = div(fullTensorChannels, streamingPool["K"]);

        size_t remainderChannels = fullTensorChannels - (streamedChannels * (streamingPool["K"] - 1));
        if (remainderChannels > streamedChannels) streamedChannels = remainderChannels;

        streamedChannels = mv::round_up(streamedChannels, 16);
    }

    if (clustering == "SplitOverH") {
        streamedHeight = div(streamedHeight, totalClusters);
    }
    if ((opType == "Conv" || opType == "DepthwiseConv" || opType == "MaxPool" || opType == "Eltwise") &&
        (!isCMConv || !isInput))  // for DPU tasks we align both input (except CM) and output tensors channels
    {
        streamedChannels = mv::round_up(streamedChannels, 16);
    }

    return tensorShape[mv::IO_WIDTH_DIMENSION] * streamedHeight * streamedChannels * streamedBatch * dtypeMultiplier;
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

std::tuple<size_t, size_t, size_t> memorySize(mv::Op& op, const mv::Attribute& clustering, bool inputActivationSparsity,
    bool outputActivationSparsity, bool weightsSparsity, const mv::Shape& streamConfig, bool fakeSparsity,
    bool spilling = false, bool parentSpilling = true) {
    auto div = [](unsigned x, unsigned y) -> unsigned {
        return (x + y - 1) / y;
    };

    bool enableChannelMajorConv = true;
    int totalClusters = 4;

    size_t inputSize = 0;
    size_t outputSize = 0;
    size_t weightSize = 0;
    size_t weightTableSize = 0;
    // NOTE: here is done a trick for the sub-dilated convolutions, if you are
    // dilated on your cmx as input is the original shape tensor which is before
    // the input of the slice...
    bool dilatedLayerInputMemory = false;

    auto opType = op.getOpType();
    auto isCMConv = false;
    auto clusterStrategy = clustering.get<std::string>();

    if (enableChannelMajorConv && op.supportsCMConv()) isCMConv = true;

    if (op.hasAttr("DilatedSubConv") && (op.get<bool>("DilatedSubConv"))) dilatedLayerInputMemory = true;

    if (opType != "Input" && opType != "Concat") {
        // Note: when an operation is streaming activations, but it's parent didn't spill, the input won't be streamed
        mv::Shape temporaryStreamConfig = {
            streamConfig["W"], streamConfig["H"], streamConfig["C"], 1, streamConfig["B"]};
        if (!parentSpilling) temporaryStreamConfig = {1, 1, 1, 1, 1};
        inputSize = activationTensorSize(
            op.getInputTensor(0), clusterStrategy, temporaryStreamConfig, isCMConv, op, true, dilatedLayerInputMemory);
    }
    if (opType != "Output") {
        // NOTE: when streaming operations are not spilled, full output (not streamed size) must be counted
        // Similarly, with explicit concats. We don't call this function for ddr concats, only CMX
        mv::Shape temporaryStreamConfig = {
            streamConfig["W"], streamConfig["H"], 1, streamConfig["K"], streamConfig["B"]};
        if (!spilling) temporaryStreamConfig = {1, 1, 1, 1, 1};

        outputSize =
            activationTensorSize(op.getOutputTensor(0), clusterStrategy, temporaryStreamConfig, isCMConv, op, false);
    }

    auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

    size_t outChannels = op.outputSlots() ? op.getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] : 0;
    size_t alignedFullChannels = mv::round_up(outChannels, 16);
    size_t alignedSplittedChannels = mv::round_up(alignedFullChannels / streamConfig["K"], 16);
    if (clusterStrategy == "SplitOverK") {
        alignedSplittedChannels = mv::round_up(alignedSplittedChannels / totalClusters, 16);
    }

    if (opType == "Conv" || opType == "DepthwiseConv") {
        weightTableSize = 16 * alignedSplittedChannels;
        //std::cout << "op name is " << op.getName() << "WT size is " << weightTableSize << std::endl;
        if (opType == "Conv") {
            weightSize += alignedWeightsSize(op.getInputTensor(1), {1, 1, 1, streamConfig["K"], 1}, clusterStrategy);
            //std::cout << "op name is " << op.getName() << " weightSize size is " << weightSize << std::endl;

        } else {
            weightSize += realTensorSize(op.getInputTensor(1), {1, 1, streamConfig["C"], 1, 1}, isCMConv);
            if (clusterStrategy == "SplitOverK") weightSize = div(weightSize, totalClusters);
        }

    } else if (opType == "MaxPool") {
        weightTableSize = 16 * alignedSplittedChannels;
        weightSize = 0;
    } else if (opType == "Eltwise" && !software) {
        weightTableSize = 0;
        weightSize = 0;
        mv::Shape temporaryStreamConfig = {
            streamConfig["W"], streamConfig["H"], streamConfig["C"], 1, streamConfig["B"]};
        if (!parentSpilling) temporaryStreamConfig = {1, 1, 1, 1, 1};
        inputSize +=
            activationTensorSize(op.getInputTensor(1), clusterStrategy, temporaryStreamConfig, isCMConv, op, true);
    }

    // Additional memory footprint for sparsity
    if (fakeSparsity) {
        if (opType != "MaxPool" && opType != "DepthwiseConv" && !isCMConv) {
            // throw mv::LogicError(*this, op.getName() + ": Invalid fake Sparsity! Has to be only for MaxPool, DW or
            // CMConv!! opType is " + opType);
        }
        uint16_t kernelW, kernelH;

        auto strides = op.get<std::array<unsigned short, 2>>("stride");

        if (op.hasAttr("kSize")) {
            auto kernelShape = op.get<std::array<unsigned short, 2>>("kSize");
            kernelW = kernelShape[0];
            kernelH = kernelShape[1];
        } else {
            auto weightsShape = op.getInputTensor(1)->getShape();
            kernelW = weightsShape[mv::KERNEL_WIDTH];
            kernelH = weightsShape[mv::KERNEL_HEIGHT];
        }

        mv::DType dataType = op.getInputTensor(0)->getDType();
        if (opType != "MaxPool") dataType = op.getInputTensor(1)->getDType();

        auto windowsSize = getWindowSize(kernelW, strides[0], dataType);
        size_t fakeSparsitySize = 0;
        if ((opType == "MaxPool") || (opType == "DepthwiseConv")) {
            // inputChannels = 1
            auto bitpatternSize = windowsSize * kernelH;
            // ndims = {16 * static_cast<std::size_t>(std::ceil(bitpatternSize / 128.0)), 1, 1, 1};
            fakeSparsitySize = 16 * static_cast<std::size_t>(std::ceil(bitpatternSize / 128.0));
        }
        // Channel Major Convolution doesn't need rounding of channels
        else if (isCMConv)  // isChannelMajorConvolution
        {
            std::size_t outputChannels = op.getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
            outputChannels = outputChannels / streamConfig["K"];
            std::size_t inputChannels = op.getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

            auto windowSparsitySize =
                static_cast<std::size_t>(std::ceil(windowsSize / 8.0));  // how many bytes we need per window
            auto NumberOfRowsSparistyBytes =
                static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0));

            // ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
            fakeSparsitySize = 16 * NumberOfRowsSparistyBytes * outputChannels;
        }
        inputSize += fakeSparsitySize;
    }
    if (inputActivationSparsity) {
        // Alignment due to input channels mult of 16 requirement
        // Only ZM Conv and Elwise are sparse consumers, both need
        // input channels mult of 16
        auto tensorSize = op.getInputTensor(0)->computeTotalSize(16, false, false, true);
        size_t streamDivisor = streamConfig["W"] * streamConfig["H"] * streamConfig["C"];
        // Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseInputSize = std::ceil((double)tensorSize / (8 * op.getInputTensor(0)->getDType().getSizeInBytes()));
        // Storage element table calculation, 4 bytes pointers
        // Bigger with C streaming
        sparseInputSize += op.getInputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION] *
                           op.getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] * streamConfig["C"] * 4;
        // Alignment due to bus access requirements
        sparseInputSize = mv::round_up(sparseInputSize, 16);
        inputSize += (sparseInputSize / streamDivisor);
    }
    if (outputActivationSparsity) {
        // Alignment due to output channels mult of 16 requirement
        // Only ZM Conv and Elwise are sparse consumers
        auto tensorSize = op.getOutputTensor(0)->computeTotalSize(16, false, false, true);
        size_t streamDivisor = streamConfig["W"] * streamConfig["H"] * streamConfig["K"];
        // Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseOutputSize =
            std::ceil((double)tensorSize / (8 * op.getOutputTensor(0)->getDType().getSizeInBytes()));
        // Storage element table calculation, 4 bytes pointers
        // Bigger with K streaming
        sparseOutputSize += op.getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION] *
                            op.getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] * streamConfig["K"] * 4;
        // Alignment due to bus access requirements
        sparseOutputSize = mv::round_up(sparseOutputSize, 16);
        outputSize += (sparseOutputSize / streamDivisor);
    }
    if (weightsSparsity) {
        // Alignment due to output/input channels mult of 16 requirement
        auto tensorSize =
            op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] * op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] *
            mv::round_up(op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS], 16) * alignedSplittedChannels;
        // Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseWeightSize = std::ceil((double)tensorSize / 8);
        // Sparse pointers taken into account in weight table ...
        sparseWeightSize = mv::round_up(sparseWeightSize, 16);
        weightSize += sparseWeightSize;
    }

    weightSize += weightTableSize;

    // Note: for SOH and SOK, division by number of clusters is done in activationTensorSize
    // and alignedWeightsSize, respectively. This allows greater precision than dividing
    // totalClusters. Multiclustering doesn't perfectly split tensor, depends on subtensor size!
    if (clusterStrategy == "HKSwitch") inputSize = div(inputSize, totalClusters);
    if (clusterStrategy == "SplitOverHOverlapped") {
        inputSize = div(inputSize, totalClusters);
        outputSize = div(outputSize, totalClusters);
    }

    return std::tuple<std::size_t, std::size_t, std::size_t>(inputSize, outputSize, weightSize);
}

bool requiresFakeActivationSparsity(mv::Op& op) {
    bool enableChannelMajorConv = true;
    if (enableChannelMajorConv && op.supportsCMConv()) return true;

    if (op.getOpType() == "MaxPool") return true;

    if (op.getOpType() == "DepthwiseConv") return true;

    return false;
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

std::map<int, int> getMinWeightsPerClusterSizePerChain(std::list<subgraph_t>& chainSubgraphs, const mv::pass::PassEntry& pass,
                                         mv::ComputationModel& model) {
    mv::OpModel om(model);
    typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
    typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    std::vector<mv::Element> strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    std::map<int, int> minWeightsPerClusterPerChain;
    unsigned chainID = 0;
    std::string clustering;
    std::vector<size_t> streamsSizes;  // Store streamsSizes

    // For each chain
    for (subgraph_t chain_subgraph : chainSubgraphs) {
        streamsSizes.clear();  // clear stream sizes for chain i

        // For each operation in chain[i]
        for (auto& op : chain_subgraph.dpu_chain_) {
            pass.log(mv::Logger::MessageType::Debug,
                     "Process Op " + op->getName() + " in chain " + std::to_string(chainID));
            mv::Data::OpListIterator opIt = om.getOp(op->getName());
            opIt->set<unsigned>("chainID", chainID);

            // If its a conv and marke
            if (opIt->getOpType() == "Conv") {
                std::cout << opIt->getName() << " " << opIt->getInputTensor(1)->computeTotalSize() << std::endl;

                // Get the strategy for this conv
                for (auto layerNameStrategy : strategyList) {
                    std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

                    if (nodeName == op->getName()) {
                        auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");

                        pass.log(mv::Logger::MessageType::Debug,
                                 "Op " + op->getName() + " is streaming over K: " +
                                         std::to_string(streaming_strategy[3].get<int>("K")));

                        if (op->hasAttr("splitStrategy"))
                            clustering = op->get<std::string>("splitStrategy");

                        bool inputActivationSparsity, outputActivationSparsity, weightsSparsity = false;
                        if (op->hasAttr("inputActivationSparsity"))
                            inputActivationSparsity = op->get<bool>("inputActivationSparsity");
                        if (op->hasAttr("outputActivationSparsity"))
                            outputActivationSparsity = op->get<bool>("outputActivationSparsity");
                        if (op->hasAttr("weightsSparsity"))
                            weightsSparsity = op->get<bool>("weightsSparsity");

                        // get the memory size of the streams weights
                        size_t input, output, weightsPerCluster;
                        input = output = weightsPerCluster = 0;
                        mv::Data::OpListIterator oitr = om.getOp(op->getName());

                        std::tie(input, output, weightsPerCluster) = memorySize(
                                *oitr, clustering, inputActivationSparsity, outputActivationSparsity, weightsSparsity,
                                {1, (unsigned int)streaming_strategy[1].get<int>("H"),
                                 (unsigned int)streaming_strategy[2].get<int>("C"),
                                 (unsigned int)streaming_strategy[3].get<int>("K"),
                                 (unsigned int)streaming_strategy[4].get<int>("N")},
                                requiresFakeActivationSparsity(*oitr), true, true);

                        streamsSizes.push_back(weightsPerCluster);

                        pass.log(mv::Logger::MessageType::Info,
                                 "Op " + op->getName() + " is streaming over K: " +
                                         std::to_string(streaming_strategy[3].get<int>("K")) +
                                         " and the stream weights size per cluster is " +
                                         std::to_string(weightsPerCluster));
                    }
                }
            }
        }
        std::cout<<"here"<<std::endl;
        std::sort(streamsSizes.begin(), streamsSizes.end());
        streamsSizes.erase(unique(streamsSizes.begin(), streamsSizes.end()), streamsSizes.end());
        std::cout<<"hereee"<<std::endl;
        minWeightsPerClusterPerChain.insert({chainID, streamsSizes[0]});
        chainID++;
        
    }
    return minWeightsPerClusterPerChain;
}

void addOptimalChainPipeliningStrategiesFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    /*
    * Highlevel steps:
    *   Step 1: Get the subgraph chains using LPScheduler class
    *   Step 2: For each operation in chain[i], get the K stream size in bytes for layer that is streaming over K, so that you can get the minimum stream size in the chain
    *   Step 3: For each operation in chain[i]
    *         3a. calculate the MAXimiumNumberOfPossibleKStreams, defined as the outputChannels / 4 / 16
    *         3b. calculate the OptimumNumberOfKStreams for a layer, defined as the (Total weight and WT size in bytes) / the smallest stream size in a chain
    *         3c. compare the OptimumNumberOfKStreams versus the number of K streams determined by graph optimizer
    *             - if (OptimumNumberOfKStreams < minimiumNumberOfPossibleKStreams) && (OptimumNumberOfKStreams > KStreamsAssignedbyGO)
    *                   - assign the the OptimumNumberOfKStreams
    *               else
    *                   - retain the K streams assigned by graph optimizer
    */
    mv::OpModel om(model);
    typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
    typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;
    unsigned chainID = 0;
    std::vector<size_t> streamsSizes; //Store streamsSizes  
    bool isKStreaming = false;
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    std::vector<mv::Element> strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    auto compDesc = model.getGlobalConfigParams();
    std::string clustering;
    size_t fullweightsSize = 0;
    double maxpossibleStreams = 0.0;
    size_t optimalNumberOfKStreams = 0;
    std::vector<mv::Element> streaming_strategy;
    std::vector<mv::Element> overWrittenStreamingStrategies;
    std::vector<mv::Element> allStreamingStrategies;
    
    //Get the strategies to be included in pass
    auto strategies = getStrategies(passDesc);

    // instantiate pipeline chains class 
    pipeline_chains_t pipeliner(om);
    size_t pipeline_stages = 0UL;

    if (passDesc.hasAttr("select_stages")) {
        pipeline_stages = (size_t) passDesc.get<int>("select_stages");
    }

    // Step 1: Get the subgraph chains
    auto chainSubgraphs = pipeliner.get_chain_subgraphs(pipeline_stages);

    auto mp = getMinWeightsPerClusterSizePerChain(chainSubgraphs, pass, model);
    std::cout<<"here1"<<std::endl;
    std::cout << "KEY\tELEMENT\n";
    for (auto itr = mp.begin(); itr != mp.end(); ++itr) {
        std::cout << itr->first
             << '\t' << itr->second << '\n';
    }
    std::cout << "done" << std::endl;
    
    // // For each chain
    // for (subgraph_t chain_subgraph : chainSubgraphs) {

    //     streamsSizes.clear(); //clear stream sizes for chain i

    //     // Step 2: For each operation in chain[i]
    //     for (auto& op : chain_subgraph.dpu_chain_) { 
            
    //         isKStreaming = false;

    //         pass.log(mv::Logger::MessageType::Debug, "Process Op " + op->getName() + " in chain " + std::to_string(chainID));
    //         mv::Data::OpListIterator opIt = om.getOp(op->getName());
    //         opIt->set<unsigned>("chainID", chainID);

    //         // If its a conv and marke
    //         if (opIt->getOpType() == "Conv") 
    //         {
    //             std::cout << opIt->getName() << " " << opIt->getInputTensor(1)->computeTotalSize() << std::endl;
    //             // Get the strategy for this conv
    //             for (auto layerNameStrategy : strategyList) {

    //                 std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

    //                 if (nodeName == op->getName()) {
    //                     auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
    //                     isKStreaming = streaming_strategy[3].get<int>("K") > 1 ? true : false;

    //                     pass.log(mv::Logger::MessageType::Debug,
    //                              "Op " + op->getName() + " is streaming over K: " +
    //                                      std::to_string(streaming_strategy[3].get<int>("K")));

    //                     if (op->hasAttr("splitStrategy"))
    //                         clustering = op->get<std::string>("splitStrategy");

    //                     bool inputActivationSparsity, outputActivationSparsity, weightsSparsity = false;
    //                     if (op->hasAttr("inputActivationSparsity"))
    //                         inputActivationSparsity = op->get<bool>("inputActivationSparsity");
    //                     if (op->hasAttr("outputActivationSparsity"))
    //                         outputActivationSparsity = op->get<bool>("outputActivationSparsity");
    //                     if (op->hasAttr("weightsSparsity"))
    //                         weightsSparsity = op->get<bool>("weightsSparsity");

    //                     // get the memory size of the streams weights
    //                     size_t input, output, weightsPerCluster;
    //                     input = output = weightsPerCluster = 0;
    //                     mv::Data::OpListIterator oitr = om.getOp(op->getName());

    //                     std::tie(input, output, weightsPerCluster) = memorySize(
    //                             *oitr, clustering, inputActivationSparsity, outputActivationSparsity, weightsSparsity,
    //                             {1, (unsigned int)streaming_strategy[1].get<int>("H"),
    //                              (unsigned int)streaming_strategy[2].get<int>("C"),
    //                              (unsigned int)streaming_strategy[3].get<int>("K"),
    //                              (unsigned int)streaming_strategy[4].get<int>("N")},
    //                             requiresFakeActivationSparsity(*oitr), true, true);

    //                     streamsSizes.push_back(weightsPerCluster);

    //                     pass.log(mv::Logger::MessageType::Info,
    //                              "Op " + op->getName() + " is streaming over K: " +
    //                                      std::to_string(streaming_strategy[3].get<int>("K")) +
    //                                      " and the stream weights size per cluster is " +
    //                                      std::to_string(weightsPerCluster));
    //             }
    //             }
    //         }
    //         std::cout << "*********" << std::endl;
    //     }
    //     //Sort streams by sizes
    //     std::sort(streamsSizes.begin(), streamsSizes.end());
    //     streamsSizes.erase(unique(streamsSizes.begin(), streamsSizes.end()), streamsSizes.end());

    //     std::size_t minStreamSize = 0;
    //     // if(streamsSizes[0] < 34816)
    //     //     minStreamSize = streamsSizes[0];
    //     // else
    //         minStreamSize = 34816;
        
    //     std::cout << "min stream size for chain " << chainID << " is " << streamsSizes[0] << std::endl;
    //     std::cout << "using " << minStreamSize << " as the size for " << chainID << std::endl;
    //     // Print the stream sizes in chain i
    //     pass.log(mv::Logger::MessageType::Debug, "Stream sizes for chain " + std::to_string(chainID) + " are: ");
    //     std::vector<size_t>::iterator itr;
    //     for (itr = streamsSizes.begin(); itr != streamsSizes.end(); ++itr)
    //         pass.log(mv::Logger::MessageType::Debug, std::to_string(*itr));
        
    //     /* Step 3: For each operation in chain[i]
    //      *    3a. calculate the maximiumNumberOfPossibleKStreams, defined as the outputChannels / 4 / 16
    //      *    3b. calculate the OptimumNumberOfKStreams for a layer, defined as the (Total weight and WT size in bytes) / the smallest stream size in a chain
    //      *    3c. compare the OptimumNumberOfKStreams versus the number of K streams determined by graph optimizer
    //      *            - if (OptimumNumberOfKStreams < maximumNumberOfPossibleKStreams) && (OptimumNumberOfKStreams > KStreamsAssignedbyGO)
    //      *                   - assign the the OptimumNumberOfKStreams
    //      *               else
    //      *                   - retain the K streams assigned by graph optimizer
    //      */

    //     for (auto& op : chain_subgraph.dpu_chain_) {
    //         for (auto layerNameStrategy : strategyList) {
    //             std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
    //             mv::Data::OpListIterator opIt;

    //             if (nodeName == op->getName()) {
    //                 opIt = om.getOp(nodeName);
    //                 optimalNumberOfKStreams = 0;

    //                 if (opIt->getOpType() == "Conv") 
    //                 {
    //                     pass.log(mv::Logger::MessageType::Debug, "Op " + opIt->getName());
    //                     auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
    //                     isKStreaming = streaming_strategy[3].get<int>("K") > 1 ? true : false;

    //                     // Add number of streams to list
    //                     if (isKStreaming) {
    //                         int originalKStreaming = streaming_strategy[3].get<int>("K");
    //                         fullweightsSize = opIt->getInputTensor(1)->computeTotalSize();

    //                         size_t outChannels = opIt->outputSlots()
    //                                                  ? opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION]
    //                                                  : 0;

    //                         size_t alignedFullChannels = mv::round_up(outChannels, 16);
    //                         double weightTableSize = 0.0;
    //                         weightTableSize = 16 * alignedFullChannels;
    //                         int weightSize = opIt->getInputTensor(1)->computeTotalSize();

    //                         pass.log(mv::Logger::MessageType::Info,
    //                             "Op " + opIt->getName() + " full weights size is: " + std::to_string(weightSize) +
    //                                 " and weight table is " + std::to_string(16 * alignedFullChannels) +
    //                                 " total size is (byes) " + std::to_string(fullweightsSize + weightTableSize) +
    //                                 " and number of streams is " + std::to_string(originalKStreaming));

    //                         maxpossibleStreams = opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] / 64;

    //                         pass.log(mv::Logger::MessageType::Info, "Op " + opIt->getName() +
    //                                                                     " min possible sreams are " +
    //                                                                     std::to_string(maxpossibleStreams));

                            
    //                             double optimalNumberOfKStreams =
    //                                 std::round((fullweightsSize + weightTableSize) / (minStreamSize * 4.0));

    //                             pass.log(mv::Logger::MessageType::Info, "Op " + opIt->getName() +
    //                                                                         " optimalNumberOfKStreams is " +
    //                                                                         std::to_string(optimalNumberOfKStreams));

    //                             if ((optimalNumberOfKStreams <= maxpossibleStreams) &&
    //                                 (optimalNumberOfKStreams > originalKStreaming)) {
    //                                 pass.log(mv::Logger::MessageType::Info,
    //                                     "Op " + opIt->getName() +
    //                                         " optimal number of streams based on min chain size should be: " +
    //                                         std::to_string(optimalNumberOfKStreams) + " strategy manager assigned " +
    //                                         std::to_string(originalKStreaming));

    //                                 pass.log(mv::Logger::MessageType::Info,
    //                                     "Op " + opIt->getName() +
    //                                         " changing to optimal number of streams based on min chain size: " +
    //                                         std::to_string(optimalNumberOfKStreams) + " strategy manager assigned " +
    //                                         std::to_string(originalKStreaming));

    //                                 opIt->set<unsigned>("optimalNumberOfKStreams", optimalNumberOfKStreams);
    //                                 std::cout <<   "Op " << opIt->getName() << " changing to optimal number of streams based on min chain size: " << 
    //                                         optimalNumberOfKStreams << " strategy manager assigned " <<
    //                                         originalKStreaming << std::endl;
    //                                 break;
    //                             }
    //                             else if ((optimalNumberOfKStreams > maxpossibleStreams) &&
    //                                 (optimalNumberOfKStreams > originalKStreaming))
    //                             {
    //                                 opIt->set<unsigned>("optimalNumberOfKStreams", maxpossibleStreams);
    //                                 std::cout <<   "Op " << opIt->getName() << " changing to optimal number of streams to be the max number of streams: " << 
    //                                 maxpossibleStreams << " strategy manager assigned " <<
    //                                 originalKStreaming << std::endl;

    //                             }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     chainID++;
    // }
    // assignNewSrategies(model, allStreamingStrategies, overWrittenStreamingStrategies, globalParams);
    // compDesc->set("streaming_strategy", allStreamingStrategies);
    // saveNewStreamingStrategiesToJson(pass, overWrittenStreamingStrategies);
}
