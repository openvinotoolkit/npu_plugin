#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/graphOptimizations/strategy_utils.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/pass/graphOptimizations/simple_strategy_manager.hpp"

namespace mv {
namespace graphOptimizer {

StrategyManagerSimple::StrategyManagerSimple(OpModel& model,mv::Element& passDesc, mv::TargetDescriptor& td) :
    StrategyManager(model,passDesc)
{
    auto globalParams = model.getGlobalConfigParams();
    referenceDevice = globalParams->get<std::string>("referenceDevice");
    totalClusters = globalParams->get<int>("Number_of_Clusters");
    clusterMemory = globalParams->get<int>("cmx");
    dpuPerCluster = globalParams->get<int>("Number_of_DPUs") / totalClusters;
    target = td.getTarget();

    //Seems to work better for THB?
    if(target == mv::Target::ma3100)
        cmxPipeLineWeightsOverhead = 74241.0;

    globalEnableStreaming = globalParams->get<bool>("enableStreaming");
    globalEnablePipelining = globalParams->get<bool>("enablePipelining");
    globalEnablePrefetching = globalParams->get<bool>("enablePrefetching");
    globalEnableWeightsSparsity = globalParams->get<bool>("enableWeightsSparsity");
}



bool StrategyManagerSimple::requiresActivationSparsity(Op& op, std::string clustering)
{
    if(requiresRealActivationSparsity(op, clustering))
        return true;

    if(requiresCompilerActivationSparsity(op))
        return true;

    return false;
}

bool StrategyManagerSimple::requiresWeightsSparsity(Op& op)
{
    // If Z-major Conv in Float precision then need to have weights Sparsity
    bool isCMConv = op.hasAttr("supportsCM") && op.get<bool>("supportsCM");

    if(op.getOpType() == "Conv" &&
        op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") &&
        !isCMConv )
            return true;

    return false;
}

// In these cases parent output sparsity does matter, but child input sparsity must be true
bool StrategyManagerSimple::requiresCompilerActivationSparsity(Op& op)
{
    bool isCMConv = op.hasAttr("supportsCM") && op.get<bool>("supportsCM");

    if (op.getOpType() == "Conv" && !isCMConv
            && (op.hasAttr("DilatedSubConv") && op.get<bool>("DilatedSubConv")))
        return true;

    return false;
}

bool StrategyManagerSimple::requiresRealActivationSparsity(Op& op, std::string clustering){
    //An fp16 Conv Z-major must have activation sparsity
    bool isCMConv = op.hasAttr("supportsCM") && op.get<bool>("supportsCM");

    if (op.isSparsityConsumer() &&
        op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") &&
        !isCMConv && checkA0Sparsity(model_))
    {
        return true;
    }


    // Check for need for A0 SOH Sparsity workaround, (SOH conv with kernel > 1)
    // if needed, check memory constraints as for sparse tensor
    if (op.getOpType() == "Conv" ) {
        if( clustering == "SplitOverH" &&
            (op.getInputTensor(1)->getShape()[KERNEL_HEIGHT] > 1) &&
            !isCMConv && (target == mv::Target::ma3100 ||  // Apply the W/A also for TBH to overcome accuracy regression
            (target == mv::Target::ma2490 && referenceDevice == "A0")))
            {
                return true;
            }
    }

    return false;
}

//Channel major conv, pooling and depthwise will get fake sparsity, so need to check memory constraints as if real sparsity
bool StrategyManagerSimple::requiresFakeActivationSparsity(Op& op)
{
    if(op.hasAttr("supportsCM") && op.get<bool>("supportsCM") && target != mv::Target::ma3720)
        return true;

    if(op.getOpType() == "MaxPool")
        return true;

    if(op.getOpType() == "DepthwiseConv")
        return true;

    return false;
}

bool StrategyManagerSimple::decideWeightsSparsity(mv::Op op, float floatOverhead = 0.0625, float intOverhead = 0.125)
{
    // Only Z-major convolutions support weights sparsity, this is codified in the compilation descriptors
    if( !createStrategyFromBool(op,"weightsSparsity") )
        return false;

    // If CM convolutions are enabled, don't sparsify these
    if(op.hasAttr("supportsCM") && op.get<bool>("supportsCM"))
        return false;
    
    auto inputTensor = op.getInputTensor()[mv::IO_TENSOR_INPUT];
    auto weightTensor = op.getInputTensor()[mv::IO_TENSOR_WEIGHTS_SET];
    auto outputTensor = op.getOutputTensor()[mv::IO_TENSOR_OUTPUT];
    auto sparsityOverhead = inputTensor->isFloatingPointType() ?
        floatOverhead : intOverhead;

    auto inputTensorChannels = inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];
    auto outputTensorChannels = outputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];
    auto demandPaddingInputChannels = (inputTensorChannels % 16 != 0);
    auto demandPaddingOutputChannels = (outputTensorChannels % 16 != 0);
    if (demandPaddingOutputChannels || demandPaddingInputChannels)
    {
        float limitInputChannels = 0;
        float limitOutputChannels = 0;
        if (demandPaddingOutputChannels)
            limitOutputChannels = mv::round_up(outputTensorChannels, 16) * sparsityOverhead;
        if (demandPaddingInputChannels)
            limitInputChannels = mv::round_up(inputTensorChannels, 16) * sparsityOverhead;
        if(inputTensorChannels <= std::round(limitInputChannels) ||
            outputTensorChannels <= std::round(limitOutputChannels))
            return true;

    }

    // Size of weights, actual sparsity of tensor determine speedup
    auto weightsSize = realTensorSize(weightTensor, {1,1,1,1}, false);
    auto zeroPoints = weightTensor->getZeroValuesCount();
    double actualSparsity = (double) zeroPoints/ (double)weightsSize;

    // Enable weights sparsity if actual sparsity level observed in the tensor
    // is high enough to warrant the overhead of enabling sparsity
    if(std::isgreaterequal(actualSparsity, sparsityOverhead))
        return true;

    return false;
}

// Note: This function will return the potential streams over H for this op. For simplicity, we want to limit
// the options to reasonable configurations. This always includes H=1, or in other words no streaming over H.
// If H streaming fits at all (i.e. weights fit), find the H just big enough to fit into CMX. If CMX concat,
// spilling will be false, and H stream will be higher accordingly.
std::vector<size_t> StrategyManagerSimple::getStreamsOverH(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling, bool parentSpilling)
{
    auto minSplitsToFit = getMinStreamOverH(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, parentSpilling);
    if( minSplitsToFit == 0 || // stream over h alone doesn't fit
        minSplitsToFit == 1 || // no streams required to fit
        clustering.get<std::string>() == "HKSwitch")
        return {1};

    return {minSplitsToFit, 1};
}

// Gives the minimum number of streams over H to fit this layer, or if no number of streams enable streaming
// (for example, weights don't fit) then return 0
unsigned StrategyManagerSimple::getMinStreamOverH(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                            bool wSparsity, bool fSparsity, bool spilling,  bool parentSpilling)
{
    size_t input, output, weights;
    // in case initialization in memorySize fails
    input = output = weights = 0;
    std::tie(input, output, weights) = memorySize(op,totalClusters,clustering.get<std::string>(),iSparsity,oSparsity,wSparsity,streams,fSparsity,spilling,parentSpilling);
    auto activationsSize = input + output;
    auto weightsSize = weights;
    double availableMemory = (double) clusterMemory - (double) weightsSize;

    if (availableMemory <= 0) // Weights don't fit, can't stream over H
        return 0;

    // Keep increasing H until we find one big enough to fit, or we run out of H dimension to stream
    auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];

    // Every output slice must have at least one line to compute
    unsigned upperBoundH = outputHeight;
    if(clustering.get<std::string>() == "SplitOverH")
    {
        upperBoundH = upperBoundH/totalClusters;
    }

    // Start searching for min stream at naive requirement for splits to fit, rather than 1
    for(unsigned splits = ceil((double)activationsSize/availableMemory); splits <= upperBoundH; splits++)
    {
        Shape updatedStreams({1,splits,1,streams["K"],streams["B"]});
        auto memFitCheck = memorySize(op,totalClusters, clustering.get<std::string>(),iSparsity,oSparsity,wSparsity,updatedStreams,fSparsity,spilling,parentSpilling);

        if((std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                validateHStream(op, clustering, splits))
        {
            return splits;
        }
    }

    return 0;
}

bool StrategyManagerSimple::validateHStream(mv::Op& op, mv::Attribute clustering, std::size_t splits)
{
    if( op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
    {
        if(clustering.get<std::string>() == "SplitOverH")
        {
            auto weightsShape = op.getInputTensor(1)->getShape();
            //Try to guess subtensor height, and avoid situations where kernel is bigger than last workload dimension
            auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
            auto workloadHeight = ceil((double)outputHeight / (double)(totalClusters * splits));
            if(totalClusters > 1) //last
                workloadHeight = outputHeight - (workloadHeight * (totalClusters-1)); //get remaining height
            if(workloadHeight < weightsShape[KERNEL_HEIGHT])
                return false;
        }
    }

    //check that the inputSize will not be smaller than kernel size
    if (op.getOpType() == "MaxPool" ||
        op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
    {
        uint16_t kernelH;
        std::array<unsigned short, 4> padding;

        auto originalH = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
        auto newOutputSizes = tileSpatialOutputSize(originalH, splits);

        //Reject H streams were the last stream isn't equal or smaller than the rest
        //Reject H streams were the last stream is 1, unless they are all 1
        if(newOutputSizes.back() > newOutputSizes.front() ||
            (newOutputSizes.back() == 1 && newOutputSizes.front() != 1)) 
                return false;

        unsigned short kernelStride;
        if (op.hasAttr("stride"))
            kernelStride = op.get<std::array<unsigned short, 2>>("stride")[1];
        else
            kernelStride = 1;//fake stride

        if (op.hasAttr("padding"))
            padding = op.get<std::array<unsigned short, 4>>("padding");
        else
            padding = {0, 0, 0, 0};

        int padStart = 0;
        int padEnd = padding[3];

        if (op.hasAttr("kSize"))
        {
            auto kernelShape = op.get<std::array<unsigned short, 2>>("kSize");
            kernelH = kernelShape[1];
        }
        else
        {
            auto weightsShape = op.getInputTensor(1)->getShape();
            kernelH = weightsShape[mv::KERNEL_HEIGHT];
        }

        //Reject H streams where the last subtensor will be less than kernel height
        if(clustering.get<std::string>() == "SplitOverH" && ((newOutputSizes.back()/totalClusters) < kernelH))
            return false;

        int inputSizeForLastSplit = ((newOutputSizes.back() -1) * kernelStride)  -padStart - padEnd + kernelH;
        if ((inputSizeForLastSplit + padEnd) < kernelH)
            return false;

        //Reject H streams where height of last subtensor will be less than kernel height
        //in case where input size is different because of padding
        int heightOfLastSubtensor = inputSizeForLastSplit - (std::ceil((float)inputSizeForLastSplit/totalClusters) * (totalClusters-1));
        if(clustering.get<std::string>() == "SplitOverH" && heightOfLastSubtensor < kernelH)
            return false;
    }

    return true;
}

// Note: This function produces the potential stream over K strategies for each layer
// Try to find 2 possible combinations of K, in addition ot K=1 (no streams in this dimension)
// First, just enough to fit in cmx. Second, enough to enable pipelining.
std::vector<std::size_t> StrategyManagerSimple::getStreamsOverK(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling, bool parentSpilling)
{
    auto minSplitsToFit = getMinStreamOverK(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, parentSpilling);
    if(minSplitsToFit == 0) // No suitable stream over K found
        return {1};

    std::vector<std::size_t> splits;
    splits.push_back(1);
    if(minSplitsToFit != 1)
        splits.push_back(minSplitsToFit);

    if( globalEnablePipelining && //TODO only consdier pipelining when resulting wieghts will be bigger than min
        (clustering.get<std::string>() == "SplitOverK" || clustering.get<std::string>() == "Clustering") &&
        createStrategyFromBool(op, "pipelining"))
    {
        auto pipelinedMinSplitsToFit = getMinStreamOverK(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, parentSpilling, true);
        if(pipelinedMinSplitsToFit != 0)
        {
            if(pipelinedMinSplitsToFit != minSplitsToFit)
                splits.push_back(pipelinedMinSplitsToFit);
            auto nextKStream = getNextStreamOverK(op, clustering, pipelinedMinSplitsToFit, spilling);
            if(nextKStream > 0)
                splits.push_back(nextKStream);
        }
    }

    return splits;
}

unsigned StrategyManagerSimple::getMinStreamOverK(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling, bool parentSpilling, bool pipelined)
{
    auto outputShape = op.getOutputTensor(0)->getShape();
    size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
    size_t alignedOutputChannelSize = mv::round_up(outputChannelSize, 16);

    auto maxSplit = alignedOutputChannelSize / 16;

    if(clustering.get<std::string>() == "SplitOverK")
        maxSplit = maxSplit / totalClusters;

    for(unsigned split = 1; split <= maxSplit; split++)
    {
        auto memFitCheck = memorySize(op,totalClusters, clustering.get<std::string>(),iSparsity,oSparsity,wSparsity,{1,1,1,split,streams["B"]},fSparsity, spilling, parentSpilling);
        if( pipelined && //pipelining weights requires 2 weights streams to fit
            (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + 2*std::get<2>(memFitCheck) < clusterMemory) &&
            validateKStream(op, clustering, split, spilling) )
        {
            return split;
        }
        else if(!pipelined &&
                (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                validateKStream(op, clustering, split, spilling) )
        {
            return split;
        }
    }

    return 0;
}

unsigned StrategyManagerSimple::getNextStreamOverK(mv::Op& op, mv::Attribute clustering, size_t startSplit, bool spilling)
{
    auto outputShape = op.getOutputTensor(0)->getShape();
    size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
    size_t alignedOutputChannelSize = mv::round_up(outputChannelSize, 16);

    //Find max split
    auto maxSplit = alignedOutputChannelSize/16;

    for(size_t split = startSplit+1; split <= maxSplit; split++)
    {
        //TODO can we steal some logic from nested streaming to jump to the next "best" K
        // would be useful for when many streams over K are needed just to fit and we
        // run into +1 doesn't result in a differing number of channels in final task...
        if(validateKStream(op, clustering, split, spilling))
            return split;
    }

    return 0;
}

bool StrategyManagerSimple::validateKStream(mv::Op& op, mv::Attribute clustering, size_t split, bool spilling)
{
    if( op.getOpType() == "Conv" &&
        clustering.get<std::string>() == "SplitOverK")
    {
        auto weightsShape = op.getInputTensor(1)->getShape();
        auto numOutChannels = weightsShape[KERNEL_OUTPUT_CHANNELS];
        if((numOutChannels/split * totalClusters) < 16)
            return false;
    }
    if(!spilling)
    {
        auto outputShape = op.getOutputTensor(0)->getShape();
        size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
        //ok it fits, now make sure that if we are !spilling that there's no crop
        size_t outputChannelSlice = ceil((double)outputChannelSize/(double)split);
        size_t lastSlice = outputChannelSize - outputChannelSlice*(split - 1);
        if (!(outputChannelSlice%16 == 0 && lastSlice%16 == 0)) //would need crop
            return false;
    }

    return true;
}

// Note: Find suitable stream over C values
std::vector<std::size_t> StrategyManagerSimple::getStreamsOverC(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling)
{
    auto minSplitsToFit = getMinStreamOverC(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling);
    if(minSplitsToFit == 0) // No suitbale stream over C found
        return {1};

    std::vector<std::size_t> splits;
    splits.push_back(1);

    if(minSplitsToFit != 1)
        splits.push_back(minSplitsToFit);

    return splits;
}

unsigned StrategyManagerSimple::getMinStreamOverC(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling)
{
    auto inputShape = op.getInputTensor(0)->getShape();
    size_t inputChannelSize = inputShape[IO_CHANNEL_DIMENSION];

    unsigned startSplit = 1;
    if(inputChannelSize > mv::MAX_DIM_SIZE)
        startSplit = 2;

    for(unsigned split = startSplit; split <= inputChannelSize; split++)
    {
        auto memFitCheck = memorySize(op, totalClusters,clustering.get<std::string>(),iSparsity,oSparsity,wSparsity,{1,1,split,1,streams["B"]},fSparsity, spilling);
        if((std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory))
            return split;
    }

    return 0;
}

unsigned findBestK(unsigned alignedSize, unsigned channels){
    return std::ceil((double)alignedSize / ((alignedSize/2) - channels));
}

//Note: this function only used to generate many stream over k options when we NESTED stream
std::vector<size_t> StrategyManagerSimple::getMaxStreamOverK(mv::Op& op)
{
    auto outputShape = op.getOutputTensor(0)->getShape();
    size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
    size_t alignedOutputChannelSize = mv::round_up(outputChannelSize, 16);

    std::vector<size_t> splits;

    //Add max split
    splits.push_back(alignedOutputChannelSize/16);

    // For each aligned-to-16 number of output channels possibility, add only the
    // minimum number of streams over k that will be aligned to that number
    for(int channels = (alignedOutputChannelSize/2 -16); channels >= 16; channels=channels-16){
        auto possibleK = findBestK(alignedOutputChannelSize, channels);
        if(splits.back() != possibleK && possibleK >= 1)
            splits.push_back(possibleK);
    }
    if(splits.back() > 2)
        splits.push_back(2);

    if(splits.back() > 1)
        splits.push_back(1);

    return splits;
}

bool StrategyManagerSimple::createStrategyFromBool(mv::Op op, std::string name)
{
    auto& streamingStrategy = getStrategy(op,name);

    bool value = streamingStrategy.get<bool>();
    if(value)
        return true;
    else
        return false;
}

std::vector<Attribute> StrategyManagerSimple::createTFStrategyPoolFromBool(mv::Op op,std::string name)
{
    auto& streamingStrategy = getStrategy(op,name);

    bool value = streamingStrategy.get<bool>();
    if(value)
        return std::vector<Attribute>{true,false};
    else
        return std::vector<Attribute>{false};
}

std::vector<mv::Attribute> StrategyManagerSimple::createTStrategyPoolFromBool(mv::Op op,std::string name)
{
    auto& streamingStrategy = getStrategy(op,name);

    bool value = streamingStrategy.get<bool>();
    if(value)
        return std::vector<mv::Attribute>{true};
    else
        return std::vector<mv::Attribute>{true,false};
}


std::vector<mv::Attribute> StrategyManagerSimple::createStrategyPoolFromStrategySet(mv::Op op, std::string name)
{
    auto streamingStrategy = getStrategy(op,name);

    std::vector<mv::Attribute> attr;

    for (auto elem : streamingStrategy.get<std::vector<std::string>>())
    {
        attr.push_back(elem);
    }

    return attr;
}

bool opInCMX(mv::Op& op, StrategyManagerSimple::StrategySet& strategy)
{
    auto spilling = strategy["spilling"].get<bool>();

    auto opType = op.getOpType();
    if(opType == "Input" || opType == "Output")
        return false;

    if(!op.hasTypeTrait("optimizable"))
        return false;

    if(op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted"))
        return false;

    if(opType == "Concat" && spilling)
        return false;

    return true;
}
//Check to see if a given stategy is internally consistent for performance
//Strategies that can only have infinite edges because they are illegal should never be added to the graph
// Note: IF ADDING A NEW FAILURE CASE, must add new description to failure_causes
StrategyManagerSimple::FailCause StrategyManagerSimple::validateStrategy(mv::Op& op,StrategySet& strategy)
{
    auto clustering = strategy["clustering"].get<std::string>();
    auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
    auto streamShape = strategy["streaming"].get<Shape>();
    auto spilling = strategy["spilling"].get<bool>();
    auto parentSpilling = strategy["parentSpilling"].get<bool>();

    // For SOH to stream over H, enforce both input and output must be in DDR
    if((!parentSpilling || !spilling) && clustering == "SplitOverH" && streamShape["H"] > 1)
        return FailCause::cmxConcatDecision;

    // If previous op is SOH in CMX, then each cluster won't have the required tensor "block" 
    // to support streaming in next op. Only the first cluster's first subtensor will be valid
    // if(streamShape["H"] > 1 && (!parentSpilling || !spilling))
    //     return FailCause::cmxConcatDecision;

    //NOTE: funny part you can spill even if you are not streaming, fasten your seatbelts!!
    bool isStreaming = ((streamShape["W"] * streamShape["H"] * streamShape["C"]
                                                * streamShape["K"] * streamShape["B"]) > 1) ? true : false;

    // NOTE: This is a temporary workaround till we are able to identify the chains before graph
    // optimizer and control the cmx percentage that we want the weigths to receive described in
    // https://jira.devtools.intel.com/browse/CVS-43222
    {
        if (op.getOpType() == "Conv")
        {
            if (op.getInputTensor()[0]->getShape() == mv::Shape({13,13,512,1}) &&
                op.getInputTensor()[1]->getShape() == mv::Shape({3,3,512,1024}) &&
                op.getOutputTensor()[0]->getShape() == mv::Shape({13,13,1024,1}))
            {
                if (globalEnablePipelining && streamShape["K"] != 8 &&
                    op.getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType() == mv::DType("UInt8"))
                    return FailCause::cmxConcatDecision;
            }

            if (op.getInputTensor()[0]->getShape() == mv::Shape({13,13,1024,1}) &&
                op.getInputTensor()[1]->getShape() == mv::Shape({3,3,1024,1024}) &&
                op.getOutputTensor()[0]->getShape() == mv::Shape({13,13,1024,1}))
            {
                if (globalEnablePipelining && streamShape["K"] != 8 &&
                    op.getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType() == mv::DType("UInt8"))
                    return FailCause::cmxConcatDecision;
            }

            if (op.getInputTensor()[0]->getShape() == mv::Shape({1,1,2048,1}) &&
                op.getInputTensor()[1]->getShape() == mv::Shape({1,1,2048,1000}) &&
                op.getOutputTensor()[0]->getShape() == mv::Shape({1,1,1000,1}))
            {
                if (globalEnablePipelining && streamShape["K"] != 4 && streamShape["K"] != 2 &&
                    op.getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType() == mv::DType("UInt8"))
                    return FailCause::cmxConcatDecision;
            }
        }

    }

    if(op.getOpType() == "Eltwise" && op.hasAttr("andConversion") &&
        op.get<bool>("andConversion") && clustering != "Clustering")
            return FailCause::cmxConcatDecision;
    // A proper decision on CMX concat for explicit concat or eltwise streaming cannot
    // be made with the information on hand. Will not optimize strategies for these.
    // In later pass, we will mark those that can fit in CMX
    // as CMX-able.
    //Note: Removing eltwise from here becuase of course hkswitch needs to be in cmx
    if(op.getOpType() == "Concat" && !spilling 
        && op.getOutputTensor(0)->getShape().totalSize() >= clusterMemory)
        return FailCause::cmxConcatDecision;

    if(opInCMX(op, strategy))
    {
        size_t input, output, weights;
        // in case initialization in memorySize fails
        input = output = weights = 0;
        std::tie(input, output, weights) = memorySize(op,
                                                        totalClusters,
                                                        clustering,
                                                        strategy["inputSparsity"],
                                                        strategy["outputSparsity"],
                                                        weightsSparsity,
                                                        streamShape,
                                                        requiresFakeActivationSparsity(op),
                                                        spilling,
                                                        parentSpilling);
        if (input + output + weights >= clusterMemory)
            return FailCause::MemorySize;


        // To do a CMX concat, the channels must be aligned to 16
        if (!spilling && isStreaming)
        {
            if (op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION]%16 != 0)
            {
                return FailCause::cmxConcatDecision;
            }
            if (op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION]%16 != 0 &&
                    op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] > 16)
            {
                return FailCause::cmxConcatDecision;
            }
        }
    }

    bool isCMConv = op.hasAttr("supportsCM") && op.get<bool>("supportsCM");
    //If spilling, HKSwitch makes no sense
    if( (parentSpilling || spilling) && (clustering == "HKSwitch"))
        return FailCause::SpillHKSwitch;

    if( isStreaming && (clustering == "HKSwitch"))
        return FailCause::SpillHKSwitch;

    if( op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
    {
        auto weightsShape = op.getInputTensor(1)->getShape();
        auto numInChannels = weightsShape[KERNEL_INPUT_CHANNELS];
        auto numOutChannels = weightsShape[KERNEL_OUTPUT_CHANNELS];
        if (op.getOpType() == "Conv")
        {
            if((clustering == "SplitOverK") && (numOutChannels/(streamShape["K"] * totalClusters) < 16))
                return FailCause::SOKNotAlign16;
        }
        else
        {
            if((clustering == "SplitOverK") && (numInChannels/(streamShape["C"] * totalClusters) < 16))
                return FailCause::SOKNotAlign16;
        }
        if(clustering == "SplitOverH")
        {
            // TODO should we use padding here too?
            //Try to guess subtensor height, and avoid situations where kernel is bigger than last workload dimension
            auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
            auto workloadHeight = ceil((double)outputHeight / (double)(totalClusters * streamShape["H"]));
            if(totalClusters > 1) //last
                workloadHeight = outputHeight - (workloadHeight * (totalClusters-1)); //get remaining height
            if(workloadHeight < weightsShape[KERNEL_HEIGHT])
                return FailCause::WorkloadLessKernelSOH;
        }
    }

    //Input and Output must have Spilled==True
    //Note: To ensure correctness of the algorithm that will validate clustering
    // transitions, these must be Spilled=False
    //Then, they will get marked as spilled when saved
    if((op.getOpType() == "Input" || op.getOpType() == "ImplicitInput") && (!spilling))
        return FailCause::InputNotSpilled;

    if((op.getOpType() == "Output") && (!spilling))
        return FailCause::OutputNotSpilled;

    if(isCMConv && (strategy["inputSparsity"].get<bool>() || strategy["weightsSparsity"].get<bool>()))
        return FailCause::ChannelMjr2;

    //Guide early on the proposal of a valid strategy
    if (op.getOpType() == "DepthwiseConv")
    {
        if ((op.getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] > mv::MAX_DIM_SIZE)
                && (streamShape["C"] == 1))
            return FailCause::DWChannels;
    }

    //For every dpuTask if we splitOverH, workloads are over H dimension, so they need to have at
    //least one line to be assigned with
    if (op.isHardwarizable() && clustering == "SplitOverH")
    {
        auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
        auto estimatedClusterH = (unsigned)floor((double)outputHeight/totalClusters);
        if (estimatedClusterH < dpuPerCluster || (outputHeight - (totalClusters - 1) * estimatedClusterH) < dpuPerCluster)
            return FailCause::SOHheight;
    }
    //To match above, no need to consider HKSwitch if input tensor can't be SOH
    if (op.isHardwarizable() && (clustering == "HKSwitch" || clustering == "SplitOverH"))
    {
        auto inputHeight = op.getInputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
        auto estimatedClusterH = (unsigned)floor((double)inputHeight/totalClusters);
        if (estimatedClusterH < dpuPerCluster || (inputHeight - (totalClusters - 1) * estimatedClusterH) < dpuPerCluster)
            return FailCause::SOHheight;
    }

    // TODO: future task tackle SE table allocation to
    // to enable sparse channel segmented ops
    // Additionally each K tile will HAVE to be a power of 2 (IDU restriction)
    // which will create a feedback loop on the K stream rountine
    if(strategy["outputSparsity"].get<bool>() &&
        (clustering == "SplitOverK" ||
        clustering == "HKSwitch" ||
        streamShape["K"] > 1))
        return FailCause::SparsityKSegmented;

    // TODO: much more harder restriction to overcome
    // joint compiler and runtime coordination needed
    if(strategy["outputSparsity"].get<bool>() &&
        spilling)
        return FailCause::SparsitySpilling;

    // TODO: sparsity should be okay with streaming as long as not k segmented!
    if(strategy["outputSparsity"].get<bool>() &&
        isStreaming)
        return FailCause::SparsitySpilling;

    if(requiresFakeActivationSparsity(op) && strategy["inputSparsity"].get<bool>())
        return FailCause::RealSparseForFakeSparseOp;

    if (op.getOpType() == "DepthwiseConv" && op.hasAttr("DWWithReplaceLargeStrides")
            && op.get<bool>("DWWithReplaceLargeStrides") && (clustering == "SplitOverK"))
        return FailCause::DWLargeStrideReplacementSOK;

    //NOTE: Subdilation storage element population is not implemented for the SOH case
    if (op.getOpType() == "Conv"  && op.hasAttr("DilatedSubConv")
            && op.get<bool>("DilatedSubConv")
            && clustering == "SplitOverH")
        return FailCause::DilatedSOH;

    if (op.getOpType() == "Conv"  && op.hasAttr("DilatedSubConv")
            && op.get<bool>("DilatedSubConv")
            && !spilling)
        return FailCause::DilatedSOH;

    //Note: This is a temporary workaround for ICnet. In general, we are able to stream over H
    //for dilated sub convolutions. Root cause unidentified.
    if (op.getOpType() == "Conv"  &&
        op.hasAttr("DilatedSubConv") && op.get<bool>("DilatedSubConv") &&
        op.hasAttr("originalShape") && op.get<mv::Shape>("originalShape")[mv::IO_HEIGHT_DIMENSION] == 23 &&
        streamShape["H"] > 1)
        return FailCause::DilatedSOH;

    //Note: This is a workaround for Unet, root cause unidentified.
    //Unet non-DepthwiseDeConv subConv, avoiding splits < # of clusters, to avoid indeterministic outputs on back to back runs
    if(op.getOpType() == "Conv" && op.hasAttr("DeconvSubConv") && clustering == "SplitOverK")
    {
        auto originalH = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
        auto numberOfStreamSplits = streamShape["H"];
        if ((originalH % numberOfStreamSplits) != 0)
        {
            auto newOutputSizes = tileSpatialOutputSize(originalH, numberOfStreamSplits);
            auto remainderOutputSize = newOutputSizes.back();

            if (remainderOutputSize < totalClusters)
            {
                return FailCause::DeConvSubConvSOKHeight;
            }
        }
    }

    if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isCMConv &&
        (streamShape["K"] > 1 ) && spilling)
        return FailCause::SpiltOverHWithStreamOverK;

    //NOTE: This is not a HACK!!! if an operation is assigned with streamOverH + SplitOverH
    //and we concatenate on cmx the data are going to have a strange format...so nothing can be done later, so spill...
    /*For example, let's think a convolution with splitOverH and streams = 2, the tensor will be splitted to
        * 8 tiles, where every single tile is going to be assigned to a cluster with the round robin logic.
        * That means that cluster 0, will have tiles0,4. Cluster1 will have tiles1,5 and so on...
        * The way the data are splitted between clusters and the order of the tiles, do not allow us to concatenate
        * in the initial order inside CMX*/
    if (clustering == "SplitOverH" &&
        (streamShape["H"] > 1) && !spilling &&
        op.getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType() == mv::DType("UInt8"))
        return FailCause::SpiltOverHWithStreamOverHInCMX;
    
    // This is intended to be a temporary workaround for ModelE, layer '97' & '113', which does work with SOH
    // It has not been root caused to the compiler or runtime but as of now the compiler logic seems OK
    if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isCMConv && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 64 &&
        op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 80 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 48 &&
        op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 64 && op.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 80 &&
        op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 48 && op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 3 &&
        op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 3 &&
        op.getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType() == mv::DType("UInt8"))
        return FailCause::SpiltOverHForLayer97and113ModelE;

    // This is intended to be a temporary workaround for ACLnet, layer '79', which does work with SOH
    // It has not been root caused to the compiler or runtime but as of now the compiler logic seems OK
    if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isCMConv && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 1 &&
        op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 100 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 64 &&
        op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 64 && op.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 100 &&
        op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 64 && op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 3 &&
        op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 3 &&
        op.getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType() == mv::DType("UInt8"))
        return FailCause::SpiltOverHForLayer79InACLNet;

    // This is intended to be a temporary workaround for FaceDetectionRetail, layer fire6/suqeeze1x1/WithoutBiases, which does work with SOH
    // It has not been root caused to the compiler or runtime but as of now the compiler logic seems OK
    if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isCMConv && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 128 &&
        op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 38 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 38 &&
        op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 24 && op.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 38 &&
        op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 38 && op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 1 &&
        op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 1 &&
        op.getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType() == mv::DType("UInt8"))
        return FailCause::SpiltOverHForFaceDetectionRetail0004;

    //NOTE: we need a ticket for that failure, blob looks fine for streaming overH = 12 which means every stream assigned with 2 lines
    //last one with 1, and the last one seems not to function correctly
    if (op.hasAttr("floatPrecision"))
    {
            if (op.getOpType() == "Conv" && op.get<bool>("floatPrecision") && 
                op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 1024 &&
                op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 30 && 
                op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 23 &&
                op.getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType() == mv::DType("UInt8"))
            {
            auto outputTilesShape = tileSpatialOutputSize(op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION], streamShape["H"]);
            for (auto tileShape:outputTilesShape)
                if (tileShape == 1)
                    return FailCause::SoftwareDeconvolutionSet;
            }
    }

    //temporarily disable the SplitOverHOverlapped for custom network kernel size 7x7 subtensors not correct
    if (clustering == "SplitOverH" && op.getOpType() == "Conv" && isCMConv && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 3 &&
        op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 72 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 72 &&
        op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 7 && op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 7)
        return FailCause::SplitOverHOverlappedWronglyComputed;

        //temporarily disable the SplitOverHOverlapped for custom network kernel size 1x7 subtensors not correct
    if (clustering == "SplitOverH" && op.getOpType() == "Conv" && isCMConv && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 1 &&
        op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 1024 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 64 &&
        op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 1 && op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 7)
        return FailCause::SplitOverHOverlappedWronglyComputed;

    // This is intended to be a temporary workaround for ModelF, layer 'af_01/01_conv/Conv2D', which does work with SOH
    // It has not been root caused to the compiler or runtime but as of now the compiler logic seems OK
    if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isCMConv && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 16 &&
        op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 1024 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 64 &&
        op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 16 && op.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 1024 &&
        op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 64 && op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 7 &&
        op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 1)
        return FailCause::SpiltOverHForConvModelF;

    return FailCause::Pass; //good strategy
}

bool StrategyManagerSimple::decideWeightsPipelineable(mv::Op& op, StrategySet& strategy, bool allowSpilling)
{
    if(!globalEnablePipelining)
        return false;

    if(op.hasAttr("DilatedSubConv") && op.get<bool>("DilatedSubConv"))
        return false;

    // Is this op type enabled for pipelining? For now, default turned on for just conv and dw
    if(!createStrategyFromBool(op, "pipelining"))
        return false;

    auto stream = strategy["streaming"].get<Shape>();
    auto clustering = strategy["clustering"].get<std::string>();
    auto spilling = strategy["spilling"].get<bool>();
    auto parentSpilling = strategy["parentSpilling"].get<bool>();

    //Note: For now, only consider weights pipelining in the case where both activations
    // will live in CMX
    if(parentSpilling || (!allowSpilling && spilling))
        return false;

    //TODO re-enable pipelining in this case with a new overhead number
    if(clustering == "SplitOverH")
        return false;

    if(clustering == "HKSwitch")
        return false;

    //Note: at this stage, only check pipelining of weights over K
    if((stream["B"] * stream["C"] * stream["H"]) > 1)
        return false;

    auto inputSparsity = strategy["inputSparsity"].get<bool>();
    auto outputSparsity = strategy["outputSparsity"].get<bool>();
    auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
    size_t input, output, weights;
    // in case initialization in memorySize fails
    input = output = weights = 0;
    std::tie(input, output, weights) = memorySize(op,
                                                    totalClusters,
                                                    clustering,
                                                    inputSparsity,
                                                    outputSparsity,
                                                    weightsSparsity,
                                                    stream,
                                                    requiresFakeActivationSparsity(op),
                                                    spilling,
                                                    parentSpilling);

    //Note: avoid pipelining small weights. This number came out of experiments, when
    // the overhead starts to appear...
    if(weights < cmxPipeLineWeightsOverhead)
        return false;

    if(stream["K"] > 1) // Full activation in CMX, stream weights
    {
        //Note: memory size function is smart enough to take care of input/output size relative to spilling
        auto memReq = input + output + 2*weights;
        if(memReq < clusterMemory)
        {
            return true;
        }
    }
    
    return false; // without streaming, there is no pipelining
}

bool StrategyManagerSimple::isSOKCompatible(mv::Op& op)
{
    auto weightsShape = op.getInputTensor(1)->getShape();
    auto numInChannels = weightsShape[KERNEL_INPUT_CHANNELS];
    auto numOutChannels = weightsShape[KERNEL_OUTPUT_CHANNELS];
    if (op.getOpType() == "Conv")
    {
        if((numOutChannels/totalClusters < 16))
            return false;
    }
    else
    {
        if( (numInChannels/totalClusters < 16))
            return false;
    }
    return true;
}

bool StrategyManagerSimple::decideCMXable(mv::Op& op, bool input)
{
    if(op.getOpType() == "Concat")
        return false;
    if(input && op.getOpType() != "Input")
    {
        size_t inputSize = op.getInputTensor(0)->computeTotalSize();

        if(op.getOpType() == "Eltwise")
            inputSize += op.getInputTensor(1)->computeTotalSize();

        if(((double)inputSize/(double)totalClusters) < clusterMemory)
            return true;
    }
    if(!input && op.getOpType() != "Output")
    {
        auto outputTensor = op.getOutputTensor(0);
        size_t outputSize = outputTensor->computeTotalSize();

        if(((double)outputSize/(double)totalClusters) < clusterMemory)
            return true;
    }
    return false;
}

int8_t StrategyManagerSimple::checkInOutSizes(mv::Op& op, size_t input_gate)
{
    int8_t executableInHW = 0;
    if (op.getInputTensor(input_gate)->getShape()[mv::IO_WIDTH_DIMENSION] > mv::MAX_DIM_SIZE ||
        op.getInputTensor(input_gate)->getShape()[mv::IO_HEIGHT_DIMENSION] > mv::MAX_DIM_SIZE ||
        op.getInputTensor(input_gate)->getShape()[mv::IO_CHANNEL_DIMENSION] > mv::MAX_DIM_SIZE ||

        op.getOutputTensor(input_gate)->getShape()[mv::IO_WIDTH_DIMENSION] > mv::MAX_DIM_SIZE ||
        op.getOutputTensor(input_gate)->getShape()[mv::IO_HEIGHT_DIMENSION] > mv::MAX_DIM_SIZE ||
        op.getOutputTensor(input_gate)->getShape()[mv::IO_CHANNEL_DIMENSION] > mv::MAX_DIM_SIZE )
            executableInHW = 1;
    return executableInHW;
}

int8_t StrategyManagerSimple::checkKernelSizes(mv::Op& op)
{
    int8_t executableInHW = 0;
    std::array<unsigned short, 4> kernel = {1,1,1,1};//for non conv IN OUT CHANNEL dims = 1
    if (op.hasAttr("kSize"))
    {
        if (op.getOpType() == "MaxPool" || op.isEltwiseTypeOp())
        {
            kernel[mv::KERNEL_WIDTH] = op.get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_WIDTH];
            kernel[mv::KERNEL_HEIGHT] = op.get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_HEIGHT];
        }
        else if (op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
        {
            kernel[mv::KERNEL_WIDTH] = op.getInputTensor(1)->getShape()[mv::IO_WIDTH_DIMENSION];
            kernel[mv::KERNEL_HEIGHT] = op.getInputTensor(1)->getShape()[mv::IO_HEIGHT_DIMENSION];
            kernel[mv::KERNEL_INPUT_CHANNELS] = op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS];
            kernel[mv::KERNEL_OUTPUT_CHANNELS] = op.getInputTensor(1)->getShape()[mv::KERNEL_OUTPUT_CHANNELS];
        }
    }
    if (kernel[mv::KERNEL_WIDTH] > mv::MAX_KERNEL ||
            kernel[mv::KERNEL_HEIGHT] > mv::MAX_KERNEL ||
            kernel[mv::KERNEL_INPUT_CHANNELS] > mv::MAX_DIM_SIZE ||
            kernel[mv::KERNEL_OUTPUT_CHANNELS] > mv::MAX_DIM_SIZE  )
        executableInHW = 3;

    if (kernel[mv::KERNEL_WIDTH] != kernel[mv::KERNEL_HEIGHT])
        log(mv::Logger::MessageType::Debug, op.getName() + "has asymmetric kernel sizes" + \
                                    " k_w" + std::to_string(kernel[mv::KERNEL_WIDTH]) + \
                                    " k_h" + std::to_string(kernel[mv::KERNEL_HEIGHT]));
    return executableInHW;
}


int8_t StrategyManagerSimple::checkStrideSizes(mv::Op& op)
{
    int8_t executableInHW = 0;
    std::array<unsigned short, 2> stride = {1,1};
    if (op.hasAttr("stride"))
        stride = op.getAttrs().at("stride").get<std::array<unsigned short, 2>>();
    if (stride[mv::STRIDE_WIDTH] > mv::MAX_STRIDE || stride[mv::STRIDE_HEIGHT] > mv::MAX_STRIDE)
        executableInHW += 3;

    if (stride[mv::STRIDE_WIDTH] != stride[mv::STRIDE_HEIGHT])
        log(mv::Logger::MessageType::Debug, op.getName() + "has asymmetric strides" + \
                                        " s_w" + std::to_string(stride[mv::STRIDE_WIDTH]) + \
                                        " s_h" + std::to_string(stride[mv::STRIDE_HEIGHT]));
    return executableInHW;
}

int8_t StrategyManagerSimple::checkHWUnsupportedOp(mv::Op& op)
{
    int8_t executableInHW = 0;
    if (op.isHardwarizable())
    {
        for (std::size_t input_gates = 0; input_gates < op.getInputTensor().size(); input_gates++)
        {
            if (input_gates == 0)
            {
                executableInHW += checkInOutSizes(op, input_gates);
                executableInHW += checkKernelSizes(op); //Note: all the ops have maximum a second input (weights) at G.O stage
                executableInHW += checkStrideSizes(op);
            }
            else if (input_gates == 1 && op.getOpType() == "Eltwise")
            {
                if (op.getInputTensor(input_gates)->getShape()[mv::IO_WIDTH_DIMENSION] > mv::MAX_DIM_SIZE ||
                    op.getInputTensor(input_gates)->getShape()[mv::IO_HEIGHT_DIMENSION] > mv::MAX_DIM_SIZE ||
                    op.getInputTensor(input_gates)->getShape()[mv::IO_CHANNEL_DIMENSION] > mv::MAX_DIM_SIZE)
                    executableInHW += 1;
            }
        }
    }
    return executableInHW;
}

void StrategyManagerSimple::generateStrategySetForLayer(mv::Op& op,std::vector<StrategySet>& strategyVec)
{
    auto findStrategy = [](std::vector<Attribute>& vec,const std::string& str) ->bool { for(const auto elem : vec) if(str==elem.get<std::string>()) return true; return false;};
    int8_t success = checkHWUnsupportedOp(op);
    if (success != 0)
    {
        if (success == 1)
            log(mv::Logger::MessageType::Warning, "The limitation of the tensor dimension 8192 might be exceeded for \
                the operation " + op.getName());
            else
            log(mv::Logger::MessageType::Error, "Unsupported kernel/stride combination for DpuTask for \
                the operation " + op.getName());
    }

    std::vector<Attribute> spillingPool;
    if(globalForceSpilling)
        spillingPool.push_back(true);
    else
        spillingPool = createTStrategyPoolFromBool(op, "forceSpilling");

    auto opType = op.getOpType();
    std::vector<Attribute> parentSpillingPool;
    if(globalForceSpilling || opType == "Input" || opType == "Output")
        parentSpillingPool.push_back(true);
    else
        parentSpillingPool = {true, false};

    std::vector<Attribute> clusteringStrategyPool;

    if(totalClusters == 1)
        clusteringStrategyPool.push_back(std::string("Clustering"));
    else if (totalClusters > 1)
        clusteringStrategyPool = createStrategyPoolFromStrategySet(op,"clusteringStrategies");
    else
        throw LogicError(*this, "Graph Optimizer unable to determine number of clusters");

    
    std::vector<Attribute> streamingStrategyPool = createStrategyPoolFromStrategySet(op,"streamingStrategies");

    bool hasStreamOverK = false;
    bool hasStreamOverH = false;
    bool hasStreamOverC = false;
    bool hasStreamOverN = false;

    if(globalEnableStreaming)
    {
        hasStreamOverK = findStrategy(streamingStrategyPool,"StreamOverK");
        hasStreamOverH = findStrategy(streamingStrategyPool,"StreamOverH");
        hasStreamOverC = findStrategy(streamingStrategyPool,"StreamOverC");
        hasStreamOverN = findStrategy(streamingStrategyPool,"StreamOverN");
    }

    bool weightsSparsity = false;
    if(requiresWeightsSparsity(op))
        weightsSparsity = true;
    else if(globalEnableWeightsSparsity)
        weightsSparsity = decideWeightsSparsity(op);


    for(auto spilling : spillingPool)
    {
    for(auto parentSpilling : parentSpillingPool)
    {
    for(auto clustering : clusteringStrategyPool)
    {
        // Make decision about input activation sparsity, depending on clustering strategy
        std::vector<Attribute> inputActivationSparsity = createTFStrategyPoolFromBool(op, "inputActivationSparsity");
        std::vector<Attribute> outputActivationSparsity = createTFStrategyPoolFromBool(op,"outputActivationSparsity");

        if(requiresActivationSparsity(op, clustering.get<std::string>()))
        {
            inputActivationSparsity.clear();
            inputActivationSparsity.push_back(true);
        }

        for( auto inputSparsity : inputActivationSparsity)
        {
        for( auto outputSparsity : outputActivationSparsity)
        {
            bool fakeSparsity = requiresFakeActivationSparsity(op);

            // Determine streaming options
            // 0. Determine if streams over H are possible
            // 1. Determine if streams over N are required
            // 2. Determine if streams over K are possible
            // 3. If no streams over H or K will fit, enable nested streaming
            // 4. Nested loops over generated streaming options to produce all strategy options
            std::vector<size_t> streamsOverH;
            if(hasStreamOverH)
                streamsOverH = getStreamsOverH(op, clustering, {1,1,1,1,1}, inputSparsity.get<bool>(),
                                                    outputSparsity.get<bool>(), weightsSparsity, fakeSparsity,
                                                    spilling.get<bool>(), parentSpilling.get<bool>());
            else
                streamsOverH.push_back(1);

            // Stream over batch, each batch must be it's own stream
            unsigned n = 1;
            if(hasStreamOverN && op.getInputTensor(0)->getShape()["N"] > 1)
                n = op.getInputTensor(0)->getShape()["N"];

            std::vector<size_t> streamsOverK;
            if(hasStreamOverK)
            {
                // streamsOverK = getMaxStreamOverK(op);
                streamsOverK = getStreamsOverK(op, clustering, {1,1,1,1,n}, inputSparsity.get<bool>(),
                                                outputSparsity.get<bool>(), weightsSparsity, fakeSparsity, 
                                                spilling.get<bool>(), parentSpilling.get<bool>());
            }
            else
            {
                streamsOverK.push_back(1);
            }

            // Sad hack - due to recent changes on sparse data calculation
            // now the minimal streams for K in some cases are lower, and this
            // bring some performance regressions, on SSD512 for example
            // https://jira.devtools.intel.com/browse/EISW-7241
            // TODO: Future taks to add heuristics that increase the amount of
            // streaming, so we can get rid of this harcoding
            if (op.getOpType() == "Conv" &&
                op.getInputTensor(0)->getDType().getSizeInBytes() == 1)
            {
                if (op.getInputTensor(0)->getShape() == mv::Shape({32,32,512,1}) &&
                    op.getInputTensor(1)->getShape() == mv::Shape({3,3,512,512}) &&
                    op.getOutputTensor(0)->getShape() == mv::Shape({32,32,512,1}) &&
                    clustering.get<std::string>() == "SplitOverH" &&
                    hasStreamOverK) {
                    streamsOverK = {8};
                    model_.getInput()->set<bool>("hardcoded_streams", true);
                    log(mv::Logger::MessageType::Warning, "Following op " + op.getName() +
                        "  has been forced to be scheduled with fixed K stream of " +
                        std::to_string(streamsOverK[0]));
                }
            }

            std::vector<size_t> streamsOverC;
            if (hasStreamOverC)
                streamsOverC = getStreamsOverC(op, clustering, {1,1,1,1,n}, inputSparsity.get<bool>(),
                                                outputSparsity.get<bool>(), weightsSparsity, fakeSparsity, spilling.get<bool>());
            else
                streamsOverC.push_back(1);

            bool enableNestedStreaming = false;
            auto maxK = streamsOverK.back();
            auto memK = memorySize(op, totalClusters, clustering.get<std::string>(),inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,1,1,maxK,n},fakeSparsity, spilling.get<bool>(), parentSpilling.get<bool>());
            auto memoryMaxK = std::get<0>(memK) + std::get<1>(memK) + std::get<2>(memK);
            auto maxH = streamsOverH.front();
            auto memH = memorySize(op,totalClusters, clustering.get<std::string>(),inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,maxH,1,1,n},fakeSparsity, spilling.get<bool>(), parentSpilling.get<bool>());
            auto memoryMaxH =  std::get<0>(memH) + std::get<1>(memH) + std::get<2>(memH);


            // If streaming is enabled, but streaming over k or h alone doesn't fit, enable nested streaming
            //nested streaming is a trick to fit an operation when both weight tensor and input tensors do not fit on cmx
            //not much sense to try fit the output tensor there, disabling in case of no spilling
            if( hasStreamOverK && hasStreamOverH && spilling &&
                (memoryMaxH > clusterMemory) &&
                (memoryMaxK > clusterMemory) )
            {
                //Note: We generate only 1 stream over K possibility now, just enough to fit
                // If we need nested streaming, generate a range of possibilities as we don't know what
                // will fit with the range of streams over H
                streamsOverK = getMaxStreamOverK(op);
                enableNestedStreaming = true;
            }
            for(const auto k : streamsOverK)
            {
                if(enableNestedStreaming) // generate h ranges on the fly
                {
                    streamsOverH = getStreamsOverH(op, clustering, {1,1,1,k,1}, inputSparsity.get<bool>(),
                                                    outputSparsity.get<bool>(), weightsSparsity, fakeSparsity, 
                                                    spilling.get<bool>(),  parentSpilling.get<bool>());
                }
                for(const auto h : streamsOverH)
                {
                    for(const auto c : streamsOverC)
                    {
                        if((h > 1) && (c > 1)) //Fast hack to disable nested streaming with C
                            continue;
                        if((h > 1) && (n > 1)) //Fast hack to disable nested streaming with n
                            continue;
                        if( !enableNestedStreaming && ((h>1) && (k>1))) // Skip nested streams unless necessary
                            continue;
                        if( enableNestedStreaming && ((h==1) || (k==1))) // If need nested streams, ignore non-nested
                            continue;
                        if( enableNestedStreaming && (!parentSpilling || !spilling) )
                            continue; // Only nested stream if we're doing it to fit at all

                        Shape streamShape({1,h,c,k,n}); //Stream over W is 1. Not implemented.

                        StrategySet s;
                        s["name"] = op.getName();
                        s["id"] = (unique_ctr++);
                        s["inputSparsity"] = inputSparsity;
                        s["outputSparsity"] = outputSparsity;
                        s["weightsSparsity"] = weightsSparsity;
                        s["spilling"] = spilling;
                        s["parentSpilling"] = parentSpilling;
                        s["clustering"] = clustering;
                        s["streaming"] = streamShape;
                        s["pipeline"] = decideWeightsPipelineable(op, s, false);
                        s["spillPipeline"] = decideWeightsPipelineable(op, s, true);
                        s["inputMustSpill"] = decideCMXable(op, true);
                        s["outputMustSpill"] = decideCMXable(op, false);

                        //Function to prune strategies that will have only infinite edges in or out (or both), improves performance
                        auto strategyCheck = validateStrategy(op,s);
                        // std::cout << op.getName() << " : " << clustering.toString() << " : " << streamShape.toString() << " : pS " << parentSpilling.toString() << " : S " << spilling.toString() << " : I " << inputSparsity.toString() << " : O " << outputSparsity.toString() << " = " << failure_causes[strategyCheck]<< std::endl;
                        if(strategyCheck != FailCause::Pass)
                            continue;

                        strategyVec.push_back(s);

                        //    std::cout << "Name: " + op.getName() << " ID " << s["id"].toString()<< std::endl;
                        //    std::cout << "Input Sparsity: " + inputSparsity.toString() << std::endl;
                        //    std::cout << "Output Sparsity: " + outputSparsity.toString() << std::endl;
                        //    std::cout << "Weights Sparsity: " + weightsSparsity << std::endl;
                        //    std::cout << "Spilling: " + spilling.toString() << std::endl;
                        //    std::cout << "MCStrategy: " + clustering.toString() << std::endl;
                        //    std::cout << "Streaming(W,H,C,K,N): " + streamShape.toString() << std::endl<<std::endl;

                    }
                }
            }
            }
            }
            }
        }
    }
    if(strategyVec.empty())
        throw LogicError(*this,"No strategies created for layer " + op.getName() + ". Layer possibly unsupported.");
}
}
}