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

static void validateStreamingOverKForChainPipeliningFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void addActivationStreamingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

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

static void addActivationStreamingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AddActivationStreaming)
        .setFunc(addActivationStreamingFcn)
        .setDescription(
            "This pass increases activation streaming over the H dimension for performance."
        );
    }
}

std::size_t activationTensorSize(mv::Data::OpListIterator opIt, std::string clustering, size_t hStream, bool isCMConv, bool isInput, size_t totalClusters)
{
    auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
    auto inputTensor = opIt->getInputTensor(0);
    auto outputTensor = opIt->getOutputTensor(0);
    mv::Data::TensorIterator tensorToSize;
    if(isInput)
        tensorToSize = inputTensor;
    else
        tensorToSize = outputTensor;
    
    auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits()/8.0);
    auto opType = opIt->getOpType();
    auto tensorShape = tensorToSize->getShape();
    size_t fullOutputTensorHeight = outputTensor->getShape()[mv::IO_HEIGHT_DIMENSION];

    //Note: For now, all batched operations stream over batch so that N = 1
    size_t streamedBatch = 1;

    size_t fullTensorHeight = tensorShape[mv::IO_HEIGHT_DIMENSION];
    size_t streamedHeight = fullTensorHeight;

    size_t fullTensorChannels = tensorShape[mv::IO_CHANNEL_DIMENSION];
    size_t streamedChannels = fullTensorChannels;

    if(hStream > 1)
    {
        auto newOutputSizes = mv::tileSpatialOutputSize(fullOutputTensorHeight, hStream);
        streamedHeight = newOutputSizes.front();
        if(streamedHeight < newOutputSizes.back())
            streamedHeight = newOutputSizes.back();

        if(isInput)
        {
            unsigned short kernelStride;
            if (opIt->hasAttr("stride"))
                kernelStride = opIt->get<std::array<unsigned short, 2>>("stride")[mv::KERNEL_HEIGHT];
            else
                kernelStride = 1;//fake stride
        
            streamedHeight = streamedHeight * kernelStride;
        }

        // Kernel and padding will add extra lines to final size of streamed portion
        size_t kHeight = 1;
        std::array<unsigned short, 4> padding;
        if(  (opIt->getOpType() == "Conv") || (opIt->getOpType() == "DepthwiseConv") )
            kHeight = opIt->getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT];
        else if (opIt->getOpType() == "MaxPool")
            kHeight = opIt->get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_HEIGHT];
        if (opIt->hasAttr("padding"))
            padding = opIt->get<std::array<unsigned short, 4>>("padding");
        else
            padding = {0, 0, 0, 0};

        size_t extraLines = 0;

        if(extraLines < kHeight-1)
        {
            extraLines = kHeight -1;
        }

        if(padding[2] > padding[3])
        {
            if(padding[2] > extraLines)
                extraLines = padding[2];
        }
        else
        {
            if(padding[3] > extraLines)
                extraLines = padding[3];
        }

        streamedHeight += extraLines;
    }
    if(clustering == "SplitOverH")
    {
        streamedHeight = div(streamedHeight,totalClusters);
    }
    if((opType == "Conv" || opType == "DepthwiseConv" || 
        opType == "MaxPool" || opType == "Eltwise") && 
        !(isCMConv && isInput)) //for DPU tasks we align both input (except CM) and output tensors channels
    {
        streamedChannels = mv::round_up(streamedChannels, 16);
    }

    return tensorShape[mv::IO_WIDTH_DIMENSION] * streamedHeight * streamedChannels * streamedBatch * dtypeMultiplier;
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

bool validateHStream(mv::Data::OpListIterator opIt, std::string clustering, std::size_t splits, size_t totalClusters)
{
    if( opIt->getOpType() == "Conv" || opIt->getOpType() == "DepthwiseConv")
    {
        if(clustering == "SplitOverH")
        {
            auto weightsShape = opIt->getInputTensor(1)->getShape();
            //Try to guess subtensor height, and avoid situations where kernel is bigger than last workload dimension
            auto outputHeight = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
            auto workloadHeight = ceil((double)outputHeight / (double)(totalClusters * splits));
            if(totalClusters > 1) //last
                workloadHeight = outputHeight - (workloadHeight * (totalClusters-1)); //get remaining height
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

std::tuple<size_t,size_t,size_t> memorySize(mv::ComputationModel& model, mv::Data::OpListIterator opIt, size_t hStream)
{
    auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
    mv::OpModel om(model);
    size_t inputSize = 0;
    size_t outputSize = 0;
    size_t weightSize = 0;
    size_t weightTableSize = 0;

    auto opType = opIt->getOpType();
    auto isCMConv = false;
    auto clusterStrategy = opIt->get<std::string>("splitStrategy");
    bool spilling = opIt->get<bool>("goPredictsSpill");
    
    auto prevOp = om.getSourceOp(opIt->getInputTensor(0));
    bool parentSpilling = prevOp->get<bool>("goPredictsSpill");
    auto inputSparse = opIt->get<bool>("inputActivationSparsity");
    auto outputSparse = opIt->get<bool>("outputActivationSparsity");
    auto weightsSparse = opIt->get<bool>("weightsSparsity");

    auto globalParams = model.getGlobalConfigParams();
    bool enableChannelMajorConv = globalParams->get<bool>("enable_channel_major_conv");
    size_t totalClusters = globalParams->get<int>("Number_of_Clusters");

    if(enableChannelMajorConv && opIt->supportsCMConv())
        isCMConv = true;

    // Note: when an operation is streaming activations, but it's parent didn't spill, the input won't be streamed
    size_t inputTempStreamConfig = hStream;
    if(!parentSpilling)
        inputTempStreamConfig = 1;
    inputSize = activationTensorSize(opIt,clusterStrategy,inputTempStreamConfig, isCMConv, true, totalClusters);

    //Note: when streaming operations are not spilled, full output (not streamed size) must be counted
    size_t outputTempStreamConfig = hStream;
    if (!spilling)
        outputTempStreamConfig = 1;
    outputSize = activationTensorSize(opIt,clusterStrategy,outputTempStreamConfig, isCMConv, false, totalClusters);

    size_t outChannels = opIt->outputSlots() ? opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] : 0;
    size_t alignedFullChannels = mv::round_up(outChannels, 16);
    size_t alignedSplittedChannels = alignedFullChannels;
    if(clusterStrategy == "SplitOverK") {
        alignedSplittedChannels =  mv::round_up(alignedSplittedChannels/totalClusters, 16);
    }

    if(opType == "Conv" || opType == "DepthwiseConv")
    {
        mv::Shape weightStreamConfig = {1,1,1,1,1};
        weightTableSize = 16 * alignedSplittedChannels;
        if (opType == "Conv")
        {
            weightSize += alignedWeightsSize(opIt->getInputTensor(1), {1, 1, 1, 1, 1}, clusterStrategy);
        }
        else
        {
            weightSize += opIt->getInputTensor(1)->computeTotalSize(16, false, false, true);
            if(clusterStrategy == "SplitOverK")
                weightSize = div(weightSize,totalClusters);
        }

    }
    else if(opType == "MaxPool")
    {
        weightTableSize = 16 * alignedSplittedChannels;
        weightSize = 0;
    }

    // Fake Sparsity
    if(opType == "MaxPool" || opType == "DepthwiseConv" || isCMConv)
    {
        uint16_t kernelW, kernelH;


        auto strides = opIt->get<std::array<unsigned short, 2>>("stride");

        if (opIt->hasAttr("kSize"))
        {
            auto kernelShape = opIt->get<std::array<unsigned short, 2>>("kSize");
            kernelW = kernelShape[0];
            kernelH = kernelShape[1];
        }
        else
        {
            auto weightsShape = opIt->getInputTensor(1)->getShape();
            kernelW = weightsShape[mv::KERNEL_WIDTH];
            kernelH = weightsShape[mv::KERNEL_HEIGHT];
        }

        mv::DType dataType = opIt->getInputTensor(0)->getDType();
        if (opType != "MaxPool")
            dataType = opIt->getInputTensor(1)->getDType();

        auto windowsSize = getWindowSize(kernelW, strides[0], dataType);
        size_t fakeSparsitySize = 0;
        if ((opType == "MaxPool") || (opType == "DepthwiseConv"))
        {
            //inputChannels = 1
            auto bitpatternSize = windowsSize*kernelH;
            //ndims = {16 * static_cast<std::size_t>(std::ceil(bitpatternSize / 128.0)), 1, 1, 1};
            fakeSparsitySize = 16 * static_cast<std::size_t>(std::ceil(bitpatternSize / 128.0));
        }
        // Channel Major Convolution doesn't need rounding of channels
        else if (isCMConv)//isChannelMajorConvolution
        {

            std::size_t outputChannels =  opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
            outputChannels = outputChannels;///streamConfig["K"];
            std::size_t inputChannels = opIt->getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

            auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8.0)); //how many bytes we need per window
            auto NumberOfRowsSparistyBytes = static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0 ));

            //ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
            fakeSparsitySize = 16*NumberOfRowsSparistyBytes*outputChannels;

        }
        inputSize += fakeSparsitySize;
    }
    if(inputSparse){
        //Alignment due to input channels mult of 16 requirement
        //Only ZM Conv and Elwise are sparse consumers, both need
        //input channels mult of 16
        auto tensorSize = opIt->getInputTensor(0)->computeTotalSize(16, false, false, true);
        size_t streamDivisor = hStream;
        //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseInputSize = std::ceil((double)tensorSize /
            (8 * opIt->getInputTensor(0)->getDType().getSizeInBytes()));
        //Storage element table calculation, 4 bytes pointers
        //Bigger with C streaming
        sparseInputSize += opIt->getInputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION] *
            opIt->getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] * 4;
        //Alignment due to bus access requirements
        sparseInputSize = mv::round_up(sparseInputSize, 16);
        inputSize += (sparseInputSize / streamDivisor);
    }
    if(outputSparse){
        //Alignment due to output channels mult of 16 requirement
        //Only ZM Conv and Elwise are sparse consumers
        auto tensorSize = opIt->getOutputTensor(0)->computeTotalSize(16, false, false, true);
        size_t streamDivisor = hStream;
        //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseOutputSize = std::ceil((double)tensorSize /
            (8 * opIt->getOutputTensor(0)->getDType().getSizeInBytes()));
        //Storage element table calculation, 4 bytes pointers
        //Bigger with K streaming
        sparseOutputSize += opIt->getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION] *
            opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] * 4;
        //Alignment due to bus access requirements
        sparseOutputSize = mv::round_up(sparseOutputSize, 16);
        outputSize += (sparseOutputSize / streamDivisor);
    }
    if(weightsSparse){
        //Alignment due to output/input channels mult of 16 requirement
        auto tensorSize = opIt->getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] *
            opIt->getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] *
            mv::round_up(opIt->getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS], 16) *
            alignedSplittedChannels;
        //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseWeightSize = std::ceil((double)tensorSize / 8);
        //Sparse pointers taken into account in weight table ...
        sparseWeightSize = mv::round_up(sparseWeightSize, 16);
        weightSize += sparseWeightSize;
    }

    weightSize += weightTableSize;

    return std::tuple<std::size_t,std::size_t,std::size_t>(inputSize, outputSize,weightSize);
}

// Gives the minimum number of streams over H to fit this layer, or if no number of streams enable streaming
// (for example, weights don't fit) then return 0
unsigned getMinStreamOverH(mv::ComputationModel& model, mv::Data::OpListIterator opIt)
{
    auto globalParams = model.getGlobalConfigParams();
    size_t totalClusters = globalParams->get<int>("Number_of_Clusters");
    size_t clusterMemory = globalParams->get<unsigned>("cmx");
    auto clusterStrategy = opIt->get<std::string>("splitStrategy");

    size_t input, output, weights;
    // in case initialization in memorySize fails
    input = output = weights = 0;
    std::tie(input, output, weights) = memorySize(model, opIt, 1);
    auto activationsSize = input + output;
    auto weightsSize = weights;
    double availableMemory = (double) clusterMemory - (double) weightsSize;

    if (availableMemory <= 0) // Weights don't fit, can't stream over H
        return 0;

    // Keep increasing H until we find one big enough to fit, or we run out of H dimension to stream
    auto outputHeight = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];

    // Every output slice must have at least one line to compute
    unsigned upperBoundH = outputHeight;
    if(clusterStrategy == "SplitOverH")
    {
        upperBoundH = upperBoundH/totalClusters;
    }

    // Start searching for min stream at naive requirement for splits to fit, rather than 1
    for(unsigned splits = ceil((double)activationsSize/availableMemory); splits <= upperBoundH; splits++)
    {
        auto memFitCheck = memorySize(model, opIt, splits);

        if((std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                validateHStream(opIt, clusterStrategy, splits, totalClusters))
        {
            return splits;
        }
    }

    return 0;
}

// Note: Validate a stream so that its largest slice fits in CMX and no workload issues
unsigned findOptimalValidStream(mv::ComputationModel& model, mv::Data::OpListIterator opIt, size_t startStream)
{
    auto globalParams = model.getGlobalConfigParams();
    size_t totalClusters = globalParams->get<int>("Number_of_Clusters");
    size_t clusterMemory = globalParams->get<unsigned>("cmx");
    auto clusterStrategy = opIt->get<std::string>("splitStrategy");

    for(unsigned splits = startStream; splits >= 1; splits--)
    {
        auto memFitCheck = memorySize(model, opIt, splits);
        if( (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
            validateHStream(opIt, clusterStrategy, splits, totalClusters))
                return splits;
    }

    return 1;
}

bool isStreamOptimizable(mv::ComputationModel& model, mv::Data::OpListIterator opIt, std::vector<mv::Element> streaming_strategy)
{
    auto opType = opIt->getOpType();
    if(!(opIt->hasTypeTrait("optimizable") && (opType == "Conv" || opType == "MaxPool" || opType == "DepthwiseConv")))
        return false;

    if (opIt->hasAttr("DilatedSubConv") && (opIt->get<bool>("DilatedSubConv")))
        return false;

    mv::OpModel om(model);
    auto globalParams = model.getGlobalConfigParams();
    auto prevOp = om.getSourceOp(opIt->getInputTensor(0));
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
        auto minSplits = getMinStreamOverH(model, opIt);
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
std::size_t findOptimalStream(mv::ComputationModel& model, mv::Data::OpListIterator opIt, size_t originalHStream)
{
    mv::OpModel om(model);
    auto globalParams = model.getGlobalConfigParams();
    size_t clusterMemory = globalParams->get<unsigned>("cmx");
    size_t totalClusters = globalParams->get<int>("Number_of_Clusters");
    auto clusteringStrategy = opIt->get<std::string>("splitStrategy");

    // Step 1. Decide which tensor will be the benchmark for how many streams we should do
    size_t input, output, weights;
    input = output = weights = 0;
    std::tie(input, output, weights) = memorySize(model, opIt, 1);

    // Step 2. Calculate a possible number of streams using experimetnally found magic number
    // Idea is, if possible, allow multiple slices to fit in CMX to maximize paths
    size_t magicStreams = std::ceil((2*input + output)/ ((clusterMemory-weights)*0.6));
    if(magicStreams < originalHStream)
        magicStreams = originalHStream; //If GO gave carved it up into smaller pieces, must be for a reason
    else if(magicStreams > originalHStream*3)
        magicStreams = originalHStream*3; // Let's not get crazy with the H streams

    // Can't exceed the max, which ensures at least one line of output for each stream to compute
    size_t maxStreams = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
    if(clusteringStrategy == "SplitOverH") maxStreams = maxStreams/totalClusters;

    size_t proposedStreams = std::min(magicStreams, maxStreams); //will be in range [originalHStream, maxStream]

    std::cout << "  proposed stream is h=" << proposedStreams << std::endl;

    // Step 3. Find valid stream starting from proposedStreams and decreasing towards originalHStreams
    // Ensures lines are divided in such a way that it still fits in CMX, no workload issues etc
    auto optStream = findOptimalValidStream(model, opIt, proposedStreams);

    if(optStream < originalHStream)
        return originalHStream; // Never return fewer streams than GO assigned
    
    return optStream;
}

size_t perClusterWeightsSize(mv::Op& op, const mv::Attribute& clustering, bool weightsSparsity, const mv::Shape& streamConfig) {
    
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
                                  mv::ComputationModel& model,
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
     size_t minWeightsPerClusterPerChainConstant = 66560; // This value was derived from emperical testing 

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
                         streamingStrategyList, multiClusterStrategyList, model, opIt->getName());
                 
                 // Get the streaming and multicluster strategy assigned by GO for this operation 
                 auto graphOptimizerStreamingStrategy = graphOptimizerAssignedStategies.first;
                 auto graphOptimizerMultiClusterStrategy = graphOptimizerAssignedStategies.second;

                 bool isKStreaming = graphOptimizerStreamingStrategy[3].get<int>("K") > 1 ? true : false;
                 bool isHStreaming = graphOptimizerStreamingStrategy[1].get<int>("H") > 1 ? true : false;

                 // Get the output channels to determine the max possible K streams so we know the limit
                 alignedFullOutputChannels = mv::round_up(opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION], 16);

                 // Calculate the max possible K streams based on the multi-cluster strategy
                 maxpossibleStreams = floor(alignedFullOutputChannels / minOutputChannels[graphOptimizerMultiClusterStrategy]);
                
                 // Get the weights per cluster for this op
                 weightsPerCluster = weightsPerClusterPerOp.find(opIt->getName())->second;

                  // The operation must be already assigned stream over K and SOK and not be sream over H to be considered for a new K stream strategy 
                  if (isKStreaming && graphOptimizerMultiClusterStrategy == "SplitOverK" && !isHStreaming) {
                    
                    fullWeightsSizeOptimalKStreaming = {0,0};
                    if(minWeightsPerClusterPerChain[chainID] > 0)
                        fullWeightsSizeOptimalKStreaming = fullWeightsSizeForOpandOptimalKStreaming(graphOptimizerMultiClusterStrategy, weightsPerCluster, minWeightsPerClusterPerChain[chainID], isKStreaming,graphOptimizerStreamingStrategy[3].get<int>("K"), nClusters);
                    
                    fullWeightsSize = fullWeightsSizeOptimalKStreaming.first;
                    optimalNumberOfKStreams = fullWeightsSizeOptimalKStreaming.second;

                    // Assign the new streaming strategies
                    // The optimalNumberOfKStreams must be > 0, less than the max possible K streams and must not decrease the K streams assinged from the GO
                     if ((optimalNumberOfKStreams > 0) && (optimalNumberOfKStreams <= maxpossibleStreams) && (optimalNumberOfKStreams > graphOptimizerStreamingStrategy[3].get<int>("K"))) {

                        if(minWeightsPerClusterPerChain[chainID] < minWeightsPerClusterPerChainConstant)
                            minWeightsPerClusterPerChain[chainID] = minWeightsPerClusterPerChainConstant;

                         printInfoToFile(chainID, (opIt->getName()).c_str(), graphOptimizerStreamingStrategy[3].get<int>("K"),
                                         graphOptimizerStreamingStrategy[1].get<int>("H"), graphOptimizerMultiClusterStrategy.c_str(),
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
                                         graphOptimizerStreamingStrategy[1].get<int>("H"), graphOptimizerMultiClusterStrategy.c_str(),
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
     auto compDesc = model.getGlobalConfigParams();
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
        compDesc->set("streaming_strategy", newStreamingStrategies);
        saveNewStreamingStrategiesToJson(pass, newStreamingStrategies);
     }
 }

// Note: The idea of this pass is to increase streaming over the height dimension in specific cases
// to increase performance. Specifically, we consider DPU tasks (convs, dw, maxpool) that have their
// input tensor in DDR. Performance increase results because smaller DMA of input slices leads to 
// earlier start to compute, and the resulting smaller pieces are often easier for the scheduler 
// to schedule efficiently.
//
// Side Note: There are several reasons an input tensor might be in DDR, it could be the network
// input, or a spilled activation due to tensor size or need to change clustering strategy. In this
// pass we don't care why the tensor is in DDR, we just use the GO pass' prediction for where the 
// tensor will be located. We skip extra streams in the case that the GO can't predict tensor location
// such as after an explicit concat (could be CMXed later). For simplicity, we also only consider ops
// that were already streaming over H, but this pass could be extended to consider non-streaming ops.
void addActivationStreamingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("split_strategy"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No custom splitting strategy provided, exiting...");
        return;
    }
    auto streamingStrategies = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    std::vector<mv::Element> newStreamingStrategies;

    for(auto streamingStrategy : streamingStrategies)
    {
        std::string nodeName = streamingStrategy.get<std::string>("name_filter");
        // In case of manual strategy
        if (!om.checkOp(nodeName))
            continue;

        auto opIt = om.getOp(nodeName);
        bool updated = false;
        auto streams = streamingStrategy.get<std::vector<mv::Element>>("splits");

        // Step 0. Decide if we can insert activation streaming for this op
        if(isStreamOptimizable(model, opIt, streams))
        {
            std::cout << "stream for " << nodeName << " is optimizable "<< std::endl;
            size_t originalHStream = streams[1].get<int>("H");
            
            // Step 1. Choose optimal stream over H number for this op
            auto newHstream = findOptimalStream(model, opIt, originalHStream);
            std::cout << "  old stream was h=" << originalHStream << " and new stream is h=" << newHstream << std::endl;
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
    }

    //Step 4. Save the streaming strategies into the compilation descriptor to be read by the streaming pass
    globalParams->set<std::vector<mv::Element>>("streaming_strategy", newStreamingStrategies);
    saveNewStreamingStrategiesToJson(pass, newStreamingStrategies);
}