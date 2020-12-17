#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include <iostream>
#include <iomanip>
#include "chrono"


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

size_t alignedWeightsSize(const mv::Data::TensorIterator tensorToSize, std::string clustering, size_t totalClusters){
    auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
    auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits()/8.0);
    size_t alignedFullInputChannels = mv::round_up(tensorToSize->getShape()[mv::KERNEL_INPUT_CHANNELS], 16);

    size_t alignedFullOutputChannels = mv::round_up(tensorToSize->getShape()[mv::KERNEL_OUTPUT_CHANNELS], 16);

    if(clustering == "SplitOverK")
    {
        size_t alignedSplittedOutputChannels = div(alignedFullOutputChannels,totalClusters);
        alignedSplittedOutputChannels = mv::round_up(alignedSplittedOutputChannels, 16);

        return (alignedFullInputChannels * alignedSplittedOutputChannels *
                tensorToSize->getShape()[mv::KERNEL_WIDTH] * tensorToSize->getShape()[mv::KERNEL_HEIGHT])
                * dtypeMultiplier;
    }
    else{
        return (alignedFullInputChannels * alignedFullOutputChannels *
                tensorToSize->getShape()[mv::KERNEL_WIDTH] * tensorToSize->getShape()[mv::KERNEL_HEIGHT])
                * dtypeMultiplier;
    }
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
            weightSize += alignedWeightsSize(opIt->getInputTensor(1), clusterStrategy, totalClusters);
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

    // Step 3. Find valid stream starting from proposedStreams and decreasing towards originalHStreams
    // Ensures lines are divided in such a way that it still fits in CMX, no workload issues etc
    auto optStream = findOptimalValidStream(model, opIt, proposedStreams);

    if(optStream < originalHStream)
        return originalHStream; // Never return fewer streams than GO assigned
    
    return optStream;
}

void saveNewStreamingStrategiesToJson(const mv::pass::PassEntry& pass, const mv::Attribute& streamingStrategyElements) {
    pass.log(mv::Logger::MessageType::Debug, "Saving New Streaming Strategies to JSON file");
    std::ofstream jsonOutputFile;
    std::string jsonOutFileName = "./output/mcmCompiler_new_h_streaming_strategy_output.json";
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
            size_t originalHStream = streams[1].get<int>("H");
            
            // Step 1. Choose optimal stream over H number for this op
            auto newHstream = findOptimalStream(model, opIt, originalHStream);
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