#include "include/mcm/pass/graphOptimizations/strategy_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

std::size_t mv::realTensorSize(const mv::Data::TensorIterator tensorToSize, const mv::Shape& streamingPool, bool isCMConv)
{
    mv::Shape worstStreamPool = streamingPool;

    //TODO harmonize this, for now only consider worst shape for nested streams
    if(streamingPool["H"] > 1 && streamingPool["K"] > 1)
    {
        mv::Shape tensorShape = tensorToSize->getShape();
        //update the streamingPool to the worst combination, based on slice sizes
        auto outputSize = tensorShape[mv::IO_HEIGHT_DIMENSION];
        auto numberOfSplits = streamingPool[mv::IO_HEIGHT_DIMENSION];

        auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
        auto newOutputSize = newOutputSizes.front();

        auto worstNumberOfSplits = outputSize/newOutputSize;
        worstStreamPool[mv::IO_HEIGHT_DIMENSION] = worstNumberOfSplits;
    }

    //TODO add handling for weights case if we dont align it to 16 always
    std::size_t streamDivisor = 1;
    for(std::size_t dim = 0; dim <  worstStreamPool.ndims(); ++dim)
    {
        streamDivisor = streamDivisor * worstStreamPool[dim];
    }

    if(isCMConv)
        return tensorToSize->computeTotalSize(16, false, false, false)/streamDivisor;

    return tensorToSize->computeTotalSize(16, false, false, true)/streamDivisor;
}

std::size_t inferInputSize(std::size_t outputSize, std::size_t padding_start, std::size_t padding_end, std::size_t kernel_size, std::size_t kernel_stride)
{
    return ((outputSize -1) * kernel_stride) - padding_start - padding_end + kernel_size;
}

std::size_t inferOutputSize(std::size_t inputSize, std::size_t padding_start, std::size_t padding_end, std::size_t kernel_size, std::size_t kernel_stride)
{
    return ( inputSize + padding_start + padding_end - kernel_size) / kernel_stride + 1;
}

std::size_t mv::activationTensorSize(mv::Op& op, const mv::Data::TensorIterator tensorToSize, std::string clustering, const mv::Shape& streamingPool, bool isCMConv, int totalClusters, bool isInput, bool dilation)
{
    auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
    auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits()/8.0);
    auto opType = op.getOpType();
    auto tensorShape = tensorToSize->getShape();
    if (dilation)
        tensorShape = tensorToSize->get<mv::Shape>("originalShape");
    //Note: For now, all batched operations stream over batch so that N = 1
    size_t streamedBatch = 1;

    size_t fullTensorHeight = tensorShape[mv::IO_HEIGHT_DIMENSION];
    size_t streamedHeight = fullTensorHeight;

    size_t fullTensorChannels = tensorShape[mv::IO_CHANNEL_DIMENSION];
    size_t streamedChannels = fullTensorChannels;

    if(streamingPool["H"] > 1)
    {
        size_t kernelSize = 1;
        if(  (opType == "Conv") || (opType == "DepthwiseConv") )
            kernelSize = op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT];
        else if (opType == "MaxPool")
            kernelSize = op.get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_HEIGHT];

        unsigned short kernelStride;
        if (op.hasAttr("stride"))
            kernelStride = op.get<std::array<unsigned short, 2>>("stride")[mv::KERNEL_HEIGHT];
        else
            kernelStride = 1;//fake stride

        std::array<unsigned short, 4> padding;
        if (op.hasAttr("padding"))
            padding = op.get<std::array<unsigned short, 4>>("padding");
        else
            padding = {0, 0, 0, 0};
        int padStart = padding[2];
        int padEnd = padding[3];

        auto outputSize = inferOutputSize(op.getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION], padStart, padEnd, kernelSize, kernelStride);
        auto newOutputSizes = tileSpatialOutputSize(outputSize, streamingPool["H"]);
        streamedHeight = newOutputSizes.front();
        if(streamedHeight < newOutputSizes.back())
            streamedHeight = newOutputSizes.back();

        if(isInput)
        {
            std::size_t largestTileHeight = 0;
            for (std::size_t split = 0; split < streamingPool["H"]; split++)
            {
                std::size_t inferSize = 0;
                if (split == 0)
                    inferSize = inferInputSize(newOutputSizes[split],padStart,0,kernelSize,kernelStride);
                else if (split == (streamingPool["H"]-1))
                    inferSize = inferInputSize(newOutputSizes[split],0,padEnd,kernelSize,kernelStride);
                else
                    inferSize = inferInputSize(newOutputSizes[split],0,0,kernelSize,kernelStride);

                //Remember the largest tile size
                if(inferSize >  largestTileHeight)
                    largestTileHeight = inferSize;
            }
            streamedHeight = largestTileHeight;
        }
    }
    if(streamingPool["C"] > 1)
    {
        streamedChannels = div(fullTensorChannels,streamingPool["C"]);
    }
    if (streamingPool["K"] > 1)
    {
        streamedChannels =  div(fullTensorChannels, streamingPool["K"]);

        size_t remainderChannels = fullTensorChannels - (streamedChannels*(streamingPool["K"] -1));
        if (remainderChannels > streamedChannels)
            streamedChannels = remainderChannels;

        streamedChannels = mv::round_up(streamedChannels, 16);
    }

    if(clustering == "SplitOverH")
    {
        streamedHeight = div(streamedHeight,totalClusters);
    }
    if((opType == "Conv" || opType == "DepthwiseConv" || opType == "MaxPool" ||
        opType == "Eltwise") && (!isCMConv || !isInput)) //for DPU tasks we align both input (except CM) and output tensors channels
    {
        streamedChannels = mv::round_up(streamedChannels, 16);
    }

    return tensorShape[mv::IO_WIDTH_DIMENSION] * streamedHeight * streamedChannels * streamedBatch * dtypeMultiplier;
}

std::size_t mv::alignedWeightsSize(const mv::Data::TensorIterator tensorToSize, const Shape& streamConfig, std::string clustering, int totalClusters){
    auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
    auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits()/8.0);
    size_t alignedFullInputChannels = mv::round_up(tensorToSize->getShape()[mv::KERNEL_INPUT_CHANNELS], 16);

    size_t alignedFullOutputChannels = mv::round_up(tensorToSize->getShape()[mv::KERNEL_OUTPUT_CHANNELS], 16);
    size_t alignedStreamedOutputChannels = mv::round_up(alignedFullOutputChannels/streamConfig["K"], 16);

    if(clustering == "SplitOverK")
    {
        size_t alignedSplittedOutputChannels = div(alignedStreamedOutputChannels,totalClusters);
        alignedSplittedOutputChannels = mv::round_up(alignedSplittedOutputChannels, 16);

        return (alignedFullInputChannels * alignedSplittedOutputChannels *
                tensorToSize->getShape()[mv::KERNEL_WIDTH] * tensorToSize->getShape()[mv::KERNEL_HEIGHT])
                * dtypeMultiplier;
    }
    else{
        return (alignedFullInputChannels * alignedStreamedOutputChannels *
                tensorToSize->getShape()[mv::KERNEL_WIDTH] * tensorToSize->getShape()[mv::KERNEL_HEIGHT])
                * dtypeMultiplier;
    }
}

std::tuple<std::size_t,std::size_t,std::size_t> mv::memorySize(mv::Op& op, int totalClusters, bool enableChannelMajorConv, std::string clustering, 
                                            bool inputActivationSparsity, bool outputActivationSparsity, bool weightsSparsity, 
                                            const Shape& streamConfig, bool fakeSparsity, bool spilling, bool parentSpilling)
{
    auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };

    size_t inputSize = 0;
    size_t outputSize = 0;
    size_t weightSize = 0;
    size_t weightTableSize = 0;
    //NOTE: here is done a trick for the sub-dilated convolutions, if you are
    //dilated on your cmx as input is the original shape tensor which is before
    //the input of the slice...
    bool dilatedLayerInputMemory = false;

    auto opType = op.getOpType();
    auto isCMConv = false;

    if(enableChannelMajorConv && op.supportsCMConv())
        isCMConv = true;

    if (op.hasAttr("DilatedSubConv") && (op.get<bool>("DilatedSubConv")))
        dilatedLayerInputMemory = true;

    if(opType != "Input" && opType != "Concat")
    {
        // Note: when an operation is streaming activations, but it's parent didn't spill, the input won't be streamed
        Shape temporaryStreamConfig = {streamConfig["W"],streamConfig["H"],streamConfig["C"],1,streamConfig["B"]};
        if(!parentSpilling)
            temporaryStreamConfig = {1,1,1,1,1};
        inputSize = activationTensorSize(op, op.getInputTensor(0),clustering,temporaryStreamConfig, isCMConv, totalClusters, true, dilatedLayerInputMemory);
    }
    if(opType != "Output")
    {
        //NOTE: when streaming operations are not spilled, full output (not streamed size) must be counted
        // Similarly, with explicit concats. We don't call this function for ddr concats, only CMX
        Shape temporaryStreamConfig = {streamConfig["W"],streamConfig["H"],1,streamConfig["K"],streamConfig["B"]};
        if (!spilling)
            temporaryStreamConfig = {1,1,1,1,1};

        outputSize = activationTensorSize(op, op.getOutputTensor(0),clustering,temporaryStreamConfig, isCMConv, totalClusters, false);
    }

    auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

    size_t outChannels = op.outputSlots() ? op.getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION] : 0;
    size_t alignedFullChannels = mv::round_up(outChannels, 16);
    size_t alignedSplittedChannels = mv::round_up(alignedFullChannels/streamConfig["K"], 16);
    if(clustering == "SplitOverK") {
        alignedSplittedChannels =  mv::round_up(alignedSplittedChannels/totalClusters, 16);
    }

    if(opType == "Conv" || opType == "DepthwiseConv")
    {
        weightTableSize = 16 * alignedSplittedChannels;
        if (opType == "Conv")
        {
            weightSize += alignedWeightsSize(op.getInputTensor(1),{1,1,1,streamConfig["K"],1}, clustering, totalClusters);
        }
        else
        {
            weightSize += realTensorSize(op.getInputTensor(1),{1,1,streamConfig["C"],1,1}, isCMConv);
            if(clustering == "SplitOverK")
                weightSize = div(weightSize,totalClusters);
        }

    }
    else if(opType == "MaxPool")
    {
        weightTableSize = 16 * alignedSplittedChannels;
        weightSize = 0;
    }
    else if(opType == "Eltwise" && !software)
    {
        weightTableSize = 0;
        weightSize = 0;
        Shape temporaryStreamConfig = {streamConfig["W"],streamConfig["H"],streamConfig["C"],1,streamConfig["B"]};
        if(!parentSpilling)
            temporaryStreamConfig = {1,1,1,1,1};
        inputSize += activationTensorSize(op, op.getInputTensor(1),clustering,temporaryStreamConfig, isCMConv, totalClusters, true);
    }

    //Additional memory footprint for sparsity
    if(fakeSparsity)
    {
        uint16_t kernelW, kernelH;

        auto strides = op.get<std::array<unsigned short, 2>>("stride");

        if (op.hasAttr("kSize"))
        {
            auto kernelShape = op.get<std::array<unsigned short, 2>>("kSize");
            kernelW = kernelShape[0];
            kernelH = kernelShape[1];
        }
        else
        {
            auto weightsShape = op.getInputTensor(1)->getShape();
            kernelW = weightsShape[mv::KERNEL_WIDTH];
            kernelH = weightsShape[mv::KERNEL_HEIGHT];
        }

        mv::DType dataType = op.getInputTensor(0)->getDType();
        if (opType != "MaxPool")
            dataType = op.getInputTensor(1)->getDType();

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

            std::size_t outputChannels =  op.getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION];
            outputChannels = outputChannels/streamConfig["K"];
            std::size_t inputChannels = op.getInputTensor(0)->getShape()[IO_CHANNEL_DIMENSION];

            auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8.0)); //how many bytes we need per window
            auto NumberOfRowsSparistyBytes = static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0 ));

            //ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
            fakeSparsitySize = 16*NumberOfRowsSparistyBytes*outputChannels;

        }
        inputSize += fakeSparsitySize;
    }
    if(inputActivationSparsity){
        //Alignment due to input channels mult of 16 requirement
        //Only ZM Conv and Elwise are sparse consumers, both need
        //input channels mult of 16
        auto tensorSize = op.getInputTensor(0)->computeTotalSize(16, false, false, true);
        size_t streamDivisor = streamConfig["W"] * streamConfig["H"] * streamConfig["C"];
        //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseInputSize = std::ceil((double)tensorSize /
            (8 * op.getInputTensor(0)->getDType().getSizeInBytes()));
        //Storage element table calculation, 4 bytes pointers
        //Bigger with C streaming
        sparseInputSize += op.getInputTensor(0)->getShape()[IO_WIDTH_DIMENSION] *
            op.getInputTensor(0)->getShape()[IO_HEIGHT_DIMENSION] *
            streamConfig["C"] * 4;
        //Alignment due to bus access requirements
        sparseInputSize = mv::round_up(sparseInputSize, 16);
        inputSize += (sparseInputSize / streamDivisor);
    }
    if(outputActivationSparsity){
        //Alignment due to output channels mult of 16 requirement
        //Only ZM Conv and Elwise are sparse consumers
        auto tensorSize = op.getOutputTensor(0)->computeTotalSize(16, false, false, true);
        size_t streamDivisor = streamConfig["W"] * streamConfig["H"] * streamConfig["K"];
        //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseOutputSize = std::ceil((double)tensorSize /
            (8 * op.getOutputTensor(0)->getDType().getSizeInBytes()));
        //Storage element table calculation, 4 bytes pointers
        //Bigger with K streaming
        sparseOutputSize += op.getOutputTensor(0)->getShape()[IO_WIDTH_DIMENSION] *
            op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION] *
            streamConfig["K"] * 4;
        //Alignment due to bus access requirements
        sparseOutputSize = mv::round_up(sparseOutputSize, 16);
        outputSize += (sparseOutputSize / streamDivisor);
    }
    if(weightsSparsity){
        //Alignment due to output/input channels mult of 16 requirement
        auto tensorSize = op.getInputTensor(1)->getShape()[KERNEL_WIDTH] *
            op.getInputTensor(1)->getShape()[KERNEL_HEIGHT] *
            mv::round_up(op.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS], 16) *
            alignedSplittedChannels;
        //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
        auto sparseWeightSize = std::ceil((double)tensorSize / 8);
        //Sparse pointers taken into account in weight table ...
        sparseWeightSize = mv::round_up(sparseWeightSize, 16);
        weightSize += sparseWeightSize;
    }

    weightSize += weightTableSize;

    // Note: for SOH and SOK, division by number of clusters is done in activationTensorSize
    // and alignedWeightsSize, respectively. This allows greater precision than dividing
    // totalClusters. Multiclustering doesn't perfectly split tensor, depends on subtensor size!
    if(clustering == "HKSwitch")
        inputSize = div(inputSize,totalClusters);
    if(clustering == "SplitOverHOverlapped")
    {
        inputSize = div(inputSize,totalClusters);
        outputSize = div(outputSize,totalClusters);
    }

    return std::tuple<std::size_t,std::size_t,std::size_t>(inputSize, outputSize,weightSize);
}

void mv::saveNewStreamingStrategiesToJson(const mv::pass::PassEntry& pass, const mv::Attribute& streamingStrategyElements, std::string passName) {
    pass.log(mv::Logger::MessageType::Debug, "Saving New Streaming Strategies to JSON file");
    std::ofstream jsonOutputFile;
    std::string jsonOutFileName = "./output/" + std::string(passName) + ".json";
    jsonOutputFile.open(jsonOutFileName, std::ios::out);
    if (!(jsonOutputFile.is_open()))
        pass.log(mv::Logger::MessageType::Debug, std::string(passName) + " could not open output file " + jsonOutFileName);

    auto currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string timeStamp(ctime(&currentTime));
    if (!timeStamp.empty() && timeStamp[timeStamp.length() - 1] == '\n') timeStamp.erase(timeStamp.length() - 1);

    mv::Element SSA("Streaming strategies generated by mcmCompiler " + timeStamp);
    SSA.set("streaming_strategy", streamingStrategyElements);
    auto jsonSStrategy = SSA.toJSON(true);
  
    jsonOutputFile << jsonSStrategy.stringifyPretty() << "," << std::endl;
    jsonOutputFile.close();
}

void mv::saveNewStreamingStrategiesToJson1(const mv::Attribute& streamingStrategyElements) {
    std::ofstream jsonOutputFile;
    std::string jsonOutFileName = "./output/" + std::to_string('Streaming_activations_and_weights_performance_strategies') + ".json";
    jsonOutputFile.open(jsonOutFileName, std::ios::out);
    // if (!(jsonOutputFile.is_open()))
    //     pass.log(mv::Logger::MessageType::Debug, std::string(passName) + " could not open output file " + jsonOutFileName);

    auto currentTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string timeStamp(ctime(&currentTime));
    if (!timeStamp.empty() && timeStamp[timeStamp.length() - 1] == '\n') timeStamp.erase(timeStamp.length() - 1);

    mv::Element SSA("Streaming strategies generated by mcmCompiler " + timeStamp);
    SSA.set("streaming_strategy", streamingStrategyElements);
    auto jsonSStrategy = SSA.toJSON(true);
  
    jsonOutputFile << jsonSStrategy.stringifyPretty() << "," << std::endl;
    jsonOutputFile.close();
}

bool mv::validateKStream(mv::Op& op, mv::Attribute clustering, size_t split, bool spilling, size_t nClusters) {
    if (op.getOpType() == "Conv" && clustering.get<std::string>() == "SplitOverK") {
        auto weightsShape = op.getInputTensor(1)->getShape();
        auto numOutChannels = weightsShape[KERNEL_OUTPUT_CHANNELS];
        if ((numOutChannels / split * nClusters) < 16)
            return false;
    }
    if (!spilling) {
        auto outputShape = op.getOutputTensor(0)->getShape();
        size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
        // ok it fits, now make sure that if we are !spilling that there's no crop
        size_t outputChannelSlice = ceil((double)outputChannelSize / (double)split);
        size_t lastSlice = outputChannelSize - outputChannelSlice * (split - 1);
        if (!(outputChannelSlice % 16 == 0 && lastSlice % 16 == 0))  // would need crop
            return false;
    }

    return true;
}