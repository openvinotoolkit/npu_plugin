#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/compression/hde.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "mcm/utils/custom_strings.hpp"


static void GraphParameterOptimizationFcn(const mv::pass::PassEntry&,
    mv::ComputationModel& model,
    mv::TargetDescriptor&, mv::Element& passDesc,
    mv::Element&
);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GraphParameterOptimization)
                .setFunc(GraphParameterOptimizationFcn)
                .setDescription("Analyzes graph, and tries to come up with optimal schedule");

    }

}

namespace mv
{

    namespace graphOptimizer
    {

        class StrategyManagerKmb : public StrategyManager
        {

        public:
            StrategyManagerKmb(OpModel& model,mv::Element& passDesc, const mv::TargetDescriptor& td) :
                StrategyManager(model,passDesc)
            {
                auto globalParams = model.getGlobalConfigParams();
                enableChannelMajorConv = globalParams->get<bool>("enable_channel_major_conv");

                // Load the HDE hardware specs
                auto hdeDef = td.hdeDef();
                hde_.reset(new Hde(hdeDef.bitPerSymbol, hdeDef.maxNumberEncodedSymbols, 0, hdeDef.blockSize, false, hdeDef.bypassMode));
            }

            std::unique_ptr<Hde> hde_ = nullptr;
            size_t totalClusters=4;
            size_t clusterMemoryKb=896;
            size_t dpuPerCluster=5;
            std::string referenceDevice = "A0";
            int ddrBandwidth=128;
            int sysClock=500;
            bool globalEnableStreaming=true;
            bool globalEnableActivationSparsity=true;
            bool globalForceActivationSparsity=false;
            bool globalEnableWeightsSparsity=false;
            bool globalForceSpilling=false;
            bool enableChannelMajorConv=false;
            double safetyFactor=1.0;
            double clusterMemory=(double)clusterMemoryKb * 1024.0 * safetyFactor;
            std::vector<string> failure_causes = {"Unknown", "MemorySize", "Stream+ClusterComp",
            "SpillHKSwitch", "SOKNotAlign16", "InputNotSpilled", "OutputNotSpilled", "StreamingNotSpilled",
            "Workload<KernelSOH", "ChannelMjr1", "ChannelMjr2", "DWChannels", "SOHheight", "RequiresSparsity", "SparsitySOK"};


            void readGlobalConfigs()
            {
                referenceDevice = globalConfig_["referenceDevice"].get<string>();
                totalClusters = globalConfig_["totalClusters"].get<int>();
                clusterMemoryKb = globalConfig_["clusterMemory"].get<int>();
                dpuPerCluster = globalConfig_["dpuPerCluster"].get<int>();
                ddrBandwidth = globalConfig_["ddrBandwidth"].get<int>();
                sysClock = globalConfig_["systemClockMhz"].get<int>();
                createStrategyDots = globalConfig_["createStrategyDots"].get<bool>();
                dotFileLocation = globalConfig_["dotFileLocation"].get<string>();
                jsonOutFileName = globalConfig_["jsonOutFileName"].get<string>();
                safetyFactor = globalConfig_["FathomSafetyFactor"].get<double>();
                //Input is in Kb
                clusterMemory = (double)clusterMemoryKb * 1024.0 * safetyFactor;

                globalEnableStreaming = globalStrategies_["enableStreaming"].get<bool>();
                // globalEnableActivationSparsity = globalStrategies_["enableActivationSparsity"].get<bool>();
                globalForceActivationSparsity = globalStrategies_["forceActivationSparsity"].get<bool>();
                globalEnableWeightsSparsity = globalStrategies_["enableWeightsSparsity"].get<bool>();
                globalForceSpilling =  globalStrategies_["forceSpilling"].get<bool>();
            }

            /*
             * This method calculates a compression ratio of compressed weight size / orignal weight size.
             * This ratio could be used by the calculation of execution time.
             * Execution time is calculated by this formula and theoretically the DMA of compressed data should be
             * faster than non compressed data
             * execTime += WSize / ddrBandwidth;
             * So execution time calculation could be extended to be:
             * execTime += (WSize * weightscompressionRatio / ddrBandwidth;
             *
             * Empirical testing has found this does not change final strategy section as the same amount of data is
             * ultimately DMA'ed to CMX. So for now the ratio is not used until a more sensitive cost function is
             * developed as it does not warrant the increase in compilation time caused by calling the HDE library in strategy manager.
             */
            double calculateWeightsCompressionRatio(mv::Op layer)
            {
                double weightsCompressionRatio = 1;
                auto inputTensor = layer.getInputTensor(0);
                auto weightsTensor = layer.getInputTensor(1);
                auto outputTensor = layer.getOutputTensor(0);
                auto weightsTensorShape = weightsTensor->getShape();
                auto inputTensorShape = inputTensor->getShape();
                auto outputTensorShape = outputTensor->getShape();

                auto globalConfigParams = model_.getGlobalConfigParams();
                int pad = globalConfigParams->hasAttr("VPU2ChannelPadding") ? globalConfigParams->get<int>("VPU2ChannelPadding") : 16;

                auto alignedInputChannels = ((inputTensorShape[mv::IO_CHANNEL_DIMENSION] + pad- 1) / pad) * pad;
                auto alignedOutputChannels = ((outputTensorShape[mv::IO_CHANNEL_DIMENSION] + pad - 1) / pad) * pad;

                mv::Shape alignedShape = mv::Shape({weightsTensorShape[mv::KERNEL_WIDTH], weightsTensorShape[mv::KERNEL_HEIGHT],
                                                alignedInputChannels, alignedOutputChannels});

                // HDE should only compress weights larger than 4 kB
                // At this point sparsity has not yet been decided for weights
                // So using alignedShape.totalSize() is a conservative estimate as it assumes
                // non-sparse size
                if(alignedShape.totalSize() / 1024 > 4)
                {
                    // If weights are already aligned to 16 channels, then compute the HDE compression ratio
                    if (weightsTensorShape[mv::KERNEL_OUTPUT_CHANNELS] == alignedShape[mv::KERNEL_OUTPUT_CHANNELS] &&
                        weightsTensorShape[mv::KERNEL_INPUT_CHANNELS] == alignedShape[mv::KERNEL_INPUT_CHANNELS])
                    {
                        auto weightsdata = weightsTensor->getIntData();
                        auto compressedData =  hde_->hdeCompress(weightsdata, weightsTensor);
                        weightsCompressionRatio = (double)compressedData.second / weightsdata.size();
                        layer.set<double>("weightsCompressionRatio", weightsCompressionRatio);
                        return weightsCompressionRatio;
                    }
                    // Else align weights to 16 channels and compute the HDE compression ratio
                    else
                    {
                        auto weightsTensorQuantizationParams = weightsTensor->get<mv::QuantizationParams>("quantParams");
                        auto zeroPoint = weightsTensorQuantizationParams.getZeroPoint()[0];

                        auto alignedWeightsdata = weightsTensor->getIntData();
                        auto weightsAlignmentData = std::vector<int64_t>(alignedShape.totalSize() - alignedWeightsdata.size() , zeroPoint);
                        alignedWeightsdata.insert(alignedWeightsdata.end(), weightsAlignmentData.begin(), weightsAlignmentData.end());

                        auto compressedData = hde_->hdeCompress(alignedWeightsdata, weightsTensor);
                        weightsCompressionRatio = (double)compressedData.second / alignedWeightsdata.size();
                        layer.set<double>("weightsCompressionRatio", weightsCompressionRatio);

                        return weightsCompressionRatio;
                    }
                }
                return weightsCompressionRatio;
            }

            //TODO:: figure out more efficient and cleaner way to handle these....

            vector<Attribute> createTFStrategyPoolFromBool(mv::Op op,string name)
            {
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return vector<Attribute>{true,false};
                else
                    return vector<Attribute>{false};
            }

            vector<Attribute> createTStrategyPoolFromBool(mv::Op op,string name)
            {
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return vector<Attribute>{true};
                else
                    return vector<Attribute>{true,false};
            }

            bool createStrategyFromBool(mv::Op op, string name){
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return true;
                else
                    return false;
            }

            vector<Attribute> createStrategyPoolFromStrategySet(mv::Op op, string name)
            {
                auto streamingStrategy = getStrategy(op,name);

                vector<Attribute> attr;

                for (auto elem : streamingStrategy.get<vector<string>>())
                {
                    attr.push_back(elem);
                }

                return attr;
            }

            size_t realTensorSize(const mv::Data::TensorIterator tensorToSize, const Shape& streamingPool, bool isCMConv)
            {
                auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };

                Shape worstStreamPool = streamingPool;

                //TODO harmonize this, for now only consider worst shape for nested streams
                if(streamingPool["H"] > 1 and streamingPool["K"] > 1)
                {
                    Shape tensorShape = tensorToSize->getShape();
                    //update the streamingPool to the worst combination, based on slice sizes
                    auto outputSize = tensorShape[mv::IO_HEIGHT_DIMENSION];
                    auto numberOfSplits = streamingPool[mv::IO_HEIGHT_DIMENSION];

                    auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
                    int newOutputSize = newOutputSizes.front();
                    int remainderOutputSize = newOutputSizes.back();

                    auto worstNumberOfSplits = outputSize/newOutputSize;
                    worstStreamPool[mv::IO_HEIGHT_DIMENSION] = worstNumberOfSplits;
                }

                //TODO add handling for weights case if we dont align it to 16 always
                size_t streamDivisor = 1;
                for(size_t dim = 0; dim <  worstStreamPool.ndims(); ++dim)
                {
                    streamDivisor = streamDivisor * worstStreamPool[dim];
                }

                if(isCMConv)
                    return tensorToSize->computeTotalSize(16, false, false, false)/streamDivisor;

                return tensorToSize->computeTotalSize(16, false, false, true)/streamDivisor;
            }

            size_t maxTensorSize(const mv::Data::TensorIterator tensorToSize, string clustering, const Shape& streamingPool, bool isCMConv, mv::Op& op, bool dilation = false)

            {
                size_t kHeight = 1;
                if(  (op.getOpType() == "Conv") || (op.getOpType() == "DepthwiseConv") )
                    kHeight = op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT];
                else if (op.getOpType() == "MaxPool")
                    kHeight = op.get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_HEIGHT];
                //NOTE: assuming order of paddings: left,right,top,bottom
                std::array<unsigned short, 4> padding;
                if (op.hasAttr("padding"))
                    padding = op.get<std::array<unsigned short, 4>>("padding");
                else
                    padding = {0, 0, 0, 0};

                std::array<unsigned short, 2> kStride;
                if (op.hasAttr("stride"))
                    kStride = op.get<std::array<unsigned short, 2>>("stride");
                else
                    kStride = {1,1};//fake stride


                // Shape worstStreamPool = streamingPool;
                vector<double> worstStreamPool;
                for(size_t dim = 0; dim <  streamingPool.ndims(); ++dim)
                {
                    worstStreamPool.push_back(streamingPool[dim]);
                }

                Shape tensorShape = tensorToSize->getShape();
                if (dilation)
                    tensorShape = tensorToSize->get<mv::Shape>("originalShape");

                //update the streamingPool to the worst combination, based on slice sizes
                size_t outputSize;
                size_t numberOfSplits;
                if(streamingPool["H"] > 1) // If streaming over H
                {
                    outputSize = tensorShape[mv::IO_HEIGHT_DIMENSION];
                    numberOfSplits = streamingPool[mv::IO_HEIGHT_DIMENSION];
                    auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
                    int newOutputSize = newOutputSizes.front();
                    // Always returns biggest first
                    // int remainderOutputSize = newOutputSizes.back();
                    // if (remainderOutputSize > newOutputSize)
                    //     newOutputSize = remainderOutputSize;
                    int extraLines = 0;

                    if(extraLines < kHeight-1)
                        extraLines = kHeight -1;

                    if(padding[2] > padding[3])
                        if(padding[2] > extraLines)
                            extraLines = padding[2];
                    else
                        if(padding[3] > extraLines)
                            extraLines = padding[3];


                    // extraLines += (padding[2]? kHeight/2 : 0);
                    // extraLines += (padding[3]? kHeight/2 : 0);

                    // Note: worst number of splits needs to be a floating point
                    // The idea is that even if we split by some number, because of alignment and padding
                    // that will come later, splitting into some number of streams is not equivalent to simply dividing
                    // the whole tensor size by that number of splits.
                    // Instead, we calculate the worstNumberOfSplits, which will be the actual divisor to use
                    // for whole tensor size to represent the proportion the largest streamed chunk is of the whole
                    // tensor. In other words, worstNumberOfSplits should be a floating point number SMALLER
                    // than or equal to the real number of splits we are evaluating.

                    double worstNumberOfSplits = (double)outputSize/(newOutputSize + extraLines);

                    if(worstNumberOfSplits <= 0) worstNumberOfSplits = 1;
                    worstStreamPool[mv::IO_HEIGHT_DIMENSION] = worstNumberOfSplits;
                }
                else if(streamingPool["B"] > 1) // If streaming over N
                {
                    // Note: all streaming over batch must equal size of batch, other not enabled from runtime+workloads
                    // worstStreamPool[mv::IO_BATCH_DIMENSION] = streamingPool["B"];
                    // outputSize = tensorShape[mv::IO_BATCH_DIMENSION];
                    // numberOfSplits = streamingPool[mv::IO_BATCH_DIMENSION];
                    // auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
                    // int newOutputSize = newOutputSizes.front();

                    // int remainderOutputSize = newOutputSizes.back();
                    // if (remainderOutputSize > newOutputSize)
                    //     newOutputSize = remainderOutputSize;

                    // double worstNumberOfSplits = outputSize/newOutputSize;
                    // worstStreamPool[mv::IO_BATCH_DIMENSION] = worstNumberOfSplits;
                }
                if (streamingPool["K"] > 1)
                {
                    outputSize = tensorShape[mv::IO_CHANNEL_DIMENSION];
                    numberOfSplits = streamingPool["K"];
                    int newOutputSize =  ceil( ((double)outputSize) / ((double)numberOfSplits));

                    int remainderOutputSize = outputSize - (newOutputSize*(numberOfSplits -1));
                    if (remainderOutputSize > newOutputSize)
                        newOutputSize = remainderOutputSize;

                    newOutputSize = mv::round_up(newOutputSize, 16);

                    // TODO determine when there will be overlap
                    //double worstNumberOfSplits = (double)outputSize/(newOutputSize+2);
                    double worstNumberOfSplits = (double)outputSize/(newOutputSize);

                    if(worstNumberOfSplits <= 0) worstNumberOfSplits = 1;
                    worstStreamPool[mv::KERNEL_OUTPUT_CHANNELS] = worstNumberOfSplits;
                }

                double clusteringDivisor = 1;
                if(clustering == "SplitOverH")
                {
                    outputSize = tensorShape[mv::IO_HEIGHT_DIMENSION];
                    numberOfSplits = streamingPool[mv::IO_HEIGHT_DIMENSION];
                    auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
                    int newOutputSize = newOutputSizes.front();

                    auto workloadHeight = ceil((double)newOutputSize / (double)totalClusters);

                    clusteringDivisor = (double)outputSize/(workloadHeight * numberOfSplits);

                    // std::cout << op.getName() << " clusteringDivisor is " << clusteringDivisor << ", streaming H = " << streamingPool["H"]<< std::endl;
                }
                worstStreamPool.push_back(clusteringDivisor);

                double streamDivisor = 1;
                for(auto stream: worstStreamPool)
                {
                    streamDivisor = streamDivisor * stream;
                }

                if(isCMConv)
                    return std::ceil((double)tensorToSize->computeTotalSize(16, false, false, false)/streamDivisor);
                //NOTE: dilation case will need the original shape that is located on cmx
                return std::ceil((double)tensorToSize->computeTotalSize(16, false, false, true, dilation)/streamDivisor);
            }

            size_t alignedWeightsSize(const mv::Data::TensorIterator tensorToSize, const Shape& streamConfig, string clustering){
                auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
                auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits()/8.0);
                size_t alignedFullInputChannels = mv::round_up(tensorToSize->getShape()[KERNEL_INPUT_CHANNELS], 16);

                size_t alignedFullOutputChannels = mv::round_up(tensorToSize->getShape()[KERNEL_OUTPUT_CHANNELS], 16);
                size_t alignedStreamedOutputChannels = mv::round_up(alignedFullOutputChannels/streamConfig["K"], 16);

                if(clustering == "SplitOverK")
                {
                    //size_t alignedSplittedOutputChannels = ceil(alignedStreamedOutputChannels/totalClusters)
                    size_t alignedSplittedOutputChannels = div(alignedStreamedOutputChannels,totalClusters);
                    if(alignedSplittedOutputChannels < 64)
                        alignedSplittedOutputChannels = mv::round_up(alignedSplittedOutputChannels, 16);

                    return (alignedFullInputChannels * alignedSplittedOutputChannels *
                            tensorToSize->getShape()[KERNEL_WIDTH] * tensorToSize->getShape()[KERNEL_HEIGHT])
                            * dtypeMultiplier;
                }
                else{
                    return (alignedFullInputChannels * alignedStreamedOutputChannels *
                            tensorToSize->getShape()[KERNEL_WIDTH] * tensorToSize->getShape()[KERNEL_HEIGHT])
                            * dtypeMultiplier;
                }
            }

            pair<size_t,size_t> memorySize(mv::Op& op, const Attribute& clustering, bool inputActivationSparsity,
                                            bool outputActivationSparsity, bool weightsSparsity, const Shape& streamConfig, bool fakeSparsity)
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

                size_t totalWeightsSize = 0;
                size_t totalActivationSize = 0;
                auto opType = op.getOpType();
                auto isCMConv = false;
                auto clusterStrategy = clustering.get<string>();

                if(enableChannelMajorConv and opType == "Conv" and
                   op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] % 16)
                    isCMConv = true;

                if (op.hasAttr("DilatedSubConv") && (op.get<bool>("DilatedSubConv")))
                    dilatedLayerInputMemory = true;

                if(opType != "Input"){
                    inputSize = maxTensorSize(op.getInputTensor(0),clusterStrategy,{streamConfig["W"],streamConfig["H"],streamConfig["C"],1,streamConfig["B"]}, isCMConv, op, dilatedLayerInputMemory);

                }
                if(opType != "Output"){
                    outputSize = maxTensorSize(op.getOutputTensor(0),clusterStrategy,{streamConfig["W"],streamConfig["H"],1,streamConfig["K"],streamConfig["B"]}, isCMConv, op);
                }

                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

                size_t outChannels = op.outputSlots() ? op.getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION] : 0;
                size_t alignedFullChannels = mv::round_up(outChannels, 16);
                size_t alignedSplittedChannels = mv::round_up(alignedFullChannels/streamConfig["K"], 16);
                if(opType == "Conv" || opType == "DepthwiseConv")
                {
                    weightTableSize = 16 * alignedSplittedChannels;
                    if (opType == "Conv")
                    {
                        weightSize += alignedWeightsSize(op.getInputTensor(1),{1,1,streamConfig["C"],streamConfig["K"],1}, clusterStrategy);
                    }
                    else
                    {
                        weightSize += realTensorSize(op.getInputTensor(1),{1,1,streamConfig["C"],1,1}, isCMConv);
                        if(clusterStrategy == "SplitOverK")
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
                    weightSize = 0; //TODO think about
                    inputSize += maxTensorSize(op.getInputTensor(1),clusterStrategy,{streamConfig["W"],streamConfig["H"],streamConfig["C"],1,1}, isCMConv, op);
                }

                //Additional memory footprint for sparsity
                if(fakeSparsity)
                {
                    if (opType != "MaxPool" && opType != "DepthwiseConv" && !isCMConv)
                    {
                        //error
                        throw LogicError(*this, op.getName() + ": Invalid fake Sparsity! Has to be only for MaxPool, DW or CMConv!! opType is " + opType);
                    }
                    std::vector<std::size_t> ndims(4);
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

                    mv::DType dataType =op.getInputTensor(0)->get<mv::DType>("dType");
                    if (opType != "MaxPool")
                        dataType = op.getInputTensor(1)->get<mv::DType>("dType");

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
                    size_t streamDivisor = streamConfig["W"] * streamConfig["H"] * streamConfig["C"];
                    //w*h*c, 1 bit per byte of tensor.
                    auto sparseInputSize = (op.getInputTensor(0)->getShape()[0] * op.getInputTensor(0)->getShape()[1]* op.getInputTensor(0)->getShape()[2]) / 8;
                    //storage element
                    sparseInputSize += (op.getInputTensor(0)->getShape()[0] * op.getInputTensor(0)->getShape()[1]);
                    sparseInputSize = mv::round_up(sparseInputSize, 16);
                    inputSize += (sparseInputSize / streamDivisor);
                }
                if(outputActivationSparsity){
                    size_t streamDivisor = streamConfig["W"] * streamConfig["H"] * streamConfig["K"];
                    //w*h*c, 1 bit per byte of tensor.
                    auto sparseOutputSize = (op.getOutputTensor(0)->getShape()[0] * op.getOutputTensor(0)->getShape()[1]* op.getOutputTensor(0)->getShape()[2]) / 8;
                    //storage element
                    sparseOutputSize += (op.getOutputTensor(0)->getShape()[0] * op.getOutputTensor(0)->getShape()[1]);
                    sparseOutputSize = mv::round_up(sparseOutputSize, 16);
                    outputSize += (sparseOutputSize / streamDivisor);
                }
                if(weightsSparsity){
                    auto sparseWeightSize = (op.getInputTensor(1)->getShape()[0] * op.getInputTensor(1)->getShape()[1]* op.getInputTensor(1)->getShape()[2]) / 8;
                    sparseWeightSize += (op.getInputTensor(1)->getShape()[0] * op.getInputTensor(1)->getShape()[1]);
                    sparseWeightSize = mv::round_up(sparseWeightSize, 16);
                    weightSize += sparseWeightSize; //TODO probably overcounting now if SOK
                }

                weightSize += weightTableSize; // todo probably overcounts for sok now

                if(clusterStrategy == "Clustering" || clusterStrategy == "SplitOverH" || clusterStrategy == "SplitOverK")
                {
                    // Note: for SOH and SOK, division by number of clusters is done in maxTensorSize
                    // and alignedWeightsSize, respectively. This allows greater precision than dividing
                    // totalClusters. Multiclustering doesn't perfectly split tensor, depends on subtensor size!
                    totalActivationSize = inputSize + outputSize;
                    totalWeightsSize = weightSize;
                }
                else if(clusterStrategy == "HKSwitch")
                {
                    totalActivationSize = div(inputSize,totalClusters) + outputSize;
                    totalWeightsSize = weightSize;
                }//TODO: proper calculation here
                else if(clusterStrategy == "SplitOverHOverlapped")
                {
                    totalActivationSize = div(inputSize,totalClusters) + div(outputSize,totalClusters);
                    totalWeightsSize = weightSize;
                }
                else
                {
                    //todo raise rerrr
                }


                return pair<size_t,size_t>(totalActivationSize,totalWeightsSize);
            }

            unsigned getStreamsOverH(mv::Op& op, mv::Attribute clustering, bool iSparsity, bool oSparsity, bool wSparsity, Shape streams, bool fSparsity)
            {
                auto memSize = memorySize(op,clustering,iSparsity,oSparsity,wSparsity,streams,fSparsity);
                auto activationsSize = memSize.first;
                auto weightsSize = memSize.second;
                double availableMemory = (double) clusterMemory - (double) weightsSize;

                unsigned splits = 1;

                if (availableMemory < 0) // Weights don't fit, can't stream over H
                    return splits;

                unsigned splitsToFit = ceil((double)activationsSize/availableMemory);
                if (splitsToFit < 1)
                    return splits;

                splits = splitsToFit;

                // Keep increasing H until we find one big enough to fit, or we run out of H dimension to stream
                auto inputHeight = op.getInputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                unsigned upperBoundH = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                if(upperBoundH > inputHeight) upperBoundH = inputHeight;
                upperBoundH = floor(upperBoundH/2); // TODO
                if(clustering.toString() == "SplitOverH") upperBoundH = upperBoundH/totalClusters;
                do
                {
                    Shape updatedStreams({1,splits,1,streams["K"],streams["B"]});
                    auto memFitCheck = memorySize(op,clustering,iSparsity,oSparsity,wSparsity,updatedStreams,fSparsity);
                    if(memFitCheck.first + memFitCheck.second < clusterMemory) break;
                    splits++;
                }while(splits <= upperBoundH);

                //NOTE: the idea here is that when the number of streams lead to less than one line of output
                //->means kernel size in the input the result is that we can not stream
                if(op.getOpType() == "Conv")
                {
                    auto outDim = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                    auto linesPerOutputSlice = outDim/splits;
                    if(linesPerOutputSlice >= 1)
                        return splits;
                    else
                        return 1;
                    if(splits < 1)
                        return 1;
                }

                return splits + 1; // consider one extra H stream, just in case
            }

            unsigned findBestK(unsigned alignedSize, unsigned channels){
                return std::ceil((double)alignedSize / ((alignedSize/2) - channels));
            }

            vector<size_t> getMaxStreamOverK(const string& clustering,mv::Op& op)
            {
                auto outputShape = op.getOutputTensor(0)->getShape();
                size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
                size_t alignedOutputChannelSize = mv::round_up(outputChannelSize, 16);

                vector<size_t> splits;

                //Add max split
                splits.push_back(alignedOutputChannelSize/16);

                // For each aligned-to-16 number of output channels possibility, add only the
                // minimum number of streams over k that will be aligned to that number
                for(int channels = (alignedOutputChannelSize/2 -16); channels >= 16; channels=channels-16){
                    auto possibleK = findBestK(alignedOutputChannelSize, channels);
                    if(splits.back() != possibleK and possibleK >= 1)
                        splits.push_back(possibleK);
                }
                if(splits.back() > 2)
                    splits.push_back(2);

                if(splits.back() > 1)
                    splits.push_back(1);

                return splits;
            }

            size_t getMaxSplitsOverSpatial(const string& clustering,const Shape& shape,char dim)
            {
                return 0;
            }

            double executionTime(Op& op,StrategySet& strategySet)
            {
                auto opType = op.getOpType();
                if( (opType == "Input") or
                    (opType == "ImplicitInput") or
                    (opType == "Output"))
                    return 0;

                auto outputShape = op.getOutputTensor(0)->getShape();
                auto inputShape = op.getInputTensor(0)->getShape();
                auto clustering = strategySet["clustering"].get<string>();
                auto streaming = strategySet["streaming"].get<Shape>();
                auto sparsity = strategySet["weightsSparsity"].get<bool>();

                Shape contexts,isiSplit;

                if( (opType == "MaxPool") or (opType == "DepthwiseConv"))
                {
                    contexts = {16,1,16,1};
                }
                else
                {
                    contexts = {4,4,16,1};
                }

                if( (clustering == "SplitOverH") or (clustering == "SplitOverHOverlapped") or (clustering == "HKSwitch"))
                {
                    isiSplit = {1,totalClusters,1,1};
                }
                else if(clustering == "SplitOverK")
                {
                    isiSplit = {1,1,totalClusters,1};
                }
                else
                {
                    isiSplit = {1,1,1,1};
                }

                mv::Shape streamNumerator;
                if(streaming["B"] > 1){ // Note: Won't stream over both H and N
                    streamNumerator = {streaming["W"], streaming["B"], streaming["C"], streaming["K"]};
                }
                else {
                    streamNumerator = {streaming["W"], streaming["H"], streaming["C"], streaming["K"]};
                }


                //naively emulate the workload cost
                //TODO: find cleaner solution
                unsigned baseKernelCost;
                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

                if ((opType == "Eltwise" && !(software)) or (opType == "Concat"))
                {
                    baseKernelCost = 1;
                }
                else if (opType == "MaxPool")
                {
                    auto kernel = op.get<array<unsigned short,2>>("kSize");
                    baseKernelCost = kernel[0] * kernel[1];
                }
                else if ((opType == "DepthwiseConv") or (opType == "Conv"))
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    baseKernelCost = weightsShape[KERNEL_WIDTH] * weightsShape[KERNEL_HEIGHT];
                }
                else if (!(op.hasTypeTrait("optimizable")) || software)
                {
                    baseKernelCost = 1;
                }
                else
                {
                    throw LogicError(*this,"Invalid operation type " + opType);
                }

                bool channelAccum =  (opType == "Conv") ? true : false;
                if(channelAccum)
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    baseKernelCost *= weightsShape[KERNEL_INPUT_CHANNELS];
                }

                Shape streamShape = {streaming["W"], streaming["H"], streaming["K"], 1};
                //the actual compute
                if (outputShape.ndims() != streamShape.ndims())
                    outputShape = outputShape.augment(outputShape, streamShape.ndims());
                Shape dpuOutShape = ( outputShape / streamShape ) / isiSplit;
                Shape contextsInOp = dpuOutShape / contexts;
                unsigned numContextsInOp = contextsInOp.totalSize();

                if(numContextsInOp == 0)
                    throw LogicError(*this,"error in contexts");

                unsigned contextsPerDpu = (unsigned)ceil( (double)numContextsInOp / (double)dpuPerCluster);

                return contextsPerDpu * streamShape.totalSize() * baseKernelCost;
            }

            bool requiresActivationSparsity(Op& op, string clustering){
                // if(op.getOpType() == "Input" or op.getOpType() == "Output")
                //     return false;

                if(requiresRealActivationSparsity(op, clustering))
                    return true;

                if(requiresFakeActivationSparsity(op))
                    return true;

                return false;
            }

            bool requiresWeightsSparsity(Op& op)
            {
                // If Z-major Conv in Float precision then need to have weights Sparsity
                if(op.getOpType() == "Conv" and
                    op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] >= 16 and
                    op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") and
                        referenceDevice == "A0")
                        return true;

                return false;
            }

            bool requiresRealActivationSparsity(Op& op, string clustering){
                //An fp16 Conv Z-major must have activation sparsity
                if ((op.getOpType() == "Conv") and  (op.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] >= 16)
                        and op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") and
                        referenceDevice == "A0")
                    return true;
                else if (((op.getOpType() == "Conv") or (op.getOpType() == "DepthwiseConv"))
                         and (op.hasAttr("DilatedSubConv") and op.get<bool>("DilatedSubConv")))
                    return true;
                // Check for need for A0 SOH Sparsity workaround, (SOH conv with kernel > 1)
                // if needed, check memory constraints as for sparse tensor
                if ( op.getOpType() == "Conv" ) {
                    if( clustering == "SplitOverH" and
                        (op.getInputTensor(1)->getShape()[KERNEL_HEIGHT] > 1 or
                         op.getInputTensor(1)->getShape()[KERNEL_WIDTH]  > 1)
                         and (op.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] >= 16) and
                            referenceDevice == "A0")
                            return true;
                }

                return false;
            }

             //Channel major conv, pooling and depthwise will get fake sparsity, so need to check memory constraints as if real sparsity
            bool requiresFakeActivationSparsity(Op& op){
                if(enableChannelMajorConv and
                  (op.getOpType() == "Conv") and
                  (op.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] % 16))
                {
                    return true;
                }

                if(op.getOpType() == "MaxPool")
                    return true;

                if(op.getOpType() == "DepthwiseConv")
                    return true;

                return false;
            }

            int8_t checkInOutSizes(mv::Op& op, size_t input_gate)
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

            int8_t checkKernelSizes(mv::Op& op)
            {
                int8_t executableInHW = 0;
                std::array<unsigned short, 4> kernel = {1,1,1,1};//for non conv IN OUT CHANNEL dims = 1
                if (op.hasAttr("kSize"))
                    if (op.getOpType() == "MaxPool" || op.getOpType() == "Eltwise")
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

            int8_t checkStrideSizes(mv::Op& op)
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

            int8_t checkHWUnsupportedOp(mv::Op& op)
            {
                int8_t executableInHW = 0;
                if (op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv" ||
                        op.getOpType() == "MaxPool" ||
                        op.getOpType() == "Eltwise")
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
            //Check to see if a given stategy is internally consistent for performance
            //Strategies that can only have infinite edges because they are illegal should never be added to the graph
            // Note: IF ADDING A NEW FAILURE CASE, must add new description to failure_causes
            int checkForBadStrategy(mv::Op& op,StrategySet& strategy)
            {
                auto clustering = strategy["clustering"].get<string>();
                auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
                auto streamShape = strategy["streaming"].get<Shape>();
                auto spilling = strategy["spilling"].get<bool>();
                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

                if(op.getOpType() != "Output" && op.getOpType() != "Input" &&
                    (op.hasTypeTrait("optimizable") && !software)) //SW layers we dont care about size
                {
                    auto fit = memorySize(op,clustering,requiresActivationSparsity(op, clustering), false,weightsSparsity,streamShape,
                                    requiresFakeActivationSparsity(op));
                    // cout << "Check for Bad Strategy Memsize: " << fit.first + fit.second << " = " << fit.first << " + " << fit.second << endl;
                    // cout << op.getName() << " : " << clustering << " : " << streamShape.toString() << " : " << fit.first << " + " << fit.second << " = " << (fit.first + fit.second) << std::endl;
                    if(fit.first + fit.second >= clusterMemory)
                        return 1;
                }

                //If spilling, HKSwitch makes no sense
                if( (spilling) and (clustering == "HKSwitch"))
                    return 3;

                if( op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    auto numInChannels = weightsShape[KERNEL_INPUT_CHANNELS];
                    auto numOutChannels = weightsShape[KERNEL_OUTPUT_CHANNELS];
                    if (op.getOpType() == "Conv")
                    {
                        if((numOutChannels/(streamShape["K"] * totalClusters) < 16) and (clustering == "SplitOverK"))
                            return 4;
                    }
                    else
                    {
                        if((numInChannels/(streamShape["K"] * totalClusters) < 16) and (clustering == "SplitOverK"))
                            return 4;
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
                            return 8;
                    }
                }

                 //Input and Output must have Spilled==True
                if( (op.getOpType() == "Input") and (not spilling))
                    return 5;

                if( (op.getOpType() == "Output") and (not spilling))
                    return 6;

                //iIf the layer is streaming over H or W, output of this layer has to be spilled
                if( (not spilling) and ((streamShape["H"] * streamShape["W"]) > 1))
                    return 7;

                //Special rules for Channel Major Convolutions
                //No need for SOHOverlapped input unless using channel major
                if( !enableChannelMajorConv and clustering == "SplitOverHOverlapped")
                    return 9;

                if( enableChannelMajorConv and op.getOpType() == "Conv")
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    auto numInChannels = weightsShape[KERNEL_INPUT_CHANNELS];
                    if ( numInChannels % 16) //assume channel major conv
                        if(clustering == "SplitOverH" and streamShape["H"] > 1)
                            return 10;
                }


                if (op.getOpType() == "DepthwiseConv")
                {
                    if ((op.getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192)
                            && (streamShape["C"] == 1))
                        return 11;
                }

                //For every dpuTask if we splitOverH, workloads are over H dimension, so they need to have at
                //least one line to be assigned with
                if ((op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv" || op.getOpType() == "MaxPool"
                        || op.getOpType() == "Eltwise") && clustering == "SplitOverH")
                {
                    auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                    auto estimatedClusterH = (int)floor((double)outputHeight/totalClusters);
                    if (estimatedClusterH < dpuPerCluster || (outputHeight - (totalClusters - 1) * estimatedClusterH) < dpuPerCluster)
                    {
                        return 12;
                    }
                }

                // For CM Conv, as after DW, we spill to DDR, SOH gets chosen for DW. For larger input sizes, (416,416) DW when spilled
                // seems to fail CRC. Without CM Conv enabled, StreamOverH gets chosen, so with CMConv, forcing No SOH for CRC pass
                // To do: Fix (416,416) DW only CRC fail on master
                if (op.getOpType() == "DepthwiseConv" && spilling && enableChannelMajorConv)
                {
                    if ((op.getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] > 302)
                            && (clustering == "SplitOverH"))
                        return 11;
                }

                if(requiresRealActivationSparsity(op, clustering) && !strategy["inputSparsity"].get<bool>())
                    return 12;

                if(strategy["outputSparsity"].get<bool>() && (clustering == "SplitOverK" || clustering == "HKSwitch"))
                    return 13;

                return 0; //good strategy
            }
            // CM Conv needs if follows a DW Conv (like OV models) or CM Conv follows another Conv, spilling is needed
            // As OV models have DW->Conv and DW is always 1x1, as long as DW height & width are multiples are 8, no need to spill
            // Reason: CM Conv uses overlaps of rows and so the output subtensors of CM Conv may not be aligned to 8 as needed for next op if conv (and has SOH)
            bool needForceSpillingForCM(Op& parentOp, Op& childOp, std::string& parentClustering, std::string& childClustering)
            {
                bool forceSpill = false;
                if (parentOp.getOpType() == "DepthwiseConv" && childOp.getOpType() == "Conv" && childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
                    if (childClustering == "SplitOverH")
                            forceSpill = true;
                // sorry for a back hack. Spilling needed for TFLite inceptionv3 after CM Conv. But that seems to fail Facenet.
                // so conditon added so that facenet CM Conv doesn't spill its output
                if (parentOp.getOpType() == "Conv" && childOp.getOpType() == "Conv")
                    if (parentClustering == "SplitOverH" && parentOp.getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] != 160)
                            forceSpill = true;
                return forceSpill;
            }

            double transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {
                //TODO: expose these conditionals more cleanly
                auto INF = inf_;
                auto parentClustering = parent["clustering"].get<string>();
                auto childClustering = child["clustering"].get<string>();
                bool spillForCM = false;
                if (enableChannelMajorConv and ((parentOp.getOpType() == "Conv" and
                   parentOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16) or (childOp.getOpType() == "Conv" and
                   childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)))
                   spillForCM = needForceSpillingForCM(parentOp, childOp, parentClustering, childClustering);

                if(!enableChannelMajorConv && parentOp.getOpType() == "Input" && childOp.getOpType() == "Conv" && childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
                {
                    if (parentClustering == "SplitOverHOverlapped")
                    {
                    log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                        + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by no CM Conv but Input has SOHOverlapped");
                    return INF;
                    }

                }

                // forgive me for this hack. Multiple input topologies are not qualifed for CM Conv, so this hack
                if (parentOp.getOpType() == "ImplicitInput" or childOp.getOpType() == "ImplicitInput")
                {
                    if (parentClustering == "SplitOverHOverlapped")
                    {
                    log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                        + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by SOHOverlapped/CMConv not supported for multiple Input scenario");
                    return INF;

                    }
                    if (childClustering == "SplitOverHOverlapped")
                    {
                    log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                        + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by SOHOverlapped/CMConv not supported for multiple Input scenario");
                    return INF;
                    }
                }

                if (spillForCM and !(parent["spilling"].get<bool>()))
                {
                    log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                        + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by spill needed for CM Conv but parent doesn't spill");
                    return INF;
                }
                int8_t success = checkHWUnsupportedOp(parentOp);
                if (success != 0)
                {
                    if (success == 1)
                        log(mv::Logger::MessageType::Warning, "The limitation of the tensor dimension 8192 might be hitted with the \
                            operation " + parentOp.getName());
                     else
                        log(mv::Logger::MessageType::Error, "Unsupported kernel/stride combination for DpuTask for \
                            the operation " + parentOp.getName());
                }

                if(createStrategyDots)
                {
                    int strategyCheck = checkForBadStrategy(parentOp,parent);
                    if(strategyCheck > 0)
                    {
                        const mv::Attribute str = failure_causes[strategyCheck];
                        parent["infCause"] = str;
                        return INF;
                    }
                    strategyCheck = checkForBadStrategy(childOp, child);
                    if(strategyCheck > 0)
                    {
                        const mv::Attribute str = failure_causes[strategyCheck];
                        child["infCause"] = str;
                        return INF;
                    }
                }

                //NOTE: The logic dynamically enables "Concate" with SplitOverH and SplitOverK
                //to indicate splitoverH/splitoverK are allowed, so that the upper conv can
                //choose both SOH/SOK. For now we add conditions to align split strategy
                //before and after Concate and avoid Concate's parents choose different split strategy.
                //NOTE: Normally in ddr concatenation input and output tensor strategies are not mandatory to share same
                //split strategies, solving it like that temporary till all the pair-concats on ddr strategies are tested
                if (child["concat"].get<string>() == "SplitOverH")
                {
                    if(parentClustering == "SplitOverK" || parentClustering == "HKSwitch" || parentClustering == "Clustering")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by SOK/HKSwitch/clustering to concat SOH");
                            return INF;
                    }
                }
                else if (parent["concat"].get<string>() == "SplitOverH")
                {
                    if(childClustering == "SplitOverK")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by concat SOH to SOK");
                            return INF;
                    }
                }
                else if (child["concat"].get<string>() == "SplitOverK")
                {
                    if(parentClustering == "SplitOverH")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by SOH to concat SOK");
                            return INF;
                    }
                }
                else if (parent["concat"].get<string>() == "SplitOverK")
                {
                    if(childClustering == "SplitOverH" || childClustering == "HKSwitch")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by concat SOK to SOH/HKSwitch");
                            return INF;
                    }
                }
                //NOTE: If you Spill a parent a child can be everything...the only thing
                //that has no sense if is your parent is spilling to be HKSwitch as
                //this strategy exists in order to reverse strategies in CMX
                else if (parent["spilling"].get<bool>())
                {
                    if (childClustering == "HKSwitch")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by spilling before HKSwitch");
                            return INF;
                    }
                    //NOTE: For now I disable parent spill SOH->child (Clustering, K)
                    if (parentClustering == "SplitOverH" and ((childClustering == "Clustering" and childOp.getOpType() !=  "Output") ||
                                                              childClustering == "SplitOverK"))
                    {
                        if (!(enableChannelMajorConv and ((parentOp.getOpType() == "Conv" and
                           parentOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16) or (childOp.getOpType() == "Conv" and
                           childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16))) )
                            {
                                log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                 + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by SOH to SOK/clustering");
                            return INF;
                            }
                    }
                }
                else
                {
                    //NOTE: If your parent is SplitOverH, your childs should be only Soh,HKSwitch
                    if (parentClustering == "SplitOverH")
                    {
                        if (childClustering == "SplitOverK" || childClustering == "Clustering")
                        {
                            log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by incompatible clustering strategies");
                                return INF;
                        }
                    }
                    if (parentClustering == "SplitOverK" || parentClustering == "Clustering"
                            || parentClustering == "HKSwitch")
                    {
                        if (childClustering == "SplitOverH" || childClustering == "HKSwitch" || childClustering == "SplitOverHOverlapped")
                        {
                            log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by incompatible clustering strategies");
                                return INF;
                        }
                    }
                    //NOTE: If the child layer is streamed over H or C the parent/input tensors needs to be in DDR
                    if ((child["streaming"].get<Shape>()["H"] * child["streaming"].get<Shape>()["C"]
                         * child["streaming"].get<Shape>()["W"]) > 1)
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by stream after not spilling");
                            return INF;
                    }
                }
                //NOTE: Subdilation storage element population is not implemented for the SOH case
                if (childOp.getOpType() == "Conv"  && childOp.hasAttr("DilatedSubConv")
                        && childOp.get<bool>("DilatedSubConv")
                        && childClustering == "SplitOverH")
                    return INF;
                if (parentOp.getOpType() == "Conv" && parentOp.hasAttr("DilatedSubConv")
                        && parentOp.get<bool>("DilatedSubConv")
                        && parentClustering == "SplitOverH")
                    return INF;
                if( childOp.getOpType() == "Conv")
                {
                    auto weightsShape = childOp.getInputTensor(1)->getShape();
                    auto numInChannels = weightsShape[KERNEL_INPUT_CHANNELS];

                    //This rule only relevant for channel major convs
                    if( enableChannelMajorConv and numInChannels % 16)
                    {
                        if(childClustering == "SplitOverH" and parentOp.getOpType() == "Input" and not (parentClustering == "SplitOverHOverlapped"))
                        {
                            log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by SOH chmjconv");
                                return INF;
                        }
                        if(parentClustering == "SplitOverHOverlapped" and not (childClustering == "SplitOverH"))
                        {
                            log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by SOH chmjconv");
                                return INF;
                        }
                    }
                    //If we aren't CM conv, kernel > 1 requires sparsity for SOH, so parent can't spill
                    else if((parent["spilling"].get<bool>()) and (childClustering == "SplitOverH")
                            and  weightsShape[KERNEL_WIDTH] > 1 and referenceDevice == "A0")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by spill to SOH conv>1");
                            return INF;
                    }
                    else if(childClustering == "SplitOverH")
                    {
                        auto outputTensorShape = childOp.getOutputTensor(0)->getShape();
                        unsigned int W = outputTensorShape[IO_WIDTH_DIMENSION];
                        unsigned int H = outputTensorShape[IO_HEIGHT_DIMENSION];
                        unsigned int C = outputTensorShape[IO_CHANNEL_DIMENSION];
                        unsigned dy = std::ceil(static_cast<double>(H) / totalClusters);

                        if ((W*dy*C)%128 != 0)
                        {
                            log(mv::Logger::MessageType::Debug, child["name"].toString()+"_"+child["id"].toString() + " INF caused by incorrect SOH");
                            return INF;
                        }
                    }
                }
                //Note: last op should not be HKSwitch
                else if (childOp.getOpType() == "Output")
                {
                    if (parentClustering == "HKSwitch")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by final op HKSwitch");
                        return INF;
                    }
                }

                //NOTE: IF you have to spill your parent and the child is fp16 you are going to assign clustering on child
                if(parentOp.getOpType() == "Eltwise" and parent["spilling"].get<bool>() &&
                    childOp.hasAttr("floatPrecision") && childOp.get<bool>("floatPrecision") && childClustering != "Clustering"
                        and (referenceDevice == "A0"))
                {
                    return INF;
                }

                //Note: Input clustering strategy should match first layer, if it is Z-major
                if(parentOp.getOpType() == "Input" and not
                    (childOp.getOpType() == "Conv" and enableChannelMajorConv
                    and childOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] % 16))
                {
                    if(parentClustering != childClustering)
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by input not matching first layer");
                        return INF;
                    }
                }


                // These sparsity rules apply pairwise, and effect memory size and execution time.
                // Note: Activation sparsity now decided by GO
                bool parentOutputSparsity = parent["outputSparsity"].get<bool>();
                bool childInputSparsity = child["inputSparsity"].get<bool>();

                // Sparsity must match
                if(parentOutputSparsity != childInputSparsity)
                {
                    log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by sparsity pair mismatch");
                           return INF;
                }

                // In cases where real activation sparsity  will be required later
                // ensure there is enough memory for them
                if(requiresRealActivationSparsity(childOp, childClustering))
                {
                    // Note: equivalent child check happens in checkForBadStrategy
                    if(!parent["outputSparsity"].get<bool>()){
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by dense output, sparse input");
                           return INF;
                    }
                }

                bool requiresFakeSparsity = requiresFakeActivationSparsity(childOp);
                if(requiresFakeSparsity){
                    parentOutputSparsity = false;
                    childInputSparsity = true;
                }


                // Note: Should no longer need to recheck for sparse memory size - we check the actual sparse memory size
                // when generating the strategies now!
                //If activation sparsity is occuring between this pair, recheck that the increased memory footprint
                //does not exceed CMX
                if(childInputSparsity)
                {
                    auto parentMem = memorySize(parentOp,
                                            parentClustering,
                                            false,
                                            parentOutputSparsity,
                                            parent["weightsSparsity"].get<bool>(),
                                            parent["streaming"].get<Shape>(),
                                            requiresFakeActivationSparsity(parentOp));

                    auto childMem = memorySize(childOp,
                                            childClustering,
                                            childInputSparsity,
                                            false,
                                            child["weightsSparsity"].get<bool>(),
                                            child["streaming"].get<Shape>(),
                                            requiresFakeSparsity);


                    if( (childOp.getOpType() != "Output") and
                      ( (childMem.first + childMem.second) > clusterMemory) )
                    {
                            log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by child sparsityMemorySize");
                            return INF;
                    }
                    if( (parentOp.getOpType() != "Input") and parentOp.hasTypeTrait("optimizable") and
                      ( (parentMem.first + parentMem.second) > clusterMemory) )
                    {
                            log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by parent sparsityMemorySize");
                            return INF;
                    }
                }

                auto execTime1 = executionTime(parentOp,parent);
                auto execTime2 = executionTime(childOp,child);

                if(parent["spilling"].get<bool>())
                {
                    for(auto output : parentOp.getOutputTensor())
                        execTime1 += ((double)output->getShape().totalSize()) / ((double)ddrBandwidth);
                }
                if(child["spilling"].get<bool>())
                {
                    for(auto output : childOp.getOutputTensor() )
                        execTime2 += ((double)output->getShape().totalSize()) / ((double)ddrBandwidth);
                }

                double extra_stream_decay = 1.5; //TODO: expose in config
                if(parentOp.getOpType() == "Conv")
                {
                    auto streamOverK = parent["streaming"].get<Shape>()["K"];
                    auto WSize = parentOp.getInputTensor(1)->getShape().totalSize();

                    // Technically you could scale weight size here if the weight are compressed
                    // See technical note at calculateWeightsCompressionRatio() above
                    if( streamOverK == 1)
                        execTime1 += (double)WSize / (double)ddrBandwidth;
                    else if( streamOverK == 2)
                        execTime1 += ((double)WSize / (double)ddrBandwidth) * 2;
                    else if( streamOverK > 2)
                        execTime1 += ((double)WSize / (double)ddrBandwidth) * (extra_stream_decay*streamOverK);
                }

                if(childOp.getOpType() == "Conv")
                {
                    auto streamOverK = child["streaming"].get<Shape>()["K"];
                    auto WSize = childOp.getInputTensor(1)->getShape().totalSize();
                    if( streamOverK == 1)
                        execTime2 += (double)WSize / (double)ddrBandwidth;
                    else if( streamOverK == 2)
                        execTime2 += ((double)WSize  / (double)ddrBandwidth) * 2;
                    else if( streamOverK > 2)
                        execTime2 += ((double)WSize / (double)ddrBandwidth) * (extra_stream_decay*streamOverK);
                }

                auto parentStreamOverH = parent["streaming"].get<Shape>()["H"];
                if(parentStreamOverH > 1)
                {
                    //assuming here that if have streaming, then inOut is spilled. There is condition above to check this
                    // this is just current "make it work fast" assumption. Will be replaced with proper BW_to_compute calucation
                    auto iSize = parentOp.getInputTensor(0)->getShape().totalSize();
                    auto oSize = parentOp.getOutputTensor(0)->getShape().totalSize();

                    execTime1 += ((double)(iSize + oSize) / (double)ddrBandwidth) * (extra_stream_decay * parentStreamOverH);
                }
                auto childStreamOverH = child["streaming"].get<Shape>()["H"];
                if(childStreamOverH > 1)
                {
                    auto iSize = childOp.getInputTensor(0)->getShape().totalSize();
                    auto oSize = childOp.getOutputTensor(0)->getShape().totalSize();

                    execTime2 += ((double)(iSize + oSize) / (double)ddrBandwidth)  * (extra_stream_decay * childStreamOverH);
                }

                auto parentStreamOverN = parent["streaming"].get<Shape>()["B"];
                if(parentStreamOverN > 1)
                {
                    auto iSize = parentOp.getInputTensor(0)->getShape().totalSize();
                    auto oSize = parentOp.getOutputTensor(0)->getShape().totalSize();

                    execTime1 += ((double)(iSize + oSize) / (double)ddrBandwidth) * (extra_stream_decay * parentStreamOverN);
                }
                auto childStreamOverN = child["streaming"].get<Shape>()["B"];
                if(childStreamOverN > 1)
                {
                    auto iSize = childOp.getInputTensor(0)->getShape().totalSize();
                    auto oSize = childOp.getOutputTensor(0)->getShape().totalSize();

                    execTime2 += ((double)(iSize + oSize) / (double)ddrBandwidth)  * (extra_stream_decay * childStreamOverN);
                }

                //When streaming C we have to stream both activations and weights, so include both in cost
                //Note, only ops with weights should be trying to stream C (depthwise only enabled)
               auto parentStreamOverC = parent["streaming"].get<Shape>()["C"];
                if(parentStreamOverC > 1)
                {
                    auto iSize = parentOp.getInputTensor(0)->getShape().totalSize();
                    auto oSize = parentOp.getOutputTensor(0)->getShape().totalSize();
                    auto WSize = parentOp.getInputTensor(1)->getShape().totalSize();

                    execTime1 += ((double)WSize/ (double)ddrBandwidth) * (extra_stream_decay * parentStreamOverC);
                    execTime1 += ((double)(iSize + oSize)/ (double)ddrBandwidth) * (extra_stream_decay * parentStreamOverC);
                }
                auto childStreamOverC = child["streaming"].get<Shape>()["C"];
                if(childStreamOverC > 1)
                {
                    auto iSize = childOp.getInputTensor(0)->getShape().totalSize();
                    auto oSize = childOp.getOutputTensor(0)->getShape().totalSize();
                    auto WSize = childOp.getInputTensor(1)->getShape().totalSize();

                    execTime2 += ((double)WSize / (double)ddrBandwidth) * (extra_stream_decay * childStreamOverC);
                    execTime2 += ((double)(iSize + oSize) / (double)ddrBandwidth)  * (extra_stream_decay * childStreamOverC);
                }

                //TODO remove this hack. currently ensures when cluster and soh are equal, soh occurs. only matters for CMconv
                if(parentClustering == "SplitOverHOverlapped")
                    execTime1 = execTime1 - 1;

                if(parentOutputSparsity)
                    execTime1 = execTime1 + 1;

                if(childInputSparsity)
                    execTime2 = execTime2 + 1;

                return execTime1 + execTime2;
        }

            bool decideWeightsSparsity(mv::Op op)
            {
                //These values come from empircal data using three layer tests, in a matrix of size vs. sparsity
                //For performance, implemented as piece-wise if/else rather than polynomial
                double CMX_THRESHOLD_LOW = .05;
                double CMX_THRESHOLD_HIGH = .5;
                double ZEROPOINT_THRESHOLD_LOW = .3;
                double ZEROPOINT_THRESHOLD_HIGH = .2;

                // Only Z-major convolutions support weights sparsity, this is codified in the compilation descriptors
                if( !createStrategyFromBool(op,"weightsSparsity") )
                    return false;

                // If CM convolutions are enabled, don't sparsify these
                if(enableChannelMajorConv and op.getOpType() == "Conv" and
                   op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] % 16)
                    return false;

                //Size of weights, actual sparsity of tensor determine speedup
                auto weightsSize = realTensorSize(op.getInputTensor(1), {1,1,1,1}, false);

                //If weights are less than 5% of CMX, sparsity benefit does not outweigh overhead
                //no matter how sparse the weights are
                if(weightsSize < (clusterMemory * CMX_THRESHOLD_LOW))
                    return false;

                auto zeroPoints = op.getInputTensor(1)->getNumZeroPoints();
                double actualSparsity = (double) zeroPoints/ (double)weightsSize;

                //Weights between 5% and 50% of CMX, enable sparsity at threshold 30% sparse weights
                if(weightsSize < (clusterMemory * CMX_THRESHOLD_HIGH) and
                    actualSparsity < ZEROPOINT_THRESHOLD_LOW)
                    return false;

                //Weights larger than 50% of CMX, enable sparsity at threshold 20% sparse weights
                if(weightsSize >= (clusterMemory * CMX_THRESHOLD_HIGH) and
                    actualSparsity < ZEROPOINT_THRESHOLD_HIGH)
                    return false;

                return true;
            }

            void generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& strategyVec)
            {
                auto findStrategy = [](vector<Attribute>& vec,const string& str) ->bool { for(const auto elem : vec) if(str==elem.get<string>()) return true; return false;};

                vector<Attribute> spillingPool;
                if(globalForceSpilling)
                    spillingPool.push_back(true);
                else
                    spillingPool = createTStrategyPoolFromBool(op, "forceSpilling");

                vector<Attribute> clusteringStrategyPool;

                if(totalClusters == 1 or op.hasAttr("forceClustering"))
                    clusteringStrategyPool.push_back(string("Clustering"));
                else if (totalClusters > 1)
                    clusteringStrategyPool = createStrategyPoolFromStrategySet(op,"clusteringStrategies");
                else
                    throw LogicError(*this, "Graph Optimizer unable to determine number of clusters");

                vector<Attribute> streamingStrategyPool = createStrategyPoolFromStrategySet(op,"streamingStrategies");

                bool hasStreamOverK = false;
                bool hasStreamOverW = false;
                bool hasStreamOverH = false;
                bool hasStreamOverC = false;
                bool hasStreamOverN = false;

                if(globalEnableStreaming)
                {
                    hasStreamOverK = findStrategy(streamingStrategyPool,"StreamOverK");
                    hasStreamOverW = findStrategy(streamingStrategyPool,"StreamOverW");
                    hasStreamOverH = findStrategy(streamingStrategyPool,"StreamOverH");
                    hasStreamOverC = findStrategy(streamingStrategyPool,"StreamOverC");
                    hasStreamOverN = findStrategy(streamingStrategyPool,"StreamOverN");
                }


                bool weightsSparsity = false;
                if(requiresWeightsSparsity(op))
                    weightsSparsity = true;
                else if(globalEnableWeightsSparsity)
                    weightsSparsity = decideWeightsSparsity(op);

                vector<Attribute> concatPool = {string("None")};
                if(op.getOpType() == "Concat")
                {
                    concatPool = {string("SplitOverH"), string("SplitOverK")};
                }

                //TODO:: replace nested loops with clean cartesian product function
                for( const auto concat : concatPool){
                for( const auto spilling : spillingPool)
                {
                    for( const auto clustering : clusteringStrategyPool)
                    {
                        // Make decision about input activation sparsity, depending on clustering strategy
                        vector<Attribute> inputActivationSparsity = {false};
                        vector<Attribute> outputActivationSparsity = {false};
                        if(globalEnableActivationSparsity)
                        {
                            inputActivationSparsity = createTFStrategyPoolFromBool(op,"inputActivationSparsity");
                            outputActivationSparsity = createTFStrategyPoolFromBool(op,"outputActivationSparsity");
                        }
                        if(globalForceActivationSparsity)
                        {
                            inputActivationSparsity = {createStrategyFromBool(op,"inputActivationSparsity")};
                            outputActivationSparsity = {createStrategyFromBool(op,"outputActivationSparsity")};
                        }
                        for(const auto inputSparsity : inputActivationSparsity)
                        {
                        for(const auto outputSparsity : outputActivationSparsity)
                        {
                        bool fakeSparsity = requiresFakeActivationSparsity(op);


                        // Determine streaming options
                        // 0. Determine if streams over H are possible
                        // 1. Determine if streams over N are possible
                        // 2. Determine if streams over K are possible
                        // 3. If no streams over H or K will fit, enable nested streaming
                        // 4. Nested loops over generated streaming options to produce all strategy options

                        unsigned maxSplitOverH = 1;
                        unsigned minSplitOverH = 1;
                        if(hasStreamOverH)
                        {
                            maxSplitOverH = getStreamsOverH(op,clustering,inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,1,1,1,1},fakeSparsity);
                            if(maxSplitOverH < 1) maxSplitOverH = 1;
                        }

                        // Stream over batch, match number of streams over H
                        unsigned n = 1;
                        if(hasStreamOverN and op.getInputTensor(0)->getShape()["N"] > 1)
                        {
                            n = op.getInputTensor(0)->getShape()["N"];
                        }

                        vector<size_t> streamsOverK;
                        if(hasStreamOverK)
                            streamsOverK = getMaxStreamOverK(clustering.get<string>(),op);
                        else
                            streamsOverK.push_back(1);

                        vector<size_t> streamsOverC;
                        if (hasStreamOverC)
                            streamsOverC = {1,2,3,4}; // TODO calculate properly
                        else
                            streamsOverC.push_back(1);

                        bool enableNestedStreaming = false;
                        auto maxK = streamsOverK.front();
                        auto memK = memorySize(op,clustering,inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,1,1,maxK,n},fakeSparsity);
                        auto memoryMaxK = memK.first + memK.second;
                        auto memH = memorySize(op,clustering,inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,maxSplitOverH,1,1,n},fakeSparsity);
                        auto memoryMaxH = memH.first + memH.second;


                        // If streaming is enabled, but streaming over k or h alone doesn't fit, enable nested streaming
                        if(hasStreamOverK and (streamsOverK.size() > 1) and hasStreamOverH
                                and ((memoryMaxH > clusterMemory) and (memoryMaxK > clusterMemory))){
                            enableNestedStreaming = true;
                            // Note: Adjusting maxSplitOverH appropriately for nested is now handled on the fly
                            // for each possible stream over K, a single stream over H option that fits is chosen
                        }

                    // for(unsigned n = 1; n <= maxSplitOverN; n++)
                    // {
                        for(const auto k : streamsOverK)
                        {
                            if(enableNestedStreaming) // generate h on the fly
                            {
                                maxSplitOverH = getStreamsOverH(op,clustering,inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,1,1,k,1},fakeSparsity);
                                minSplitOverH = maxSplitOverH -1;
                            }
                            if(minSplitOverH < 1) minSplitOverH = 1;
                            if(maxSplitOverH < 1) maxSplitOverH = 1;
                            // cout << "K = " << k <<", H Streams loop: " << minSplitOverH << " --> " << maxSplitOverH << endl;
                            for(unsigned h = minSplitOverH; h <= maxSplitOverH; h++)
                            {
                                for(const auto c : streamsOverC)
                                {
                                    if((h > 1) and (c > 1)) //Fast hack to disable nested streaming with C
                                        continue;
                                    if((h > 1) and (n > 1)) //Fast hack to disable nested streaming with n
                                        continue;
                                    if( !enableNestedStreaming and ((h>1) and (k>1))) // Skip nested streams unless necessary
                                        continue;
                                    if( enableNestedStreaming and ((h==1) or (k==1))) // If need nested streams, ignore non-nested
                                       continue;
                                    if( ((h*k*c*n) > 1) and !spilling.get<bool>()) // If streaming and not spilling, skip
                                       continue;

                                    Shape streamShape({1,h,c,k,n});//Stream over W is 1 for now . TODO: implement stream W

                                    StrategySet s;
                                    s["name"] = op.getName();
                                    s["id"] = (unique_ctr++);
                                    //Input sparsity is always enabled/disabled by global switch, except in this case were it is disallowed
                                    // if(clustering.get<string>() == "SplitOverK")
                                    //     s["inputSparsity"] = false;
                                    // else
                                    //     s["inputSparsity"] = inputActivationSparsity;
                                    s["inputSparsity"] = inputSparsity;
                                    s["outputSparsity"] = outputSparsity;
                                    s["weightsSparsity"] = weightsSparsity;
                                    s["spilling"] = spilling;
                                    s["clustering"] = clustering;
                                    s["streaming"] = streamShape;
                                    s["concat"] = concat;

                                    //Function to prune strategies that will have only infinite edges in or out (or both), improves performance
                                    auto strategyCheck = checkForBadStrategy(op,s);
                                    // cout << op.getName() << " {" << clustering.toString() << ", " << streamShape.toString() << "} : " << strategyCheck << endl;
                                    if(!createStrategyDots and (strategyCheck > 0))
                                        continue;

                                    strategyVec.push_back(s);
                                }
                            }}
                            }
                        }
                    }
                }
				}
                if(strategyVec.empty())
                    throw LogicError(*this,"No strategies created for layer " + op.getName() + ". Layer possibly unsupported.");
            }

        };

    }

}

static void GraphParameterOptimizationFcn(
    const mv::pass::PassEntry& ,
    mv::ComputationModel& model,
    mv::TargetDescriptor& td, mv::Element& passDesc,
    mv::Element&
)
{
    mv::OpModel om(model);
    mv::graphOptimizer::StrategyManagerKmb strategyManager(om,passDesc, td);

    strategyManager.updateValuesFromJSON();
    strategyManager.updateDefaultValues();
    strategyManager.readGlobalConfigs();

    strategyManager.graphParameterOptimizations();

    return;
}
