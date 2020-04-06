#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/utils/custom_math.hpp"


static void GraphParameterOptimizationFcn(
    const mv::pass::PassEntry& pass,
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
            StrategyManagerKmb(OpModel& model,mv::Element& passDesc) :
                StrategyManager(model,passDesc)
            {
                auto globalParams = model.getGlobalConfigParams();
                enableChannelMajorConv = globalParams->get<bool>("enable_channel_major_conv");
            }

            size_t totalClusters=4;
            size_t clusterMemoryKb=896;
            size_t dpuPerCluster=5;
            int ddrBandwidth=128;
            int sysClock=500;
            bool globalEnableStreaming=true;
            bool globalEnableActivationSparsity=false;
            bool globalEnableWeightsSparsity=false;
            bool globalForceSpilling=false;
            bool enableChannelMajorConv=false;
            double safetyFactor=1.0;
            double clusterMemory=(double)clusterMemoryKb * 1024.0 * safetyFactor;
            std::vector<string> failure_causes = {"Unknown", "MemorySize", "Stream+ClusterComp", 
            "SpillHKSwitch", "SOKNotAlign16", "InputNotSpilled", "OutputNotSpilled", "StreamingNotSpilled", 
            "Workload<KernelSOH", "ChannelMjr1", "ChannelMjr2", "DWChannels"};


            void readGlobalConfigs()
            {
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
                globalEnableActivationSparsity = globalStrategies_["enableActivationSparsity"].get<bool>();
                globalEnableWeightsSparsity = globalStrategies_["enableWeightsSparsity"].get<bool>();
                globalForceSpilling =  globalStrategies_["forceSpilling"].get<bool>();
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

                // auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };

                // Shape worstStreamPool = streamingPool;

                // Shape tensorShape = tensorToSize->getShape();
                // //update the streamingPool to the worst combination, based on slice sizes
                // size_t outputSize;
                // size_t numberOfSplits;
                // if(streamingPool["H"] > 1) // If streaming over H
                // {
                //     outputSize = tensorShape[mv::IO_HEIGHT_DIMENSION];
                //     numberOfSplits = streamingPool[mv::IO_HEIGHT_DIMENSION];
                //     auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
                //     int newOutputSize = newOutputSizes.first;

                //     int remainderOutputSize = newOutputSizes.second;
                //     if (remainderOutputSize > newOutputSize)
                //         newOutputSize = remainderOutputSize;

                //     auto worstNumberOfSplits = outputSize/newOutputSize;
                //     worstStreamPool[mv::IO_HEIGHT_DIMENSION] = worstNumberOfSplits;
                // }
                // else if(streamingPool["B"] > 1) // If streaming over N
                // {
                //     outputSize = tensorShape[mv::IO_BATCH_DIMENSION];
                //     numberOfSplits = streamingPool[mv::IO_BATCH_DIMENSION];
                //     auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
                //     int newOutputSize = newOutputSizes.first;

                //     int remainderOutputSize = newOutputSizes.second;
                //     if (remainderOutputSize > newOutputSize)
                //         newOutputSize = remainderOutputSize;

                //     auto worstNumberOfSplits = outputSize/newOutputSize;
                //     worstStreamPool[mv::IO_BATCH_DIMENSION] = worstNumberOfSplits;
                // }

                // //TODO add handling for weights case if we dont align it to 16 always
                // size_t streamDivisor = 1;
                // for(size_t dim = 0; dim <  worstStreamPool.ndims(); ++dim)
                // {
                //     streamDivisor = streamDivisor * worstStreamPool[dim];
                // }

                // if(isCMConv)
                //     return tensorToSize->computeTotalSize(16, false, false, false)/streamDivisor;

                // return tensorToSize->computeTotalSize(16, false, false, true)/streamDivisor;
            }

            size_t alignedWeightsSize(const mv::Data::TensorIterator tensorToSize, const Shape& streamConfig){
                size_t alignedFullInputChannels = mv::round_up(tensorToSize->getShape()[KERNEL_INPUT_CHANNELS], 16);
                
                size_t alignedFullOutputChannels = mv::round_up(tensorToSize->getShape()[KERNEL_OUTPUT_CHANNELS], 16);
                size_t alignedSplittedOutputChannels = mv::round_up(alignedFullOutputChannels/streamConfig["K"], 16);

                return (alignedFullInputChannels * alignedSplittedOutputChannels * 
                    tensorToSize->getShape()[KERNEL_WIDTH] * tensorToSize->getShape()[KERNEL_HEIGHT]);
            }

            pair<size_t,size_t> memorySize(mv::Op& op, const Attribute& clustering, bool inputActivationSparsity,
                                            bool outputActivationSparsity, bool weightsSparsity, const Shape& streamConfig)
            {
                auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };

                size_t inputSize = 0;
                size_t outputSize = 0;
                size_t weightSize = 0;
                size_t weightTableSize = 0;

                size_t totalWeightsSize = 0;
                size_t totalActivationSize = 0;
                auto isCMConv = false;

                if(enableChannelMajorConv and op.getOpType() == "Conv" and 
                   op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
                        isCMConv = true;

                if(op.getOpType() != "Input")
                    inputSize = realTensorSize(op.getInputTensor(0),{streamConfig["W"],streamConfig["H"],streamConfig["C"],1,streamConfig["B"]}, isCMConv);
                if(op.getOpType() != "Output")
                    outputSize = realTensorSize(op.getOutputTensor(0),{streamConfig["W"],streamConfig["H"],streamConfig["K"],1,streamConfig["B"]}, isCMConv);
                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

                if(op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
                {
                    size_t alignedFullChannels = mv::round_up(op.getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION], 16);
                    size_t alignedSplittedChannels = mv::round_up(alignedFullChannels/streamConfig["K"], 16);
                    weightTableSize = 4 * alignedSplittedChannels;
                    if (op.getOpType() == "Conv")
                    {
                        weightSize += alignedWeightsSize(op.getInputTensor(1),{1,1,streamConfig["C"],streamConfig["K"],1});
                        //weightSize += realTensorSize(op.getInputTensor(1),{1,1,streamConfig["C"],streamConfig["K"]}, isCMConv);
                    }
                    else
                        weightSize += realTensorSize(op.getInputTensor(1),{1,1,streamConfig["C"],1,1}, isCMConv);
                }
                else if(op.getOpType() == "MaxPool")
                {
                    weightTableSize = 0;
                    weightSize = 0;
                }
                else if(op.getOpType() == "Eltwise" && !software)
                {
                    weightTableSize = 0;
                    weightSize = 0; //TODO think about
                    inputSize += realTensorSize(op.getInputTensor(1),{streamConfig["W"],streamConfig["H"],streamConfig["C"],1,1}, isCMConv);
                }

                //Additional memory footprint for sparsity
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
                    weightSize += sparseWeightSize;
                }

                weightSize += weightTableSize;

                auto clusterStrategy = clustering.get<string>();

                if(clusterStrategy == "Clustering")
                {
                    totalActivationSize = inputSize + outputSize;
                    totalWeightsSize = weightSize;
                }
                else if(clusterStrategy == "SplitOverH")
                {
                    totalActivationSize = div(inputSize,totalClusters) + div(outputSize,totalClusters);
                    totalWeightsSize = weightSize;
                }
                else if(clusterStrategy == "SplitOverK")
                {
                    totalActivationSize = inputSize + outputSize;
                    totalWeightsSize =  div(weightSize,totalClusters);
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

            unsigned getMaxStreamOverH(const string& clustering,mv::Op& op, vector<size_t> streamsOverK){
                for(auto k : streamsOverK){
                    auto memH = memorySize(op,clustering,true,true,true,{1,1,1,k,1});
                    auto activationsSize = memH.first;
                    auto weightsSize = memH.second;

                    double availableMemory = (double) clusterMemory - (double) weightsSize;

                    if (availableMemory > 0){ // Weights can fit, determine number of splits for activations
                        unsigned splitsToFit = ceil((double)activationsSize/availableMemory);

                        //Special case for convs: Max split over H cannot be higher than dimension/kernel
                        if(op.getOpType() == "Conv")
                        {
                            auto kernelSize = op.getInputTensor(1)->getShape()[KERNEL_HEIGHT];
                            auto dim = op.getInputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                            if(splitsToFit < dim/kernelSize)
                                return splitsToFit;
                            else
                                return dim/kernelSize; // return this and try to mix with high values of k for viable strategy
                        }
                        else //normal case, return just enough splits to fit
                            return splitsToFit;
                    }
                }
                throw LogicError(*this,"Unable to generate nested streaming strategies for layer " + op.getName() + ". Layer size is unsupported.");
            }

            vector<size_t> getMaxStreamOverK(const string& clustering,mv::Op& op)
            {
                auto opType = op.getOpType();

                if( opType == "Input" or opType == "Output" )
                    return vector<size_t>(0);

                auto outputShape = op.getOutputTensor(0)->getShape();
                size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
                size_t alignedOutputChannelSize = mv::round_up(outputChannelSize, 16);

                if(clustering == "SplitOverK")
                    alignedOutputChannelSize = alignedOutputChannelSize / totalClusters;

                vector<size_t> splits;
                size_t maxSplits = 1;

                if(globalEnableStreaming)
                    maxSplits = (alignedOutputChannelSize/16);

                if(maxSplits > 64)
                    maxSplits = 64;

                splits.push_back(1);
                for(unsigned split = 2; split <= maxSplits; split=split+2)
                {
                    bool validSplit = true;

                    if(alignedOutputChannelSize/split < 16)
                        validSplit = false;
                    
                    if(!validSplit)
                        continue;

                    splits.push_back(split);
                }

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

                //the actual compute
                if (outputShape.ndims() != streamNumerator.ndims())
                    outputShape = outputShape.augment(outputShape, streamNumerator.ndims());
                Shape dpuOutShape = ( outputShape / streamNumerator ) / isiSplit;
                Shape contextsInOp = dpuOutShape / contexts;
                unsigned numContextsInOp = contextsInOp.totalSize();

                if(numContextsInOp == 0)
                    throw LogicError(*this,"error in contexts");

                unsigned contextsPerDpu = (unsigned)ceil( (double)numContextsInOp / (double)dpuPerCluster);

                return contextsPerDpu * streamNumerator.totalSize() * baseKernelCost;
            }

            bool requiresActivationSparsity(Op& op, string clustering){
                // if(op.getOpType() == "Input" or op.getOpType() == "Output") 
                //     return false;

                if(requiresRealActivationSparsity(op, clustering)) 
                    return true;

                if(requiresFakeActivationSparsity(op, clustering)) 
                    return true;

                return false;
            }

            bool requiresWeightsSparsity(Op& op)
            {
                // If Z-major Conv in Float precision then need to have weights Sparsity
                if(op.getOpType() == "Conv" and
                    op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] >= 16 and
                    op.get<mv::DType>("dType") == mv::DType("Float16"))
                        return true;

                return false;
            }

            bool requiresRealActivationSparsity(Op& op, string clustering){
                //An fp16 Conv Z-major must have activation sparsity
                if ((op.getOpType() == "Conv") and  (op.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] >= 16)
                        and op.get<mv::DType>("dType") == mv::DType("Float16"))
                {
                    return true;
                }


                // Check for need for A0 SOH Sparsity workaround, (SOH conv with kernel > 1)
                // if needed, check memory constraints as for sparse tensor
                if ( op.getOpType() == "Conv" ) {
                    if( clustering == "SplitOverH" and 
                        (op.getInputTensor(1)->getShape()[KERNEL_HEIGHT] > 1 or
                         op.getInputTensor(1)->getShape()[KERNEL_WIDTH]  > 1) 
                         and (op.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] >= 16))
                         {
                            return true;
                         }
                }

                return false;
            }

             //Channel major conv, pooling and depthwise will get fake sparsity, so need to check memory constraints as if real sparsity
            bool requiresFakeActivationSparsity(Op& op, string clustering){
                if(enableChannelMajorConv and 
                  (op.getOpType() == "Conv") and  
                  (op.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16) )
                {
                    return true;
                }

                if(op.getOpType() == "MaxPool")
                    return true;

                if(op.getOpType() == "Depthwise")
                    return true;

                return false;
            }

            int8_t checkHWUnsupportedOp(mv::Op& op)
            {
                int8_t executableInHW = 0;
                if (op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv" || op.getOpType() == "MaxPool"
                        || op.getOpType() == "Eltwise")
                {
                    for (std::size_t input_gates = 0; input_gates < op.getInputTensor().size(); input_gates++)
                    {
                        if (input_gates == 0)
                        {
                            if (op.getInputTensor(input_gates)->getShape()[mv::IO_WIDTH_DIMENSION] > 8192 ||
                                op.getInputTensor(input_gates)->getShape()[mv::IO_HEIGHT_DIMENSION] > 8192 ||
                                op.getInputTensor(input_gates)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192 ||
                                op.getOutputTensor(input_gates)->getShape()[mv::IO_WIDTH_DIMENSION] > 8192 ||
                                op.getOutputTensor(input_gates)->getShape()[mv::IO_HEIGHT_DIMENSION] > 8192 ||
                                op.getOutputTensor(input_gates)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192 )
                                    executableInHW = 1;
                        }
                        else if (input_gates == 1 && op.getOpType() == "Eltwise")
                        {
                            if (op.getInputTensor(input_gates)->getShape()[mv::IO_WIDTH_DIMENSION] > 8192 ||
                                op.getInputTensor(input_gates)->getShape()[mv::IO_HEIGHT_DIMENSION] > 8192 ||
                                op.getInputTensor(input_gates)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192)
                                executableInHW = 1;
                        }
                        else if (op.getOpType() != "MaxPool" && op.getOpType() != "Eltwise")
                        //Note: all the ops have maximum a second input (weights) at G.O stage
                        {
                            if (op.getInputTensor(input_gates)->getShape()[mv::KERNEL_WIDTH] > 11 ||
                                op.getInputTensor(input_gates)->getShape()[mv::KERNEL_HEIGHT] > 11 ||
                                op.getInputTensor(input_gates)->getShape()[mv::KERNEL_INPUT_CHANNELS] > 8192 ||
                                op.getInputTensor(input_gates)->getShape()[mv::KERNEL_OUTPUT_CHANNELS] > 8192)
                                executableInHW = 2;
                            auto stride_array = op.getAttrs().at("stride").get<std::array<unsigned short, 2>>();
                            if (stride_array[0] > 8 || stride_array[1] > 8)
                                executableInHW = 3;
                        }
                    }
                }
                return executableInHW;
            }
            //Check to see if a given stategy is internally consistent for performance
            //Strategies that can only have infinite edges because they are illegal should never be added to the graph
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
                    auto fit = memorySize(op,clustering,requiresActivationSparsity(op, clustering), false,weightsSparsity,streamShape);
                    // std::cout << op.getName() << ": [" <<clustering << "][" <<streamShape.toString()<<"]    " << fit.first << " + " << fit.second << " = " << fit.first + fit.second << std::endl;
                    if(fit.first + fit.second > clusterMemory)
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
                    if ( numInChannels < 16 ) //assume channel major conv
                        if(clustering == "SplitOverH" and streamShape["H"] > 1)
                            return 10;
                }


                if (op.getOpType() == "DepthwiseConv")
                {
                    if ((op.getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192)
                            && (streamShape["C"] == 1))
                        return 11;
                }

                return 0; //good strategy
            }

            double transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {

                //TODO: expose these conditionals more cleanly
                auto INF = inf_;

                auto parentClustering = parent["clustering"].get<string>();
                auto childClustering = child["clustering"].get<string>();
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
                //NOTE: If you Spill a parent a child can be everything...the only thing
                //that has no sense if is your parent is spilling to be HKSwitch as
                //this strategy exists in order to reverse strategies in CMX
                if (parent["spilling"].get<bool>())
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
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString() 
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by SOH to SOK/clustering");
                            return INF;
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
                        if (childClustering == "SplitOverH" || childClustering == "HKSwitch")
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
                    //NOTE: Temporary Hack for InceptionV3...General solution change rectHeuristic
                    if (parentClustering == "SplitOverH" && childClustering == "SplitOverH" && requiresRealActivationSparsity(childOp, "SplitOverH"))
                    {
                        auto H = childOp.getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
                        auto W = childOp.getInputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION];
                        auto C = childOp.getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
                        auto estimatedClusterH = (int)floor((double)H/totalClusters);
                        if ((estimatedClusterH*W*C)%128 != 0)
                        {
                            return INF;
                        }
                    }
                }

                if( childOp.getOpType() == "Conv")
                {
                    auto weightsShape = childOp.getInputTensor(1)->getShape();
                    auto numInChannels = weightsShape[KERNEL_INPUT_CHANNELS];
                    auto numOutChannels = weightsShape[KERNEL_OUTPUT_CHANNELS];

                    //This rule only relevant for channel major convs
                    if( enableChannelMajorConv and numInChannels < 16)
                    {
                        if(childClustering == "SplitOverH" and not (parentClustering == "SplitOverHOverlapped"))
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
                            and  weightsShape[KERNEL_WIDTH] > 1)
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString() 
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by spill to SOH conv>1");
                            return INF;
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

                //Note: Input clustering strategy should match first layer, if it is Z-major
                if(parentOp.getOpType() == "Input" and not
                    (childOp.getOpType() == "Conv" and enableChannelMajorConv 
                    and childOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16))
                {
                    if(parentClustering != childClustering)
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString() 
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by input not matching first layer");
                        return INF;
                    }
                }


                //These sparsity rules apply pairwise, and effect memory size and execution time.
                //Make a local decision to get the correct runtime and execution time, but don't persist
                //Sparsity will not be applied where disallowed in later passes
                bool parentOutputSparsity = parent["outputSparsity"].get<bool>();
                bool childInputSparsity = child["inputSparsity"].get<bool>();

                if(parent["spilling"].get<bool>()){
                    parentOutputSparsity = false;
                    childInputSparsity = false;
                }

                // In cases where real activation sparsity  will be required later
                // ensure there is enough memory for them
                if(requiresRealActivationSparsity(childOp, childClustering)){
                    if(parent["spilling"].get<bool>()) 
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString() 
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by spilling before sparsity");
                        return INF;
                    }

                    parentOutputSparsity = true;
                    childInputSparsity = true;
                }

                if(requiresFakeActivationSparsity(childOp, childClustering)){
                    parentOutputSparsity = false;
                    childInputSparsity = true;
                }

                //If activation sparsity is occuring between this pair, recheck that the increased memory footprint
                //does not exceed CMX
                if(childInputSparsity)
                {
                    auto parentMem = memorySize(parentOp,
                                            parentClustering,
                                            false,
                                            parentOutputSparsity,
                                            parent["weightsSparsity"].get<bool>(),
                                            parent["streaming"].get<Shape>());

                    auto childMem = memorySize(childOp,
                                            childClustering,
                                            childInputSparsity,
                                            false,
                                            child["weightsSparsity"].get<bool>(),
                                            child["streaming"].get<Shape>());


                    if( (childOp.getOpType() != "Output") and 
                      ( (childMem.first + childMem.second) > clusterMemory) )
                    {
                            log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString() 
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by child sparsityMemorySize");
                            return INF;
                    }
                    if( (parentOp.getOpType() != "Input") and (parentOp.getOpType() != "Concat") and
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
                    if( streamOverK == 1)
                        execTime1 += (double)WSize / (double)ddrBandwidth;
                    else if( streamOverK == 2)
                        execTime1 += ((double)WSize  / (double)ddrBandwidth) * 2;
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
                   op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
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

                if(totalClusters == 1)
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


                bool inputActivationSparsity = false;
                bool outputActivationSparsity = false;
                if(globalEnableActivationSparsity)
                {
                    inputActivationSparsity = createStrategyFromBool(op,"inputActivationSparsity");
                    outputActivationSparsity = createStrategyFromBool(op,"outputActivationSparsity");
                }

                bool weightsSparsity = false;
                if(requiresWeightsSparsity(op))
                    weightsSparsity = true;
                else if(globalEnableWeightsSparsity)
                    weightsSparsity = decideWeightsSparsity(op);


                //TODO:: replace nested loops with clean cartesian product function
                for( const auto spilling : spillingPool)
                {
                    for( const auto clustering : clusteringStrategyPool)
                    {
                        // Make decision about input activation sparsity, depending on clustering strategy
                        bool iAS = inputActivationSparsity;
                        if (!iAS and requiresActivationSparsity(op, clustering.get<string>())) 
                            iAS = true;

                        // Determine streaming options
                        // 0. Determine if streams over H are possible
                        // 1. Determine if streams over N are possible
                        // 2. Determine if streams over K are possible
                        // 3. If no streams over H or K will fit, enable nested streaming
                        // 4. Nested loops over generated streaming options to produce all strategy options

                        unsigned maxSplitOverH = 1;
                        if(hasStreamOverH)
                        {
                            auto memH = memorySize(op,clustering,iAS,outputActivationSparsity,weightsSparsity,{1,1,1,1,1});
                            auto activationsSize = memH.first;
                            auto weightsSize = memH.second;
                            double availableMemory = (double) clusterMemory - (double) weightsSize;
                            if (availableMemory < 0) // Weights don't fit, can't stream over H
                                maxSplitOverH = 1;
                            else 
                            {
                                unsigned splitsToFit = ceil((double)activationsSize/availableMemory);
                                if (splitsToFit < 1)
                                    maxSplitOverH = 1;
                                else
                                    maxSplitOverH = splitsToFit;
                            }
                        }
                        // Stream over batch, match number of streams over H
                        // unsigned maxSplitOverN = 1;
                        // if(hasStreamOverN and op.getInputTensor(0)->getShape()["N"] > 1)
                        // {
                        //     // Split enough times to fit into HCW into memory, without any further streaming
                        //     maxSplitOverN = maxSplitOverH;
                        //     if (maxSplitOverN > op.getInputTensor(0)->getShape()["N"])
                        //         maxSplitOverN = op.getInputTensor(0)->getShape()["N"];
                        // }
                        // Temporarily force all streams over N = N, stream to batch 1
                        unsigned n = 1;
                        if(hasStreamOverN and op.getInputTensor(0)->getShape()["N"] > 1)
                        {
                            n = op.getInputTensor(0)->getShape()["N"];
                        }

                        //Max split over H cannot be higher than dimension/kernel
                        if(op.getOpType() == "Conv")
                        {
                            auto kernelSize = op.getInputTensor(1)->getShape()[KERNEL_HEIGHT];
                            auto dim = op.getInputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                            if(maxSplitOverH > dim/kernelSize)
                                maxSplitOverH = dim/kernelSize;
                            if(maxSplitOverH < 1)
                                maxSplitOverH = 1;
                        }

                        vector<size_t> streamsOverK;
                        if(hasStreamOverK)
                            streamsOverK = getMaxStreamOverK(clustering.get<string>(),op);
                        else
                            streamsOverK.push_back(1);

                        vector<size_t> streamsOverC;
                        if (hasStreamOverC)
                            streamsOverC = {1,2,3,4};
                        else
                            streamsOverC.push_back(1);

                        bool enableNestedStreaming = false;
                        auto maxK = streamsOverK.back();
                        auto memK = memorySize(op,clustering,iAS,outputActivationSparsity,weightsSparsity,{1,1,1,maxK,n});
                        auto memoryMaxK = memK.first + memK.second;


                        // If streaming is enabled, but streaming over k or h alone doesn't fit, enable nested streaming
                        if(hasStreamOverK and hasStreamOverH and ((maxSplitOverH == 1) and (memoryMaxK > clusterMemory))){
                            //If we're doing nested streaming, only consider SOK for multicluster, hack for decrease compile time
                            if(totalClusters > 1 and clustering.get<string>() != "SplitOverK"){
                                continue;
                            }
                            enableNestedStreaming = true;
                            //adjust streamsOverK to remove the smallest K possibilities
                            if(streamsOverK.size() > 2){
                                streamsOverK.erase(streamsOverK.begin());
                                streamsOverK.erase(streamsOverK.begin());
                            }
                            // Adjust maxSplitOverH appropriately for nested
                            maxSplitOverH = getMaxStreamOverH(clustering.get<string>(),op, streamsOverK);
                        }

                    // for(unsigned n = 1; n <= maxSplitOverN; n++)
                    // {
                        for(const auto k : streamsOverK)
                        {
                            for(unsigned h = 1; h <= maxSplitOverH; h++)
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
                                    if(clustering.get<string>() == "SplitOverK")
                                        s["inputSparsity"] = false;
                                    else
                                        s["inputSparsity"] = inputActivationSparsity;
                                    s["outputSparsity"] = outputActivationSparsity;
                                    s["weightsSparsity"] = weightsSparsity;
                                    s["spilling"] = spilling;
                                    s["clustering"] = clustering;
                                    s["streaming"] = streamShape;

                                    //Function to prune strategies that will have only infinite edges in or out (or both), improves performance
                                    if(!createStrategyDots and (checkForBadStrategy(op,s) > 0))
                                        continue;

                                    strategyVec.push_back(s);
                                }
                            }
                        }
                    }
                }
                // }
                if(strategyVec.empty())
                    throw LogicError(*this,"No strategies created for layer " + op.getName() + ". Layer possibly unsupported.");
            }

        };

    }

}

static void GraphParameterOptimizationFcn(
    const mv::pass::PassEntry& pass,
    mv::ComputationModel& model,
    mv::TargetDescriptor&, mv::Element& passDesc,
    mv::Element&
)
{
    mv::OpModel om(model);
    mv::graphOptimizer::StrategyManagerKmb strategyManager(om,passDesc);

    strategyManager.updateValuesFromJSON();
    strategyManager.updateDefaultValues();
    strategyManager.readGlobalConfigs();

    strategyManager.graphParameterOptimizations();

    return;
}
