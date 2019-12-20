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

            size_t totalClusters;
            size_t clusterMemoryKb;
            size_t dpuPerCluster;
            int ddrBandwidth;
            int sysClock;
            bool globalEnableStreaming;
            bool globalEnableActivationSparsity;
            bool globalEnableWeightsSparsity;
            bool enableChannelMajorConv;
            double safetyFactor;
            double clusterMemory;
            std::vector<string> failure_causes = {"Unknown", "MemorySize", "Stream+ClusterComp", "SpillHKSwitch", 
            "SOKNotAlign16", "InputNotSpilled", "OutputNotSpilled", "StreamingNotSpilled", "Workload<KernelSOH", "ChannelMjr1", "ChannelMjr2"};


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

            }

            //TODO:: figure out more efficient and cleaner way to handle these....

            vector<Attribute> createTFStrategyPoolFromBool(mv::Op op,string name)
            {
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                {
                    return vector<Attribute>{true,false};
                }
                else
                {
                    return vector<Attribute>{false};
                }
            }

            vector<Attribute> createTStrategyPoolFromBool(mv::Op op,string name)
            {
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                {
                    return vector<Attribute>{true};
                }
                else
                {
                    return vector<Attribute>{true,false};
                }
            }

            bool createStrategyFromBool(mv::Op op, string name){
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                {
                    return true;
                }
                else
                {
                    return false;
                }
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
                    int newOutputSize = newOutputSizes.first;

                    int remainderOutputSize = newOutputSizes.second;
                    if (remainderOutputSize > newOutputSize)
                        newOutputSize = remainderOutputSize;

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

            pair<size_t,size_t> memorySize(mv::Op& op, const Attribute& clustering, bool inputActivationSparsity,
                                            bool outputActivationSparsity, bool weightsSparsity, const Shape& streamConfig, bool prefetch)
            {
                auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };

                //StreamingPool noSplit( {{'W',1},{'H',1},{'C'},{'K'}});
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
                    inputSize = realTensorSize(op.getInputTensor(0),{streamConfig["W"],streamConfig["H"],streamConfig["C"],1}, isCMConv);
                if(op.getOpType() != "Output")
                    outputSize = realTensorSize(op.getOutputTensor(0),{streamConfig["W"],streamConfig["H"],streamConfig["K"],1}, isCMConv);
                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

                if(op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
                {
                    size_t alignedFullChannels = mv::round_up(op.getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION], 16);
                    size_t alignedSplittedChannels = mv::round_up(alignedFullChannels/streamConfig["K"], 16);
                    weightTableSize = 4 * alignedSplittedChannels ;
                    if (op.getOpType() == "Conv")
                        weightSize += realTensorSize(op.getInputTensor(1),{1,1,streamConfig["C"],streamConfig["K"]}, isCMConv);
                    else
                        weightSize += realTensorSize(op.getInputTensor(1),{1,1,streamConfig["C"],1}, isCMConv);
                }
                else if(op.getOpType() == "MaxPool")
                {
                    weightTableSize = 0;
                    weightSize = 0;
                }
                else if(op.getOpType() == "Eltwise" && !software)
                {
                    weightTableSize = 0;
                    weightSize = 0;
                    inputSize += realTensorSize(op.getInputTensor(1),{streamConfig["W"],streamConfig["H"],streamConfig["C"],1}, isCMConv);
                }

                //Additional memory footprint for sparsity
                if(inputActivationSparsity){
                    //w*h*c, 1 bit per byte of tensor.
                    auto sparseInputSize = (op.getInputTensor(0)->getShape()[0] * op.getInputTensor(0)->getShape()[1]* op.getInputTensor(0)->getShape()[2]) / 8;
                    //storage element
                    sparseInputSize += (op.getInputTensor(0)->getShape()[0] * op.getInputTensor(0)->getShape()[1]);
                    sparseInputSize = mv::round_up(sparseInputSize, 16);
                    inputSize += sparseInputSize;
                }
                if(outputActivationSparsity){
                    //w*h*c, 1 bit per byte of tensor.
                    auto sparseOutputSize = (op.getOutputTensor(0)->getShape()[0] * op.getOutputTensor(0)->getShape()[1]* op.getOutputTensor(0)->getShape()[2]) / 8;
                    //storage element
                    sparseOutputSize += (op.getOutputTensor(0)->getShape()[0] * op.getOutputTensor(0)->getShape()[1]);
                    sparseOutputSize = mv::round_up(sparseOutputSize, 16);
                    outputSize += sparseOutputSize;
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
                    auto memH = memorySize(op,clustering,true,true,true,{1,1,1,k},false);
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
                            {
                               // for(auto iter = streamsOverK.begin(); iter != streamsOverK.end(); )
                                 //   if(*iter < k) iter = streamsOverK.erase(iter);
                                return splitsToFit;
                            }else
                            {
                                return dim/kernelSize; // return this and try to mix with high values of k for viable strategy
                            }
                        }
                        else //normal case, return just enough splits to fit
                        {
                           // for(auto iter = streamsOverK.begin(); iter != streamsOverK.end(); )
                             //   if(*iter < k) iter = streamsOverK.erase(iter);
                            return splitsToFit;
                        }
                    }
                }
                throw LogicError(*this,"Unable to generate nested streaming strategies for layer " + op.getName());
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
                if (outputShape.ndims() != streaming.ndims())
                    outputShape = outputShape.augment(outputShape, streaming.ndims());
                Shape dpuOutShape = ( outputShape / streaming ) / isiSplit;
                Shape contextsInOp = dpuOutShape / contexts;
                unsigned numContextsInOp = contextsInOp.totalSize();

                if(numContextsInOp == 0)
                    throw LogicError(*this,"error in contexts");

                unsigned contextsPerDpu = (unsigned)ceil( (double)numContextsInOp / (double)dpuPerCluster);

                return contextsPerDpu * streaming.totalSize() * baseKernelCost;
            }

            //check if strategy+streaming+tensorSize is incompatible
            bool checkStreamClusterComp(Op& op,StrategySet& strategySet)
            {
                auto clustering = strategySet["clustering"].get<string>();
                auto s  = strategySet["streaming"].get<Shape>();

                auto one_shape = Shape({1,1,1,1});
                //currently we check activations.
                //for Eltwise we will assume that the 2 inputs are of equal size
                //TODO:: check only for DPU tasks
                if(op.getOpType() == "Input" or
                op.getOpType() == "Output")
                    return false;

                //stream over K is the C dim for the OutputTensor
                //steram over C is the C dim for the InputTensor
                auto outStreaming = mv::Shape({s["W"],s["H"],s["K"],1});
                auto inStreaming  = mv::Shape({s["W"],s["H"],s["C"],1});

                auto inTensor = op.getInputTensor(0);
                auto outTensor = op.getOutputTensor(0);

                //this will assume that the first N streams will have the max shape, and the subsequent will have
                //whatever is remained
                auto outTensor_shape = outTensor->getShape();
                if (outTensor_shape.ndims() != outStreaming.ndims())
                    outTensor_shape = outTensor_shape.augment(outTensor_shape, outStreaming.ndims());
                auto streamedShape = outTensor_shape / outStreaming;
                auto remainderShape = outTensor_shape - ((outStreaming - one_shape) * streamedShape);

                //todo:: check if needed for inTensor too
                if( clustering == "SplitOverH" and
                        ((streamedShape["H"] % totalClusters) or
                        (remainderShape["H"] % totalClusters)))
                    return true;

                if( clustering == "SplitOverK" and
                        ((streamedShape["C"] % totalClusters) or
                        (remainderShape["C"] % totalClusters)))
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
                        if (input_gates == 0 || (input_gates == 1 && op.getOpType() == "Eltwise"))
                        {
                            if (op.getInputTensor(input_gates)->getShape()[mv::IO_WIDTH_DIMENSION] > 8192 ||
                                op.getInputTensor(input_gates)->getShape()[mv::IO_HEIGHT_DIMENSION] > 8192 ||
                                op.getInputTensor(input_gates)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192 ||
                                op.getOutputTensor(input_gates)->getShape()[mv::IO_WIDTH_DIMENSION] > 8192 ||
                                op.getOutputTensor(input_gates)->getShape()[mv::IO_HEIGHT_DIMENSION] > 8192 ||
                                op.getOutputTensor(input_gates)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192 )
                            {
                                if (input_gates == 0)
                                {
                                    executableInHW = 1;
                                    if (op.getOpType() != "Eltwise")
                                    {
                                        if (op.getInputTensor(input_gates)->getShape()[mv::IO_WIDTH_DIMENSION]
                                                == op.getInputTensor(input_gates)->getShape()[mv::KERNEL_WIDTH] &&
                                                    op.getInputTensor(1)->getShape()[mv::IO_HEIGHT_DIMENSION]
                                                     == op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT])
                                            executableInHW = 2;
                                        else if (op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] <
                                                 op.getInputTensor(1)->getShape()[mv::IO_HEIGHT_DIMENSION])
                                            executableInHW = 3;
                                        else if (op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] <
                                                 op.getInputTensor(1)->getShape()[mv::IO_WIDTH_DIMENSION])
                                            executableInHW = 4;
                                    }
                                }
                            }
                        }
                        else
                        //Note: all the ops have maximum a second input (weights) at G.O stage
                        {
                            if (op.getInputTensor(input_gates)->getShape()[mv::KERNEL_WIDTH] > 11 ||
                                op.getInputTensor(input_gates)->getShape()[mv::KERNEL_HEIGHT] > 11 ||
                                op.getInputTensor(input_gates)->getShape()[mv::KERNEL_INPUT_CHANNELS] > 8192 ||
                                op.getInputTensor(input_gates)->getShape()[mv::KERNEL_OUTPUT_CHANNELS] > 8192)
                                executableInHW = 5;
                            auto stride_array = op.getAttrs().at("stride").get<std::array<unsigned short, 2>>();
                            if (stride_array[0] > 8 || stride_array[1] > 8)
                                executableInHW = 6;
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
                    auto fit = memorySize(op,clustering,false, false,weightsSparsity,streamShape,false);
                   // std::cout << op.getName() << ": [" <<clustering << "][" <<streamShape.toString()<<"]    " << fit.first << " + " << fit.second << " = " << fit.first + fit.second << std::endl;
                    if(fit.first + fit.second > clusterMemory)
                        return 1;
                }

                // if(checkStreamClusterComp(op, strategy))
                //     return 2;

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

                return 0; //good strategy
            }

            double transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {

                //TODO: expose these conditionals more cleanly
                auto INF = inf_;

                auto parentClustering = parent["clustering"].get<string>();
                auto childClustering = child["clustering"].get<string>();
//                int8_t success = checkHWUnsupportedOp(parentOp);
//                if (success != 0)
//                {
//                    log(mv::Logger::MessageType::Warning, "The limitation of the tensor dimension 8192 might be hitted with the \
//                        operation " + parentOp.getName());
//                }
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
                    //NOTE: If the child layer is streamed over H the parent/input tensors needs to be in DDR
                    if ((child["streaming"].get<Shape>()["H"] * child["streaming"].get<Shape>()["W"]) > 1)
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString() 
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by stream after not spilling");
                            return INF;
                    }
                    //NOTE: Temporary Hack for InceptionV3...General solution change rectHeuristic
                    if (parentClustering == "SplitOverH")
                    {
                        if (childClustering == "SplitOverH" &&
                                childOp.getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] == 73)
                            return INF;
                    }
                }

                if (parentOp.getOpType() == "DepthwiseConv")
                {
                    if ((parentOp.getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192)
                            && (parent["streaming"].get<Shape>()["K"] == 1))
                        return INF;
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
                        return INF;
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

                //Channel major conv cannot have sparsity
                if( (childOp.getOpType() == "Conv") and  (childOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16))
                    childInputSparsity = false;

                if(childInputSparsity == false)
                    parentOutputSparsity = false;

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
                                            false);

                    auto childMem = memorySize(childOp,
                                            child["clustering"],
                                            childInputSparsity,
                                            false,
                                            child["weightsSparsity"].get<bool>(),
                                            child["streaming"].get<Shape>(),
                                            false);


                    if( ((childOp.getOpType() != "Output") and (childMem.first + childMem.second) > clusterMemory) or
                            ((parentOp.getOpType() != "Input") and (parentMem.first + parentMem.second) > clusterMemory))
                    {
                            log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString() 
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by sparsityMemorySize");
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

                //TODO:: do the prefetching
                double extra_stream_decay = 1.5; //TODO: expose in config
                if(parentOp.getOpType() == "Conv")
                {
                    auto streamOverK = parent["streaming"].get<Shape>()["K"];
                    auto WSize = parentOp.getInputTensor(1)->getShape().totalSize();
                    if( streamOverK == 1)
                        execTime1 += WSize / ddrBandwidth;
                    else if( streamOverK == 2)
                        execTime1 += (WSize / 2) / ddrBandwidth;
                    else if( streamOverK > 2)
                        execTime1 += ((WSize / streamOverK) / ddrBandwidth) * (extra_stream_decay * streamOverK);
                }

                if(childOp.getOpType() == "Conv")
                {
                    auto streamOverK = child["streaming"].get<Shape>()["K"];
                    auto WSize = childOp.getInputTensor(1)->getShape().totalSize();
                    if( streamOverK == 1)
                        execTime2 += (double)WSize / (double)ddrBandwidth;
                    else if( streamOverK == 2)
                        execTime2 += ((double)WSize / 2) / (double)ddrBandwidth;
                    else if( streamOverK > 2)
                        execTime2 += (((double)WSize / (double)streamOverK) / (double)ddrBandwidth) * (extra_stream_decay*streamOverK);
                }

                auto parentStreamOverH = parent["streaming"].get<Shape>()["H"];
                if(parentStreamOverH > 1)
                {
                    //assuming here that if have streaming, then inOut is spilled. There is condition above to check this
                    // this is just current "make it work fast" assumption. Will be replaced with proper BW_to_compute calucation
                    auto iSize = parentOp.getInputTensor(0)->getShape().totalSize();
                    auto oSize = parentOp.getOutputTensor(0)->getShape().totalSize();

                    execTime1 += (((double)(iSize + oSize) / (double)parentStreamOverH) / (double)ddrBandwidth) * (extra_stream_decay * parentStreamOverH);
                }
                auto childStreamOverH = child["streaming"].get<Shape>()["H"];
                if(childStreamOverH > 1)
                {
                    auto iSize = childOp.getInputTensor(0)->getShape().totalSize();
                    auto oSize = childOp.getOutputTensor(0)->getShape().totalSize();

                    execTime2 += (((double)(iSize + oSize) / (double)childStreamOverH) / (double)ddrBandwidth)  * (extra_stream_decay * childStreamOverH);
                }

                //TODO do a proper sparsity calculation here depending on zeros in weights
                if(parent["weightsSparsity"].get<bool>()){
                    double factor = estimateSparsityPerformanceBoost(parentOp);
                    execTime1 = execTime1 * factor;
                }
                if(child["weightsSparsity"].get<bool>()){
                    double factor = estimateSparsityPerformanceBoost(childOp);
                    execTime2 = execTime2 * factor;
                }

                if(parentClustering == "SplitOverHOverlapped")
                    execTime1 = execTime1 - 1;

                return execTime1 + execTime2;
            }

            double estimateSparsityPerformanceBoost(mv::Op op){
                auto weights = op.getInputTensor(1);
                //TODO put calculation here for sparsity percentage and calculate expected performance gain
                //Return x>1 if performance loss, 0 < x < 1 for performance gain. Lower number, faster perf
                return 0.85; //dummy value, assume weights sparsity always 15% faster
            }

            void generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& strategyVec)
            {
                globalEnableStreaming = globalStrategies_["enableStreaming"].get<bool>();
                globalEnableActivationSparsity = globalStrategies_["enableActivationSparsity"].get<bool>();
                globalEnableWeightsSparsity = globalStrategies_["enableWeightsSparsity"].get<bool>();

                auto findStrategy = [](vector<Attribute>& vec,const string& str) ->bool { for(const auto elem : vec) if(str==elem.get<string>()) return true; return false;};

                vector<Attribute> weightsSparsityPool;
                if(globalEnableWeightsSparsity){
                    weightsSparsityPool = createTFStrategyPoolFromBool(op,"weightsSparsity");
                } else {
                    weightsSparsityPool.push_back({false});
                }

                bool inputActivationSparsity, outputActivationSparsity;
                if(globalEnableActivationSparsity){
                    inputActivationSparsity = createStrategyFromBool(op,"inputActivationSparsity");
                    outputActivationSparsity = createStrategyFromBool(op,"outputActivationSparsity");
                }else{
                    inputActivationSparsity = false;
                    outputActivationSparsity = false;
                }

                vector<Attribute> doubleBufferPool = createTFStrategyPoolFromBool(op,"doubleBuffering");
                vector<Attribute> spillingPool = createTStrategyPoolFromBool(op,"forceSpilling");

                vector<Attribute> clusteringStrategyPool;

                if(totalClusters == 1)
                    clusteringStrategyPool.push_back(string("Clustering"));
                else if (totalClusters >1)
                    clusteringStrategyPool= createStrategyPoolFromStrategySet(op,"clusteringStrategies");
                else
                {
                    throw LogicError(*this, "Graph Optimizer unable to determine number of clusters");
                }
                vector<Attribute> streamingStrategyPool = createStrategyPoolFromStrategySet(op,"streamingStrategies");

                //TODO:: write better codew for this
                bool hasStreamOverK = findStrategy(streamingStrategyPool,"StreamOverK");
                bool hasStreamOverW = findStrategy(streamingStrategyPool,"StreamOverW");
                bool hasStreamOverH = findStrategy(streamingStrategyPool,"StreamOverH");
                if(globalEnableStreaming == false)
                {
                    hasStreamOverH = false;
                    hasStreamOverW = false;
                    hasStreamOverK = false;
                }

                //TODO:: replace nested loops with clean cartesian product function
                for( const auto weightsSparsity : weightsSparsityPool) {
                    for( const auto spilling : spillingPool) {
                        for( const auto clustering : clusteringStrategyPool){
//std::cout << "Enter clustering loop " << std::endl;                            
// Determine streaming options
                            // 1. Determine if streams over H are possible
                            // 2. Determine if streams over K are possible
                            // 3. If no streams over H or K will fit, enable nested streaming
                            // 4. Nested loops over generated streaming options to produce all strategy options
                            unsigned maxSplitOverH = 1;
                            if(hasStreamOverH)
                            {
                                auto memH = memorySize(op,clustering,inputActivationSparsity,outputActivationSparsity,weightsSparsity.get<bool>(),{1,1,1,1},false);
                                auto activationsSize = memH.first;
                                auto weightsSize = memH.second;
                                double availableMemory = (double) clusterMemory - (double) weightsSize;
                                if (availableMemory < 0) // Weights don't fit, can't stream over H
                                    maxSplitOverH = 1;
                                else {
                                    unsigned splitsToFit = ceil((double)activationsSize/availableMemory);
                                    if (splitsToFit < 1)
                                        maxSplitOverH = 1;
                                    else
                                        maxSplitOverH = splitsToFit;
                                }
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
                            if (op.getOpType() == "DepthwiseConv")
                            {
                                streamsOverK = {1,2};
                            }

                            bool enableNestedStreaming = false;
                            auto maxK = streamsOverK.back();
                            auto memK = memorySize(op,clustering,inputActivationSparsity,outputActivationSparsity,weightsSparsity.get<bool>(),{1,1,1,maxK},false);
                            auto memoryMaxK = memK.first + memK.second;


                            // If streaming is enabled, but streaming over k or h alone doesn't fit, enable nested streaming
                            if(hasStreamOverK and hasStreamOverH and ((maxSplitOverH == 1) and (memoryMaxK > clusterMemory))){
                                //If we're doing nested streaming, only consider SOK for multicluster, hack for decrease compile time
				if(totalClusters > 1 and clustering.get<string>() != "SplitOverK"){
					//std::cout << "Skipping strategy for " << clustering.toString() << ", spilling " << spilling.toString() << std::endl;
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

			//std::cout << "Considering strategy for " << clustering.toString() << ", spilling " << spilling.toString() << ", maxH " << maxSplitOverH << ", numK " << streamsOverK.size() << std::endl;

                            for(const auto k : streamsOverK)
                            {
                                for(unsigned h = 1; h <= maxSplitOverH; h++)
                                {
                                    if( !enableNestedStreaming and ((h>1) and (k>1))) // Skip nested streams unless necessary
                                        continue;
                                    if( enableNestedStreaming and ((h==1) or (k==1))) // If need nested streams, ignore non-nested
                                        continue;
                                    if( ((h*k) > 1) and (spilling.get<bool>() == false))
                                        continue;
                                    // if ((spilling.get<bool>() == true) and (h*k == 1)
                                    //     and op.getOpType() != "Input" and op.getOpType() != "Output"
                                    //     and op.getOpType() != "Concat" and (op.hasTypeTrait("optimizable")))
                                    //     continue;

                                    Shape streamShape({1,h,1,k});//Stream over W and C are 1 for now . TODO: implement stream W/C

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
                                    //s["doubleBuffering"] = doubleBuffering;
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
		//std::cout << "Generated " << strategyVec.size() << " stratgies for layer " << op.getName() << std::endl;
                if(strategyVec.empty())
                    throw LogicError(*this,"No strategies added to the graph for layer " + op.getName());
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
