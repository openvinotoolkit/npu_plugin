#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"


static void GraphParameterOptimizationFcn(const mv::pass::PassEntry& pass,
                                    mv::ComputationModel& model,
                                    mv::TargetDescriptor&, mv::Element& passDesc,
                                    mv::Element&);

namespace mv {
namespace pass {

    MV_REGISTER_PASS(GraphParameterOptimization)
            .setFunc(GraphParameterOptimizationFcn)
            .setDescription("Analyzes graph, and tries to come up with optimal schedule");
}
}

namespace mv {
namespace graphOptimizer  {

class StrategyManagerKeembay : public StrategyManager
{
public:
    StrategyManagerKeembay(OpModel& model,mv::Element& passDesc) :
        StrategyManager(model,passDesc),unique_ctr(0)
    {
    }

    //TODO:: figure out cleaner and faster way to represent this...
    // the pass will do tremendous amounts of ops accessing this map, and constant string hashing will be slow.
    //
//    class StreamingPool
//    {
//        std::map<char,int> streamingPool_;
//
//        //TODO: inherit dims somehow from Shape
//        StreamingPool()
//        {
//            streamingPool_['H'] = 0;
//            streamingPool_['W'] = 0;
//            streamingPool_['K'] = 0;
//            streamingPool_['C'] = 0;
//        }
//
//        StreamingPool(initializer_list<pair<char,size_t>> initList)
//        {
//            StreamingPool();
//            for( auto elem : initList)
//            {
//                streamingPool_[elem.first] = elem.second;
//            }
//        }
//
////        StreamingPool(initializer_list<size_t> initList)
////        {
////
////        }
//
//        int& operator[](char dim)
//        {
//            auto elem = streamingPool_.find(dim);
//            if(elem == streamingPool_.end())
//            {
//                //TODO::Throw error for unspupported DIM
//            }
//            return elem->second;
//        }
//
//    };

    size_t totalClusters;
    size_t clusterMemoryKb;
    size_t dpuPerCluster;
    int ddrBandwidth;
    int sysClock;
    bool globalEnableStreaming;
    double safetyFactor;
    double clusterMemory;

    atomic<int> unique_ctr;

    void readGlobalConfigs()
    {
        totalClusters = globalConfig_["totalClusters"].get<int>();
        clusterMemoryKb = globalConfig_["clusterMemory"].get<int>();
        dpuPerCluster = globalConfig_["dpuPerCluster"].get<int>();
        ddrBandwidth = globalConfig_["ddrBandwidth"].get<int>();
        sysClock = globalConfig_["systemClockMhz"].get<int>();
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

    size_t realTensorSize(const mv::Data::TensorIterator tensorToSize,const Shape& streamingPool)
    {
        auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };

        size_t streamDivisor = 1;
        for( size_t dim = 0; dim <  streamingPool.ndims(); ++dim)
        {
            streamDivisor=streamDivisor*streamingPool[dim];
        }

        return tensorToSize->computeTotalSize()/streamDivisor;

    }

/*
    size_t tensorSize(const Shape& shape,const Shape& streamingPool)
    {
        auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
        Shape splittedShape(shape.ndims());

//        auto vec = vector<unsigned>(shape);
//        transform(vec.begin(), vec.end(),splittedShape.begin()[](unsigned dim) -> unsigned { return dim}
        for( size_t dim = 0; dim <  shape.ndims(); ++dim)
        {
            splittedShape[dim] = div(shape[dim],streamingPool[dim]);
        }

        return splittedShape.totalSize();

    }
*/
    pair<size_t,size_t> memorySize(mv::Op& op,const Attribute& clustering,const Attribute& sparsity,const Shape& streamConfig, bool prefetch)
    {
        auto inputTensors = op.getInputTensor();
        auto outputTensors = op.getOutputTensor();
        auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };

//        StreamingPool noSplit( {{'W',1},{'H',1},{'C'},{'K'}});
        size_t inputSize = 0;
        size_t outputSize = 0;
        size_t weightSize = 0;
        size_t weightTableSize = 0;
        size_t totalMemSize = 0;

        size_t totalWeightsSize = 0;
        size_t totalActivationSize = 0;

        if(op.getOpType() != "Input")
//            inputSize = tensorSize(op.getInputTensor(0)->getShape(),{streamConfig["W"],streamConfig["H"],streamConfig["C"],1});
            inputSize = realTensorSize(op.getInputTensor(0),{streamConfig["W"],streamConfig["H"],streamConfig["C"],1});
        if(op.getOpType() != "Output")
//            outputSize = tensorSize(op.getOutputTensor(0)->getShape(),{streamConfig["W"],streamConfig["H"],1,1});
            outputSize = realTensorSize(op.getOutputTensor(0),{streamConfig["W"],streamConfig["H"],1,1});

        if(op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv" || op.getOpType() == "DepthWiseConv")
        {
            //weightTableSize = 16*((op.getInputTensor(1)->getShape()["K"] + (streamConfig["K"]*streamConfig["H"]) - 1) / (streamConfig["K"]* streamConfig["H"])) ;
            weightTableSize = 16*((op.getInputTensor(1)->getShape()["K"] + (streamConfig["K"]) - 1) / (streamConfig["K"])) ;
//            weightSize += tensorSize(op.getInputTensor(1)->getShape(),{1,1,streamConfig["C"],streamConfig["K"]});
            weightSize += realTensorSize(op.getInputTensor(1),{1,1,streamConfig["C"],streamConfig["K"]});
        } else if(op.getOpType() == "MaxPool")
        {
            weightTableSize = 0;
            weightSize = 0;
        } else if(op.getOpType() == "Add" || op.getOpType() == "Multiply")
        {
            weightTableSize = 0;
            weightSize = 0;
//            inputSize += tensorSize(op.getInputTensor(1)->getShape(),{streamConfig["W"],streamConfig["H"],streamConfig["C"],1});
            inputSize += realTensorSize(op.getInputTensor(1),{streamConfig["W"],streamConfig["H"],streamConfig["C"],1});
        }

        weightSize += weightTableSize;

        auto clusterStrategy = clustering.get<string>();
        auto sparseStrategy = sparsity.get<bool>();

        if(clusterStrategy == "Clustering")
        {
            totalActivationSize = inputSize + outputSize;
            totalWeightsSize = weightSize;
            totalMemSize = inputSize + outputSize + weightSize;
        }
        else if(clusterStrategy == "SplitOverH")
        {
            totalActivationSize = div(inputSize,totalClusters) + div(outputSize,totalClusters);
            totalWeightsSize = weightSize;
            totalMemSize = div(inputSize,totalClusters) + div(outputSize,totalClusters) + weightSize;
        }
        else if(clusterStrategy == "SplitOverK")
        {
            totalActivationSize = inputSize + outputSize;
            totalWeightsSize =  div(weightSize,totalClusters);
            totalMemSize = inputSize + outputSize + div(weightSize,totalClusters);
        }
        else if(clusterStrategy == "HKSwitch")
        {
            totalActivationSize = div(inputSize,totalClusters) + outputSize;
            totalWeightsSize = weightSize;
            totalMemSize = div(inputSize,totalClusters) + outputSize + weightSize;
        }//TODO: proper calculation here
        else if(clusterStrategy == "SplitOverHOverlapped")
        {
            totalActivationSize = div(inputSize,totalClusters) + div(outputSize,totalClusters);
            totalWeightsSize = weightSize;
            totalMemSize = div(inputSize,totalClusters) + div(outputSize,totalClusters) + weightSize;
        }
        else
        {
            //todo raise rerrr
        }
        return pair<size_t,size_t>(totalActivationSize,totalWeightsSize);
    }

    vector<size_t> getMaxStreamOverK(const string& clustering,mv::Op& op)
    {
        auto opType = op.getOpType();

        if( opType == "Input" or opType == "Output")
            return vector<size_t>(0);

        auto outputShape = op.getOutputTensor(0)->getShape();

        vector<size_t> splits;
        size_t clusterOutChannelSize = outputShape["C"];
        vector<size_t> clusterChannelSizes(totalClusters);
        //auto roundUp = [](unsigned in,unsigned val) -> unsigned {return (in & val)+val;};
        auto roundUpToStep = [](unsigned numberToRound,unsigned step) -> unsigned {return (((numberToRound+(step-1))/step)*step);};

        if(clustering == "SplitOverK")
            clusterOutChannelSize = clusterOutChannelSize / totalClusters;

        //TODO::why is this needed?
        if((clusterOutChannelSize%16) and (clustering == "SplitOverK"))
        {
            //clusterOutChannelSize = (totalClusters -1) + roundUp(clusterOutChannelSize,16);
            clusterOutChannelSize = (totalClusters -1) + roundUpToStep(clusterOutChannelSize,16);
            fill_n(clusterChannelSizes.begin(),totalClusters-1,clusterOutChannelSize);
            clusterChannelSizes[totalClusters-1] =outputShape["C"] - (totalClusters-1)*clusterOutChannelSize;
        }
        else
        {
            fill_n(clusterChannelSizes.begin(),totalClusters,clusterOutChannelSize);
        }

        size_t maxSplits = 1;

        if(globalEnableStreaming)
        	maxSplits = (clusterOutChannelSize/2);

        std::cout << "maxStreamingSplits for " << op.getName() << " is "<< maxSplits << std::endl;

        splits.push_back(1);

        //for(unsigned split = 1; split <= maxSplits; split++)
        for(unsigned split = 2; split <= maxSplits; split=split+2)
        {
            bool validSplit = true;

            for(auto clusterSize : clusterChannelSizes)
            {
                if( ((clusterSize / split) <16) or ((clusterSize%split) !=0) or ((clusterSize/split)%16 != 0))
                    validSplit = false;
            }
            if(!validSplit)
                continue;

            splits.push_back(split);
        }

        std::cout << "Streaming options for " << op.getName() << std::endl;
        for (auto s:splits)
        {
            std::cout << "      " << s << std::endl;
        }

        return splits;
    }

    size_t getMaxSplitsOverSpatial(const string& clustering,const Shape& shape,char dim)
    {

        return 0;
    }

//    int simulateDpuExecutionTime(Op& op,StrategySet strategy)
//    {
//        auto opType = op.getOpType();
//        auto clustering = strategy["clustering"].get<string>();
//
//        vector<Shape> contexts;
//        Shape clusterSplit;
//
//        if( (opType == "Pooling") or (opType == "DepthWiseConv")) //TODO:: OR CHMAJOR CONV
//        {
//            contexts.push_back({16,1,16,1});
//        }
//        else
//        {
//            contexts.push_back({4,4,16,1});
//            contexts.push_back({16,1,16,1});
//        }
//
//        if( (clustering == "SplitOverH") or
//                (clustering == "SplitOverHOverlapped") or
//                (clustering == "HKSwitch"))
//            clusterSplit = {1,totalClusters,1,1};
//        else if(clustering == "SplitOverK")
//            clusterSplit = {1,1,totalClusters,1};
//        else
//            clusterSplit = {1,1,1,1};
//
//    }

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

        Shape contexts,isiSplit;

        // TODO:: check for CHMAJOR CONV
        if( (opType == "MaxPool") or (opType == "DepthWiseConv") or (opType == "DepthwiseConv"))
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
        if ((opType == "Add") or (opType == "Concat"))
        {
              baseKernelCost = 1;
        }
        else if (opType == "MaxPool")
        {
            auto kernel = op.get<array<unsigned short,2>>("kSize");
            baseKernelCost = kernel[0] * kernel[1];
        }
        else if ((opType == "DepthWiseConv") or (opType == "DepthwiseConv") or
                 (opType == "Conv"))
        {
            auto weightsShape = op.getInputTensor(1)->getShape();
            baseKernelCost = weightsShape[KERNEL_WIDTH] * weightsShape[KERNEL_HEIGHT];
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
        auto streamedShape = outTensor->getShape() / outStreaming;
        auto remainderShape = outTensor->getShape() - ((outStreaming - one_shape) * streamedShape);

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

    double transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
    {

        //TODO: expose these conditionals more cleanly
        auto INF = inf_;

        auto parentClustering = parent["clustering"].get<string>();
        auto childClustering = child["clustering"].get<string>();


        if((parentClustering == "HKSwitch" or
                parentClustering == "SplitOverK") and
                (childClustering == "SplitOverH")){
                     return INF;
                }

        //HK Switch requires previous layer to be SOH
        if((not (parentClustering == "SplitOverH")) and
                childClustering == "HKSwitch"){
                     return INF;
                }

        //HK Switch requires next layer to be SOK
        if( parentClustering == "HKSwitch" and
                (not (childClustering == "SplitOverK"))){
            return INF;
                }

        //Cannot pass directly from SoH to SoK
        if( parentClustering == "SplitOverH" and
                childClustering == "SplitOverK"){
                                return INF;
                }

        //cannot pass directly from SoK to SoH
        if( parentClustering == "SplitOverK" and
                childClustering == "SplitOverH"){
                     return INF;
                }
           

        if(checkStreamClusterComp(parentOp,parent)){
            return INF;
        }
        
        if(checkStreamClusterComp(childOp,child)){
            return INF;
        }


        //if child is spilled, HKSwitch makes no sense
        if( (child["spilling"].get<bool>() == true ) and
                (childClustering == "HKSwitch")){
                       return INF;
                }
           

        //TODO: Only the input can be SOH-Overlapped

        //SOH-Overlapped can only go to SOH layers
        if( parentClustering == "SplitOverHOverlapped" and
                (not (childClustering == "SplitOverH"))){
                       return INF;
                }


        //TODO: SOH channelMajor conv requires SoHOverlapped input
        if( parentClustering == "SplitOverHOverlapped" and
                (not (parentOp.getOpType() == "Input"))){
                       return INF;
                }


        if( childOp.getOpType() == "Conv")
        {
            auto weightsShape = childOp.getInputTensor(1)->getShape();
            auto numInChannels = weightsShape[KERNEL_INPUT_CHANNELS];

            if(numInChannels < 16)
            {
                //with this we will assume ChMajorConvolution
                if( (childClustering == "SplitOverH") and
                        (not (parentClustering == "SplitOverHOverlapped"))){
                       return INF;
                        }
            }
        }
        //TODO: disable sparsity for eltwise layer predecessors

        //Input and Output must have Spilled==True
        if( (parentOp.getOpType() == "Input") and
                parent["spilling"].get<bool>() == false){
                       return INF;
                }

        if( (childOp.getOpType() == "Output") and
                child["spilling"].get<bool>() == false){
                       return INF;
                }

        //iIf the layer is streaming over H or W, output of this layer has to be spilled
        if( (parent["spilling"] == false) and
                ((parent["streaming"].get<Shape>()["H"] * parent["streaming"].get<Shape>()["W"]) > 1)){
                       return INF;
                }

        //If the child layer is streaming over H or W output of this layer has to be spilled
        if( (parent["spilling"] == false) and
                ((child["streaming"].get<Shape>()["H"] * child["streaming"].get<Shape>()["W"]) > 1)){
                       return INF;
                }

        // allow only evenly divisible split over H
        //if ((childOp.getOpType()!="Output")&&(childOp.getInputTensor(0)->getShape()[1]%child["streaming"].get<Shape>()["H"] != 0)) return INF; 
        //if ((parentOp.getOpType()!="Input")&&(parentOp.getInputTensor(0)->getShape()[1]%parent["streaming"].get<Shape>()["H"] != 0)) return INF; 

        auto parentMem = memorySize(parentOp,
                                    parentClustering,
                                    parent["sparsity"],
                                    parent["streaming"].get<Shape>(),
                                    false);

        auto childMem = memorySize(childOp,
                                    child["clustering"],
                                    child["sparsity"],
                                    child["streaming"].get<Shape>(),
                                    false);

        if( ((childMem.first + childMem.second) > clusterMemory) or
            ((parentMem.first + parentMem.second) > clusterMemory)){
                       return INF;
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
        return execTime1 + execTime2;
    }

    void generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& strategyVec)
    {
        globalEnableStreaming = globalStrategies_["enableStreaming"].get<bool>();

        auto findStrategy = [](vector<Attribute>& vec,const string& str) ->bool { for(const auto elem : vec) if(str==elem.get<string>()) return true; return false;};
//        auto roundUp = [](unsigned in,unsigned val) -> unsigned {return (in & val)+val;};
 //       auto roundUp = [](unsigned val,unsigned in) -> unsigned {return (in & val)+val;};
        auto roundUpToStep = [](unsigned numberToRound,unsigned step) -> unsigned {return (((numberToRound+(step-1))/step)*step);};

        vector<Attribute> sparsityPool = createTFStrategyPoolFromBool(op,"sparsity");
        vector<Attribute> doubleBufferPool = createTFStrategyPoolFromBool(op,"doubleBuffering");
        vector<Attribute> spillingPool = createTStrategyPoolFromBool(op,"forceSpilling");

        vector<Attribute> clusteringStrategyPool;

        if(totalClusters == 1)
            clusteringStrategyPool.push_back(string("Clustering"));
        else if (totalClusters >1)
            clusteringStrategyPool= createStrategyPoolFromStrategySet(op,"clusteringStrategies");
        else
        {
            //TODO::raise error
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

//        cout<<"Generating strategies for " << op.getName() << endl;
//        auto func = std::bind(createStrategy,op,strategyVec,_1,_2,_3,_4);
//        applyDescartes(func,sparsityPool,doubleBufferPool,spillingPool,clusteringStrategyPool);

        //TODO:: replace nested loops with clean cartesian product function

        for( const auto sparsity : sparsityPool)
        {
//            cout << "\tsparsity :" << sparsity.toString() << endl;
//            for( const auto doubleBuffering : doubleBufferPool)
//            {
//                cout <<"\tdoubleBuff " << doubleBuffering.toString() << endl;
                for( const auto spilling : spillingPool)
                {
//                    cout<<"\tspilling " << spilling.toString() << endl;
                    for( const auto clustering : clusteringStrategyPool)
                    {
                        auto mem = memorySize(op,clustering,sparsity,{1,1,1,1},false);
                        auto activationsSize = mem.first;
                        auto weightsSize = mem.second;

                        unsigned maxSplitOverH;
                        if(!hasStreamOverH)
                        {
                            maxSplitOverH = 1;
                        }
                        else
                        {
                            unsigned splitsToFit = ceil((double)(activationsSize)/(double)(clusterMemory-weightsSize));

                            if (splitsToFit < 1)     // handle case when weightsSize > clusterMemory
                                maxSplitOverH = 1;
                            else
                                maxSplitOverH = splitsToFit;
/*
                            //find smallest split that is even dvisor of H, and >=splitsToFit
                            maxSplitOverH = 1;
                            auto inputHeight=op.getInputTensor(0)->getShape()[1]; 
                        
                            for (auto findMax=splitsToFit; findMax<=inputHeight/2; findMax++)
                            {
                                if ((inputHeight%findMax) == 0)
                                {
                                    maxSplitOverH=findMax;
                                    break;
                                }
                                std::cout << "GraphOptimizer ERROR op: " << op.getName() << " input shape: " << op.getInputTensor(0)->getShape().toString() << std::endl;
                                std::cout << "GraphOptimizer can't find fitting H-split that is even divisor of " << inputHeight << std::endl;
                            }
*/
                        }
                        //cout<<"hasStreamH " << hasStreamOverH << " k " << hasStreamOverK << endl;
                        //cout<<"\tclusterMem " << clusterMemory << " ceil " << ceil((double)activationsSize/(double)clusterMemory) << endl;
                        cout<<op.getName() << " clusterMem " << clusterMemory << " activationSize " << activationsSize <<" weightsSize " << weightsSize<< " maxSplitH " << maxSplitOverH << endl;

                        vector<size_t> streamsOverK;
                        if(hasStreamOverK)
                            streamsOverK = getMaxStreamOverK(clustering.get<string>(),op);
                        else
                            streamsOverK.push_back(1);

                        for(const auto k : streamsOverK)
                        {
                            for(unsigned h=1; h <= maxSplitOverH; h++)
                            {
                                //TODO: these are very fast hacks. Delete after we can allow nested streams and
                                if( (h>1) and (k>1))
                                    continue;
                                if( ((h*k) > 1) and (spilling.get<bool>() == false))
                                    continue;

                                // cout<<"\tStrH: " << h << endl;
                                Shape streamShape({1,h,1,k});//Stream over W and C are 1 for now . TODO: implement stream W/C
                                StrategySet s;
                                s["name"] = op.getName();
                                s["sparsity"] = sparsity;
//                                s["doubleBuffering"] = doubleBuffering;
                                s["spilling"] = spilling;
                                s["clustering"] = clustering;
                                s["streaming"] = streamShape;
                                s["id"] = (unique_ctr++);
                                strategyVec.push_back(s);
                            }
                        }
                    }
                }
//            }
        }
    }
};
}
}

static void GraphParameterOptimizationFcn(const mv::pass::PassEntry& pass,
                                    mv::ComputationModel& model,
                                    mv::TargetDescriptor&, mv::Element& passDesc,
                                    mv::Element&)
{

    mv::OpModel om(model);
    mv::graphOptimizer::StrategyManagerKeembay strategyManager(om,passDesc);

    strategyManager.updateValuesFromJSON();
    strategyManager.updateDefaultValues();
    strategyManager.printStrategy();

    strategyManager.readGlobalConfigs();
//    strategyManager.recursiveDijkstra(om.opBegin());
    strategyManager.graphParameterOptimizations();

    std::cout << "ran the optimizer" << std::endl;
    std::exit(0);
    return;
}

