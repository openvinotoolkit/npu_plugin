#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/graphOptimizations/strategy_utils.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/pass/graphOptimizations/heuristic_strategy.hpp"
#include "include/mcm/pass/graphOptimizations/simple_strategy_manager.hpp"

static void heuristicStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(SimpleStrategyManager)
        .setFunc(heuristicStrategyFcn)
        .setDescription(
            "This pass quickly makes simple strategy choices"
        );

    }
}

void printStrategy(StrategySet s)
{
    std::cout << s["name"].toString() << ", " << s["clustering"].toString() << ", ";
    std::cout << "\""<<s["streaming"].get<mv::Shape>().toString() << "\", ";
    std::cout << std::boolalpha << s["spilling"].get<bool>() << ", "<< s["parentSpilling"].get<bool>() << ", ";
    std::cout << std::boolalpha << s["inputSparsity"].get<bool>() << ", "<< s["outputSparsity"].get<bool>()<<  ", " <<s["weightsSparsity"].get<bool>() <<std::endl;
    std::cout << s["heuristicCost"].get<double>() << "  : " << s["cost"].get<double>() << std::boolalpha << ",  " << s["pipeline"].get<bool>()<< ",  " << s["spillPipeline"].get<bool>()<< std::endl;
}

void heuristicStrategyFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::graphOptimizer::StrategyManagerSimple strategyManager(om,passDesc, td);
    mv::graphOptimizer::HeuristicGraphOptimizer heuristicGO(om,passDesc);

    bool loadStrategiesFromFile = false;
    std::string jsonInFileName = "./output/mcmCompiler_simple_strategy_output.json";
    if (passDesc.hasAttr("loadStrategiesFromFile"))
        loadStrategiesFromFile = passDesc.get<bool>("loadStrategiesFromFile");

    if (passDesc.hasAttr("jsonInFileName"))
        jsonInFileName = passDesc.get<std::string>("jsonInFileName");

    if (loadStrategiesFromFile)
    {
        strategyManager.loadSavedStrategies(jsonInFileName);
        strategyManager.updateTensorLocationAfterLoadStrategies();
    } 
    else 
    {
        strategyManager.updateDefaultValues();
        // STEP 0 - GENERATION. Leverage the SM architecture to generate a set of 
        // potential strategies for every layer (THIS IS BY FAR THE LONGEST TASK IN THIS PASS)
        strategyManager.initLayerStrategySets();
        // STEP 1 - GREEDY. Assign each op with the best strategy, in isolation
        // Setup the variables from the CD and TD as needed
        heuristicGO.init(td);
        // Walk through the topological model, assign every strategy cost, and choose lowest cost
        // strategy as starting point for graph
        heuristicGO.assignMultiClusteringGreedy();

        //STEP 2 - VALIDATION AND SPILLING
        bool clusteringChanged = false;
        bool hardcodedStreams = om.getInput()->hasAttr("hardcoded_streams") && om.getInput()->get<bool>("hardcoded_streams") ? true : false;
        do
        {
            if(!hardcodedStreams) // For now, WA for SSD512 where we want to use the "sad hack"
                heuristicGO.forceConnectedSOH(); // TODO when HK and KH work for all ops, remove this
            heuristicGO.verifySpillStrategies(false);
            clusteringChanged = heuristicGO.addSpillsAtStrategyTransitions();
        } while (clusteringChanged);

        // For any given parent->child pair of ops, the intermediate tensor location
        // must be the same (as in reality, it is one tensor). This ensures we do not
        // have exceeding ops. For ex, if child strategy streams enough to fit in CMX if input
        // were in DDR, but parent leaves that tensor in CMX.
        heuristicGO.verifySpillStrategies(false); // this run doesn't lock clustering strategies

        // STEP 3 - FINALIZE CLUSTERING STRATEGY
        heuristicGO.chooseRollbackOrSpill(); // N.B. after this algo runs, clustering strategies are locked
        heuristicGO.verifySpillStrategies(true);
        heuristicGO.alignAndValidateSpecialOps();

        // STEP 4 - SPARSITY. Where activation sparsity is enabled, try to service with runtime output sparsity from parent(s)
        // N.B. weights sparsity is always per op, and handled at the GREEDY step
        heuristicGO.serviceActivationSparsity();

        // STEP 5 - STREAMING. We can increase streams for performance, or to allow vertical fusion. 
        heuristicGO.increaseWeightsPipelining();

        // Save into the CD for consumption by later passes
        strategyManager.saveLayerStrategies(heuristicGO.getChosenStrategies());
    }
}


namespace mv
{
namespace graphOptimizer
{

HeuristicGraphOptimizer::HeuristicGraphOptimizer(OpModel& model,mv::Element& passDesc) :
        model_(model),passDesc_(passDesc)
{
}

StrategyMap& HeuristicGraphOptimizer::getChosenStrategies()
{
    return bestStrategies_;
}

std::string HeuristicGraphOptimizer::getLogID() const
{
    return "SimpleStrategyManager";
}

void HeuristicGraphOptimizer::init(mv::TargetDescriptor& td)
{
    //TODO these values should be pulled from the target descriptor...
    // these are the THB values...
    target = td.getTarget();
    totalClusters_ = model_.getGlobalConfigParam("Number_of_Clusters").get<int>();
    dpuPerCluster_ = model_.getGlobalConfigParam("Number_of_DPUs").get<int>() / totalClusters_;
    referenceDevice_ = model_.getGlobalConfigParam("referenceDevice").get<std::string>();
    clusterMemory_ = model_.getGlobalConfigParam("cmx").get<int>();

    if(target == mv::Target::ma2490) // based on KMB B0
    {
        //Note: for CMX bandwidth, use measurements from arch team, rather than db spec
        CMX_BANDWIDTH_ = 15;//32 * 1.0; //bytes per cycle times derating factor
        DDR_BANDWIDTH_ = 8 * 0.6; //bytes per cycle times derating factor
        PIPELINE_STAGES = 2;
        SOH_HEURISTIC_MULTIPLIER = 1.37;
        SPILL_COST = 2.0;
    }
    else // THB.. ma3100
    {
        CMX_BANDWIDTH_ = 30;//64 * 1.0; //bytes per cycle times derating factor
        DDR_BANDWIDTH_ = 8 * 0.6; //bytes per cycle times derating factor
        PIPELINE_STAGES = 3;
        SOH_HEURISTIC_MULTIPLIER = 1.2;
    }
    //TODO include params for ma3720
    
    //These latency numbers inferred from KMB db v1.2
    LATENCY_ = 5; // Cycles, attempt to capture cost accessing CMX
    // DDR latency also measured for kmb at ~100 cycles per dma
    LATENCY_DDR_ = 100; // Cycles, attempt to capture cost of setup DMA
}

void HeuristicGraphOptimizer::assignMultiClusteringGreedy()
{
    auto sortedOps = model_.topologicalSort();
    for(auto opIt : sortedOps)
    {
        // Pre-req, every op we care about has been assigned the attribute StrategySet
        if(opIt->hasAttr("StrategySet"))
        {
            auto opStrategiesPtr = opIt->get<std::shared_ptr<std::vector<StrategySet>>>("StrategySet");
            allStrategies_.push_back(opStrategiesPtr);
            auto& opStrategies = *(allStrategies_.back());

            auto bestStrategy = assignStrategyCost(opIt, opStrategies);
            // Save it into our local data structures for processing
            strategy_model_.insert(std::make_pair(opIt->getName(),opStrategies));
            bestStrategies_.insert(std::make_pair(opIt->getName(), bestStrategy));
            // Use this for debugging strategy choice
            // opIt->set<std::string>("heuristicClustering", bestStrategy["clustering"].get<std::string>());
        }
    }
}

void HeuristicGraphOptimizer::forceConnectedSOH()
{
    mv::DataModel dm(model_);
    auto sortedOps = model_.topologicalSort();
    for(auto opIt : sortedOps)
    {
        // Pre-req, every op we care about has been assigned the attribute StrategySet
        if(!opIt->hasAttr("StrategySet")) continue;

        checkMultipleInputOp(opIt);
    }
    // For performance, if we hit an earlier software layer, we allow SOH to continue beyond
    // it. Only an optimizable task will count as breaking the SOH chain
    for(auto opIt : sortedOps)
    {
        // Pre-req, every op we care about has been assigned the attribute StrategySet
        if(!opIt->hasAttr("StrategySet")) continue;
        // Output op has no output tensor
        if(opIt->getOpType() == "Output") continue;

        if(isKCompatible(opIt, true))
        {
            auto sinkLayers = findSinkLayers(dm, opIt->getOutputTensor(0));
            for(auto sinkLayer : sinkLayers)
            {
                if(!sinkLayer->hasAttr("StrategySet")) continue;

                if(!isKCompatible(sinkLayer))
                {
                    findKCompatible(sinkLayer, true, false);
                    abandonSOH(sinkLayer, false);
                }
            }
        }
    }
}

void HeuristicGraphOptimizer::abandonSOH(mv::Data::OpListIterator opIt, bool allowHK)
{
    auto& opStrategies = strategy_model_.at(opIt->getName());

    for(auto& strategy : opStrategies)
    {
        if(strategy["clustering"].get<std::string>() == "SplitOverH" ||
            (!allowHK && strategy["clustering"].get<std::string>() == "HKSwitch"))
            strategy["skip"] = true;
    }
}

double HeuristicGraphOptimizer::computeTime(mv::Data::OpListIterator opIt, StrategySet& strategySet)
{
    auto opType = opIt->getOpType();
    auto software = opIt->hasAttr("softwareExecuted") && opIt->get<bool>("softwareExecuted");
    if( !opIt->isHardwarizable() || software)
        return 0;

    auto inputShape = opIt->getInputTensor(0)->getShape();
    auto outputShape = opIt->getOutputTensor(0)->getShape();
    auto clustering = strategySet["clustering"].get<std::string>();
    auto streaming = strategySet["streaming"].get<Shape>();

    Shape tileConfig,isiSplit;

    if( (opType == "MaxPool") || (opType == "DepthwiseConv"))
    {
        tileConfig = {16,1,16,1};
    }
    else
    {
        tileConfig = {4,4,16,1};
    }
    if( (clustering == "SplitOverH") || (clustering == "SplitOverHOverlapped") || (clustering == "HKSwitch"))
    {
        isiSplit = {1,totalClusters_,1,1};
    }
    else if(clustering == "SplitOverK")
    {
        isiSplit = {1,1,totalClusters_,1};
    }
    else
    {
        isiSplit = {1,1,1,1};
    }

    size_t baseKernelCost;
    if ((opType == "Eltwise" && !(software)) || (opType == "Concat") || opIt->isEltwiseSingleInputTypeOp())
    {
        baseKernelCost = 1;
    }
    else if (opType == "MaxPool")
    {
        auto kernel = opIt->get<std::array<unsigned short,2>>("kSize");
        baseKernelCost = kernel[0] * kernel[1];
    }
    else if ((opType == "DepthwiseConv") || (opType == "Conv"))
    {
        auto weightsShape = opIt->getInputTensor(1)->getShape();
        baseKernelCost = weightsShape[mv::KERNEL_WIDTH] * weightsShape[mv::KERNEL_HEIGHT];
    }
    else if (!(opIt->hasTypeTrait("optimizable")) || software)
    {
        baseKernelCost = 1;
    }
    else
    {
        throw LogicError(*this, "Invalid operation type " + opType);
    }

    bool channelAccum = (opType == "Conv") ? true : false;
    if (channelAccum)
    {
        auto weightsShape = opIt->getInputTensor(1)->getShape();
        baseKernelCost *= weightsShape[KERNEL_INPUT_CHANNELS];
    }

    // Calculate the output shape, given this stream and mc strategy
    Shape streamShape = {1, streaming["H"], streaming["K"], 1};
    //the actual compute
    if (outputShape.ndims() != streamShape.ndims())
        outputShape = outputShape.augment(outputShape, streamShape.ndims());
    Shape finalOutShape = ( outputShape / streamShape ) / isiSplit;

    // Calculate the tiling efficiency
    Shape rdc = finalOutShape / tileConfig;
    unsigned numTiles = rdc.totalSize();

    if(numTiles == 0)
        throw LogicError(*this,"error in contexts");

    // Tile efficiency expresses what percentage of the dpu hardware is actually used by a task given the MPE config
    double tileEff = (double)finalOutShape.totalSize() / ( (double)numTiles * tileConfig.totalSize());

    size_t totalStreams = 1;
    for (size_t i = 0; i < streaming.ndims(); i++)
        totalStreams *= streaming[i];

    // We don't know what workloads will look like across DPUs or how optimal they will
    // be, so here we determine a tiling efficiency across the full output shape and then 
    // equally divide the work across the DPUs. It is an approximation, but we don't have the
    // info to be more precise
    return ((double)((double)totalStreams * finalOutShape.totalSize() * baseKernelCost) / tileEff);
}

double HeuristicGraphOptimizer::dmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet)
{
    auto opType = opIt->getOpType();

    //Each layer calculates the cost to move it's weights (if they exist), and it's activations
    double weightsCycles = 0;
    double outputTime = 0;
    double inputCycles = 0;

    auto streamShape = strategySet["streaming"].get<mv::Shape>();
    auto spilling = strategySet["spilling"].get<bool>();
    auto parentSpilling = strategySet["parentSpilling"].get<bool>();
    auto clustering = strategySet["clustering"].get<std::string>();
    auto isCMConv = false;

    if(opIt->hasAttr("supportsCM") && opIt->get<bool>("supportsCM"))
        isCMConv = true;

    // Each DMA cost is modelled as latency + size*transfer_rate
    if(opType == "Conv" || opType == "DepthwiseConv")
    {
        size_t outChannels = opIt->outputSlots() ? opIt->getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION] : 0;
        size_t alignedFullChannels = mv::round_up(outChannels, 16);
        size_t alignedSplittedChannels = mv::round_up(alignedFullChannels/streamShape["K"], 16);
        if(clustering == "SplitOverK")
            alignedSplittedChannels =  mv::round_up(alignedSplittedChannels/totalClusters_, 16);
        
        size_t weightsSize = 0;
        size_t stream = 1;

        if(opType == "Conv")
        {
            stream = streamShape["K"];
            weightsSize += alignedWeightsSize(opIt->getInputTensor(1),{1,1,1,stream,1}, clustering, totalClusters_);
            if(clustering == "SplitOverK")
                weightsSize *= totalClusters_; // we still need to move the full tensor size, just getting aligned
        }
        else // Depthwise
        {
            stream = streamShape["C"];
            weightsSize += realTensorSize(opIt->getInputTensor(1),{1,1,stream,1,1}, isCMConv);
        }
        std::size_t weightTableSize = 16 * alignedSplittedChannels; //weights table size
        if (streamShape["H"] > 1)
        {
            // streaming over h - shared weights
            std::size_t nestedStreams = stream;
            std::size_t nestedWeightTableStreams = stream * streamShape["H"];
            if (stream > 1)
            {
                // In nested streaming for each slice of H, we cycle through all the slices of K
                nestedStreams *= streamShape["H"];
            }
            weightsCycles = nestedStreams*(LATENCY_DDR_ + ((double)weightsSize / DDR_BANDWIDTH_));
            weightsCycles += nestedWeightTableStreams*(LATENCY_DDR_ + ((double)weightTableSize / DDR_BANDWIDTH_));
        }
        else
        {
            // if not streaming over h - tensors fused
            weightsSize += weightTableSize;
            weightsCycles = stream*(LATENCY_DDR_ + ((double)weightsSize / DDR_BANDWIDTH_));
        }
    }

    // If parent tensor stays in CMX, no further cost. Otherwise, calculate bring the input tensor back into CMX
    if(parentSpilling && opType != "Input")
    {
        auto inputSize = opIt->getInputTensor(0)->computeTotalSize();
        size_t stream = 1;
        if(streamShape["H"] > 1)
            stream = streamShape["H"];
        else if(streamShape["C"] > 1)
            stream = streamShape["C"];
        else if(streamShape["B"] > 1) //batch splits input, won't nested with anything
            stream = streamShape["B"];

        //TODO we could account for overlap in input tiles here, some data gets moved more than once
        inputCycles = (stream*LATENCY_DDR_) + ((double)inputSize/ DDR_BANDWIDTH_);
    }

    // If output tensor stays in CMX, no further dma cost. Otherwise, calculate cost to spill to DDR
    if(spilling)
    {
        //Each stream is written to DDR. Output activation tensor might be streamed over H or K
        size_t outputSize;
        if(opType != "Output")
            outputSize = opIt->getOutputTensor(0)->computeTotalSize();
        else
            outputSize = opIt->getInputTensor(0)->computeTotalSize();

        size_t stream = 1;

        if(streamShape["H"] > 1 && streamShape["K"] > 1) // Nested streaming!
            stream = streamShape["H"] * streamShape["K"];
        else if(streamShape["H"] > 1)
            stream = streamShape["H"];
        else if(streamShape["K"] > 1)
            stream = streamShape["K"];
        else if(streamShape["C"] > 1)
            stream = streamShape["C"];    
        else if(streamShape["B"] > 1)
            stream = streamShape["B"];

        outputTime += (stream*LATENCY_DDR_) + ((double)outputSize / DDR_BANDWIDTH_);
    }
    if(clustering == "SplitOverK" || clustering == "Clustering")
    {
        //This section captures the cost to multicast to all clusters

        size_t outputSize;
        if(opType != "Output")
            outputSize = opIt->getOutputTensor(0)->computeTotalSize();
        else
            outputSize = opIt->getInputTensor(0)->computeTotalSize();

        size_t stream = 1;

        if(streamShape["H"] > 1 && streamShape["K"] > 1) // Nested streaming!
            stream = streamShape["H"] * streamShape["K"];
        else if(streamShape["H"] > 1)
            stream = streamShape["H"];
        else if(streamShape["K"] > 1)
            stream = streamShape["K"];
        else if(streamShape["B"] > 1)
            stream = streamShape["B"];

        outputTime += (totalClusters_ * stream * LATENCY_) + ((double)outputSize / CMX_BANDWIDTH_);
    }

    return (inputCycles + weightsCycles + outputTime);
}

double HeuristicGraphOptimizer::outputDmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet, bool forceSpill = false)
{
    auto opType = opIt->getOpType();
    double outputTime = 0;
    auto streamShape = strategySet["streaming"].get<mv::Shape>();
    auto spilling = strategySet["spilling"].get<bool>();
    auto clustering = strategySet["clustering"].get<std::string>();

    // If output tensor stays in CMX, no further dma cost. Otherwise, calculate cost to spill to DDR
    if(spilling || forceSpill)
    {
        //Each stream is written to DDR. Output activation tensor might be streamed over H or K
        size_t outputSize;
        if(opType != "Output")
            outputSize = opIt->getOutputTensor(0)->computeTotalSize();
        else
            outputSize = opIt->getInputTensor(0)->computeTotalSize();

        size_t stream = 1;

        if(streamShape["H"] > 1 && streamShape["K"] > 1) // Nested streaming!
            stream = streamShape["H"] * streamShape["K"];
        else if(streamShape["H"] > 1)
            stream = streamShape["H"];
        else if(streamShape["K"] > 1)
            stream = streamShape["K"];
        else if(streamShape["B"] > 1)
            stream = streamShape["B"];

        outputTime += stream*(LATENCY_DDR_ + (((double)outputSize/stream) / DDR_BANDWIDTH_));
    }
    if(clustering == "SplitOverK" || clustering == "Clustering")
    {
        size_t outputSize;
        if(opType != "Output")
            outputSize = opIt->getOutputTensor(0)->computeTotalSize();
        else
            outputSize = opIt->getInputTensor(0)->computeTotalSize();

        size_t stream = 1;

        if(streamShape["H"] > 1 && streamShape["K"] > 1) // Nested streaming!
            stream = streamShape["H"] * streamShape["K"];
        else if(streamShape["H"] > 1)
            stream = streamShape["H"];
        else if(streamShape["K"] > 1)
            stream = streamShape["K"];
        else if(streamShape["B"] > 1)
            stream = streamShape["B"];

        outputTime += stream * (((double)outputSize/stream) / CMX_BANDWIDTH_);
    }

    return outputTime;
}

double HeuristicGraphOptimizer::inputDmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet)
{
    auto opType = opIt->getOpType();
    double inputTime = 0;
    auto streamShape = strategySet["streaming"].get<mv::Shape>();
    auto parentSpilling = strategySet["parentSpilling"].get<bool>();
    auto clustering = strategySet["clustering"].get<std::string>();

    // If parent tensor stays in CMX, no further cost. Otherwise, calculate bring the input tensor back into CMX
    if(parentSpilling && opType != "Input")
    {
        auto inputSize = opIt->getInputTensor(0)->computeTotalSize();
        size_t stream = 1;
        if(streamShape["H"] > 1)
            stream = streamShape["H"];
        else if(streamShape["C"] > 1)
            stream = streamShape["C"];
        else if(streamShape["B"] > 1) //batch splits input, won't nested with anything
            stream = streamShape["B"];

        //TODO we could account for overlap in input tiles here, some data gets moved more than once
        inputTime = (stream*LATENCY_DDR_) + ((double)inputSize/ DDR_BANDWIDTH_);
    }

    return inputTime;
}

double HeuristicGraphOptimizer::averageWeightsDmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet)
{
    auto streamShape = strategySet["streaming"].get<mv::Shape>();
    auto clustering = strategySet["clustering"].get<std::string>();
    auto isCMConv = false;

    if(opIt->hasAttr("supportsCM") && opIt->get<bool>("supportsCM"))
        isCMConv = true;

    // Each DMA cost is modelled as latency + size*transfer_rate
    if(opIt->getOpType() == "Conv" || opIt->getOpType() == "DepthwiseConv")
    {
        size_t outChannels = opIt->outputSlots() ? opIt->getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION] : 0;
        size_t alignedFullChannels = mv::round_up(outChannels, 16);
        size_t alignedSplittedChannels = mv::round_up(alignedFullChannels/streamShape["K"], 16);
        if(clustering == "SplitOverK")
            alignedSplittedChannels =  mv::round_up(alignedSplittedChannels/totalClusters_, 16);
        
        size_t weightsSize = 0;
        size_t stream = 1;

        if(opIt->getOpType() == "Conv")
        {
            stream = streamShape["K"];
            weightsSize += alignedWeightsSize(opIt->getInputTensor(1),{1,1,1,stream,1}, clustering, totalClusters_);
            if(clustering == "SplitOverK")
                weightsSize *= totalClusters_; // we still need to move the full tensor size, just getting aligned
        }
        else // Depthwise
        {
            stream = streamShape["C"];
            weightsSize += realTensorSize(opIt->getInputTensor(1),{1,1,stream,1,1}, isCMConv);
        }
        std::size_t weightsCycles = 0UL;
        std::size_t weightTableSize = 16 * alignedSplittedChannels; //weights table size
        if (streamShape["H"] > 1)
        {
            // streaming over h - shared weights
            std::size_t nestedStreams = stream;
            std::size_t nestedWeightTableStreams = stream * streamShape["H"];
            if (stream > 1)
            {
                // In nested streaming for each slice of H, we cycle through all the slices of K
                nestedStreams *= streamShape["H"];
            }
            weightsCycles = nestedStreams*(LATENCY_DDR_ + ((double)weightsSize / DDR_BANDWIDTH_));
            weightsCycles += nestedWeightTableStreams*(LATENCY_DDR_ + ((double)weightTableSize / DDR_BANDWIDTH_));
        }
        else
        {
            // if not streaming over h - tensors fused
            weightsSize += weightTableSize;
            weightsCycles = stream*(LATENCY_DDR_ + ((double)weightsSize / DDR_BANDWIDTH_));
        }
        return weightsCycles;
    }

    return 0;
}

double HeuristicGraphOptimizer::averageOutputDmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet)
{
    auto streamShape = strategySet["streaming"].get<mv::Shape>();
    auto spilling = strategySet["spilling"].get<bool>();
     auto clustering = strategySet["clustering"].get<std::string>();
    // If output tensor stays in CMX, no further dma cost. Otherwise, calculate cost to spill to DDR
    if(spilling)
    {
        //Each stream is written to DDR. Output activation tensor might be streamed over H or K
        size_t outputSize;
        if(opIt->getOpType() != "Output")
            outputSize = opIt->getOutputTensor(0)->computeTotalSize();
        else
            outputSize = opIt->getInputTensor(0)->computeTotalSize();

        size_t stream = 1;

        if(streamShape["H"] > 1 && streamShape["K"] > 1) // Nested streaming!
            stream = streamShape["H"] * streamShape["K"];
        else if(streamShape["H"] > 1)
            stream = streamShape["H"];
        else if(streamShape["K"] > 1)
            stream = streamShape["K"];
        else if(streamShape["B"] > 1)
            stream = streamShape["B"];

        return (LATENCY_DDR_ + (((double)outputSize/stream) / DDR_BANDWIDTH_));
    }
    else if(clustering == "SplitOverK" || clustering == "Clustering")
    {
        size_t outputSize;
        if(opIt->getOpType() != "Output")
            outputSize = opIt->getOutputTensor(0)->computeTotalSize();
        else
            outputSize = opIt->getInputTensor(0)->computeTotalSize();

        size_t stream = 1;

        if(streamShape["H"] > 1 && streamShape["K"] > 1) // Nested streaming!
            stream = streamShape["H"] * streamShape["K"];
        else if(streamShape["H"] > 1)
            stream = streamShape["H"];
        else if(streamShape["K"] > 1)
            stream = streamShape["K"];
        else if(streamShape["B"] > 1)
            stream = streamShape["B"];

        return (((double)outputSize/stream) / CMX_BANDWIDTH_);
    }
    return 0;
}

// Note: This function captures the crux of what the full GO pass did
// In order to build on the backbone of the current Strategy Manager class for creating
// and saving strategies, it is necessary to fully cost each of these things at this point
// Future strategy generation does not need to do this. It would be possible to instead assign each
// strategy item separately from one another
// in a defined order of precedence and achieve the same results in less compilation time
StrategySet HeuristicGraphOptimizer::assignStrategyCost(mv::Data::OpListIterator opIt, 
                                                std::vector<mv::graphOptimizer::StrategyManager::StrategySet>& opStrategies)
{
    mv::DataModel dm(model_);
    StrategySet bestStrat;
    double bestCost = COST_MAX;
    StrategySet greedyStrat;
    double greedyCost = COST_MAX;
    bool hasSOK = false;
    bool isFirstOp = false;
    if(opIt->getOpType() != "Input" && opIt->isHardwarizable())
    {
        auto inputs = opIt->getInputTensor();
        for(auto input : inputs)
        {
            auto inputOp = model_.getSourceOp(input);
            if(inputOp->getOpType() == "Input" || inputOp->getOpType() == "ImplicitInput")
            {
                isFirstOp = true;
                break;
            }
        }
    }
    for(auto& strategy : opStrategies)
    {
        auto parentSpilling = strategy["parentSpilling"].get<bool>();
        auto spilling = strategy["spilling"].get<bool>();
        auto streamShape = strategy["streaming"].get<mv::Shape>();
        auto clustering = strategy["clustering"].get<std::string>();
        if(clustering == "SplitOverK")
            hasSOK = true;
        double cost = 0.0;
        double spillPipelineCost = 0.0;
        if(!strategy["spillPipeline"].get<bool>())
            spillPipelineCost = COST_MAX;
        double heuristicCost = 0.0;
        double computeCycles = 0.0;
        double dmaCycles = 0.0;
        if(strategy["pipeline"].get<bool>() || strategy["spillPipeline"].get<bool>()) // If pipelining weights
        {
            auto kStreams = streamShape["K"];
            auto compPerStream = ((double) computeTime(opIt, strategy) / kStreams) / (double) dpuPerCluster_;
            auto weightDma = averageWeightsDmaTime(opIt, strategy);
            auto outputDma = averageOutputDmaTime(opIt, strategy);
            double pipeline = 0.0;
            if(PIPELINE_STAGES == 3)
            {
                auto dmaMax = std::max(weightDma, outputDma);
                pipeline = std::max(dmaMax, compPerStream);
            }
            else
                pipeline = std::max((weightDma+outputDma), compPerStream);
            //Pipeline means overlap K stream compute with weights read for next K stream, assume input in CMX
            if(strategy["pipeline"].get<bool>())
                cost += weightDma + (kStreams-1)*pipeline + outputDma;

            spillPipelineCost =  (weightDma + (kStreams-1)*pipeline + outputDma) / kStreams;
        }

        if(!strategy["pipeline"].get<bool>())
        {
            computeCycles = computeTime(opIt, strategy) / (double) dpuPerCluster_;
            dmaCycles = dmaTime(opIt, strategy);
            cost = cost + computeCycles +dmaCycles;
        }
        

        // Note: for performance, here we ensure if that MC strategies are preferenced in order
        // SOH, HKSwitch, SOK, Clustering
        auto opType = opIt->getOpType(); 
        if(opType == "Input" ||
            opType == "Output" ||
            opType == "Concat")
            cost *= clusteringStrategyCost.at(clustering);

        if(parentSpilling)
        {
            cost += inputDmaTime(opIt, strategy); //we'll have to read this back in
            // std::cout << "extra parentSpill cost " << inputDmaTime(opIt, strategy);
            if (!strategy["inputMustSpill"].get<bool>())
            {
                cost *= SPILL_COST;
            }
        }
        if(spilling)
        {
            cost += outputDmaTime(opIt, strategy); //we'll have to read this back in
            // std::cout << "  extra spill cost " << outputDmaTime(opIt, strategy) << std::endl;
            if (!strategy["outputMustSpill"].get<bool>())
            {
                cost *= SPILL_COST;
            }
            // Single DMA controller, spilling is even worse!
            if(target == mv::Target::ma2490 && isCMXable(opIt, strategy, false))
            {
                cost *= SPILL_COST;
            }
            if(target == mv::Target::ma2490)
            {
                cost += (outputDmaTime(opIt, strategy) * 100);
            }
        }
        //Start the first dpu task faster...
        if(isFirstOp && streamShape["H"] > 1)
        {
            cost *= 0.95; //this heuristic could/should be moved to the activation streaming pass...
        }
        if(isFirstOp && clustering == "SplitOverH" && opIt->isHardwarizable() && opIt->hasAttr("kSize")
            && (!opIt->hasAttr("supportsCM") || !opIt->get<bool>("supportsCM")))
        {
            cost = COST_MAX; // prevent SOH until SOHOverlapped implemented 
        }
            
        // TODO add sparsity for performance calculation
        if(strategy["inputSparsity"].get<bool>())
        {
            auto sparsityOverhead = opIt->getInputTensor(0)->isFloatingPointType() ?
                    0.0625 : 0.125;
            // Assume in these cases we won't be able to provide runtime sparsity, so pure overhead
            if(clustering == "SplitOverK" || parentSpilling)
                cost += (cost*sparsityOverhead);
            // TODO update for perf, but for now, assume sparse version is worse
            if(!requiresRealActivationSparsity(opIt, strategy))
                cost += (cost*sparsityOverhead);
                // strategy["inputSparsity"] = false;
                

            // The SSM works on the logical assumption that strategy can be improved iteratively. In the case
            // of sparsity workarounds, the ability to service that required sparsity from runtime is actually
            // an inherent part of the cost of the multiclustering strategy, and so must be cost at this stage
            // before final multiclustering stratgies are chosen.
            if(requiresRealActivationSparsity(opIt, strategy) && !canServiceActivationSparsity(opIt, strategy))
                cost += (cost*sparsityOverhead);
        }
        if(strategy["outputSparsity"].get<bool>())
        {
            // Assume its all overhead at this point, if we need runtime sparsity
            // we will enable it later
            auto sparsityOverhead = opIt->getOutputTensor(0)->isFloatingPointType() ?
                    0.0625 : 0.125;
            cost += (cost*sparsityOverhead);
        }

        if(hasLayerWorkaroundAvoidPipeline(opIt, strategy))
        {
            spillPipelineCost = COST_MAX;
        }
        if (hasLayerWorkaroundAvoidStrategy(opIt, strategy) ||
            (requiresSparseInput(opIt, strategy) && !strategy["inputSparsity"].get<bool>()))
        {
            cost = COST_MAX;
            spillPipelineCost = COST_MAX;
        }

        strategy["spillPipelineCost"] = spillPipelineCost;
        strategy["cost"] = cost;
        heuristicCost = cost * clusteringStrategyCost.at(clustering);
        strategy["heuristicCost"] = heuristicCost;
        strategy["skip"] = false;
        strategy["prevInputSparsity"] = false;

        if(heuristicCost < bestCost)
        {
            bestStrat = strategy;
            bestCost = heuristicCost;
        }

        if(cost < greedyCost)
        {
            greedyStrat = strategy;
            greedyCost = cost;
        }
        //     printStrategy(strategy);
    }
    if(hasSOK && (opIt->getOpType() != "Input" && opIt->getOpType() !=  "Output"))
    {
        // If SplitOverK is an option, don't bother with single cluster
        bestCost = COST_MAX;
        greedyCost = COST_MAX;
        for(auto& strategy : opStrategies)
        {
            if(strategy["clustering"].get<std::string>() == "Clustering") 
                strategy["skip"] = true;

            if(strategy["skip"].get<bool>()) continue;

            if(strategy["heuristicCost"].get<double>() < bestCost)
            {
                bestStrat = strategy;
                bestCost = strategy["heuristicCost"].get<double>();
            }
            if(strategy["cost"].get<double>() < greedyCost)
            {
                greedyStrat = strategy;
                greedyCost = strategy["cost"].get<double>();
            }
        }
    }
    // Use this for debuggin strategy choice
    // opIt->set<std::string>("greedyClustering", greedyStrat["clustering"].get<std::string>());
    return bestStrat;
}

// For each of these workarounds specify the reason why this layer is disallowed, and the network it applies to
bool HeuristicGraphOptimizer::hasLayerWorkaroundAvoidStrategy(mv::Data::OpListIterator opIt, StrategySet& strategy)
{
    auto opType = opIt->getOpType();
    auto streamShape = strategy["streaming"].get<mv::Shape>();
    auto clustering = strategy["clustering"].get<std::string>();

    //This set of layer workarounds are for emotions recog retail network, pipelining
    //could help for some of these layers, and there is no spilling, but the scheduler makes bad choices..
    if(target == mv::Target::ma2490 && opType == "Conv" && streamShape["K"] > 1 && clustering == "SplitOverK")
    {
        auto inputShape = opIt->getInputTensor(0)->getShape();
        auto weightsShape = opIt->getInputTensor(1)->getShape();
        auto outputShape = opIt->getOutputTensor(0)->getShape();
        if(outputShape[mv::IO_HEIGHT_DIMENSION] == 4 && outputShape[mv::IO_WIDTH_DIMENSION] == 4 && 
            outputShape[mv::IO_CHANNEL_DIMENSION] == 256 &&
            weightsShape[mv::KERNEL_INPUT_CHANNELS] == 128 && weightsShape[mv::KERNEL_HEIGHT] == 3 &&
            inputShape[mv::IO_HEIGHT_DIMENSION] == 8 && inputShape[mv::IO_HEIGHT_DIMENSION] == 8)
                return true;

        if(outputShape[mv::IO_HEIGHT_DIMENSION] == 2 && outputShape[mv::IO_WIDTH_DIMENSION] == 2 && 
            outputShape[mv::IO_CHANNEL_DIMENSION] == 256 &&
            weightsShape[mv::KERNEL_INPUT_CHANNELS] == 256 && weightsShape[mv::KERNEL_HEIGHT] == 3 &&
            inputShape[mv::IO_HEIGHT_DIMENSION] == 4 && inputShape[mv::IO_HEIGHT_DIMENSION] == 4)
                return true;

        if(outputShape[mv::IO_HEIGHT_DIMENSION] == 2 && outputShape[mv::IO_WIDTH_DIMENSION] == 2 && 
            outputShape[mv::IO_CHANNEL_DIMENSION] == 256 &&
            weightsShape[mv::KERNEL_INPUT_CHANNELS] == 256 && weightsShape[mv::KERNEL_HEIGHT] == 3 &&
            inputShape[mv::IO_HEIGHT_DIMENSION] == 2 && inputShape[mv::IO_HEIGHT_DIMENSION] == 2)
                return true;

        if(outputShape[mv::IO_HEIGHT_DIMENSION] == 4 && outputShape[mv::IO_WIDTH_DIMENSION] == 4 && 
            outputShape[mv::IO_CHANNEL_DIMENSION] == 256 &&
            weightsShape[mv::KERNEL_INPUT_CHANNELS] == 256 && weightsShape[mv::KERNEL_HEIGHT] == 3 &&
            inputShape[mv::IO_HEIGHT_DIMENSION] == 4 && inputShape[mv::IO_HEIGHT_DIMENSION] == 4)
                return true;
    }

    return false;
}

// For each of these workarounds specify the reason why this layer is disallowed, and the network it applies to
bool HeuristicGraphOptimizer::hasLayerWorkaroundAvoidPipeline(mv::Data::OpListIterator opIt, StrategySet& strategy)
{
    auto opType = opIt->getOpType();
    auto streamShape = strategy["streaming"].get<mv::Shape>();
    auto clustering = strategy["clustering"].get<std::string>();

    //TODO check just for output channels not aligned to 16, because these cant CMX concat...
    if (target == mv::Target::ma2490 && opType == "Conv" && streamShape["K"] > 1 && opIt->getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION]%16 != 0)
        return true;

    // For performance in tiny-yolo-v2 vehicle detection, this last dpu task shouldn't stream
    // reason unknown, spillng anyway so lack of cmx concat shouldn't cause problem?
    if(streamShape["K"] > 1 && clustering == "SplitOverK" &&
        opType == "Conv" && opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] == 13 &&
        opIt->getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION] == 13 && 
        opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] == 425 &&
        opIt->getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] == 512)
            return true;

    // For performance in squeezenet, this last zm conv task shouldn't stream
    // spill (unaligned channels cant cmx concat) kills performance
    if(streamShape["K"] > 1 && clustering == "SplitOverK" &&
        opType == "Conv" && opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] == 14 &&
        opIt->getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION] == 14 && 
        opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] == 1000 &&
        opIt->getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] == 512)
            return true;

    return false;
}

bool HeuristicGraphOptimizer::isCMXable(mv::Data::OpListIterator opIt, StrategySet& strategy, bool isInput)
{
    auto clustering = strategy["clustering"].get<std::string>();
    auto iSparse = strategy["inputSparsity"].get<bool>();
    auto oSparse = strategy["outputSparsity"].get<bool>();
    auto wSparse = strategy["weightsSparsity"].get<bool>();
    auto streams = strategy["streaming"].get<mv::Shape>();
    bool fSparse = false;
    if(opIt->getOpType() == "Depthwise" || opIt->getOpType() == "MaxPool" || 
        (opIt->hasAttr("supportsCM") && opIt->get<bool>("supportsCM")))
        fSparse = true;

    bool spilling = strategy["spilling"].get<bool>();
    if(!isInput)
        spilling = false;

    bool parentSpilling = strategy["parentSpilling"].get<bool>();
    if(isInput)
        parentSpilling = false;

    int input, output, weights;
    input = output = weights = 0;
    std::tie(input, output, weights) = memorySize(*opIt,totalClusters_,clustering,iSparse,oSparse,wSparse,streams,fSparse,spilling,parentSpilling);

    if(input+output+weights < clusterMemory_)
        return true;

    return false;
}

bool HeuristicGraphOptimizer::attemptToSpillOp(mv::Data::OpListIterator opIt, bool lockClustering)
{
    auto opStrategy = bestStrategies_.at(opIt->getName());
    auto opClustering = opStrategy["clustering"].get<std::string>();
    auto& potentialStrategies = strategy_model_.at(opIt->getName());
    StrategySet bestStrat;
    double bestCost = COST_MAX;
    bool foundReplacement = false;
    for(auto& potentialStrategy : potentialStrategies)
    {
        if(!lockClustering && (opStrategy["id"].get<int>() == potentialStrategy["id"].get<int>()))
            potentialStrategy["skip"] = true;

        if(potentialStrategy["skip"].get<bool>()) continue;

        if((!lockClustering || opClustering == potentialStrategy["clustering"].get<std::string>()) &&
            potentialStrategy["spilling"].get<bool>())
        {
            if(potentialStrategy["cost"].get<double>() < bestCost)
            {
                bestStrat = potentialStrategy;
                bestCost = potentialStrategy["cost"].get<double>();
                foundReplacement  = true;
            }
        }
    }
    if(foundReplacement)
    {
        bestStrategies_.erase(opIt->getName());
        bestStrategies_.insert(std::make_pair(opIt->getName(), bestStrat));
        return true;
    }
    return false;
}

// Note: At this point, the strategies have been greedily assigned, so we may have 
// transitions that are valid if a tensor spills to DDR, but are not valid if that tensor
// stays in CMX, marked as staying in CMX (because not spilling is greedy performant!)
// We mark all these as requiring a spill, and they will be removed later, if that is
// deemed performant by the cost function
bool HeuristicGraphOptimizer::addSpillsAtStrategyTransitions()
{
    mv::DataModel dm(model_);
    std::vector<std::pair<std::string, std::string>>incompatibleStrategiesWithOutSpilling =
    {
        {"SplitOverHOverlapped", "Clustering"},
        {"SplitOverHOverlapped", "SplitOverK"},
        {"SplitOverH", "Clustering"},
        {"SplitOverH", "SplitOverK"},
        {"SplitOverK", "SplitOverH"},
        {"SplitOverK", "HKSwitch"},
        {"Clustering", "SplitOverH"},
        {"Clustering", "HKSwitch"},
        {"HKSwitch", "SplitOverH"},
        {"HKSwitch", "HKSwitch"}
    };
    bool clusteringChanged = false;

    auto sortedOps = model_.topologicalSort();
    for(auto opIt: sortedOps)
    {
        if(!opIt->hasAttr("StrategySet")) continue;

        auto opStrategy = bestStrategies_.at(opIt->getName());
        if(!opStrategy["spilling"].get<bool>()) // If we're already spilling, nothing to do
        {
            bool spillReq = false;
            auto sinkLayers = findSinkLayers(dm, opIt->getOutputTensor(0));
            for(auto sinkLayer : sinkLayers)
            {
                if(!sinkLayer->hasAttr("StrategySet")) continue;

                auto sinkStrategy = bestStrategies_.at(sinkLayer->getName());

                std::pair<std::string, std::string> possibleCombination(opStrategy["clustering"].get<std::string>(), 
                                                                        sinkStrategy["clustering"].get<std::string>());
                for (auto restrictedCombination : incompatibleStrategiesWithOutSpilling)
                {
                    if (possibleCombination == restrictedCombination)
                    {
                        spillReq = true;
                    }
                }
            }
            if(spillReq && opStrategy["clustering"].get<std::string>() == "SplitOverH")
            {
                bool success = assignBestStrategyOfType(opIt, "HKSwitch");
                if(success)
                {
                    clusteringChanged = true;
                    spillReq = false;
                }
            }
            if(spillReq)
            {
                // std::cout << "Adding spilling to op " << opIt->getName() <<std::endl;
                bool success = attemptToSpillOp(opIt, true);
                // TODO if unsuccessful (at least one op type must be in CMX), we should change child?
                // Or do we need to hack around these copy ops?
                // will also fail in cases where original guy was HKSwitch, because thats not spillable
                if(!success && opStrategy["clustering"].get<std::string>() != "HKSwitch")
                {
                    clusteringChanged = attemptToSpillOp(opIt, false);
                    if(!clusteringChanged)
                        log(mv::Logger::MessageType::Debug, "SSM leaving unspilled op " + opIt->getName());
                }
            }
        }
        else
        {
            //We're spilling, just check that none of the children are HKSwitch because it makes no snese
            //to spill in that case
            if(opIt->getOpType() == "Output") continue;

            auto sinkLayers = findSinkLayers(dm, opIt->getOutputTensor(0));
            for(auto sinkLayer : sinkLayers)
            {
                if(!sinkLayer->hasAttr("StrategySet")) continue;

                auto sinkStrategy = bestStrategies_.at(sinkLayer->getName());

                if(sinkStrategy["clustering"].get<std::string>() == "HKSwitch")
                {
                    findKCompatible(sinkLayer, true, false);
                    abandonSOH(sinkLayer, false);
                }
                
            }
        }
        
    }

    return clusteringChanged;
}

//Note: let's don't consider the spill portion of the SOH->SOK transition point here
// just consider, from a pure computational time persepective, will this layer be
// better suited to SOH or SOK
bool HeuristicGraphOptimizer::hasGreedySOK(mv::Data::OpListIterator opIt)
{
    // auto strategy = bestStrategies_.at(opIt->getName());
    auto hCost = findHCompatible(opIt, false, true);
    auto kCost = findKCompatible(opIt, false, true).first;

    if(kCost < hCost)
        return true; // If this op could be better by itself in K-compatible

    return false;
}

//Note: make sure that all the children are K compatible 
bool HeuristicGraphOptimizer::isGreedyEligible(mv::Data::OpListIterator opIt)
{
    for (auto child = opIt.leftmostChild(); child != model_.opEnd(); ++child)
    {
        if (!isKCompatible(child))
            return false;
    }
    return true;
}

void HeuristicGraphOptimizer::doSingleRollback(mv::Data::OpListIterator opIt)
{
    findKCompatible(opIt, true, true);
    abandonSOH(opIt, true);
}

// Check if this spill is strategy related
bool HeuristicGraphOptimizer::isRemoveableSpill(mv::Data::OpListIterator opIt)
{
    bool opKCompatible = isKCompatible(opIt);

    // Check K-compatability for this op and all children matches
    for(auto child = opIt.leftmostChild(); child != model_.opEnd(); ++child)
    {
        if(!child->hasAttr("StrategySet")) continue; //Note, this shouldn't happen for children

        // Need to resolve
        if(opKCompatible != isKCompatible(child))
            return true;
    }

    return false;
}

bool isZMconv(mv::Data::OpListIterator opIt)
{
    if(opIt->getOpType() != "Conv")
        return false;
    
    if(opIt->hasAttr("supportsCM") && !opIt->get<bool>("supportsCM"))
        return true;

    return false;
}

//Port this model A perf WA until the SOH->spill->SOK for ZM convs is fixed more generally
bool isModelAWA(mv::Data::OpListIterator opIt)
{
    if (opIt->getOpType() == "Conv" && opIt->getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 80 &&
        opIt->getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 44 && opIt->getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 44 &&
        opIt->getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 72 && opIt->getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 22 &&
        opIt->getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 22 && opIt->getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 3 &&
        opIt->getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 3)
    {
        return true;
    }
    if (opIt->getOpType() == "Conv" && opIt->getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 48 &&
        opIt->getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 22 && opIt->getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 22 &&
        opIt->getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 48 && opIt->getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 22 &&
        opIt->getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 22 && opIt->getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 3 &&
        opIt->getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 3)
    {
        return true;
    }
    return false;
}

bool isModelFWA(mv::Data::OpListIterator opIt)
{
    if(opIt->getOpType() == "Conv")
    {
        auto inputShape = opIt->getInputTensor(mv::IO_TENSOR_INPUT)->getShape();
        auto outputShape = opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT)->getShape();
        auto weightsShape = opIt->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET)->getShape();
        if( inputShape[mv::IO_CHANNEL_DIMENSION] == 16 &&
            inputShape[mv::IO_WIDTH_DIMENSION] == 1024 && inputShape[mv::IO_HEIGHT_DIMENSION] == 64 &&
            outputShape[mv::IO_CHANNEL_DIMENSION] == 16 && outputShape[mv::IO_WIDTH_DIMENSION] == 1024 &&
            outputShape[mv::IO_HEIGHT_DIMENSION] == 64 && weightsShape[mv::KERNEL_HEIGHT] == 7 &&
            weightsShape[mv::KERNEL_WIDTH] == 1)
            return true;
    }
    if(opIt->getOpType() == "Concat")
        return true;
    if(opIt->getOpType() == "Slice" && opIt.leftmostParent()->getOpType() == "Concat")
        return true;

    return false;
}

bool parentOpSupportsHK(mv::Data::OpListIterator opIt)
{
    auto opType = opIt->getOpType();
    if (opType == "MaxPool" || opType == "Eltwise" || opType == "HwConvert")
    {
        return true;   
    }
    return false;
}

bool HeuristicGraphOptimizer::forceRollback(mv::Data::OpListIterator opIt)
{
    bool opKCompatible = isKCompatible(opIt);

    auto strategy = bestStrategies_.at(opIt->getName());
    bool childExpectsFullInput = false;
    bool childExpectsSlicedInput = false;
    bool opHK = isHK(opIt);

    // Check K-compatability for this op and all children matches
    for(auto child = opIt.leftmostChild(); child != model_.opEnd(); ++child)
    {
        if(!child->hasAttr("StrategySet")) continue; //Note, this shouldn't happen for children

        // SOH->SOK disallowed if...
        if(!opKCompatible && isKCompatible(child))
        {
            // SOH->SOK disallowed if both are ZM conv
            if(isZMconv(opIt) && isZMconv(child)
                && !isModelAWA(opIt))
                return true;

            // SOH->SOK disallowed if one is Implicit Op in CMX
            // Note: Ops aren't marked as implicit yet, so "Default" strategy
            // assume they all happen in DDR. To be more accurate, consider Slice
            // as it's own, which can be in CMX or in DDR depending on preceeding
            // and following ops
            auto childStrategy = bestStrategies_.at(child->getName());
            if (opIt->getOpType() == "Slice" && !isModelFWA(opIt)
                && (!strategy["spilling"].get<bool>() || !childStrategy["parentSpilling"].get<bool>()))
                    return true;
            // ssd512 workaround - TODO disable when HKSwitch enabled for all ops
            if (opIt->getOpType() == "Slice" && parentOpSupportsHK(opIt) && !isModelFWA(opIt)
                && (!strategy["spilling"].get<bool>() || !childStrategy["parentSpilling"].get<bool>()))
                    return true;
        }
        if (isKCompatible(child))
            childExpectsFullInput = true;
        else
            childExpectsSlicedInput = true;

        // HK -> HK disallowed
        if(opHK && isHK(child))
            return true;
    }

    // check if all children expect the same type of input
    // split input or full input (H vs K compat)
    if (childExpectsFullInput && childExpectsSlicedInput)
        return true;

    return false;
}

bool HeuristicGraphOptimizer::isHK(mv::Data::OpListIterator opIt)
{
    // If strategy is already SOK, Clustering
    auto strategy = bestStrategies_.at(opIt->getName());
    auto clustering = strategy["clustering"].get<std::string>();
    if(clustering == "HKSwitch")
        return true;

    return false;
}

bool HeuristicGraphOptimizer::isKCompatible(mv::Data::OpListIterator opIt, bool allowHK)
{
    // If strategy is already SOK, Clustering
    auto strategy = bestStrategies_.at(opIt->getName());
    auto clustering = strategy["clustering"].get<std::string>();
    if( clustering == "SplitOverK" ||
        clustering == "Clustering" ||
        (allowHK && clustering == "HKSwitch"))
        return true;

    return false;
}

bool HeuristicGraphOptimizer::couldBeKCompatible(mv::Data::OpListIterator opIt)
{
    // If strategy is already SOK, Clustering, HKSwitch
    // Or if it has a valid HKSwitch option
    if(isKCompatible(opIt))
        return true;

    auto potentialStrategies = strategy_model_.at(opIt->getName());
    for(auto potentialStrategy : potentialStrategies)
    {
        if(potentialStrategy["clustering"].get<std::string>() == "HKSwitch")
            return true;
    }

    return false;
}

std::pair<double, StrategySet> HeuristicGraphOptimizer::findKCompatible(mv::Data::OpListIterator opIt, bool doAssignment, bool allowHK = true)
{
    auto potentialStrategies = strategy_model_.at(opIt->getName());
    StrategySet bestHKStrat;
    StrategySet bestKStrat;
    double bestHKCost = COST_MAX;
    double bestKCost = COST_MAX;
    bool foundHKReplacement = false;
    bool foundKReplacement = false;
    for(auto potentialStrategy : potentialStrategies)
    {
        if(potentialStrategy["skip"].get<bool>()) continue;

        if(potentialStrategy["clustering"].get<std::string>() == "HKSwitch")
        {
            if(potentialStrategy["cost"].get<double>() < bestHKCost)
            {
                bestHKStrat = potentialStrategy;
                bestHKCost = potentialStrategy["cost"].get<double>();
                foundHKReplacement  = true;
            }
        }
        else if(potentialStrategy["clustering"].get<std::string>() == "SplitOverK" ||
                potentialStrategy["clustering"].get<std::string>() == "Clustering")
        {
            if(potentialStrategy["cost"].get<double>() < bestKCost)
            {
                bestKStrat = potentialStrategy;
                bestKCost = potentialStrategy["cost"].get<double>();
                foundKReplacement  = true;
            }
        }
    }
    if(doAssignment)
    {
        if(foundHKReplacement && allowHK)
        {
            bestStrategies_.erase(opIt->getName());
            bestStrategies_.insert(std::make_pair(opIt->getName(), bestHKStrat));
        }
        else if(foundKReplacement)
        {
            bestStrategies_.erase(opIt->getName());
            bestStrategies_.insert(std::make_pair(opIt->getName(), bestKStrat));
        }
        abandonSOH(opIt, allowHK);
    }

    if(foundHKReplacement && allowHK)
        return std::make_pair(bestHKCost, bestHKStrat);
    else if(foundKReplacement)
        return std::make_pair(bestKCost, bestKStrat);
    else 
        return std::make_pair(COST_MAX, bestStrategies_.at(opIt->getName()));
}

double HeuristicGraphOptimizer::findHCompatible(mv::Data::OpListIterator opIt, bool doAssignment, bool allowHK = true)
{
    auto potentialStrategies = strategy_model_.at(opIt->getName());
    StrategySet bestHKStrat;
    StrategySet bestHStrat;
    double bestHKCost = COST_MAX;
    double bestHCost = COST_MAX;
    bool foundHKReplacement = false;
    bool foundHReplacement = false;
    for(auto potentialStrategy : potentialStrategies)
    {
        if(potentialStrategy["skip"].get<bool>()) continue;

        if(potentialStrategy["clustering"].get<std::string>() == "HKSwitch")
        {
            if(potentialStrategy["cost"].get<double>() < bestHKCost)
            {
                bestHKStrat = potentialStrategy;
                bestHKCost = potentialStrategy["cost"].get<double>();
                foundHKReplacement  = true;
            }
        }
        else if(potentialStrategy["clustering"].get<std::string>() == "SplitOverH" ||
                potentialStrategy["clustering"].get<std::string>() == "SplitOverHOverlapped")
        {
            if(potentialStrategy["cost"].get<double>() < bestHCost)
            {
                bestHStrat = potentialStrategy;
                bestHCost = potentialStrategy["cost"].get<double>();
                foundHReplacement  = true;
            }
        }
    }
    if(doAssignment)
    {
        if(foundHKReplacement && allowHK)
        {
            bestStrategies_.erase(opIt->getName());
            bestStrategies_.insert(std::make_pair(opIt->getName(), bestHKStrat));
        }
        else if(foundHReplacement)
        {
            bestStrategies_.erase(opIt->getName());
            bestStrategies_.insert(std::make_pair(opIt->getName(), bestHStrat));
        }
    }

    if(foundHKReplacement && allowHK)
        return bestHKCost;
    else if(foundHReplacement)
        return bestHCost;
    else return COST_MAX;
}

// This function is the heart of what the StrategyManager and MetaGraph did for the 
// Graph Optimizer pass. The idea is, when strategies on multiple ops must be changed
// together (i.e. making choices around SOH, SOK), this pass will decide the most efficient way
// to do those strategy transitions.
// The options are, 1. rollback the transition to some earlier point in the graph
// 2. Spill to do the transition on the spot
// 3. If possible, do the transition in CMX
// To decide between these options, we look at each op in turn moving backwards through the op model
// If it marks a transition point of SOH->SOK, then we search the graph for all its neighbors that would
// also require a change if this op where to change to K-compatible. As we go, we tally the cost of changing
// each op, and at the end we compare that to the current cost of this neighbor subgraph. We choose the more
// performant strategy, which either requires processing the subgraph to change each node, or leaving it be.
void HeuristicGraphOptimizer::chooseRollbackOrSpill()
{
    auto sortedOps = model_.topologicalSort();
    std::reverse(sortedOps.begin(), sortedOps.end());
    for(auto opIt: sortedOps)
    {
        if(!opIt->hasAttr("StrategySet")) continue;
        // std::cout <<std::endl<< "Processing op: " << opIt->getName() << std::endl;
        // Iff the spill is caused by strategy shift only (not CMX related, etc)
        if(isRemoveableSpill(opIt) && hasGreedySOK(opIt) && isGreedyEligible(opIt))
        {
            // This op was actually better in SOK, just got SOH b/c heuristic
            doSingleRollback(opIt);
        }
        else
        {
            processForSpillRemoval(opIt);
        }
        
        //The above algorithm moves up through the model, but looking at children
        //Also of interest, is ensuring the compatability of multiple input ops (eltwise, concat)
        //If I'm a K-compatible elt or concat, make sure all inputs are k-compatible
        //We can't get here with H-compatible with k-compatible inputs because of forceConnectedSOH pass
        // redo the addSpillsAtStrategyTransitions if the below code executes? To ensure correct cost
        // at next iteration of processForSpillRemoval?
        checkMultipleInputOp(opIt);
        // if(addNeededSpills)
            addSpillsAtStrategyTransitions();

    }
}

bool HeuristicGraphOptimizer::checkMultipleInputOp(mv::Data::OpListIterator opIt)
{
    bool addNeededSpills = false;
    auto opType = opIt->getOpType();
    auto opKCompatible = isKCompatible(opIt);
    bool foundMismatch = false;
    if(opType == "Eltwise" || opType == "Concat")
    {
        for(auto input = opIt.leftmostParent(); input != model_.opEnd(); ++input)
        {
            if(!input->hasAttr("StrategySet")) continue;
            
            if(opKCompatible != isKCompatible(input))
            {
                foundMismatch = true;
                findKCompatible(input, true, true);
                abandonSOH(input, true);
                if(isKCompatible(input)) // don't need the spill if we are HKSwitch now
                    addNeededSpills = true;
            }
        }
    }
    if(!opKCompatible && foundMismatch)
    {
        //Also update the original elt or concat#
        findKCompatible(opIt, true, false);
        abandonSOH(opIt, false); //inputs now k-comp so don't allow hk
    }
    return addNeededSpills;
}


// Note: The idea of this algorithm is to decide between spilling to change tensor split strategy,
// rolling back SOH to some point where this strategy switch can happen in CMX (HKSwitch)
// We move backwards through a topological sort of the Op Model
// For each op, if it is a strategy spill we first decide what ops would need to change for a rollback
// Consider a simple linear graph A (SOH or HK) -> B (SOH) -> C (SOH) -> D (SOK)
// When we reach C, a SOH layer that must spill we process it:
// 1. Add C to ops_to_change, mark it
// 2. Add unmarked children to Q_c, if they have SOH or HKSwitch strategy and mark them
//      ex: Q_c : remains empty
// 3. If C cannnot take HKSwitch, add unmarked parents to Q_p and mark them
//      ex: Q_p : B
// 4. Continue processing elements from Q_p while not empty
//      ex: pop B and process it from step 1 (Q_c will remain empty, Q_p: A)
//          pop A and process it from step 1 (Q_c will remain empty, Q_p is empty)
// 5. Continue processing elements from Q_c while not empty
// 6. If cost cheaper to roll back, last HK-elligble op added to ops to change is HK, 
//    rest take best compatible (SOK, clus) strategy
// 
void HeuristicGraphOptimizer::processForSpillRemoval(mv::Data::OpListIterator opIt)
{
    mv::DataModel dm(model_);
    double currentCost = 0.0;
    double changeCost = 0.0;
    std::queue<mv::Data::OpListIterator> parents;
    std::queue<mv::Data::OpListIterator> children;
    std::list<mv::Data::OpListIterator> parentOpsToChange;
    std::list<mv::Data::OpListIterator> childrenOpsToChange;
    std::set<std::string> markedOps;
    auto heuristicMultiplier = getMultiplier(opIt);

    bool opsLeftToProcess = true;
    bool rollbackReq = forceRollback(opIt); //TODO remove, requires the spill option to be accurate.
    mv::Data::OpListIterator N = opIt;

    // Determine which nodes would need to be changed in order to remove the spill
    // Calculate the potential cost as we go
    parentOpsToChange.push_back(N);
    changeCost += findKCompatible(N, false, true).first;
    do {
        // std::cout << " N is " << N->getName() <<std::endl;
        // printStrategy(bestStrategies_.at(N->getName()));
        markedOps.insert(N->getName());
        currentCost += bestStrategies_.at(N->getName())["cost"].get<double>();

        // We always add children if they are still in SOH compatible 
        for(auto child = N.leftmostChild(); child != model_.opEnd(); ++child)
        {
            if(!child->hasAttr("StrategySet")) continue; //Note, this shouldn't happen for children

            // std::cout << "    Found Child: " << child->getName() << std::endl;

            auto childStrategy = bestStrategies_.at(child->getName());
            auto childClustering = childStrategy["clustering"].get<std::string>();
            if((childClustering == "SplitOverH" || childClustering == "HKSwitch") &&
                (markedOps.find(child->getName()) == markedOps.end()))
            {
                children.push(child);
                markedOps.insert(child->getName());
                // std::cout << "       Added child to Q" << std::endl;
                // In these special cases, a spill can't fix the transition
                // So we force a rollback
                if(isKCompatible(N) && (childClustering == "HKSwitch" || 
                    (childClustering == "SplitOverH" && child->getOpType() == "Concat")))
                    // We have K -> HK, SOH
                    rollbackReq = true;
            }
        }
        // We stop moving up the graph when we find nodes that could be HK switch points
        // or nodes that are already in compatible strategies (SOK, Clus)
        if(!couldBeKCompatible(N))
        {
            for(auto parent = N.leftmostParent(); parent != model_.opEnd(); ++parent)
            {
                if(!parent->hasAttr("StrategySet")) continue;

                // std::cout << "    Found Parent: " << parent->getName() << std::endl;

                auto parentStrategy = bestStrategies_.at(parent->getName());
                auto parentClustering = parentStrategy["clustering"].get<std::string>();
                if((parentClustering == "SplitOverH" || parentClustering == "HKSwitch") &&
                    (markedOps.find(parent->getName()) == markedOps.end()))
                {
                    parents.push(parent);
                    markedOps.insert(parent->getName());
                    // std::cout << "       Added parent to Q" << std::endl;
                    if(parentClustering == "HKSwitch")
                    {
                        //We have an HK -> SOH, this should be prevented by the forceConnectedSOH pass, but leaving here 
                        rollbackReq = true;
                    }
                }
            }
        }
        if(!parents.empty())
        {
            N = parents.front();
            parents.pop();
            parentOpsToChange.push_back(N);
            auto compatStrategy = findKCompatible(N, false, true);
            changeCost += compatStrategy.first; 
            //TODO, if we allow HK that later goes, this cost isn't exact
            //If I'm going to become an HK, and any of my children have parentSpilling as their
            //best KCompatible strategy, we need to capture the cost of spilling from the HK too
            if(compatStrategy.second["clustering"].get<std::string>() == "HKSwitch")
            {
                auto sinkLayers = findSinkLayers(dm, N->getOutputTensor(0));
                bool allInCmx = true;
                for(auto sink : sinkLayers)
                {
                    auto newStrat = findKCompatible(sink, false, false);
                    if(newStrat.second["parentSpilling"].get<bool>()) 
                    {
                        // std::cout << "HKSwitch: " << N->getName() << ", and parentSpilling: " << sink->getName() << std::endl;
                        // The parent can't be sparse b/c another child needs dense
                        allInCmx = false;
                        break;
                    }
                }
                if(!allInCmx)
                    changeCost += 2*outputDmaTime(N, compatStrategy.second, true);
                     //TODO should this really be multiplied by 2, or should the heuristic be used? put here for tiny yolo v2 perf
            }
        }
        else if(!children.empty())
        {
            N = children.front();
            children.pop();
            childrenOpsToChange.push_back(N);
            changeCost += findKCompatible(N, false, false).first;
        }
        else
        {
            opsLeftToProcess = false;
        }
        // std::cout << std::boolalpha << "   CURRENT STATE: rollbackReq = " << rollbackReq << ", rollbackCost = " << changeCost << ", currentCost = " << currentCost << std::endl;
    } while (opsLeftToProcess);
    
    // std::cout << std::boolalpha << "rollbackReq = " << rollbackReq << ", rollbackCost = " << changeCost << ", rollbackHeuristic: " << changeCost*heuristicMultiplier << ", currentCost = " << currentCost << std::endl;
    if(rollbackReq || (changeCost * heuristicMultiplier) < currentCost)
    {
        for(auto op : parentOpsToChange)
        {
            findKCompatible(op, true);
        }
        for(auto op : childrenOpsToChange)
        {
            findKCompatible(op, true, false);
        }
    }
}

//Here we can put network or layer specific workarounds to the cost function
//Ensure each workaround is as specific as possible, and comment the name of the
//network or layer it is there to optimize
double HeuristicGraphOptimizer::getMultiplier(mv::Data::OpListIterator opIt)
{
    std::ignore = opIt;

    //open pose perf
    if(target == mv::Target::ma2490 && model_.getInput()->getOutputTensor(0)->getShape()["W"] == 656
        && model_.getInput()->getOutputTensor(0)->getShape()["H"] == 368)
        return 1.0;

    return SOH_HEURISTIC_MULTIPLIER;
}

void HeuristicGraphOptimizer::alignAndValidateSpecialOps()
{
    mv::DataModel dm(model_);
    //Ensure that input strategy is correct
    auto inputs = model_.getOps("Input"); // there should only be one?
    for(auto input : inputs)
    {
        auto sinkLayers = findSinkLayers(dm, input->getOutputTensor(0));
        auto inputStrategy = bestStrategies_.at(input->getName());
        auto inputClustering = inputStrategy["clustering"].get<std::string>();
        for(auto sink : sinkLayers)
        {
            auto sinkClustering = bestStrategies_.at(sink->getName())["clustering"].get<std::string>();
            //TODO handle situation where input goes multiple K-compatible ways
            if(sink->hasAttr("supportsCM") && sink->get<bool>("supportsCM") &&
                sinkClustering == "SplitOverH")
                assignBestStrategyOfType(input, "SplitOverHOverlapped");
            else if(inputClustering != sinkClustering)
                assignBestStrategyOfType(input, sinkClustering);
        }
    }
}

bool HeuristicGraphOptimizer::assignBestStrategyOfType(mv::Data::OpListIterator opIt, std::string clusteringStrategy)
{
    auto opStrategies = strategy_model_.at(opIt->getName());

    StrategySet bestStrat;
    double bestCost = COST_MAX;
    bool foundReplacement = false;
    for(auto strategy : opStrategies)
    {
        if(strategy["skip"].get<bool>()) continue;

        if(strategy["clustering"].get<std::string>() == clusteringStrategy)
        {
            if(strategy["cost"].get<double>() < bestCost)
            {
                bestStrat = strategy;
                bestCost = strategy["cost"].get<double>();
                foundReplacement = true;
            }
        }
    }
    if(foundReplacement)
    {
        bestStrategies_.erase(opIt->getName());
        bestStrategies_.insert(std::make_pair(opIt->getName(), bestStrat));
    }

    return foundReplacement;
}

double HeuristicGraphOptimizer::findBestStrategyOfLocation(mv::Data::OpListIterator opIt, bool doAssignment, 
                                                            bool inputDDR, bool lockOutput, bool outputDDR,
                                                            bool lockClustering, std::string clustering)
{
    auto opStrategies = strategy_model_.at(opIt->getName());

    StrategySet bestStrat;
    double bestCost = COST_MAX;
    bool foundReplacement = false;
    for(auto strategy : opStrategies)
    {
        if(strategy["skip"].get<bool>()) continue;

        if(strategy["parentSpilling"].get<bool>() == inputDDR &&
           (!lockOutput || strategy["spilling"].get<bool>() == outputDDR) &&
           (!lockClustering || strategy["clustering"].get<std::string>() == clustering))
        {
            if(strategy["cost"].get<double>() < bestCost)
            {
                bestStrat = strategy;
                bestCost = strategy["cost"].get<double>();
                foundReplacement = true;
            }
        }
    }
    if(doAssignment && foundReplacement)
    {
        bestStrategies_.erase(opIt->getName());
        bestStrategies_.insert(std::make_pair(opIt->getName(), bestStrat));
    }

    return bestCost;
}

bool HeuristicGraphOptimizer::strategyChangeRequiresSpill(mv::Data::OpListIterator& opIt, mv::Data::OpListIterator& pIt)
{
    return isKCompatible(opIt, false) != isKCompatible(pIt, true) && pIt->getOpType() != "Input";
}

void HeuristicGraphOptimizer::verifySpillStrategies(bool lockClusteringStrategy = false)
{
    mv::DataModel dm(model_);
    auto sortedOps = model_.topologicalSort();
    for(auto opIt : sortedOps)
    {
        if(!opIt->hasAttr("StrategySet") || opIt->getOpType() == "Input" || opIt->getOpType() == "Concat") continue;

        auto opStrategy = bestStrategies_.at(opIt->getName());
        if(!opStrategy["parentSpilling"].get<bool>()) continue; //If we work with CMX input, no need to check

        auto inputTensors = opIt->getInputTensor();
        double ddrCost = opStrategy["cost"].get<double>(); // op already expects input in ddr
        double cmxCost = findBestStrategyOfLocation(opIt, false, false, false, 
                                                    opStrategy["spilling"].get<bool>(), 
                                                    lockClusteringStrategy, 
                                                    opStrategy["clustering"].get<std::string>()); // if can take input cmx, find cost

        bool foundCMXinput = false;
        bool strategyRequiresSpill = false;
        for(auto inputTensor : inputTensors)
        {
            auto inputOp = model_.getSourceOp(inputTensor);
            if(!inputOp->hasAttr("StrategySet") || opIt->getOpType() == "Concat") continue;
            auto parentStrategy = bestStrategies_.at(inputOp->getName());
            bool parentSpilling = parentStrategy["spilling"].get<bool>();
            //Check for required spill due to strategy change
            if (strategyChangeRequiresSpill(opIt, inputOp))
            {
                strategyRequiresSpill = true;
                parentSpilling = true;
            }
            if(!parentSpilling) //input in CMX, mismatch found
            {    
                foundCMXinput = true;
                ddrCost += findBestStrategyOfLocation(inputOp, false, parentSpilling, true, true, 
                                                        lockClusteringStrategy, parentStrategy["clustering"].get<std::string>());
                cmxCost += parentStrategy["cost"].get<double>();
            }
            else
            {
                ddrCost += parentStrategy["cost"].get<double>(); // this input won't need to change, same as current
                auto cost = findBestStrategyOfLocation(inputOp, false, parentSpilling, true, false, 
                                                        lockClusteringStrategy, parentStrategy["clustering"].get<std::string>());
                if(cost < COST_MAX)
                    cmxCost += cost; //If this input could provide cmx input, use, but not required if it didn't exist
            }
            
        }
        if(foundCMXinput || strategyRequiresSpill)
        {
            bool inputDDR = cmxCost > ddrCost;
            findBestStrategyOfLocation(opIt, true, inputDDR, false, 
                                            opStrategy["spilling"].get<bool>(), 
                                            lockClusteringStrategy, 
                                            opStrategy["clustering"].get<std::string>() ); //This op takes CMX input, can spill or not
            for(auto inputTensor : inputTensors)
            {
                auto inputOp = model_.getSourceOp(inputTensor);
                if(!inputOp->hasAttr("StrategySet") || opIt->getOpType() == "Concat") continue;
                auto parentStrategy = bestStrategies_.at(inputOp->getName());
                strategyRequiresSpill = strategyChangeRequiresSpill(opIt, inputOp);
                //For each input, get the best spill=false strategy (lock parent spilling)
                findBestStrategyOfLocation(inputOp, true, (parentStrategy["parentSpilling"].get<bool>() || strategyRequiresSpill), true, inputDDR,
                                            lockClusteringStrategy, parentStrategy["clustering"].get<std::string>());
            }
        }
    }
}


//Note: This function checks more than if this op type is a sparse consumer
// on hardware. Given the other "locked" strategy values at this point, we also
// want to check that an input sparse version of the strategy exists
// Streaming might change, but mc strategy should not
bool HeuristicGraphOptimizer::findRealSparseInput(mv::Data::OpListIterator opIt, bool doAssignment)
{
    if(!opIt->isSparsityConsumer()) return false;

    auto opName = opIt->getName();
    auto currentStrategy = bestStrategies_.at(opName);
    if (currentStrategy["inputSparsity"].get<bool>() && 
        !requiresCompilerActivationSparsity(opIt, currentStrategy)) 
        return true;

    auto opStrategies = strategy_model_.at(opName);
    StrategySet bestStrat;
    double bestCost = COST_MAX;
    bool foundSparseReplacement = false;
    for (auto potentialStrategy : opStrategies)
    {
        if (potentialStrategy["skip"].get<bool>()) continue;

        // At this point, we want clustering and spilling to stay the same
        if ((potentialStrategy["clustering"].get<std::string>() == currentStrategy["clustering"].get<std::string>()) &&
            (potentialStrategy["spilling"].get<bool>() == currentStrategy["spilling"].get<bool>()) &&
            (potentialStrategy["parentSpilling"].get<bool>() == currentStrategy["parentSpilling"].get<bool>()) &&
            (potentialStrategy["inputSparsity"].get<bool>() || potentialStrategy["prevInputSparsity"].get<bool>()) &&
            !requiresCompilerActivationSparsity(opIt, potentialStrategy))
            {
                if(potentialStrategy["cost"].get<double>() < bestCost)
                {
                    bestStrat = potentialStrategy;
                    bestCost = potentialStrategy["cost"].get<double>();
                    foundSparseReplacement = true;
                }
            }
    }

    if (doAssignment && foundSparseReplacement)
    {
        bestStrategies_.erase(opName);
        bestStrategies_.insert(std::make_pair(opName, bestStrat));
    }

    return foundSparseReplacement;
}

bool HeuristicGraphOptimizer::canServiceActivationSparsity(mv::Data::OpListIterator opIt, StrategySet& strategy)
{
    if(requiresCompilerActivationSparsity(opIt, strategy))
        return false;

    mv::DataModel dm(model_);
    std::vector<std::shared_ptr<std::vector<StrategySet>>> allStrategies;
    auto inputTensors = opIt->getInputTensor();
    bool allInputsSparse = true;
    for(auto inputTensor : inputTensors)
    {
        auto inputOp = model_.getSourceOp(inputTensor);
        if(!inputOp->hasAttr("StrategySet")) continue;

        auto opStrategiesPtr = opIt->get<std::shared_ptr<std::vector<StrategySet>>>("StrategySet");
        allStrategies.push_back(opStrategiesPtr);
        auto& opStrategies = *(allStrategies.back());
        bool foundSparseOutput = false;
        for(auto inputStrategy : opStrategies)
            if(inputStrategy["outputSparsity"].get<bool>() && 
            inputStrategy["clustering"].get<std::string>() == strategy["clustering"].get<std::string>())
                foundSparseOutput = true;

        if(!foundSparseOutput)
            allInputsSparse = false;
    }

    return allInputsSparse;
}

void HeuristicGraphOptimizer::serviceActivationSparsity()
{
    mv::DataModel dm(model_);
    auto sortedOps = model_.topologicalSort();
    //TODO perf sparsity, but for now, if we don't need it, ditch it
    for(auto opIt : sortedOps)
    {
        if(!opIt->hasAttr("StrategySet")) continue;

        auto& strategy = bestStrategies_.at(opIt->getName());
        if( strategy["inputSparsity"].get<bool>() && !requiresSparseInput(opIt, strategy))
        {
            //We got here and this strategy was chosen, but we don't really need it to be sparse
            strategy["inputSparsity"] = false;
            strategy["prevInputSparsity"] = true;
        }
    }
    for(auto opIt : sortedOps)
    {
        if(!opIt->hasAttr("StrategySet")) continue;

        auto childStrategy = bestStrategies_.at(opIt->getName());
        if( childStrategy["inputSparsity"].get<bool>() &&
            !requiresCompilerActivationSparsity(opIt, childStrategy))
        {
            auto inputTensors = opIt->getInputTensor();
            for(auto inputTensor : inputTensors)
            {
                auto inputOp = model_.getSourceOp(inputTensor);
                if(!inputOp->hasAttr("StrategySet")) continue;

                //Parent already sparse, nothing more to do
                auto parentStrategy = bestStrategies_.at(inputOp->getName());
                if(parentStrategy["outputSparsity"].get<bool>()) continue;

                auto sinkLayers = findSinkLayers(dm, inputOp->getOutputTensor(0));
                bool allAcceptSparse = true;
                for(auto sink : sinkLayers)
                {
                    if(!findRealSparseInput(sink, false)) 
                    {
                        // The parent can't be sparse b/c another child needs dense
                        // or child needs compiler, not runtime sparsity
                        allAcceptSparse = false;
                        break;
                    }
                }
                if(allAcceptSparse) 
                {  
                
                    bool success = findSparseOutput(inputOp);
                    if(success)
                    {
                        for(auto sink : sinkLayers)
                        {
                            findRealSparseInput(sink, true);
                        }
                    }
                }
            }
        }
        else if(childStrategy["inputSparsity"].get<bool>() &&
                requiresCompilerActivationSparsity(opIt, childStrategy))
        {
            //Ensure no sparsity is being passed to this op
            auto inputTensors = opIt->getInputTensor();
            for(auto inputTensor : inputTensors)
            {
                auto inputOp = model_.getSourceOp(inputTensor);
                if(!inputOp->hasAttr("StrategySet")) continue;

                //Parent is sparse, need to turn this off
                auto parentStrategy = bestStrategies_.at(inputOp->getName());
                if(parentStrategy["outputSparsity"].get<bool>())
                {
                    bool success = findDenseOutput(inputOp);
                    if(!success)
                        std::cout << "WARNING: unable to denisfy op " << inputOp->getName() << std::endl;
                }
            }
        }
    }
}

void HeuristicGraphOptimizer::increaseWeightsPipelining()
{
    auto sortedOps = model_.topologicalSort();
    for(auto opIt : sortedOps)
    {
        if(!opIt->hasAttr("StrategySet")) continue;
        auto opName = opIt->getName();

        auto currentStrategy = bestStrategies_.at(opName);
        auto streamShape = currentStrategy["streaming"].get<Shape>();
        bool isStreaming = ((streamShape["W"] * streamShape["H"] * streamShape["C"]
                                                * streamShape["K"] * streamShape["B"]) > 1) ? true : false;
        if((streamShape["K"] > 1 || !isStreaming) && 
            (opIt->getOpType() == "Conv" || opIt->getOpType() == "DepthwiseConv") &&
            !(!isStreaming && currentStrategy["inputSparsity"].get<bool>())) // Streaming would require compiler sparsity, keep runtime
        {
            auto opStrategies = strategy_model_.at(opName);
            StrategySet bestStrat;
            double bestCost = currentStrategy["spillPipelineCost"].get<double>();
            bool foundReplacement = false;
            for(auto potentialStrategy : opStrategies)
            {
                if(potentialStrategy["skip"].get<bool>()) continue;

                //Lock clustering strategy, sparsity strategy, parentSpilling, okay to move from !spilling to spilling
                // and the point is to change streaming strategy of course..
                if((potentialStrategy["clustering"].get<std::string>() == currentStrategy["clustering"].get<std::string>()) &&
                    (potentialStrategy["parentSpilling"].get<bool>() == currentStrategy["parentSpilling"].get<bool>()) &&
                    (potentialStrategy["inputSparsity"].get<bool>() == currentStrategy["inputSparsity"].get<bool>()) &&
                    (potentialStrategy["outputSparsity"].get<bool>()  == currentStrategy["outputSparsity"].get<bool>()) )
                    {
                        if(potentialStrategy["spillPipelineCost"].get<double>() < bestCost)
                        {
                            bestStrat = potentialStrategy;
                            bestCost = potentialStrategy["spillPipelineCost"].get<double>();
                            foundReplacement = true;
                        }
                    }
            }

            if(foundReplacement)
            {
                bestStrategies_.erase(opName);
                bestStrategies_.insert(std::make_pair(opName, bestStrat));
            }
        }
    }
}

// Attempt to change to a similar strategy, with runtime sparsity enabled on output
bool HeuristicGraphOptimizer::findSparseOutput(mv::Data::OpListIterator opIt)
{
    auto opName = opIt->getName();
    auto currentStrategy = bestStrategies_.at(opName);
    auto opStrategies = strategy_model_.at(opName);

    StrategySet bestStrat;
    double bestCost = COST_MAX;
    bool foundSparseReplacement = false;
    for(auto potentialStrategy : opStrategies)
    {
        if(potentialStrategy["id"].get<int>() == currentStrategy["id"].get<int>()) continue;

        if(potentialStrategy["skip"].get<bool>()) continue;

        // At this point, we want clustering and spilling to stay the same
        // Enable full sparsity if possible
        // TODO should I check I'm still streaming in the same dimension?
        if((potentialStrategy["clustering"].get<std::string>() == currentStrategy["clustering"].get<std::string>()) &&
            (potentialStrategy["spilling"].get<bool>() == currentStrategy["spilling"].get<bool>()) &&
            (potentialStrategy["parentSpilling"].get<bool>() == currentStrategy["parentSpilling"].get<bool>()) &&
            (potentialStrategy["inputSparsity"].get<bool>() == currentStrategy["inputSparsity"].get<bool>()) &&
            potentialStrategy["outputSparsity"].get<bool>())
            {
                if(potentialStrategy["cost"].get<double>() < bestCost)
                {
                    bestStrat = potentialStrategy;
                    bestCost = potentialStrategy["cost"].get<double>();
                    foundSparseReplacement = true;
                }
            }
    }

    if(foundSparseReplacement)
    {
        bestStrategies_.erase(opName);
        bestStrategies_.insert(std::make_pair(opName, bestStrat));
    }

    return foundSparseReplacement;
}

// Attempt to change to a similar strategy, with runtime sparsity dis-abled on output
bool HeuristicGraphOptimizer::findDenseOutput(mv::Data::OpListIterator opIt)
{
    auto opName = opIt->getName();
    auto currentStrategy = bestStrategies_.at(opName);
    auto opStrategies = strategy_model_.at(opName);

    StrategySet bestStrat;
    double bestCost = COST_MAX;
    bool foundDenseReplacement = false;
    for(auto& potentialStrategy : opStrategies)
    {
        if(potentialStrategy["outputSparsity"].get<bool>())
            potentialStrategy["skip"] = true;

        if(potentialStrategy["skip"].get<bool>()) continue;

        // At this point, we want clustering and spilling to stay the same
        // Disable runtime sparsity if possible
        // TODO should I check I'm still streaming in the same dimension?
        if((potentialStrategy["clustering"].get<std::string>() == currentStrategy["clustering"].get<std::string>()) &&
            (potentialStrategy["spilling"].get<bool>() == currentStrategy["spilling"].get<bool>()) &&
            (potentialStrategy["parentSpilling"].get<bool>() == currentStrategy["parentSpilling"].get<bool>()) &&
            (potentialStrategy["inputSparsity"].get<bool>() == currentStrategy["inputSparsity"].get<bool>()) &&
            !potentialStrategy["outputSparsity"].get<bool>())
            {
                if(potentialStrategy["cost"].get<double>() < bestCost)
                {
                    bestStrat = potentialStrategy;
                    bestCost = potentialStrategy["cost"].get<double>();
                    foundDenseReplacement = true;
                }
            }
    }

    if(foundDenseReplacement)
    {
        bestStrategies_.erase(opName);
        bestStrategies_.insert(std::make_pair(opName, bestStrat));
    }

    return foundDenseReplacement;
}

// This op must have sparse input (for example, because of op type or clustering strategy). 
// Does not matter if sparsity comes from runtime or compiler. 
// For now, we use this to turn off sparsity unless it's necessary, in lieu of a decision on activation sparsity for performance.
bool HeuristicGraphOptimizer::requiresSparseInput(mv::Data::OpListIterator opIt, StrategySet& strategy)
{
    if(requiresRealActivationSparsity(opIt, strategy))
        return true;

    bool isCMConv = opIt->hasAttr("supportsCM") && opIt->get<bool>("supportsCM");
    if (opIt->getOpType() == "Conv" && !isCMConv
            && (opIt->hasAttr("DilatedSubConv") && opIt->get<bool>("DilatedSubConv")))
        return true;

    return false;
}

bool HeuristicGraphOptimizer::requiresRealActivationSparsity(mv::Data::OpListIterator opIt, StrategySet& strategy)
{
    auto clustering = strategy["clustering"].get<std::string>();
    //An fp16 Conv Z-major must have activation sparsity
    bool isCMConv = opIt->hasAttr("supportsCM") && opIt->get<bool>("supportsCM");

    if (opIt->isSparsityConsumer() &&
        opIt->getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") &&
        !isCMConv && checkA0Sparsity(model_))
    {
        return true;
    }


    // Check for need for A0 SOH Sparsity workaround, (SOH conv with kernel > 1)
    // if needed, check memory constraints as for sparse tensor
    // TODO: Review the need for below KMB-A0 specific W/A for KMB-B0 and TBH platforms
    if (opIt->getOpType() == "Conv" ) {
        if( clustering == "SplitOverH" &&
            (opIt->getInputTensor(1)->getShape()[KERNEL_HEIGHT] > 1) &&
            !isCMConv && (target == mv::Target::ma3100 ||  // Apply the W/A also for TBH to overcome accuracy regression
                        (target == mv::Target::ma2490 && referenceDevice_ == "A0")))

            {
                return true;
            }
    }

    // TODO: Work on removal of below temporar W/A which was added to prevent
    // unexpected inference issue on yolo-v3-darknet. Activation sparsity itself is not the source
    // of the problem but once removed for this specific case compiler makes decisions which eventually
    // makes the blob not executable on KMB
    // Note: porting, from changes in GO after SSM created. Unclear if this impacts, given the different
    // strategies generated, and the real fixes for accuracy introduced.
    if (opIt->isSparsityConsumer() &&
        opIt->getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") &&
        !isCMConv && (opIt->hasAttr("placeConversionToFloat") && opIt->get<bool>("placeConversionToFloat")))
    {
        auto inputShape = opIt->getInputTensor(0)->getShape();
        auto outputShape = opIt->getOutputTensor(0)->getShape();
        if (inputShape[mv::IO_CHANNEL_DIMENSION] == 1024 && inputShape[mv::IO_WIDTH_DIMENSION] == 19 && inputShape[mv::IO_HEIGHT_DIMENSION] == 19 &&
            outputShape[mv::IO_CHANNEL_DIMENSION] == 255 && outputShape[mv::IO_WIDTH_DIMENSION] == 19 && outputShape[mv::IO_HEIGHT_DIMENSION] == 19)
        {
            return true;
        }
    }


    return false;
}

// Not all of these require sparse input, but if any take sparse input, that sparsity must come from the compiler.  
// This is used both at cost gen (this type of sparsity is pure overhead) and 
// strategy selection (avoid enabling sparse output from parents with compiler sparse req children).
bool HeuristicGraphOptimizer::requiresCompilerActivationSparsity(mv::Data::OpListIterator opIt, StrategySet& strategy)
{
    bool isCMConv = opIt->hasAttr("supportsCM") && opIt->get<bool>("supportsCM");
    auto clusteringStrategy = strategy["clustering"].get<std::string>();

    if (opIt->getOpType() == "Conv" && !isCMConv
            && (opIt->hasAttr("DilatedSubConv") && opIt->get<bool>("DilatedSubConv")))
        return true;

    auto childStreamShape = strategy["streaming"].get<mv::Shape>();
    bool childStreaming = (childStreamShape["H"] * childStreamShape["C"] * childStreamShape["K"] * childStreamShape["B"]) > 1 ? true : false;
    if(childStreaming)
        return true;

    if( opIt->getOpType() == "Conv")
    {
        auto weightsShape = opIt->getInputTensor(1)->getShape();
        if( !isCMConv &&
            clusteringStrategy == "SplitOverH" &&
            weightsShape[mv::KERNEL_HEIGHT] > 1 )
        {
            auto parentOp = model_.getSourceOp(opIt->getInputTensor(0));
            // This should also be solveable with fake compiler provided sparsity
            // there may very well be cases where sparsity if enforced, but due to this
            // limitation proper sparsity is not a choice since cluster boundary sparse map
            // reads will fail due to misalignment
            // Fake sparsity will provide all 1's sparse map so that probem is solved
            // from the starts

            // Sparse map has to be contiguously alligned at 16 bytes
            // for first (N - 1) clusters
            auto outputTensorShape = parentOp->getOutputTensor(0)->getShape();
            unsigned int W = outputTensorShape[IO_WIDTH_DIMENSION];
            unsigned int H = outputTensorShape[IO_HEIGHT_DIMENSION];
            unsigned int C = outputTensorShape[IO_CHANNEL_DIMENSION];
            unsigned dy = std::ceil(static_cast<double>(H) / totalClusters_);

            // this limitation that sparse map should be 16 bytes aligned in each subtensors,
            // ONLY applies when DPUs are reading data from neighbor clusters
            // Each subtensor should be aligned to 16 byte boundaries. For SM we have 1 bit per elem,
            // so divide tensor by 8 get size in bytes
            // (sparse idu for SOH ZM CONV kernel h > 1)
            if( (W*dy*C)%128 != 0 ) //  equivalent with (W*dy*C/8)%16
            {
                log(mv::Logger::MessageType::Debug, strategy["name"].toString()+"_"+strategy["id"].toString() + " INF caused by incorrect SOH");
                return true;
            }
        }
    }

    if (opIt->hasAttr("floatPrecision") && opIt->get<bool>("floatPrecision") && 
        opIt->getOpType() == "Eltwise" && (clusteringStrategy == "SplitOverH" || clusteringStrategy == "HKSwitch"))
    {
        //NOTE: On floating point network, Mobilenet there is a case that if we have runtime sparsity with
        //SOH going to an eltwise the eltwise fails, so the step is to use compiler sparsity on that point
        auto inputTensors = opIt->getInputTensor();
        for(auto inputTensor : inputTensors)
        {
            auto parentOp = model_.getSourceOp(inputTensor);
            if(parentOp->getOpType() == "Conv" || parentOp->getOpType() == "Eltwise")
            {
                if( parentOp->getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 28 &&
                    parentOp->getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 28 &&
                    parentOp->getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 32)
                        return true;
            }
        }
    }

    return false;
}

}
}
