#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/graphOptimizations/strategy_utils.hpp"
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

bool requiresFakeActivationSparsity(mv::Data::OpListIterator opIt, bool enableChannelMajorConv)
{
    if(enableChannelMajorConv && opIt->supportsCMConv())
        return true;
    if(opIt->getOpType() == "MaxPool")
        return true;
    if(opIt->getOpType() == "DepthwiseConv")
        return true;

    return false;
}

std::tuple<size_t, size_t, size_t> getMemorySize(mv::ComputationModel& model,mv::Data::OpListIterator opIt, const mv::Shape& streamConfig)
{
    mv::OpModel om(model);
    auto globalParams = model.getGlobalConfigParams();
    bool enableChannelMajorConv = globalParams->get<bool>("enable_channel_major_conv");
    size_t totalClusters = globalParams->get<int>("Number_of_Clusters");

    auto clustering = opIt->get<std::string>("splitStrategy");
    auto inputSparse = opIt->get<bool>("inputActivationSparsity");
    auto outputSparse = opIt->get<bool>("outputActivationSparsity");
    auto weightsSparse = opIt->get<bool>("weightsSparsity");
    auto fakeSparse = requiresFakeActivationSparsity(opIt, enableChannelMajorConv);
    bool spilling = opIt->get<bool>("goPredictsSpill");
    auto prevOp = om.getSourceOp(opIt->getInputTensor(0));
    bool parentSpilling = prevOp->get<bool>("goPredictsSpill");

    return mv::memorySize(*opIt, totalClusters, enableChannelMajorConv, clustering, inputSparse, outputSparse, weightsSparse, streamConfig,
                        fakeSparse, spilling, parentSpilling);
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

        //Reject H streams were the last stream isn't equal or smaller than the rest
        //Reject H streams were the last stream is 1, unless they are all 1
        if(newOutputSizes.back() > newOutputSizes.front() ||
            (newOutputSizes.back() == 1 && newOutputSizes.front() != 1)) 
                return false;

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
    size_t clusterMemory = globalParams->get<int>("cmx");
    auto clusterStrategy = opIt->get<std::string>("splitStrategy");

    size_t input, output, weights;
    // in case initialization in memorySize fails
    input = output = weights = 0;
    std::tie(input, output, weights) = getMemorySize(model, opIt, {1,1,1,1,1});
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
        auto memFitCheck = getMemorySize(model, opIt, {1,splits,1,1,1});

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
    size_t clusterMemory = globalParams->get<int>("cmx");
    auto clusterStrategy = opIt->get<std::string>("splitStrategy");

    for(unsigned splits = startStream; splits >= 1; splits--)
    {
        auto memFitCheck = getMemorySize(model, opIt,{1,splits,1,1,1});
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

    // if(opIt->hasAttr("floatPrecision") && opIt->get<bool>("floatPrecision"))
    //     return false;

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
    if(!parentSpilling || prevOpType == "Concat")
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

bool isChannelMajorConvWithSOK(mv::ComputationModel& model, mv::Data::OpListIterator opIt)
{
    mv::OpModel om(model);
    auto clusteringStrategy = opIt->get<std::string>("splitStrategy");
    // accuracy issues when activation is pipelined and perf regression if streams increased
    if (opIt->supportsCMConv() && clusteringStrategy == "SplitOverK")
        return true;

    return false;
}

bool hardcodedSparsityIssue(mv::ComputationModel& model)
{
    mv::OpModel om(model);
    return om.getInput()->hasAttr("hardcoded_streams") && om.getInput()->get<bool>("hardcoded_streams");
}

//ASSUMPTION: the op must fit in cmx with just streaming over H (or no streaming at all)
std::size_t findOptimalStream(mv::ComputationModel& model, mv::Data::OpListIterator opIt, size_t originalHStream)
{
    mv::OpModel om(model);
    auto globalParams = model.getGlobalConfigParams();
    size_t clusterMemory = globalParams->get<int>("cmx");
    size_t totalClusters = globalParams->get<int>("Number_of_Clusters");
    size_t totalDpus = globalParams->get<int>("Number_of_DPUs");
    double dpuPerCluster = std::floor(totalDpus/totalClusters);
    auto clusteringStrategy = opIt->get<std::string>("splitStrategy");

    // Step 1. Decide which tensor will be the benchmark for how many streams we should do
    size_t magicStreams, input, output, weights;
    magicStreams = input = output = weights = 0;
    std::tie(input, output, weights) = getMemorySize(model, opIt, {1,1,1,1,1});

    // Step 2. Calculate a possible number of streams using experimetnally found magic number
    // Idea is, if possible, allow multiple slices to fit in CMX to maximize paths
    // For example, for first layer of YoloV2, (input=416*416*3, output=416*416*32)
    // The GO will choose to stream this over H, just enough to fit or H=7
    //      (416*416*3) + (416*416*32) = 6056960 / 7 = 865280, which is less than CMX (weights are under 1kb)
    // Experimentally, we found highest performance gain when we plan to fit input and output
    // slice, the next input slice, and the full weights in around 60% of CMX. This gives
    // the scheduler plenty of room, and creates small slices that better overlap with compute
    //      ((416*416*3)*2 + (416*416*32)) / (917504-(weights) * 0.6) = ~12.01 -> 13
    // Note that input and output in these equations are the full tensors because the 
    // getMemorySize call above is invoked without any streams {1,1,1,1,1}
    // for the case when output > input use 2 * max(input, output) + min()
    if(!isChannelMajorConvWithSOK(model, opIt) && !hardcodedSparsityIssue(model))
    {
        opIt->set<bool>("performance_optimized", true);
        magicStreams = std::ceil((2*std::max(input, output) + std::min(input, output))/ ((clusterMemory-weights)*0.6));
    }
    else
        magicStreams = std::ceil((2*input + output)/ ((clusterMemory-weights)*0.6));
    if(magicStreams < originalHStream)
        magicStreams = originalHStream; //If GO gave carved it up into smaller pieces, must be for a reason
    else if(magicStreams > originalHStream*3)
        magicStreams = originalHStream*3; // Let's not get crazy with the H streams

    // Can't exceed the max, which ensures at least one line of output for each stream to compute
    size_t maxStreams = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
    if(clusteringStrategy == "SplitOverH") maxStreams = std::ceil((double)maxStreams/totalClusters);
    maxStreams = std::ceil((double) maxStreams / dpuPerCluster);

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
        pass.log(mv::Logger::MessageType::Debug, "AddActivationStreaming could not open output file " + jsonOutFileName);

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
void addActivationStreamingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{
    mv::OpModel om(model);
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("split_strategy"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No custom splitting strategy provided, exiting...");
        return;
    }
    bool enableChannelMajorConv = globalParams->get<bool>("enable_channel_major_conv");
    if(!enableChannelMajorConv) return;


    auto streamingStrategies = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    std::vector<mv::Element> newStreamingStrategies;

    int streamCount=0;
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

        // check # streams added
        auto strategySelected = newStreamingStrategies.back();
        auto streamsSelected = strategySelected.get<std::vector<mv::Element>>("splits");
        
        // Streams are mostly 1's, so multiply total streams added
        int streamsAdded =  streamsSelected[0].get<int>("W") * 
                            streamsSelected[1].get<int>("H") * 
                            streamsSelected[2].get<int>("C") *
                            streamsSelected[3].get<int>("K") *
                            streamsSelected[4].get<int>("N");
        
        streamCount += streamsAdded;
    }

    //Step 4. Save the streaming strategies into the compilation descriptor to be read by the streaming pass
    // 1050 streams equals ~ 7000 tasks which causes the runtime load/parse time to increase to >10 seconds.
    // vpuMgr currently times out all calls to the VPU that don't return in < 10secs
    int MAX_STREAM_COUNT = 1000;
    if (passDesc.hasAttr("max_streams"))
        MAX_STREAM_COUNT = passDesc.get<int>("max_streams");

    if (streamCount < MAX_STREAM_COUNT)
    {
        globalParams->set<std::vector<mv::Element>>("streaming_strategy", newStreamingStrategies);
        saveNewStreamingStrategiesToJson(pass, newStreamingStrategies);
    }
}