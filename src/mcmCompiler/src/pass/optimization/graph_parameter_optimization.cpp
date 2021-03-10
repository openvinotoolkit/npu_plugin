#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/utils/custom_math.hpp"
// #include "include/mcm/utils/compression/hde.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/graphOptimizations/strategy_utils.hpp"

static void GraphParameterOptimizationFcn(const mv::pass::PassEntry&,
    mv::ComputationModel& model,
    mv::TargetDescriptor&, mv::Element& passDesc,
    mv::Element&
);


namespace {

struct EnumClassHash final {
    template <typename E>
    size_t operator()(E t) const {
        return std::hash<int32_t>()(static_cast<int32_t>(t));
    }
};

}  // namespace

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
            StrategyManagerKmb(OpModel& model,mv::Element& passDesc, mv::TargetDescriptor& td) :
                StrategyManager(model,passDesc)
            {
                auto globalParams = model.getGlobalConfigParams();
                enableChannelMajorConv = globalParams->get<bool>("enable_channel_major_conv");
                target = td.getTarget();
            }

            size_t totalClusters=4;
            size_t clusterMemoryKb=896;
            size_t dpuPerCluster=5;
            std::string referenceDevice = "A0";
            int ddrBandwidth=128;
            int sysClock=500;
            double sysClockHz=sysClock*1000000;
            double DMA_LATENCY = 36 / sysClockHz; // in c -> s
            double DMA_BANDWIDTH = 25769803776; // gb/s - > b/s
            double LATENCY_CMX = 5 / sysClockHz; // in c -> s
            double BANDWIDTH_CMX = (double)(1.5 * 1099511627776); // tb/s -> b/s
            bool globalEnableStreaming=true;
            bool globalEnablePipelining = true;
            bool globalEnablePrefetching = true;
            bool globalEnableActivationSparsity=true;
            bool globalForceActivationSparsity=false;
            bool globalEnableWeightsSparsity=false;
            bool globalForceSpilling=false;
            bool enableChannelMajorConv=false;
            double safetyFactor=1.0;
            mv::Target target = mv::Target::ma2490;
            double clusterMemory=(double)clusterMemoryKb * 1024.0 * safetyFactor;
            double cmxPipeLineWeightsOverhead=34816.0;
            enum class FailCause
            {
                Pass,
                cmxConcatDecision,
                MemorySize,
                StreamAndClusterComp,
                SpillHKSwitch,
                SOKNotAlign16,
                InputNotSpilled,
                OutputNotSpilled,
                StreamingNotSpilled,
                WorkloadLessKernelSOH,
                ChannelMjr1,
                ChannelMjr2,
                DWChannels,
                SOHheight,
                RequiresSparsity,
                RealSparseForFakeSparseOp,
                DilatedSOH,
                DWLargeStrideReplacementSOK,
                SpiltOverHWithStreamOverK,
                SpiltOverHWithStreamOverHInCMX,
                SparsityKSegmented,
                SparsitySpilling,
                DeConvSubConvSOKHeight,
                SpiltOverHForLayer79InACLNet,
                SpiltOverHForLayer97and113ModelE,
                SpiltOverHForFaceDetectionRetail0004,
                //SplitOverHOverlappedWronglyComputed,
                SoftwareDeconvolutionSet,
                UpaHKSwitch
            };

            std::unordered_map<FailCause, std::string, ::EnumClassHash> failure_causes = {
                {FailCause::Pass, "Pass"},
                {FailCause::cmxConcatDecision, "cmxConcatDecision"},
                {FailCause::MemorySize, "MemorySize"},
                {FailCause::StreamAndClusterComp, "Stream+ClusterComp"},
                {FailCause::SpillHKSwitch, "SpillHKSwitch"},
                {FailCause::SOKNotAlign16, "SOKNotAlign16"},
                {FailCause::InputNotSpilled, "InputNotSpilled"},
                {FailCause::OutputNotSpilled, "OutputNotSpilled"},
                {FailCause::StreamingNotSpilled, "StreamingNotSpilled"},
                {FailCause::WorkloadLessKernelSOH, "Workload<KernelSOH"},
                {FailCause::ChannelMjr1, "ChannelMjr1"},
                {FailCause::ChannelMjr2, "ChannelMjr2"},
                {FailCause::DWChannels, "DWChannels"},
                {FailCause::SOHheight, "SOHheight"},
                {FailCause::RequiresSparsity, "RequiresSparsity"},
                {FailCause::RealSparseForFakeSparseOp, "RealSparseForFakeSparseOp"},
                {FailCause::DilatedSOH, "DilatedSOH"},
                {FailCause::DWLargeStrideReplacementSOK, "DWLargeStrideReplacementSOK"},
                {FailCause::SpiltOverHWithStreamOverK, "SpiltOverHWithStreamOverK"},
                {FailCause::SpiltOverHWithStreamOverHInCMX, "SpiltOverHWithStreamOverHInCMX"},
                {FailCause::SparsityKSegmented, "SparsityKSegmented"},
                {FailCause::SparsitySpilling, "SparsitySpilling"},
                {FailCause::DeConvSubConvSOKHeight, "DeConvSubConvSOKHeight"},
                {FailCause::SpiltOverHForLayer79InACLNet, "SpiltOverHForLayer79InACLNet"},
                {FailCause::SpiltOverHForLayer97and113ModelE, "SpiltOverHForLayer97and113ModelE"},
                {FailCause::SoftwareDeconvolutionSet, "SoftwareDeconvolutionSet"},
                {FailCause::SpiltOverHForFaceDetectionRetail0004, "SpiltOverHForFaceDetectionRetail0004"},
                //{FailCause::SplitOverHOverlappedWronglyComputed, "SplitOverHOverlappedWronglyComputed"},
                {FailCause::UpaHKSwitch, "UpaHKSwitch"}
            };

            void readGlobalConfigs()
            { 
                referenceDevice = model_.getGlobalConfigParam("referenceDevice").get<std::string>();
                totalClusters = model_.getGlobalConfigParam("Number_of_Clusters").get<int>();
                clusterMemoryKb = model_.getGlobalConfigParam("cmx").get<int>() / 1024;
                dpuPerCluster = model_.getGlobalConfigParam("Number_of_DPUs").get<int>() / totalClusters;
                createStrategyDots = globalConfig_["createStrategyDots"].get<bool>();
                dotFileLocation = globalConfig_["dotFileLocation"].get<std::string>();
                jsonOutFileName = globalConfig_["jsonOutFileName"].get<std::string>();
                loadStrategiesFromFile = globalConfig_["loadStrategiesFromFile"].get<bool>();
                jsonInFileName = globalConfig_["jsonInFileName"].get<std::string>();
                //Input is in Kb
                clusterMemory = (double)clusterMemoryKb * 1024.0 * safetyFactor;
                globalEnableStreaming = globalStrategies_["enableStreaming"].get<bool>();
                globalEnablePipelining = globalStrategies_["enablePipelining"].get<bool>();
                globalEnablePrefetching = globalStrategies_["enablePrefetching"].get<bool>();
                globalForceActivationSparsity = globalStrategies_["forceActivationSparsity"].get<bool>();
                globalEnableWeightsSparsity = globalStrategies_["enableWeightsSparsity"].get<bool>();
                globalForceSpilling =  globalStrategies_["forceSpilling"].get<bool>();
            }

            /***
             *
             * File organization.
             *
             * STRATEGY GENERATION
             * STRATEGY EVAULATION
             *
             */

            void generateStrategySetForLayer(mv::Op& op,std::vector<StrategySet>& strategyVec)
            {
                auto findStrategy = [](std::vector<Attribute>& vec,const std::string& str) ->bool { for(const auto elem : vec) if(str==elem.get<std::string>()) return true; return false;};

                std::vector<Attribute> spillingPool;
                if(globalForceSpilling)
                    spillingPool.push_back(true);
                else
                    spillingPool = createTStrategyPoolFromBool(op, "forceSpilling");

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

                //TODO:: replace nested loops with clean cartesian product function
                for(auto spilling : spillingPool)
                {
                for(auto clustering : clusteringStrategyPool)
                {
                    // Make decision about input activation sparsity, depending on clustering strategy
                    std::vector<Attribute> inputActivationSparsity = {false};
                    std::vector<Attribute> outputActivationSparsity = {false};
                    if(globalEnableActivationSparsity)
                    {
                        inputActivationSparsity = createTFStrategyPoolFromBool(op,"inputActivationSparsity");
                        outputActivationSparsity = createTFStrategyPoolFromBool(op,"outputActivationSparsity");
                    }
                    if(globalForceActivationSparsity)
                    {
                        inputActivationSparsity = {createStrategyFromBool(op,"inputActivationSparsity")};
                        outputActivationSparsity = createTFStrategyPoolFromBool(op,"outputActivationSparsity");
                    }

                    if (requiresActivationSparsity(op, clustering.get<std::string>()))
                        inputActivationSparsity = {true};

                    for( auto inputSparsity : inputActivationSparsity)
                    {
                    for( auto outputSparsity : outputActivationSparsity)
                    {
                        bool fakeSparsity = requiresFakeActivationSparsity(op);
                        std::vector<Attribute> eltwiseParentPool;
                        if(op.getOpType() == "Eltwise")
                            eltwiseParentPool = {true, false};
                        else
                            eltwiseParentPool.push_back(true);


                        for( const auto eltwiseParentStrategy : eltwiseParentPool)
                        {

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
                                                                spilling.get<bool>(),eltwiseParentStrategy.get<bool>());
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
                                                            outputSparsity.get<bool>(), weightsSparsity, fakeSparsity, spilling.get<bool>());
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
                        auto memK = memorySize(op, totalClusters,enableChannelMajorConv, clustering.get<std::string>(),inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,1,1,maxK,n},fakeSparsity, spilling.get<bool>());
                        auto memoryMaxK = std::get<0>(memK) + std::get<1>(memK) + std::get<2>(memK);
                        auto maxH = streamsOverH.front();
                        auto memH = memorySize(op,totalClusters,enableChannelMajorConv, clustering.get<std::string>(),inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,maxH,1,1,n},fakeSparsity, spilling.get<bool>());
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
                                                                outputSparsity.get<bool>(), weightsSparsity, fakeSparsity, spilling.get<bool>());
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

                                    Shape streamShape({1,h,c,k,n}); //Stream over W is 1. Not implemented.

                                    StrategySet s;
                                    s["name"] = op.getName();
                                    s["id"] = (unique_ctr++);
                                    s["inputSparsity"] = inputSparsity;
                                    s["outputSparsity"] = outputSparsity;
                                    s["weightsSparsity"] = weightsSparsity;
                                    s["spilling"] = spilling;
                                    s["clustering"] = clustering;
                                    s["streaming"] = streamShape;
                                    if(op.getOpType() == "Eltwise")
                                        s["eltwiseParentSpilling"] = eltwiseParentStrategy;

                                    //Function to prune strategies that will have only infinite edges in or out (or both), improves performance
                                    auto strategyCheck = validateStrategy(op,s);
                                    //std::cout << op.getName() << " : " << clustering.toString() << " : " << streamShape.toString() << " : S " << spilling.toString() << " : I " << inputSparsity.toString() << " : O " << outputSparsity.toString() << " = " << failure_causes[strategyCheck]<< std::endl;
                                    if(strategyCheck != FailCause::Pass)
                                        continue;

                                    strategyVec.push_back(s);

                                    //    std::cout << "Name: " + op.getName() << " ID " << s["id"].toString()<< std::endl;
                                    //    std::cout << "Input Sparsity: " + inputSparsity.toString() << std::endl;
                                    //    std::cout << "Output Sparsity: " + outputSparsity.toString() << std::endl;
                                    //    std::cout << "Weights Sparsity: " + weightsSparsity.toString() << std::endl;
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

            // Note: This function will return the potential streams over H for this op. For simplicity, we want to limit
            // the options to reasonable configurations. This always includes H=1, or in other words no streaming over H.
            // If H streaming fits at all (i.e. weights fit), find the H just big enough to fit into CMX. If CMX concat,
            // spilling will be false, and H stream will be higher accordingly.
            std::vector<size_t> getStreamsOverH(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                                bool wSparsity, bool fSparsity, bool spilling, bool parentSpilling = true)
            {
                auto minSplitsToFit = getMinStreamOverH(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, false, parentSpilling);
                if(minSplitsToFit == 0 || clustering.get<std::string>() == "HKSwitch") // stream over H doesn't fit
                    return {1};

                // TODO re-enable pipeline over H
                // Case 0, Not spilling, cmx concat. If pipelined, need room for 2 input slices.
                // if(!spilling && globalEnablePipelining && createStrategyFromBool(op, "pipelining"))
                // {
                //     auto pipelinedMinSplitsToFit =  getMinStreamOverH(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, true, parentSpilling);
                //     if(pipelinedMinSplitsToFit > 1 && pipelinedMinSplitsToFit != minSplitsToFit)
                //         return {pipelinedMinSplitsToFit, minSplitsToFit, 1};
                // }
                // else
                if(spilling) // Case 1 Spilling, ddr concat. Find min stream over H for both input in cmx and input streamed.
                {
                    auto inputCmxMinSplitsToFit =  getMinStreamOverH(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, false, false);
                    if(inputCmxMinSplitsToFit > 1 && inputCmxMinSplitsToFit != minSplitsToFit)
                    {
                        if(minSplitsToFit > 1)
                            return {inputCmxMinSplitsToFit, minSplitsToFit, 1};
                        else
                            return {inputCmxMinSplitsToFit, 1};
                    }
                }

                if(minSplitsToFit == 1)
                    return {1};

                return {minSplitsToFit, 1};
            }

            // Gives the minimum number of streams over H to fit this layer, or if no number of streams enable streaming
            // (for example, weights don't fit) then return 0
            unsigned getMinStreamOverH(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                        bool wSparsity, bool fSparsity, bool spilling, bool pipelined = false, bool parentSpilling = true)
            {
                size_t input, output, weights;
                // in case initialization in memorySize fails
                input = output = weights = 0;
                std::tie(input, output, weights) = memorySize(op,totalClusters,enableChannelMajorConv,clustering.get<std::string>(),iSparsity,oSparsity,wSparsity,streams,fSparsity,spilling,parentSpilling);
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
                    auto memFitCheck = memorySize(op,totalClusters, enableChannelMajorConv, clustering.get<std::string>(),iSparsity,oSparsity,wSparsity,updatedStreams,fSparsity,spilling,parentSpilling);

                    if( pipelined && //TODO inputCMX here too
                        (2*std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                        validateHStream(op, clustering, splits) )
                    {
                        return splits;
                    }
                    else if(!pipelined &&
                            (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                            validateHStream(op, clustering, splits))
                    {
                        return splits;
                    }
                }

                return 0;
            }

            bool validateHStream(mv::Op& op, mv::Attribute clustering, std::size_t splits)
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
                    int inputSizeForLastSplit = ((newOutputSizes.back() -1) * kernelStride)  -padStart - padEnd + kernelH;
                    if ((inputSizeForLastSplit + padEnd) < kernelH)
                        return false;
                }

                return true;
            }

            // Note: This function produces the potential stream over K strategies for each layer
           // Try to find 2 possible combinations of K, in addition ot K=1 (no streams in this dimension)
           // First, just enough to fit in cmx. Second, enough to enable pipelining.
            std::vector<std::size_t> getStreamsOverK(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                                bool wSparsity, bool fSparsity, bool spilling)
            {
                auto minSplitsToFit = getMinStreamOverK(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling);
                if(minSplitsToFit == 0) // No suitable stream over K found
                    return {1};

                std::vector<std::size_t> splits;
                splits.push_back(1);
                if(minSplitsToFit != 1)
                    splits.push_back(minSplitsToFit);

                if( globalEnablePipelining &&
                    createStrategyFromBool(op, "pipelining") && //Only find extra K streams if pipelining enabled
                    (clustering.get<std::string>() == "SplitOverK"))
                {
                    auto pipelinedMinSplitsToFit = getMinStreamOverK(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, true);
                    if(pipelinedMinSplitsToFit != 0)
                    {
                        if(pipelinedMinSplitsToFit != minSplitsToFit)
                            splits.push_back(pipelinedMinSplitsToFit);
                        auto nextKStream = getNextStreamOverK(op, clustering, pipelinedMinSplitsToFit, spilling);
                        if(nextKStream > 0)
                        {
                            splits.push_back(nextKStream);
                            auto thirdKStream = getNextStreamOverK(op, clustering, nextKStream, spilling);
                            if(thirdKStream > 0)
                                splits.push_back(thirdKStream);
                        }
                    }
                }

                return splits;
            }

            unsigned getMinStreamOverK(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                                bool wSparsity, bool fSparsity, bool spilling, bool pipelined = false)
            {
                auto outputShape = op.getOutputTensor(0)->getShape();
                size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
                size_t alignedOutputChannelSize = mv::round_up(outputChannelSize, 16);

                auto maxSplit = alignedOutputChannelSize / 16;

                if(clustering.get<std::string>() == "SplitOverK")
                    maxSplit = maxSplit / totalClusters;

                for(unsigned split = 1; split <= maxSplit; split++)
                {
                    auto memFitCheck = memorySize(op,totalClusters,enableChannelMajorConv, clustering.get<std::string>(),iSparsity,oSparsity,wSparsity,{1,1,1,split,streams["B"]},fSparsity, spilling);
                    if( pipelined && //pipelining weights requires 2 weights streams to fit
                        (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + 2*std::get<2>(memFitCheck) < clusterMemory) &&
                        validateKStream(op, clustering, split, spilling, totalClusters) )
                    {
                        return split;
                    }
                    else if(!pipelined &&
                            (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                            validateKStream(op, clustering, split, spilling, totalClusters) )
                    {
                        return split;
                    }
                }

                return 0;
            }

            unsigned getNextStreamOverK(mv::Op& op, mv::Attribute clustering, size_t startSplit, bool spilling)
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
                    if(validateKStream(op, clustering, split, spilling, totalClusters))
                        return split;
                }

                return 0;
            }

            // Note: Find suitable stream over C values
            std::vector<std::size_t> getStreamsOverC(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
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

            unsigned getMinStreamOverC(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                                bool wSparsity, bool fSparsity, bool spilling, bool /*pipelined*/ = false)
            {
                auto inputShape = op.getInputTensor(0)->getShape();
                size_t inputChannelSize = inputShape[IO_CHANNEL_DIMENSION];

                unsigned startSplit = 1;
                if(inputChannelSize > mv::MAX_DIM_SIZE)
                    startSplit = 2;

                for(unsigned split = startSplit; split <= inputChannelSize; split++)
                {
                    auto memFitCheck = memorySize(op, totalClusters,enableChannelMajorConv,clustering.get<std::string>(),iSparsity,oSparsity,wSparsity,{1,1,split,1,streams["B"]},fSparsity, spilling);
                    if((std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory))
                        return split;
                }

                return 0;
            }

            //Note: this function only used to generate many stream over k options when we NESTED stream
            std::vector<size_t> getMaxStreamOverK(mv::Op& op)
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

            unsigned findBestK(unsigned alignedSize, unsigned channels){
                return std::ceil((double)alignedSize / ((alignedSize/2) - channels));
            }

            bool createStrategyFromBool(mv::Op op, std::string name){
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return true;
                else
                    return false;
            }

            std::vector<Attribute> createTFStrategyPoolFromBool(mv::Op op,std::string name)
            {
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return std::vector<Attribute>{true,false};
                else
                    return std::vector<Attribute>{false};
            }

            std::vector<mv::Attribute> createTStrategyPoolFromBool(mv::Op op,std::string name)
            {
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return std::vector<mv::Attribute>{true};
                else
                    return std::vector<mv::Attribute>{true,false};
            }


            std::vector<mv::Attribute> createStrategyPoolFromStrategySet(mv::Op op, std::string name)
            {
                auto streamingStrategy = getStrategy(op,name);

                std::vector<mv::Attribute> attr;

                for (auto elem : streamingStrategy.get<std::vector<std::string>>())
                {
                    attr.push_back(elem);
                }

                return attr;
            }

            bool decideWeightsSparsity(mv::Op op)
            {
                // Only Z-major convolutions support weights sparsity, this is codified in the compilation descriptors
                if( !createStrategyFromBool(op,"weightsSparsity") )
                    return false;

                // If CM convolutions are enabled, don't sparsify these
                if(enableChannelMajorConv && op.supportsCMConv())
                    return false;

                // Size of weights, actual sparsity of tensor determine speedup
                auto weightsSize = realTensorSize(op.getInputTensor(1), {1,1,1,1}, false);
                auto zeroPoints = op.getInputTensor(1)->getZeroValuesCount();
                double actualSparsity = (double) zeroPoints/ (double)weightsSize;

                auto sparsityOverhead = op.getInputTensor(0)->isFloatingPointType() ?
                    0.0625 : 0.125;

                // Enable weights sparsity if actual sparsity level observed in the tensor
                // is high enough to warrant the overhead of enabling sparsity
                if(std::isgreaterequal(actualSparsity, sparsityOverhead))
                    return true;

                return false;
            }

            bool opInCMX(mv::Op& op, StrategySet& strategy)
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

            /***
             *
             * Begin Strategy Evaluation section
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             *
             * */

            //Check to see if a given stategy is internally consistent for performance
            //Strategies that can only have infinite edges because they are illegal should never be added to the graph
            // Note: IF ADDING A NEW FAILURE CASE, must add new description to failure_causes
            FailCause validateStrategy(mv::Op& op,StrategySet& strategy)
            {
                auto clustering = strategy["clustering"].get<std::string>();
                auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
                auto streamShape = strategy["streaming"].get<Shape>();
                auto spilling = strategy["spilling"].get<bool>();

                //NOTE: funny part you can spill even if you are not streaming, fasten your seatbelts!!
                bool isStreaming = ((streamShape["W"] * streamShape["H"] * streamShape["C"]
                                                            * streamShape["K"] * streamShape["B"]) > 1) ? true : false;

                // These rules are necessary for Yolo v2 performance
                {
                    if (op.getOpType() == "Conv")
                    {
                        if (op.getInputTensor()[0]->getShape() == mv::Shape({13,13,512,1}) && 
                            op.getInputTensor()[0]->getDType() == mv::DType("UInt8") &&
                            op.getInputTensor()[1]->getShape() == mv::Shape({3,3,512,1024}) &&
                            op.getInputTensor()[1]->getDType() == mv::DType("UInt8") && 
                            op.getOutputTensor()[0]->getShape() == mv::Shape({13,13,1024,1}) &&
                            op.getOutputTensor()[0]->getDType() == mv::DType("UInt8"))
                        {
                            if(globalEnablePipelining && streamShape["K"] != 8)
                                return FailCause::cmxConcatDecision;
                        }

                         //conv 9,11,13
                        if (op.getInputTensor()[0]->getShape() == mv::Shape({26,26,256,1}) &&
                            op.getInputTensor()[0]->getDType() == mv::DType("UInt8") &&
                            op.getInputTensor()[1]->getShape() == mv::Shape({3,3,256,512}) &&
                            op.getInputTensor()[1]->getDType() == mv::DType("UInt8") &&
                            op.getOutputTensor()[0]->getShape() == mv::Shape({26,26,512,1}) &&
                            op.getOutputTensor()[0]->getDType() == mv::DType("UInt8"))
                        {
                            if(globalEnablePipelining && streamShape["K"] != 4)
                                return FailCause::cmxConcatDecision;
                        }

                        if (op.getInputTensor()[0]->getShape() == mv::Shape({13,13,1024,1}) &&
                            op.getInputTensor()[0]->getDType() == mv::DType("UInt8") &&
                            op.getInputTensor()[1]->getShape() == mv::Shape({3,3,1024,1024}) &&
                            op.getInputTensor()[1]->getDType() == mv::DType("UInt8") &&
                            op.getOutputTensor()[0]->getShape() == mv::Shape({13,13,1024,1}) &&
                            op.getOutputTensor()[0]->getDType() == mv::DType("UInt8"))
                        {
                            if(globalEnablePipelining && streamShape["K"] != 8)
                                return FailCause::cmxConcatDecision;
                        }
                    }

                }

                // A proper decision on CMX concat for explicit concat or eltwise streaming cannot
                // be made with the information on hand. Will not optimize strategies for these.
                // In later pass, we will mark those that can fit in CMX
                // as CMX-able.
                //Note: Removing eltwise from here becuase of course hkswitch needs to be in cmx
                if(op.getOpType() == "Concat" && !spilling)
                    return FailCause::cmxConcatDecision;

                bool eltwiseParentSpilling = true;
                if(op.getOpType() == "Eltwise")
                    eltwiseParentSpilling = strategy["eltwiseParentSpilling"].get<bool>();

                if(opInCMX(op, strategy))
                {
                    size_t input, output, weights;
                    // in case initialization in memorySize fails
                    input = output = weights = 0;
                    std::tie(input, output, weights) = memorySize(op,
                                                                    totalClusters,enableChannelMajorConv,
                                                                    clustering,
                                                                    strategy["inputSparsity"],
                                                                    strategy["outputSparsity"],
                                                                    weightsSparsity,
                                                                    streamShape,
                                                                    requiresFakeActivationSparsity(op),
                                                                    spilling,
                                                                    true);
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

                bool isChanMajor = enableChannelMajorConv && op.supportsCMConv();
                //If spilling, HKSwitch makes no sense
                if( (spilling) && (clustering == "HKSwitch"))
                    return FailCause::SpillHKSwitch;

                if( isStreaming && (clustering == "HKSwitch"))
                    return FailCause::SpillHKSwitch;

                if( op.getOpType() == "Eltwise" && eltwiseParentSpilling && (clustering == "HKSwitch"))
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
                if((op.getOpType() == "Input" || op.getOpType() == "ImplicitInput") && (!spilling))
                    return FailCause::InputNotSpilled;

                if((op.getOpType() == "Output") && (!spilling))
                    return FailCause::OutputNotSpilled;

                //Special rules for Channel Major Convolutions
                //No need for SOHOverlapped input unless using channel major
                if(!enableChannelMajorConv && clustering == "SplitOverHOverlapped")
                    return FailCause::ChannelMjr1;


                if(isChanMajor && (strategy["inputSparsity"].get<bool>() || strategy["weightsSparsity"].get<bool>()))
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

                // For CM Conv, as after DW, we spill to DDR, SOH gets chosen for DW. For larger input sizes, (416,416) DW when spilled
                // seems to fail CRC. Without CM Conv enabled, StreamOverH gets chosen, so with CMConv, forcing No SOH for CRC pass
                // To do: Fix (416,416) DW only CRC fail on master
                if (op.getOpType() == "DepthwiseConv" && spilling && enableChannelMajorConv)
                {
                    if ((op.getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] > 302)
                            && (clustering == "SplitOverH"))
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

                if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isChanMajor &&
                    (streamShape["K"]  * streamShape["H"]) > 1 && spilling)
                    return FailCause::SpiltOverHWithStreamOverK;

                //NOTE: This is not a HACK!!! if an operation is assigned with streamOverH + SplitOverH
                //and we concatenate on cmx the data are going to have a strange format...so nothing can be done later, so spill...
                /*For example, let's think a convolution with splitOverH and streams = 2, the tensor will be splitted to
                 * 8 tiles, where every single tile is going to be assigned to a cluster with the round robin logic.
                 * That means that cluster 0, will have tiles0,4. Cluster1 will have tiles1,5 and so on...
                 * The way the data are splitted between clusters and the order of the tiles, do not allow us to concatenate
                 * in the initial order inside CMX*/
                if (clustering == "SplitOverH" &&
                    (streamShape["H"] > 1) && !spilling)
                    return FailCause::SpiltOverHWithStreamOverHInCMX;
                
                // This is intended to be a temporary workaround for ModelE, layer '97' & '113', which does work with SOH
                // It has not been root caused to the compiler or runtime but as of now the compiler logic seems OK
                if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isChanMajor && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 64 &&
                    op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 80 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 48 &&
                    op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 64 && op.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 80 &&
                    op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 48 && op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 3 &&
                    op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 3)
                    return FailCause::SpiltOverHForLayer97and113ModelE;

                // This is intended to be a temporary workaround for ACLnet, layer '79', which does work with SOH
                // It has not been root caused to the compiler or runtime but as of now the compiler logic seems OK
                if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isChanMajor && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 1 &&
                    op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 100 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 64 &&
                    op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 64 && op.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 100 &&
                    op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 64 && op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 3 &&
                    op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 3)
                    return FailCause::SpiltOverHForLayer79InACLNet;

                // This is intended to be a temporary workaround for FaceDetectionRetail, layer fire6/suqeeze1x1/WithoutBiases, which does work with SOH
                // It has not been root caused to the compiler or runtime but as of now the compiler logic seems OK
                if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isChanMajor && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 128 &&
                    op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 38 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 38 &&
                    op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 24 && op.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 38 &&
                    op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 38 && op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 1 &&
                    op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 1)
                    return FailCause::SpiltOverHForFaceDetectionRetail0004;

                //NOTE: we need a ticket for that failure, blob looks fine for streaming overH = 12 which means every stream assigned with 2 lines
                //last one with 1, and the last one seems not to function correctly
                if (op.hasAttr("floatPrecision"))
                {
                     if (op.getOpType() == "Conv" && op.get<bool>("floatPrecision") && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 1024 &&
                             op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 30 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 23)
                     {
                        auto outputTilesShape = tileSpatialOutputSize(op.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION], streamShape["H"]);
                        for (auto tileShape:outputTilesShape)
                            if (tileShape == 1)
                                return FailCause::SoftwareDeconvolutionSet;
                     }
                }
           
                // //temporarily disable the SplitOverHOverlapped for custom network kernel size 7x7 subtensors not correct
                // if (clustering == "SplitOverH" && op.getOpType() == "Conv" && isChanMajor && op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 3 &&
                //     op.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 72 && op.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 72 &&
                //     op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 7 && op.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 7)
                //     return FailCause::SplitOverHOverlappedWronglyComputed;
                return FailCause::Pass; //good strategy
            }

            // If output tensor is larger than CMX, even when SplitOverH (divided by totalClusters)
            // this op will always spill back to DDR
            bool willAlwaysSpill(mv::Op& op)
            {
                auto outputTensorSize = op.getOutputTensor(0)->computeTotalSize();
                outputTensorSize = (outputTensorSize / totalClusters);
                if(outputTensorSize > clusterMemory)
                    return true;

                return false;
            }

            bool requiresActivationSparsity(Op& op, std::string clustering)
            {
                if(requiresRealActivationSparsity(op, clustering))
                    return true;

                if(requiresCompilerActivationSparsity(op))
                    return true;

                return false;
            }

            bool requiresWeightsSparsity(Op& op)
            {
                // If Z-major Conv in Float precision then need to have weights Sparsity
                bool isCMConv = enableChannelMajorConv && op.supportsCMConv();

                if(op.getOpType() == "Conv" &&
                    op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") &&
                    !isCMConv && target == mv::Target::ma2490 && referenceDevice == "A0")
                        return true;

                return false;
            }

            // In these cases parent output sparsity does matter, but child input sparsity must be true
            bool requiresCompilerActivationSparsity(Op& op)
            {
                bool isCMConv = enableChannelMajorConv && op.supportsCMConv();

                if (op.getOpType() == "Conv" && !isCMConv
                       && (op.hasAttr("DilatedSubConv") && op.get<bool>("DilatedSubConv")))
                    return true;

                return false;
            }

            bool requiresRealActivationSparsity(Op& op, std::string clustering){
                //An fp16 Conv Z-major must have activation sparsity
                bool isCMConv = enableChannelMajorConv && op.supportsCMConv();

                if (op.isSparsityConsumer() &&
                    op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") &&
                    !isCMConv &&
                    target == mv::Target::ma2490 && referenceDevice == "A0")
                {
                    return true;
                }


                // Check for need for A0 SOH Sparsity workaround, (SOH conv with kernel > 1)
                // if needed, check memory constraints as for sparse tensor
                if (op.getOpType() == "Conv" ) {
                    if( clustering == "SplitOverH" &&
                        (op.getInputTensor(1)->getShape()[KERNEL_HEIGHT] > 1) &&
                        !isCMConv &&
                        target == mv::Target::ma2490 && referenceDevice == "A0")
                        {
                            return true;
                        }
                }

                return false;
            }

             //Channel major conv, pooling and depthwise will get fake sparsity, so need to check memory constraints as if real sparsity
            bool requiresFakeActivationSparsity(Op& op){
                if(enableChannelMajorConv && op.supportsCMConv() && target != mv::Target::ma3720)
                    return true;

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
                {
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

            // SM marks some nodes as forcedClustering, indicating that they may not
            // be checked against every real flow in the model. Need to ensure these are
            // compatible with any potential strategy that could be paired with them
            bool checkValidForForcedCompatiblity(Op& op,StrategySet& strategy)
            {
                //Not forced to take any particular strategies
                if(!(op.hasAttr("forceClustering") && op.get<bool>("forceClustering")))
                    return true;

                // In future, will add additional constraints re: activation sparsity here
                // But for now, we can always create compiler gen sparsity if needed by an op
                auto clusteringStrategy = strategy["clustering"].get<std::string>();
                if(clusteringStrategy == "Clustering" || clusteringStrategy == "SplitOverK")
                    return true;

                return false;
            }


            double transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {
                auto INF = inf_;
                auto parentClustering = parent["clustering"].get<std::string>();
                auto childClustering = child["clustering"].get<std::string>();
                auto parentOpType = parentOp.getOpType();
                auto childOpType = childOp.getOpType();
                auto parentSpilling = parent["spilling"].get<bool>();
                auto childSpilling = child["spilling"].get<bool>();
                auto parentStreamShape = parent["streaming"].get<mv::Shape>();
                auto childStreamShape = child["streaming"].get<mv::Shape>();
                auto parentOutputSparsity = parent["outputSparsity"].get<bool>();
                auto childInputSparsity = child["inputSparsity"].get<bool>();

                if( !checkValidForForcedCompatiblity(parentOp, parent) ||
                    !checkValidForForcedCompatiblity(childOp, child))
                        return INF;

                //TODO re-enable runtime sparsity in this case, see spatial_split_streaming L143 for issue
                //Dummy slice prevents runtime sparsity from being activated in sparsity pass
                if (parentOutputSparsity && childStreamShape["K"] > 1)
                    return INF;

                // Runtime sparsity with streamed consumers is not currently supported
                if (parentOutputSparsity && (childStreamShape["H"] * childStreamShape["C"] * childStreamShape["W"] * childStreamShape["B"]) > 1)
                    return INF;

                // Note: Wasted output sparsity, output sparsity in this context is runtime generated
                if (parentOutputSparsity && !childInputSparsity)
                    return INF;


                // Note: no sense to stream activations if both layers stay in CMX
                if(!parentSpilling && !childSpilling && (childStreamShape["H"] > 1))
                    return INF;

                if(childOpType == "Output" && !parentSpilling)
                    return INF;

                //Note: Synchronize across parallel branches so we can keep track of eltwise parent activation in ddr or cmx
                if(childOpType == "Eltwise" && ((child["eltwiseParentSpilling"].get<bool>() != parentSpilling)))
                    return INF;
                
                //Note: Retinaface-mobilenetv2 accuracy issue, no issue found in blob
                if(childOpType == "Eltwise" && childClustering == "HKSwitch" && parentOpType == "Conv" && 
                    childOp.getInputTensor(0UL)->getShape() == mv::Shape({80,80,64,1}) &&
                    parentOp.getInputTensor(0UL)->getShape() == mv::Shape({80,80,48,1}) &&
                    parentOp.getInputTensor(1UL)->getShape() == mv::Shape({1,1,48,64}))
                    return INF;

                if(violatesClusteringStrategyRules(parentOp, childOp, parent, child))
                {
                    log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() +
                                " INF caused by incompatible clustering strategies");
                    return INF;
                }

                // Note: when a parent is cmx concating, the input activation tensor will be in CMX, need to recheck memory fits
                bool childActivationStreaming = (childStreamShape["H"] * childStreamShape["C"] * childStreamShape["W"]) > 1 ? true : false;
                if(!parentSpilling && childActivationStreaming)
                {
                    size_t input, output, weights;
                    // in case initialization in memorySize fails
                    input = output = weights = 0;
                    std::tie(input, output, weights) = memorySize(childOp, totalClusters,enableChannelMajorConv,
                                                                childClustering,
                                                                child["inputSparsity"].get<bool>(),
                                                                child["outputSparsity"].get<bool>(),
                                                                child["weightsSparsity"].get<bool>(),
                                                                childStreamShape,
                                                                requiresFakeActivationSparsity(childOp),
                                                                childSpilling, parentSpilling);
                    if(input + output + weights >= clusterMemory)
                        return INF;
                }

                if(enableChannelMajorConv && target != mv::Target::ma3720)
                {
                    if( violatesChannelMajorRules(parentOp, childOp, parent, child) )
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                            + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by CM Conv rules");
                        return INF;
                    }
                }

                if(!enableChannelMajorConv && parentOpType == "Input" &&
                        childOpType == "Conv" &&
                        childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
                {
                    if (parentClustering == "SplitOverHOverlapped")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                            + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by no CM Conv but Input has SOHOverlapped");
                        return INF;
                    }

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


                auto isChildChanMajor = childOpType == "Conv" && enableChannelMajorConv &&
                                        childOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16;

                if( childOpType == "Conv")
                {
                    auto childWeightsShape = childOp.getInputTensor(1)->getShape();
                    if( !isChildChanMajor &&
                        childClustering == "SplitOverH" &&
                        childWeightsShape[mv::KERNEL_HEIGHT] > 1 &&
                        childInputSparsity &&
                        parentOutputSparsity ) // only allow for compiler sparsity, disallow for runtime sparsity
                    {
                        // This should also be solveable with fake compiler provided sparsity
                        // there may very well be cases where sparsity if enforced, but due to this
                        // limitation proper sparsity is not a choice since cluster boundary sparse map
                        // reads will fail due to misalignment
                        // Fake sparsity will provide all 1's sparse map so that probem is solved
                        // from the starts

                        // Sparse map has to be contiguously alligned at 16 bytes
                        // for first (N - 1) clusters
                        auto outputTensorShape = parentOp.getOutputTensor(0)->getShape();
                        unsigned int W = outputTensorShape[IO_WIDTH_DIMENSION];
                        unsigned int H = outputTensorShape[IO_HEIGHT_DIMENSION];
                        unsigned int C = outputTensorShape[IO_CHANNEL_DIMENSION];
                        unsigned dy = std::ceil(static_cast<double>(H) / totalClusters);

                        // this limitation that sparse map should be 16 bytes aligned in each subtensors,
                        // ONLY applies when DPUs are reading data from neighbor clusters
                        // Each subtensor should be aligned to 16 byte boundaries. For SM we have 1 bit per elem,
                        // so divide tensor by 8 get size in bytes
                        // (sparse idu for SOH ZM CONV kernel h > 1)
                        if( (W*dy*C/8)%128 != 0 )
                        {
                            log(mv::Logger::MessageType::Debug, child["name"].toString()+"_"+child["id"].toString() + " INF caused by incorrect SOH");
                            return INF;
                        }
                    }
                }

                if (childOp.hasAttr("floatPrecision"))
                {
                    //NOTE: On floating point network, Mobilenet there is a case that if we have runtime sparsity with
                    //SOH going to an eltwise the eltwise fails, so the step is to use compiler sparsity on that point
                    if (childOpType == "Eltwise" && (parentOpType == "Conv" || parentOpType == "Eltwise")
                        && childOp.get<bool>("floatPrecision") &&
                        (parentOp.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 28 &&
                           parentOp.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 28 &&
                           parentOp.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 32) &&
                        parentOutputSparsity && childInputSparsity && (childClustering == "SplitOverH" ||
                            childClustering == "HKSwitch"))
                        return INF;
                }

                //Note: Input clustering strategy should match first layer, if it is Z-major
                if(parentOpType == "Input" && !isChildChanMajor)
                {
                    if(parentClustering != childClustering)
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by input not matching first layer");
                        return INF;
                    }
                }
                //Note: these are full cost, across all streams, used in serial computation
                //Also used to calculate sparsity overhead vs. speedup in child
                auto pFullComp = computeTime(parentOp,parent);
                auto cFullComp = computeTime(childOp,child);
                auto pFullDma = dmaTime(parentOp, parent);
                auto cFullDma = dmaTime(childOp, child, parentSpilling);

                auto pOutDma = averageOutputDmaTime(parentOp, parent);
                auto pStreams = parentStreamShape["H"] * parentStreamShape["K"];
                auto pLastComp = pFullComp / pStreams;

                //Note: these are cost per stream, used when pipelining or prefetching
                auto cInDma = averageInputDmaTime(childOp, child, parentSpilling);
                auto cWeightDma = averageWeightsDmaTime(childOp, child);
                auto cOutDma = averageOutputDmaTime(childOp, child);

                double sparsityCost = 0;

                // For performance in YoloV2 and other networks with large input which don't allow SOH to stay in CMX
                // at the start of the network, here we preference being clustering or SOK rather than SOH.
                // This will allow us to use the AddActivationStreaming pass to speed up these layers.
                // Needed because SOH can only stream over K, and that pass only speeds up streaming over H...
                if (parentOpType == "Input" && isChildChanMajor && willAlwaysSpill(childOp) &&
                    !(childClustering == "Clustering" || childClustering == "SplitOverK")) {
                    cFullDma = cFullDma * 10;
                }

                // Case in which child input sparsity will be provided by compiler
                // Compiler provided sparsity is a dummy sparsity (all 1's sparse map)
                // so no real sparse acceleration will pe provided, only sparse decoding overhead
                auto sparsityOverhead = childOp.getInputTensor(0)->isFloatingPointType() ?
                    0.0625 : 0.125;
                if (!parentOutputSparsity && childInputSparsity)
                    sparsityCost = cFullComp * sparsityOverhead;

                //TODO capture sparse speedup potential here if childInputSparsity && parentOutputSparsity both true
                // but probably only enable when activation sparsity is requested from CD. Otherwise, discourage?
                if(childInputSparsity && !requiresActivationSparsity(childOp, childClustering))
                    sparsityCost = sparsityCost + (cFullComp * 0.01); // penalize not needed sparsity

                // TODO capture this somehow in the larger cost idea below
                // but for now a simple speedup factor will have to do
                double finalLayerStreamingBoost = 0;
                if(parentOpType == "Conv" && childOpType == "Output")
                {
                    auto streams = parentStreamShape["K"];
                    if(streams > 1 && parentClustering == "SplitOverK")
                        finalLayerStreamingBoost = streams * ((pFullDma + pFullComp) * 0.1); // streaming over more K is better
                }

                // For performance in YoloV2 and other networks with large input which don't allow SOH to stay in CMX
                // at the start of the network, here we preference being clustering or SOK rather than SOH. 
                // This will allow us to use the AddActivationStreaming pass to speed up these layers.
                // Needed because SOH can only stream over K, and that pass only speeds up streaming over H...
                if(parentOpType == "Input" && isChildChanMajor && willAlwaysSpill(childOp) &&
                    !(childClustering == "Clustering" || childClustering == "SplitOverK"))
                {
                    cFullDma = cFullDma * 10;
                }

                double heuristics = sparsityCost - finalLayerStreamingBoost;

                auto pipelineable = isPipeliningPossible(childOp, child, parent["spilling"].get<bool>());
                auto prefetchable = isPrefetchPossible(parentOp, childOp, parent, child);

                auto kStreams = childStreamShape["K"];

                auto compPerStream = ((double) cFullComp / kStreams);
                auto prefetch = std::min(pLastComp, cWeightDma);
                auto pipeline = std::max((cWeightDma+cOutDma), compPerStream);

                auto prepipe_cost = (cWeightDma - prefetch) + (kStreams-1)*pipeline + cOutDma;
                //Pipeline means overlap K stream compute with weights read for next K stream, assume input in CMX
                auto pipe_cost = cWeightDma + (kStreams-1)*pipeline + cOutDma;
                //Prefetch means bring in one weights slice while still computing the parent, assume input in CMX
                auto pre_cost = cWeightDma - prefetch + (kStreams-1)*cWeightDma + cFullComp + (kStreams * cOutDma);
                auto base_cost = cFullDma + cFullComp;

                double cost = base_cost;

                // std::cout << "Strategy for " << parent["id"].toString() << " --> " << child["id"].toString() << std::endl <<
                //     "    " << prepipe_cost << " : " << pipe_cost << " : " << pre_cost << " : " << base_cost << std::endl;

                if(pipelineable && prefetchable)
                {
                    cost = prepipe_cost;
                    // std::cout << "    chose prepipe: " << prepipe_cost << std::endl;
                }
                else if(pipelineable)
                {
                    cost = pipe_cost;
                    // std::cout << "    chose pipe: " << pipe_cost << std::endl;
                }
                else if(prefetchable)
                {
                    cost = pre_cost;
                    // std::cout << "    chose pre: " << pre_cost << std::endl;
                }
                else
                {
                    // std::cout << "    chose base: " << base_cost << std::endl;
                }

                // Note: for performance, here we ensure if that MC strategies are preferenced in order
                // SOH, HKSwitch, SOK, Clustering. Required in order to remove parent cost from calculation.
                if(parentClustering == "SplitOverH" && !parentSpilling)
                        cost = cost * 0.95;
                if((childClustering == "SplitOverH" || childClustering == "HKSwitch") && !childSpilling) 
                     cost = cost * 0.95;   
                if(parentClustering == "Clustering" || childClustering == "Clustering")
                    cost = cost * 1.1;

                cost = cost + heuristics;
                // std::cout << " returning cost " << cost << std::endl;

                return cost;
        }

            bool violatesClusteringStrategyRules(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {
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

                // Note: capturing rules from master, TODO revisit
                std::vector<std::pair<std::string, std::string>>incompatibleStrategiesChildConcat =
                {
                    // Parent op, child concat
                    {"SplitOverK", "SplitOverH"},
                    {"HKSwitch", "SplitOverH"},
                    {"Clustering", "SplitOverH"},
                    {"SplitOverH", "SplitOverK"},
                    {"SplitOverH", "Clustering"} //clustering concats follow same rule as SOK concats
                };

                std::vector<std::pair<std::string, std::string>>incompatibleStrategiesParentConcat =
                {
                    // parent concat, child op
                    {"SplitOverH", "SplitOverK"},
                    {"SplitOverK", "SplitOverH"},
                    {"SplitOverK", "HKSwitch"},
                    {"Clustering", "SplitOverH"},//clustering concats follow same rule as SOK concats
                    {"Clustering", "HKSwitch"}
                };

                auto parentClustering = parent["clustering"].get<std::string>();
                auto childClustering = child["clustering"].get<std::string>();
                auto parentOpType = parentOp.getOpType();
                auto childOpType = childOp.getOpType();
                auto parentSpilling = parent["spilling"].get<bool>();
                auto childSpilling = child["spilling"].get<bool>();


                if (parentOpType == "Concat" && childOpType == "Concat")
                {
                    //NOTE: it is not possible to have a parent that is spilling, going to ddr and
                    //the child not spilling there is no concatenation through ddr->cmx concat
                    //when the child is an explicit concat, when it is 2 streaming ops that could happen
                    if (parentSpilling && !childSpilling)
                        return true;


                    //TODO should probably enforce they have same strategy here too..
                }
                else if (parentOpType == "Concat")
                {
                    if (childOpType != "Output")
                    {
                        std::pair<std::string, std::string> possibleCombination(parentClustering, childClustering);
                        for (auto restrictedCombination: incompatibleStrategiesParentConcat)
                        {
                            if (possibleCombination == restrictedCombination)
                            {
                                return true;
                            }
                        }
                    }
                    if(!parentSpilling)
                    {
                        //NOTE: It is impossible to concatenate inside cmx if the input tensors are not aligned // 16
                        for (auto inputTensor : parentOp.getInputTensor())
                        {
                            if (inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION] % 16)
                                return true;
                        }
                    }
                }
                else if (childOpType == "Concat" )
                {
                    //NOTE: This is not correct for the spilling concats, as there the strategies can change with the dma...
                    std::pair<std::string, std::string> possibleCombination(parentClustering, childClustering);
                    for (auto restrictedCombination: incompatibleStrategiesChildConcat)
                    {
                        if (possibleCombination == restrictedCombination)
                        {
                            return true;
                        }
                    }
                    if(!childSpilling)
                    {
                        //NOTE: It is impossible to concatenate inside cmx if the input tensors are not aligned // 16
                        for (auto inputTensor : childOp.getInputTensor())
                        {
                            if (inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION] % 16)
                                return true;
                        }
                    }
                }
                //NOTE: If you Spill a parent a child can be everything...the only thing
                //that has no sense if is your parent is spilling to be HKSwitch as
                //this strategy exists in order to reverse strategies in CMX
                else if (parentSpilling)
                {
                    if (childClustering == "HKSwitch")
                        return true;

                    bool modelAWA = false;
                    if (childOp.getOpType() == "Conv" && childOp.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 80 &&
                        childOp.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 44 && childOp.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 44 &&
                        childOp.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 72 && childOp.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 22 &&
                        childOp.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 22 && childOp.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 3 &&
                        childOp.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 3)
                    {
                        modelAWA = true;
                    }
                    if (childOp.getOpType() == "Conv" && childOp.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 48 &&
                        childOp.getInputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 22 && childOp.getInputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 22 &&
                        childOp.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] == 48 && childOp.getOutputTensor()[0]->getShape()[mv::IO_WIDTH_DIMENSION] == 22 &&
                        childOp.getOutputTensor()[0]->getShape()[mv::IO_HEIGHT_DIMENSION] == 22 && childOp.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT] == 3 &&
                        childOp.getInputTensor(1)->getShape()[mv::KERNEL_WIDTH] == 3)
                    {
                        modelAWA = true;
                    }

                    //NOTE: For now I disable parent spill SOH->child (Clustering, K) for Z major convs
                    //Workaround added to enable SplitOverH for ModelA - Perf
                    if (parentClustering == "SplitOverH" && ((childClustering == "Clustering" && childOpType !=  "Output") ||
                                                              childClustering == "SplitOverK") && !modelAWA)

                    {
                        if (!(enableChannelMajorConv &&
                            ((parentOpType == "Conv" &&
                            parentOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16) ||
                            (childOpType == "Conv" &&
                            childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16))) )

                            return true;

                    }
                }
                else
                {
                    std::pair<std::string, std::string> possibleCombination(parentClustering, childClustering);
                    for (auto restrictedCombination: incompatibleStrategiesWithOutSpilling)
                    {
                        if (possibleCombination == restrictedCombination)
                        {
                            return true;
                        }
                    }
                }

                //Note: last op should not be HKSwitch
                if (childOpType == "Output")
                {
                    if (parentClustering == "HKSwitch")
                        return true;
                }

                return false;
            }

            bool violatesChannelMajorRules(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {
                bool spillForCM = false;
                auto parentClustering = parent["clustering"].get<std::string>();
                auto childClustering = child["clustering"].get<std::string>();
                auto parentOpType = parentOp.getOpType();
                auto childOpType = childOp.getOpType();
                auto parentSpilling = parent["spilling"].get<bool>();


                auto isChildChanMajor = childOpType == "Conv" &&
                                        childOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16;
                auto isParentChanMajor = parentOpType == "Conv" &&
                                        parentOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16;

                if (isParentChanMajor || isChildChanMajor)
                   spillForCM = needForceSpillingForCM(parentOp, childOp, parentClustering, childClustering);

                if (spillForCM && !parentSpilling)
                    return true;

                if(isChildChanMajor)
                {
                    //Note: If SOHOverlapped input requires SOH CMconv, and vice versa
                    if(childClustering == "SplitOverH" &&
                        (parentOpType == "Input" && parentClustering != "SplitOverHOverlapped"))
                        return true;
                    if(parentClustering == "SplitOverHOverlapped" && childClustering != "SplitOverH")
                        return true;
                }

                // TODO
                // forgive me for this hack. Multiple input topologies are not qualifed for CM Conv, so this hack
                if (parentOpType == "ImplicitInput" || childOpType == "ImplicitInput")
                {
                    if (parentClustering == "SplitOverHOverlapped")
                        return true;
                    if (childClustering == "SplitOverHOverlapped")
                        return true;
                }

                return false;
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

            double dmaTime(Op& op,StrategySet& strategySet, bool parentSpilling = false)
            {
                auto opType = op.getOpType();

                //Each layer calculates the cost to move it's weights (if they exist), and it's output tensor
                double weightsTime = 0;
                double outputTime = 0;
                //Child will calculate cost for input tensor
                double inputTime = 0;

                auto streamShape = strategySet["streaming"].get<mv::Shape>();
                auto spilling = strategySet["spilling"].get<bool>();

                // Each DMA cost is modelled as latency + size*transfer_rate
                if(opType == "Conv" || opType == "DepthwiseConv")
                {
                    auto weightsSize = op.getInputTensor(1)->computeTotalSize(); // approx for now
                    size_t stream = 1;

                    if(opType == "Conv")
                    {
                        stream = streamShape["K"];
                    }
                    else // Depthwise
                    {
                        stream = streamShape["C"];
                    }

                    weightsTime = stream*(DMA_LATENCY + (((double)weightsSize/stream) / DMA_BANDWIDTH));
                }

                // If parent tensor stays in CMX, no further cost. Otherwise, calculate bring the input tensor back into CMX
                if(parentSpilling)
                {
                    auto inputSize = op.getInputTensor(0)->computeTotalSize();
                    size_t stream = 1;
                    if(streamShape["H"] > 1)
                        stream = streamShape["H"];
                    else if(streamShape["C"] > 1)
                        stream = streamShape["C"];
                    else if(streamShape["B"] > 1) //batch splits input, won't nested with anything
                        stream = streamShape["B"];

                    inputTime = stream* (DMA_LATENCY + (((double)inputSize/stream) / DMA_BANDWIDTH) );
                }

                // If output tensor stays in CMX, no further dma cost. Otherwise, calculate cost to spill to DDR
                if(spilling)
                {
                    //Each stream is written to DDR. Output activation tensor might be streamed over H or K
                    size_t outputSize;
                    if(opType != "Output")
                        outputSize = op.getOutputTensor(0)->computeTotalSize();
                    else
                        outputSize = op.getInputTensor(0)->computeTotalSize();

                    size_t stream = 1;

                    if(streamShape["H"] > 1 && streamShape["K"] > 1) // Nested streaming!
                        stream = streamShape["H"] * streamShape["K"];
                    else if(streamShape["H"] > 1)
                        stream = streamShape["H"];
                    else if(streamShape["K"] > 1)
                        stream = streamShape["K"];
                    else if(streamShape["B"] > 1)
                        stream = streamShape["B"];

                    outputTime = stream*(DMA_LATENCY + (((double)outputSize/stream) / DMA_BANDWIDTH));
                }
                return (inputTime + weightsTime + outputTime) * 1000000; //return in us
            }

            double averageInputDmaTime(Op& op,StrategySet& strategySet, bool parentSpilling = false)
            {
                auto streamShape = strategySet["streaming"].get<mv::Shape>();
                // If parent tensor stays in CMX, no further cost. Otherwise, calculate bring the input tensor back into CMX
                if(parentSpilling)
                {
                    auto inputSize = op.getInputTensor(0)->computeTotalSize();
                    size_t stream = 1;
                    if(streamShape["H"] > 1)
                        stream = streamShape["H"];
                    else if(streamShape["C"] > 1)
                        stream = streamShape["C"];
                    else if(streamShape["B"] > 1) //batch splits input, won't nested with anything
                        stream = streamShape["B"];

                   return (DMA_LATENCY + (((double)inputSize/stream) / DMA_BANDWIDTH) ) * 1000000; //return in us;
                }

                return 0;
            }

            double averageWeightsDmaTime(Op& op,StrategySet& strategySet)
            {
                auto streamShape = strategySet["streaming"].get<mv::Shape>();
                // Each DMA cost is modelled as latency + size*transfer_rate
                if(op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
                {
                    auto weightsSize = op.getInputTensor(1)->computeTotalSize(); // approx for now
                    size_t stream = 1;

                    if(op.getOpType() == "Conv")
                    {
                        stream = streamShape["K"];
                    }
                    else // Depthwise
                    {
                        stream = streamShape["C"];
                    }

                    return (DMA_LATENCY + (((double)weightsSize/stream) / DMA_BANDWIDTH)) * 1000000; //return in us;
                }

                return 0;
            }

            double averageOutputDmaTime(Op& op,StrategySet& strategySet)
            {
                auto streamShape = strategySet["streaming"].get<mv::Shape>();
                auto spilling = strategySet["spilling"].get<bool>();
                // If output tensor stays in CMX, no further dma cost. Otherwise, calculate cost to spill to DDR
                if(spilling)
                {
                    //Each stream is written to DDR. Output activation tensor might be streamed over H or K
                    size_t outputSize;
                    if(op.getOpType() != "Output")
                        outputSize = op.getOutputTensor(0)->computeTotalSize();
                    else
                        outputSize = op.getInputTensor(0)->computeTotalSize();

                    size_t stream = 1;

                    if(streamShape["H"] > 1 && streamShape["K"] > 1) // Nested streaming!
                        stream = streamShape["H"] * streamShape["K"];
                    else if(streamShape["H"] > 1)
                        stream = streamShape["H"];
                    else if(streamShape["K"] > 1)
                        stream = streamShape["K"];
                    else if(streamShape["B"] > 1)
                        stream = streamShape["B"];

                    return (DMA_LATENCY + (((double)outputSize/stream) / DMA_BANDWIDTH)) * 1000000; //return in us;
                }
                return 0;
            }

            double computeTime(Op& op,StrategySet& strategySet)
            {
                auto opType = op.getOpType();
                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");
                if( !op.isHardwarizable() || software)
                    return 0;

                double OPS = 7.168 * 1099511627776; //tops -> o/s
                if(op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16"))
                    OPS = 1.792 * 1099511627776;

                auto inputShape = op.getInputTensor(0)->getShape();
                auto outputShape = op.getOutputTensor(0)->getShape();
                auto clustering = strategySet["clustering"].get<std::string>();
                auto streaming = strategySet["streaming"].get<Shape>();

                size_t baseKernelCost;

                if ((opType == "Eltwise" && !(software)) || (opType == "Concat"))
                {
                    baseKernelCost = 1;
                }
                else if (opType == "MaxPool")
                {
                    auto kernel = op.get<std::array<unsigned short,2>>("kSize");
                    baseKernelCost = kernel[0] * kernel[1];
                }
                else if ((opType == "DepthwiseConv") || (opType == "Conv"))
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
                    throw LogicError(*this, "Invalid operation type " + opType);
                }

                bool channelAccum = (opType == "Conv") ? true : false;
                if (channelAccum)
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    baseKernelCost *= weightsShape[KERNEL_INPUT_CHANNELS];
                }

                size_t totalStreams = 1;
                for (size_t i = 0; i < streaming.ndims(); i++)
                    totalStreams *= streaming[i];

                auto isiDecay = 1.0;
                if (clustering == "SplitOverK" || clustering == "HKSwitch")
                    isiDecay = 0.1 *
                        std::max((size_t) 1, dpuPerCluster - 1) *
                        std::max((size_t) 1, totalClusters - 1);

                //TODO to capture fully, should calculate the cost to bring data to compute from cmx and output back to cmx
                auto inputSize = inputShape.totalSize();
                auto outputSize = outputShape.totalSize();
                if(opType == "Conv" || opType == "DepthwiseConv" || opType == "Eltwise")
                    inputSize += op.getInputTensor(1)->getShape().totalSize();
                double readIn = (totalStreams * LATENCY_CMX) + (inputSize / BANDWIDTH_CMX);
                auto totalToCompute = (outputSize / totalStreams);
                double readOut = (totalStreams * LATENCY_CMX) + (outputSize / BANDWIDTH_CMX);

                // Multiclustering allows parallelism
                if(clustering != "Clustering")
                    totalToCompute = totalToCompute / totalClusters;

                totalToCompute = totalToCompute * isiDecay;

                double compTime = ((totalToCompute * baseKernelCost) / OPS);

                return  (totalStreams * (readIn + readOut + compTime)) * 1000000; //return in us
            }

            bool isPipeliningPossible(mv::Op& op, StrategySet& strategy, bool parentSpilling)
            {
                if(!globalEnablePipelining)
                    return false;

                // Is this op type enabled for pipelining? For now, default turned on for just conv and dw
                if(!createStrategyFromBool(op, "pipelining"))
                    return false;

                auto stream = strategy["streaming"].get<Shape>();
                auto clustering = strategy["clustering"].get<std::string>();
                auto inputSparsity = strategy["inputSparsity"].get<bool>();
                auto outputSparsity = strategy["outputSparsity"].get<bool>();
                auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
                auto spilling = strategy["spilling"].get<bool>();

                if(parentSpilling)
                    return false;

                if(clustering == "SplitOverH" || clustering == "HKSwitch")
                    return false;

                //Note: for now, only support pipelining over H and K
                //TODO renable for H
                if((stream["B"] * stream["C"] * stream["H"]) > 1)
                    return false;

                // No sense making nested streaming any worse than it is
                if(stream["H"] > 1 && stream["K"] > 1)
                    return false;

                size_t input, output, weights;
                // in case initialization in memorySize fails
                input = output = weights = 0;
                std::tie(input, output, weights) = memorySize(op,
                                                                totalClusters,enableChannelMajorConv,
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
                else if(stream["H"] > 1)
                {
                    if(!spilling && !parentSpilling) //Activations all in CMX, nothing to pipeline
                        return false;

                    double memReq = 0;

                    //Note: memory size function is smart enough to take care of input/output size relative to spilling
                    if(parentSpilling) //streamed input, either full or streamed output
                        memReq = 2*input + weights + output;
                    else //full input, streamed output
                        memReq = input + weights + 2*output;

                    if(memReq < clusterMemory)
                    {
                        return true;
                    }
                }
                return false; // without streaming, there is no pipelining
            }

            bool isPrefetchPossible(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {
                if(!globalEnablePrefetching)
                    return false;

                //Note: No sense in prefetching weights if we just have to wait for activations
                if(parent["spilling"].get<bool>())
                    return false;

                //TODO this should work for depthwise too, but for now just check convs
                if(childOp.getOpType() != "Conv") // Need something to prefetch, i.e. weights!
                    return false;

                // Note: No sense prefetching weights if we are nested streaming
                mv::Shape& childStreams = child["streaming"].get<mv::Shape>();

                if(childStreams["H"] > 1 && childStreams["K"] > 1)
                    return false;

                //Prefetch is possible if the previous op leaves enough CMX open to fit a slice of the childs weights
                size_t childWeight = alignedWeightsSize(childOp.getInputTensor(1),
                                                        childStreams,
                                                        child["clustering"].get<std::string>(), totalClusters);

                size_t parentInput, parentOutput, parentWeight;
                // in case initialization in memorySize fails
                parentInput = parentOutput = parentWeight = 0;
                std::tie(parentInput, parentOutput, parentWeight) = memorySize( parentOp,
                                                                                totalClusters,
                                                                                enableChannelMajorConv,
                                                                                parent["clustering"].get<std::string>(),
                                                                                parent["inputSparsity"].get<bool>(),
                                                                                parent["outputSparsity"].get<bool>(),
                                                                                parent["weightsSparsity"].get<bool>(),
                                                                                parent["streaming"].get<mv::Shape>(),
                                                                                requiresFakeActivationSparsity(parentOp));

                if(parentInput+parentOutput+parentWeight + childWeight < clusterMemory)
                    return true;

                return false;
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
