//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

// clang-format off

#include "ngraph_mcm_frontend/frontend.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include "ngraph_mcm_frontend/passes/add_io_convert_ops.hpp"
#include "ngraph_mcm_frontend/passes/convert_to_mcm_conv.hpp"
#include "ngraph_mcm_frontend/passes/convert_to_mcm_model.hpp"
#include "ngraph_mcm_frontend/passes/collapse_concat_chain.hpp"
#include "ngraph_mcm_frontend/passes/convert_to_mcm_fc.hpp"
#include "ngraph_mcm_frontend/passes/merge_TopK_convert.hpp"
#include "ngraph_mcm_frontend/passes/replace_add_with_eltwise.hpp"
#include "ngraph_mcm_frontend/passes/convert_MVN6_to_MVN1.hpp"
#include "ngraph_mcm_frontend/passes/replace_scaleshift_with_mcm_scale.hpp"
#include "ngraph_mcm_frontend/passes/align_eltwise_scales.hpp"
#include "ngraph_mcm_frontend/passes/align_concat_scales.hpp"
#include "ngraph_mcm_frontend/passes/fuse_scaleshift.hpp"
#include "ngraph_mcm_frontend/passes/fuse_padding.hpp"
#include "ngraph_mcm_frontend/passes/convert_extract_image_patches_to_reorg_vpu.hpp"
#include "ngraph_mcm_frontend/passes/broadcast_eltwise_inputs.hpp"
#include "ngraph_mcm_frontend/passes/replace_onnx_pattern_to_reorg.hpp"
#include "ngraph_mcm_frontend/passes/fuse_scale_in_previous_weights_fq.hpp"
#include "ngraph_mcm_frontend/passes/insert_maxpool.hpp"
#include "ngraph_mcm_frontend/passes/replace_shuffle.hpp"
#include "ngraph_mcm_frontend/passes/handle_3d_transpose.hpp"
#include <ngraph_mcm_frontend/passes/propagate_fq.hpp>
#include <ngraph_mcm_frontend/passes/align_scales.hpp>
#include <ngraph_mcm_frontend/passes/detect_input_fq.hpp>

#include "vpux/utils/core/error.hpp"

#include <file_utils.h>
#include <vpu/utils/logger.hpp>
#include <device_helpers.hpp>

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/init_node_info.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp>

#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/utils/utils.hpp>

#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>

#include <ngraph/op/hswish.hpp>

#include <include/mcm/compiler/compilation_unit.hpp>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <transformations/serialize.hpp>

std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReference(
    const std::string& tensorName,
    const InferenceEngine::TensorDesc& tensorInfo);

namespace {
    std::map<std::string, std::string> MapInputOutputInfoToNgraphOps(const std::shared_ptr<ngraph::Function>& func,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo) {
        // Due to historical reasons, CNNNetwork::getOutputsInfo() does not match excatly
        // to ngraph::op::v0::Result::get_friendly_name(), actual get_friendly_name() may be have arbitary different.
        // Instead getOutputsInfo() returns names of nodes, who produces input to ngraph::op::v0::Result,
        // This expected to be fixed in 2021.2
        // Below ngraph function is changed and Result producers are replaced, making impossible to match.
        // Therefore Ngraph Results must be mached to OutputsInfo Here.
        // There is no API to extract actual mapping from CNNNetwork
        // See how results are converted to outputInfo in convert_function_to_cnn_network.cpp
        std::map<std::string, std::string> ioMap;
        // TBD Do we need inputs too?
        for (const auto& inputInfo : inputsInfo) {
            bool isFound = false;
            for (auto&& paramOp : func->get_parameters()) {
                IE_ASSERT(1 == paramOp->get_output_size());
                auto name = paramOp->output(0).get_tensor().get_name();
                if (name.empty())
                    name = ngraph::op::util::create_ie_output_name(paramOp->output(0));
                if (name == inputInfo.first) {
                    ioMap[inputInfo.first] = paramOp->get_friendly_name();
                    isFound = true;
                    break;
                }
            }
            if (!isFound)
                IE_THROW() << "Input not found: " << inputInfo.first;
        }

        for (const auto& outputInfo : outputsInfo) {
            bool isFound = false;
            for (auto&& resultOp : func->get_results()) {
                IE_ASSERT(1 == resultOp->get_input_size());
                const auto &input = resultOp->input_value(0);
                auto name = input.get_tensor().get_name();
                if (name.empty())
                    name = ngraph::op::util::create_ie_output_name(input);
                if (name == outputInfo.first) {
                    ioMap[outputInfo.first] = resultOp->get_friendly_name();
                    isFound = true;
                    break;
                }
            }
            if (!isFound)
                IE_THROW() << "Ouput not found: " << outputInfo.first;
        }

        return ioMap;
    }
}

std::unique_ptr<mv::CompilationUnit> createCompilationUnit(
    const std::string& netName,
    const ie::InputsDataMap& inputsInfo,
    const ie::OutputsDataMap& outputsInfo,
    const vpu::MCMConfig& config,
    std::shared_ptr<vpu::Logger> log,
    std::string & errMsg)
{
    auto mcmCompiler = std::unique_ptr<mv::CompilationUnit>(new mv::CompilationUnit(netName));
    {
        log->debug("Configure MCM Compiler");
        VPU_LOGGER_SECTION(log);

        std::string compDescName;
        std::string targetDescName;

        if (config.mcmTargetDesciptor() != "release_kmb") {
            targetDescName = !config.mcmTargetDesciptor().empty() ?
                              config.mcmTargetDesciptor() : "release_kmb";
        }
        else {
            auto platform = config.platform();
            if (platform == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
                const auto platformName = utils::getPlatformNameByDeviceName(config.deviceId());
                const auto targetPos = platformName.rfind("_EMU");
                if (targetPos == std::string::npos) {
                    errMsg = "Error: Emulator target platform is not defined.";
                    return nullptr;
                }
                const auto targetName = platformName.substr(0, targetPos);
                platform = utils::getPlatformByDeviceName(targetName);
            }

            switch (platform) {
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800:
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900: {
                    targetDescName = "release_thb";
                    break;
                }
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720: {
                    targetDescName = "release_mtl";
                    break;
                }
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400_A0:
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400:
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700:
                    targetDescName = "release_kmb";
                    break;
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO:
                default:
                    errMsg = "Error: VPUXPlatform is not defined.";
                    return nullptr;
            }
        }

        if (config.mcmCompilationDesciptor() != "release_kmb") {
            compDescName = !config.mcmCompilationDesciptor().empty() ?
                            config.mcmCompilationDesciptor() : "release_kmb";
        }
        else {
            switch (config.platform()) {
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400:
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700: {
                    compDescName = "release_kmb_B0";
                    break;
                }
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720: {
                    compDescName = "release_mtl";
                    break;
                }
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400_A0:
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800:
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900:
                    compDescName = "release_kmb";
                    break;
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR:
                    compDescName = "emulator_kmb_SC-Prefetch1";
                    break;
                case InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO:
                default:
                    errMsg = "Error: VPUXPlatform is not defined.";
                    return nullptr;
            }
        }

        const auto targetPath = ie::getIELibraryPath() + "/" + config.mcmTargetDesciptorPath() + "/" + targetDescName + ".json";
        const auto compDescPath = ie::getIELibraryPath() + "/" + config.mcmCompilationDesciptorPath() + "/" + compDescName + ".json";

        IE_ASSERT(mcmCompiler->loadTargetDescriptor(targetPath));
        IE_ASSERT(mcmCompiler->loadCompilationDescriptor(compDescPath));

        auto& mcmCompDesc = mcmCompiler->compilationDescriptor();

        mcmCompDesc.setPassArg("GlobalConfigParams", "verbose", cvtLogLevelToMCM(config.mcmLogLevel()));
        mcmCompDesc.setPassArg("GlobalConfigParams", "RemovePermuteNoOp", config.removePermuteNoOp());
        mcmCompDesc.setPassArg("GlobalConfigParams", "enable_channel_major_conv",
                                       std::find_if(inputsInfo.begin(), inputsInfo.end(),
                                                        [](const std::pair<std::string, ie::InputInfo::Ptr>& input) {
                    return input.second->getLayout() != InferenceEngine::Layout::NCHW &&
                           input.second->getLayout() != InferenceEngine::Layout::CHW;
                }) == inputsInfo.end());
        mcmCompDesc.setPassArg("GlobalConfigParams", "DeviceRevision",
                               std::string(MVCNN::EnumNameTargetDeviceRevision(getDeviceRevision(config.platform()))));


        if (config.referenceMode()) {
            mcmCompDesc.setPassArg("GlobalConfigParams", "ReferenceMode", true);
        }

        if (config.performanceCounting()) {
            mcmCompDesc.setPassArg("GlobalConfigParams", "PerformanceCounting", true);
        }

        std::function<void(MVCNN::GraphFileT&)> metaInfoSerializer =
            [&inputsInfo, &outputsInfo, &netName](MVCNN::GraphFileT& graphFileInstance) {
            if (graphFileInstance.header == nullptr) {
                IE_THROW() << "metaInfoSerializer: graph file header points to null";
            }

            for (const auto& inInfo : inputsInfo) {
                graphFileInstance.header->in_tensor_desc.push_back(
                    buildTensorReference(inInfo.first, inInfo.second->getTensorDesc()));
            }

            for (const auto& outInfo : outputsInfo) {
                graphFileInstance.header->out_tensor_desc.push_back(
                    buildTensorReference(outInfo.first, outInfo.second->getTensorDesc()));
            }

            graphFileInstance.header->identifier = netName;
        };

        if (config.platform() != InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
            mcmCompDesc.setPassArg("GenerateBlobKmb", "metaInfoSerializer", metaInfoSerializer);
        }

        if (config.numberOfClusters() > 0) {
            const int clusterCount = config.numberOfClusters();
            // number of DPU and barriers per cluster was deduced empirically
            // other values lead either to exceptions in mcmCompiler
            // or to hang-ups during inference
            constexpr int DPU_PER_CLUSTER = 5;
            const int dpuCount = clusterCount * DPU_PER_CLUSTER;
            constexpr int BARRIERS_PER_CLUSTER = 8;
            const int barrierCount = clusterCount * BARRIERS_PER_CLUSTER;
            constexpr int BARRIER_BOUNDS_PER_CLUSTER = 4;
            const int boundCount = clusterCount * BARRIER_BOUNDS_PER_CLUSTER;

            mcmCompDesc.setPassArg("GlobalConfigParams", "Number_of_DPUs", dpuCount);
            mcmCompDesc.setPassArg("GlobalConfigParams", "Number_of_Clusters", clusterCount);
            mcmCompDesc.setPassArg("GlobalConfigParams", "real_physical_barriers", barrierCount);
            mcmCompDesc.setPassArg("GlobalConfigParams", "barrier_bound", boundCount);
        }

        if (!config.mcmCompilationPassBanList().empty()) {
            std::stringstream banList{config.mcmCompilationPassBanList()};
            std::string groupPassPair;
            while (std::getline(banList, groupPassPair, ';')) {
                const auto delim = groupPassPair.find(',');
                VPUX_THROW_UNLESS(delim != std::string::npos,
                                  "McmCompilationPassBanList parsing error: provided value '{0}'"
                                  "should have comma separated Group,Pass string",
                                  groupPassPair);
                const auto group = groupPassPair.substr(0, delim);
                const auto pass = groupPassPair.substr(delim + 1, std::string::npos);
                mcmCompDesc.remove(group, pass);
            }
        }

        // override a layer's split strategy - if provided
        if (!config.layerSplitStrategies().empty()) {
            std::stringstream splitList{config.layerSplitStrategies()};
            std::string layerStrategyPair;
            
            std::vector<mv::Element> overrideStrategies;
            while (std::getline(splitList, layerStrategyPair, ',')) {
                // parse layer:strategy
                const auto delim = layerStrategyPair.find(':');
                VPUX_THROW_UNLESS(delim != std::string::npos,
                                 "layerSplitStrategies parsing error: provided value '%s'"
                                 "should have semi-colon separated layername,strategy string, eg, conv1:SplitOverK,conv2:SplitOverH",
                                 layerStrategyPair);
                const auto layerName = layerStrategyPair.substr(0, delim);
                const auto splitStrategy = layerStrategyPair.substr(delim + 1, std::string::npos);
                
                // save to vector
                mv::Element strategyElem("item");
                strategyElem.set<std::string>("name_filter", layerName);
                strategyElem.set<std::string>("strategy", splitStrategy);
                overrideStrategies.emplace_back(strategyElem);
            }
            mcmCompDesc.setPassArg("GlobalConfigParams", "split_strategy", overrideStrategies);
        }

        // override a layer's stream strategy - if provided
        if (!config.layerStreamStrategies().empty()) {
            std::stringstream splitList{config.layerStreamStrategies()};
            std::string layerStrategySet;
            try {
                std::vector<mv::Element> overrideStrategies;
                while (std::getline(splitList, layerStrategySet, ',')) {
                    // parse "layer_name:streamsW:streamsH:streamsC:streamsK:streamsN,"
                    std::vector<std::string> allVals;
                    std::string nextValue;
                    std::stringstream ss(layerStrategySet);
                    while(std::getline(ss, nextValue, ':'))
                        allVals.push_back(nextValue);

                    // save to vector
                    mv::Element strategyElem("item");
                    strategyElem.set<std::string>("name_filter", allVals[0]);
                    std::vector<mv::Element> streams;
                    
                    mv::Element itemW("W");
                    itemW.set<int>("W", std::stoi(allVals[1]));
                    streams.emplace_back(itemW);

                    mv::Element itemH("H");
                    itemH.set<int>("H", std::stoi(allVals[2]));
                    streams.emplace_back(itemH);
                    
                    mv::Element itemC("C");
                    itemC.set<int>("C", std::stoi(allVals[3]));
                    streams.emplace_back(itemC);

                    mv::Element itemK("K");
                    itemK.set<int>("K", std::stoi(allVals[4]));
                    streams.emplace_back(itemK);
                    
                    mv::Element itemN("N");
                    itemN.set<int>("N", std::stoi(allVals[5]));
                    streams.emplace_back(itemN);

                    strategyElem.set<std::vector<mv::Element>>("splits", streams);
                    overrideStrategies.emplace_back(strategyElem);
                }
                mcmCompDesc.setPassArg("GlobalConfigParams", "streaming_strategy", overrideStrategies);
            }
            catch (std::exception& ex) {
                throw std::logic_error("layerStreamStrategies parsing error: format should be semi-colon separated string "
                                 "layername:W:H:C:K:N, eg, conv1:1:2:3:4:5");
            }
        }

        // override a layer's sparsity strategy - if provided
        if (!config.layerSparsityStrategies().empty()) {
            std::stringstream splitList{config.layerSparsityStrategies()};
            std::string layerStrategySet;

            try {
                std::vector<mv::Element> overrideStrategies;
                while (std::getline(splitList, layerStrategySet, ',')) {
                    // parse "layer_name:input_sparsity:output_sparsity:weights_sparsity"
                    std::vector<std::string> allVals;
                    std::string nextValue;
                    std::stringstream ss(layerStrategySet);
                    while(std::getline(ss, nextValue, ':'))
                        allVals.push_back(nextValue);

                    mv::Element strategyElem("item");
                    strategyElem.set<std::string>("name_filter", allVals[0]);
                    strategyElem.set<bool>("inputActivationSparsity", allVals[1] == "true");
                    strategyElem.set<bool>("outputActivationSparsity", allVals[2] == "true");
                    strategyElem.set<bool>("weightsSparsity", allVals[3] == "true");
                    overrideStrategies.emplace_back(strategyElem);
                }
                mcmCompDesc.setPassArg("GlobalConfigParams", "sparsity_strategy", overrideStrategies);
            }
            catch (std::exception& ex) {
                throw std::logic_error("layerSparsityStrategies parsing error: format should be semi-colon separated string "
                                 "layername:input_sparsity:output_sparsity:weights_sparsity, e.g. conv1:true:false:true");
            }
        }

        // override a layer's location strategy - if provided
        if (!config.layerLocationStrategies().empty()) {
            std::stringstream splitList{config.layerLocationStrategies()};
            std::string layerStrategyPair;

            std::vector<mv::Element> overrideStrategies;
            while (std::getline(splitList, layerStrategyPair, ',')) {
                // parse layer:strategy
                const auto delim = layerStrategyPair.find(':');
                VPUX_THROW_UNLESS(delim != std::string::npos,
                                  "layerLocationStrategies parsing error: provided value '%s'"
                                  "should have semi-colon separated layername:strategy string, e.g. conv1:DDR",
                                  layerStrategyPair);
                const auto tensorName = layerStrategyPair.substr(0, delim);
                const auto strategy = layerStrategyPair.substr(delim + 1, std::string::npos);

                mv::Element strategyElem("item");
                strategyElem.set<std::string>("name_filter", tensorName);
                strategyElem.set<std::string>("mem_location", strategy);
                overrideStrategies.emplace_back(strategyElem);
            }
            mcmCompDesc.setPassArg("GlobalConfigParams", "tensor_placement_override", overrideStrategies);
        }


        IE_ASSERT(mcmCompiler->initialize());
    }
    return mcmCompiler;
}


void applyTransformations(
    const std::shared_ptr<ngraph::Function> func,
    std::unique_ptr<mv::CompilationUnit>& mcmCompiler,
    const ie::InputsDataMap& inputsInfo,
    const ie::OutputsDataMap& outputsInfo,
    const vpu::MCMConfig& config,
    std::shared_ptr<vpu::Logger> log,
    const bool useCompiler,
    std::shared_ptr<std::unordered_set<std::string>> supported
    )
{
    log->debug("Convert nGraph to MCM Model");

    bool needConvertInputPrecision = false;

    NodeOutputToMcmMap mcmOutputsMap;

    ngraph::pass::Manager passManager;
    passManager.register_pass<ngraph::pass::InitNodeInfo>();
    passManager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    passManager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>();
    passManager.register_pass<ngraph::pass::ConstantFolding>();
    passManager.register_pass<ngraph::pass::FusePadding>();

    if (config.scaleShiftFusing()) {
        passManager.register_pass<FuseScaleShift>();
    }

    // TODO: Add passes for rewriting parts of graph
    auto anchor = passManager.register_pass<ngraph::pass::GraphRewrite>();
    anchor->add_matcher<ngraph::pass::CollapseConcats0238>();
    anchor->set_name("ngraph::pass::mcmAdaptation");

    passManager.register_pass<OnnxReorgPatternToDarkNetReorg>();
    passManager.register_pass<ConvertExtractImagePatchesToReorgYoloVPU>();
    passManager.register_pass<PropagateFQ>();
    passManager.register_pass<AlignScales>();

    if (!config.serializeCNNBeforeCompileFile().empty()) {
        std::string origFileName = config.serializeCNNBeforeCompileFile();
        auto baseFileName = (origFileName.substr(origFileName.length() - 4, 4) == ".xml")
                            ? origFileName.substr(0, origFileName.length() - 4)
                            : origFileName;

        passManager.register_pass<ngraph::pass::Serialize>(baseFileName + ".xml", baseFileName + ".bin");
    }

    passManager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
    passManager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
    passManager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    passManager.register_pass<ngraph::pass::ConstantFolding>();

    passManager.register_pass<ngraph::pass::ConvertReduceToPooling>();

    // TBD Should be ngraph::pass too in order to be applied in between other passes.
    const auto ioMap = MapInputOutputInfoToNgraphOps(func, inputsInfo, outputsInfo);

    passManager.register_pass<FuseScaleAfterClamp>();
    passManager.register_pass<ConvertToMcmConv>();
    passManager.register_pass<ConvertToMcmFC>();
    passManager.register_pass<ReplaceScaleShiftWithMcmScale>();
    passManager.register_pass<ReplaceAddWithMcmEltwise>();
    passManager.register_pass<ConvertMVN6toMVN1>();
    passManager.register_pass<ngraph::pass::ConstantFolding>();
    passManager.register_pass<BroadcastEltwiseInputs>();
    passManager.register_pass<MergeTopKConvert>();
    passManager.register_pass<InsertMaxPool>();
    passManager.register_pass<ReplaceShuffle>();
    passManager.register_pass<Handle3DTranspose>();
    if (config.optimizeInputPrecision()) {
        passManager.register_pass<DetectInputFQ>(&needConvertInputPrecision);
    }
    // TODO: [Track number: E#13091]
    // passManager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4>();

    if (useCompiler) {
        auto& mcmModel = mcmCompiler->model();
        passManager.register_pass<ConvertToMcmModel>(mcmModel, mcmOutputsMap, inputsInfo, outputsInfo, ioMap, config, &needConvertInputPrecision);
    } else {
        passManager.register_pass<QueryModel>(supported);
    }

    const auto transformationsPredicate = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
        const bool skipLayers =
            std::dynamic_pointer_cast<const ngraph::opset4::SoftPlus>(node) ||
            std::dynamic_pointer_cast<const ngraph::opset4::HSwish>(node);

        return skipLayers;
    };

    passManager.set_callback(transformationsPredicate);

    auto passConfig = passManager.get_pass_config();
    auto disablePassPredicate = [](const std::shared_ptr<const ngraph::Node>&) {
        return true;
    };

    passConfig->set_callback<ngraph::pass::ConvertStridedSliceToCropMatcher>(disablePassPredicate);

    const auto start = std::chrono::high_resolution_clock::now();
    passManager.run_passes(func);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto process_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    log->info("Plugin processing time: %v ms", process_time.count());
}

std::unique_ptr<mv::CompilationUnit> compileNGraphIntoCompilationUnit(
        const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo,
        const vpu::MCMConfig& config,
        std::string & errMsg) {
    const auto log = std::make_shared<vpu::Logger>("VPUX nGraph Parser", config.logLevel(), vpu::consoleOutput());

    log->info("Parse nGraph %v", netName);

    auto mcmCompiler = createCompilationUnit(netName, inputsInfo, outputsInfo, config, log, errMsg);

    if (!mcmCompiler)
        return {};

    std::shared_ptr<std::unordered_set<std::string>> supportedLayersStub;
    applyTransformations(func, mcmCompiler, inputsInfo, outputsInfo, config, log, true, supportedLayersStub);

    //
    // Run MCM Compiler
    //

    {
        log->debug("Run MCM Compiler");
        try {
            const auto start = std::chrono::high_resolution_clock::now();
            mcmCompiler->run();
            const auto end = std::chrono::high_resolution_clock::now();
            const auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
            log->info("Compiler processing time: %v ms", compile_time.count());
        } catch (std::string& str) {
            log->error("MCM Compiler error: %v", str);
            errMsg = str;
            return {};
        } catch (const char* str) {
            errMsg = (str != nullptr) ? std::string(str) : "(null)";
            log->error("MCM Compiler error: %s", errMsg);
            return {};
        } catch (std::exception& ex) {
            log->error("MCM Compiler exception: %v", ex.what());
            errMsg = ex.what();
            return {};
        } catch (...) {
            log->error("MCM Compiler general exception");
            errMsg = "MCM Compiler general exception";
            return {};
        }
    }

    return mcmCompiler;
}

std::shared_ptr<std::unordered_set<std::string>> getSupportedLayers(
        const InferenceEngine::CNNNetwork& network,
        const vpu::MCMConfig& config)
{
    std::shared_ptr<std::unordered_set<std::string>> supported = std::make_shared<std::unordered_set<std::string>>();

    const auto ngraph_function = ngraph::clone_function(*(network.getFunction()));
    auto mcmCompiler = std::unique_ptr<mv::CompilationUnit>();
    ie::InputsDataMap inputsInfo = network.getInputsInfo();
    ie::OutputsDataMap outputsInfo = network.getOutputsInfo();
    const auto log = std::make_shared<vpu::Logger>("VPUX nGraph Parser", config.logLevel(), vpu::consoleOutput());

    applyTransformations(ngraph_function, mcmCompiler, inputsInfo, outputsInfo, config, log, false, supported);

    return supported;
}

std::vector<char> serializeCompilationUnit(
    const std::unique_ptr<mv::CompilationUnit>& compUnit,
    std::string & errMsg) {
    const auto blob = compUnit->getBlob();
    if (blob == nullptr) {
        errMsg = "mcmCompiler.getBlob() == nullptr";
        return {};
    }
    if (blob->empty()) {
        errMsg = "MCM Compiler general exception";
        return {};
    }
    return *blob;
}

std::vector<char> compileNGraph(
        const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo,
        const vpu::MCMConfig& config,
        std::string & errMsg) {
    const std::unique_ptr<mv::CompilationUnit> compilationUnit = compileNGraphIntoCompilationUnit(func, netName, inputsInfo, outputsInfo, config, errMsg);
    if (!errMsg.empty()) return {};
    return serializeCompilationUnit(compilationUnit, errMsg);
}
// clang-format on
