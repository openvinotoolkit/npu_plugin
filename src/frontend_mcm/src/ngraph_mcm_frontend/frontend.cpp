//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
#include "ngraph_mcm_frontend/passes/replace_scaleshift_with_mcm_scale.hpp"
#include "ngraph_mcm_frontend/passes/align_eltwise_scales.hpp"
#include "ngraph_mcm_frontend/passes/align_concat_scales.hpp"
#include "ngraph_mcm_frontend/passes/fuse_scaleshift.hpp"
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

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>

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
                THROW_IE_EXCEPTION << "Input not found: " << inputInfo.first;
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
                THROW_IE_EXCEPTION << "Ouput not found: " << outputInfo.first;
        }

        return ioMap;
    }
}

std::unique_ptr<mv::CompilationUnit> compileNGraphIntoCompilationUnit(
        const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo,
        const vpu::MCMConfig& config,
        std::string & errMsg) {
    const auto log = std::make_shared<vpu::Logger>("KMB nGraph Parser", config.logLevel(), vpu::consoleOutput());

    log->info("Parse nGraph %v", netName);

    bool needConvertInputPrecision = false;

    //
    // Configure MCM Compiler
    //

    auto mcmCompiler = std::unique_ptr<mv::CompilationUnit>(new mv::CompilationUnit(netName));

    {
        log->debug("Configure MCM Compiler");
        VPU_LOGGER_SECTION(log);

        std::string compDescName = !config.mcmCompilationDesciptor().empty() ?
                            config.mcmCompilationDesciptor() : "release_kmb";

        if (config.deviceId() == "EMULATOR") {
            compDescName = "emulator_kmb_SC-Prefetch1";
        }

        const auto targetPath = ie::getIELibraryPath() + "/" + config.mcmTargetDesciptorPath() + "/" + config.mcmTargetDesciptor() + ".json";
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


        if (config.referenceMode()) {
            mcmCompDesc.setPassArg("GlobalConfigParams", "ReferenceMode", true);
        }

        if (config.performanceCounting()) {
            mcmCompDesc.setPassArg("GlobalConfigParams", "PerformanceCounting", true);
        }

        std::function<void(MVCNN::GraphFileT&)> metaInfoSerializer =
            [&inputsInfo, &outputsInfo, &netName](MVCNN::GraphFileT& graphFileInstance) {
            if (graphFileInstance.header == nullptr) {
                THROW_IE_EXCEPTION << "metaInfoSerializer: graph file header points to null";
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
        if (config.deviceId() != "EMULATOR") {
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

        IE_ASSERT(mcmCompiler->initialize());
    }

    //
    // Convert nGraph to MCM Model
    //

    {
        log->debug("Convert nGraph to MCM Model");

        auto& mcmModel = mcmCompiler->model();
        NodeOutputToMcmMap mcmOutputsMap;

        ngraph::pass::Manager passManager;
        passManager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
        passManager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>();
        passManager.register_pass<ngraph::pass::ConstantFolding>();

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

        passManager.register_pass<ngraph::pass::ConvertPriorBox>(); // strict requirement: ConvertPriorBox should be first

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
        passManager.register_pass<ngraph::pass::ConstantFolding>();
        passManager.register_pass<BroadcastEltwiseInputs>();
        passManager.register_pass<MergeTopKConvert>();
        passManager.register_pass<InsertMaxPool>();
        passManager.register_pass<ReplaceShuffle>();
        passManager.register_pass<Handle3DTranspose>();
        passManager.register_pass<DetectInputFQ>(&needConvertInputPrecision);
        passManager.register_pass<ConvertToMcmModel>(mcmModel, mcmOutputsMap, inputsInfo, outputsInfo, ioMap, config, &needConvertInputPrecision);
        passManager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4>();

        const auto start = std::chrono::high_resolution_clock::now();

        const auto transformationsPredicate = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
            const bool skipLayers =
                std::dynamic_pointer_cast<const ngraph::opset4::SoftPlus>(node) ||
                std::dynamic_pointer_cast<const ngraph::opset4::HSwish>(node);

            return skipLayers;
        };

        passManager.set_callback(transformationsPredicate);

        passManager.run_passes(func);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto process_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
        log->info("Plugin processing time: %v ms", process_time.count());
    }

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
