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
#include "ngraph_mcm_frontend/passes/convert_to_mcm_fc.hpp"
#include "ngraph_mcm_frontend/passes/fuse_dequantize.hpp"
#include "ngraph_mcm_frontend/passes/merge_quantize_with_input.hpp"
#include "ngraph_mcm_frontend/passes/merge_result_convert.hpp"
#include "ngraph_mcm_frontend/passes/quantize_constants.hpp"
#include "ngraph_mcm_frontend/passes/quantize_conv_biases.hpp"
#include "ngraph_mcm_frontend/passes/replace_add_with_eltwise.hpp"
#include "ngraph_mcm_frontend/passes/replace_scale_shift_with_fq.hpp"
#include "ngraph_mcm_frontend/passes/replace_scaleshift_with_mcm_scale.hpp"
#include "ngraph_mcm_frontend/passes/split_fq.hpp"
#include "ngraph_mcm_frontend/passes/align_eltwise_scales.hpp"
#include "ngraph_mcm_frontend/passes/align_concat_scales.hpp"
#include <file_utils.h>
#include <vpu/utils/logger.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp>
#include <transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp>
#include <transformations/convert_reduce_to_pooling.hpp>
#include <transformations/lin_op_sequence_fusoin.hpp>
#include <transformations/convert_opset1_to_legacy/convert_convolutions.hpp>
#include <transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp>
#include <transformations/convert_opset1_to_legacy/convert_prelu_to_relu_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <include/mcm/compiler/compilation_unit.hpp>
#include <memory>
#include <string>
#include <vector>
#include <utility>

std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReference(
    const std::string& tensorName,
    const InferenceEngine::TensorDesc& tensorInfo);

std::vector<char> compileNGraph(
        const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo,
        const vpu::MCMConfig& config) {
    const auto log = std::make_shared<vpu::Logger>("KMB nGraph Parser", config.logLevel(), vpu::consoleOutput());

    log->info("Parse nGraph %v", netName);
    VPU_LOGGER_SECTION(log);

    //
    // Configure MCM Compiler
    //

    mv::CompilationUnit mcmCompiler(netName);

    {
        log->debug("Configure MCM Compiler");
        VPU_LOGGER_SECTION(log);

        const auto targetPath = ie::getIELibraryPath() + "/" + config.mcmTargetDesciptorPath() + "/" + config.mcmTargetDesciptor() + ".json";
        const auto compDescPath = ie::getIELibraryPath() + "/" + config.mcmCompilationDesciptorPath() + "/" + config.mcmCompilationDesciptor() + ".json";

        IE_ASSERT(mcmCompiler.loadTargetDescriptor(targetPath));
        IE_ASSERT(mcmCompiler.loadCompilationDescriptor(compDescPath));

        auto& mcmCompDesc = mcmCompiler.compilationDescriptor();

        mcmCompDesc.setPassArg("GlobalConfigParams", "verbose", cvtLogLevelToMCM(config.mcmLogLevel()));

        if (config.referenceMode()) {
            mcmCompDesc.setPassArg("GlobalConfigParams", "ReferenceMode", true);
        }

        std::function<void(MVCNN::GraphFileT&)> metaInfoSerializer =
            [&inputsInfo, &outputsInfo](MVCNN::GraphFileT& graphFileInstance) {
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
        };
        mcmCompDesc.setPassArg("GenerateBlobKmb", "metaInfoSerializer", metaInfoSerializer);

        IE_ASSERT(mcmCompiler.initialize());
    }

    //
    // Convert nGraph to MCM Model
    //

    {
        log->debug("Convert nGraph to MCM Model");
        VPU_LOGGER_SECTION(log);

        auto& mcmModel = mcmCompiler.model();
        NodeOutputToMcmMap mcmOutputsMap;

        ngraph::pass::Manager passManager;

        passManager.register_pass<ngraph::pass::ConstantFolding>();
        passManager.register_pass<ngraph::pass::ConvertConvolutions>();
        passManager.register_pass<ngraph::pass::LinOpSequenceFusion>();
        passManager.register_pass<ngraph::pass::ConvertMatMulToFCorGemm>();
        passManager.register_pass<ngraph::pass::ConvFusion>();
        passManager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
        passManager.register_pass<ngraph::pass::ConvertMulAddToScaleShiftOrPower>();
        passManager.register_pass<ngraph::pass::ConvertMulOrAddFinally>();
        passManager.register_pass<ngraph::pass::ConvertReduceToPooling>();
        passManager.register_pass<ngraph::pass::ConstantFolding>();
        passManager.register_pass<ngraph::pass::ConvertPriorBox>();
        passManager.register_pass<ngraph::pass::ConvertPReLUToReLUIE>();

        // TBD passManager.register_pass<ngraph::pass::ConvertQuantizeDequantize>(); // transformation for ONNX importer #34095

        passManager.register_pass<ConvertToMcmConv>();
        passManager.register_pass<ConvertToMcmFC>();
        passManager.register_pass<ReplaceScaleShiftWithMcmScale>();
        passManager.register_pass<ReplaceAddWithMcmEltwise>();
        passManager.register_pass<AlignEltwiseScales>();
        passManager.register_pass<AlignConcatScales>();
        passManager.register_pass<ConvertToMcmModel>(mcmModel, mcmOutputsMap);

        const auto start = std::chrono::high_resolution_clock::now();
        passManager.run_passes(func);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto process_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
        std::stringstream msg;
        msg << "Plugin time: " << process_time.count() << " ms" << std::endl;
        log->debug(msg.str().c_str());
    }

    //
    // Run MCM Compiler
    //

    {
        log->debug("Run MCM Compiler");
        VPU_LOGGER_SECTION(log);
        const auto start = std::chrono::high_resolution_clock::now();
        mcmCompiler.run();
        const auto end = std::chrono::high_resolution_clock::now();
        const auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
        std::stringstream msg;
        msg << "MCM Compiler time: " << compile_time.count() << " ms" << std::endl;
        log->debug(msg.str().c_str());

    }

    //
    // Return compiled blob
    //

    const auto memBlob = mcmCompiler.getBlob();
    IE_ASSERT(memBlob != nullptr);
    std::vector<char> blob;
    std::copy(memBlob->begin(), memBlob->end(), std::back_inserter(blob));

    if (blob.empty()) {
        THROW_IE_EXCEPTION << "Blob created by mcmCompiler is empty!";
    }

    return blob;
}

// clang-format on
