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
#ifdef ENABLE_MCM_COMPILER

#include "ngraph_mcm_frontend/passes/convert_to_mcm_model.hpp"
#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"
#include "ngraph_mcm_frontend/ops/mcm_fc.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include <ngraph_ops/fully_connected.hpp>
#include "ngraph/op/reduce_mean.hpp"
#include "ngraph/op/fused/clamp.hpp"

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/softmax.hpp"
// not needed #include <ngraph_ops/prior_box_ie.hpp>
#include "ngraph/op/fused/prelu.hpp"
#include "ngraph/op/experimental/layers/region_yolo.hpp"

#include "ngraph/op/experimental/layers/reorg_yolo.hpp"

#include <ngraph_ops/power.hpp>

#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"

#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"

#include <ngraph/op/fused/fake_quantize.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph_ops/scaleshift.hpp>

#include <ngraph/op/parameter.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/constant.hpp>
#include <memory>
#include <vector>
#include <map>

namespace {

using Callback = void (*)(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap);
using DispatchMap = std::map<ngraph::NodeTypeInfo, Callback>;

std::vector<mv::Data::TensorIterator> getMcmInputs(std::shared_ptr<ngraph::Node> node, const NodeOutputToMcmMap& mcmOutputsMap) {
    std::vector<mv::Data::TensorIterator> out;
    out.reserve(node->get_input_size());

    for (const auto& input : node->inputs()) {
        try {
            out.push_back(mcmOutputsMap.at(input.get_source_output()));
        } catch (const std::exception &ex) {
            std::cout << "Output not found: " << input.get_source_output().get_tensor().get_name()
                      << " " << ex.what() << std::endl;
        }
    }

    return out;
}

void registerOutputs(std::shared_ptr<ngraph::Node> node, std::vector<mv::Data::TensorIterator> mcmOutputs, NodeOutputToMcmMap& mcmOutputsMap) {
    size_t ind = 0;
    for (const auto& mcmOutput : mcmOutputs) {
        mcmOutputsMap.insert({node->output(ind), mcmOutput});
        ++ind;
    }
}

void convert(std::shared_ptr<ngraph::op::Parameter> param, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mvShape = cvtShapeToMCM(param->get_shape());
    const auto mvDType = mv::DType("UInt8"); // cvtElemTypeToMCM(param->get_element_type());
    const auto mvOrder = mv::Order("NHWC");// McmOpAttrs::getOrder(param);
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(param);
    const auto& opName = param->get_friendly_name();

    // MCM Compiler requirements
    IE_ASSERT(mv::DType("UInt8") == mvDType);
    IE_ASSERT(mv::Order("NHWC") == mvOrder);
    const auto mcmOutput = mcmModel.input(mvShape, mvDType, mvOrder, mvQuantParams, opName);

    registerOutputs(param, {mcmOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::Result> result, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(result, mcmOutputsMap);
    const auto mvDType = cvtOutputType(result->get_element_type());

    // MCM Compiler requirements
    IE_ASSERT(mv::DType("Float16") == mvDType);
    mcmModel.output(mcmInputs.at(0), mvDType, {{}, {}, {}, {}});
}

void convert(std::shared_ptr<ngraph::op::Constant> constant, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {

    const auto mvShape = cvtShapeToMCM((constant->get_shape().size()) ? (constant->get_shape()) : (ngraph::Shape {1}));
    const auto mvDType = cvtElemTypeToMCM(constant->get_element_type());
    const auto mvOrder = mv::Order::getColMajorID(mvShape.ndims()) ; //McmOpAttrs::getOrder(constant);
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(constant);
    const auto& opName = constant->get_friendly_name();


    mv::Data::TensorIterator mcmOutput;
    if (constant->get_element_type().is_real()) {
        mcmOutput = mcmModel.constant(constant->cast_vector<double>(), mvShape, mvDType, mvOrder, mvQuantParams, opName);
    } else {
        mcmOutput = mcmModel.constantInt(constant->cast_vector<int64_t>(), mvShape, mvDType, mvOrder, mvQuantParams, opName);
    }

    registerOutputs(constant, {mcmOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<McmConv> conv, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(conv, mcmOutputsMap);

    const auto mcmData = mcmInputs.at(0);
    const auto mcmWeights = mcmInputs.at(1);

    const auto& strides = conv->get_strides();
    const auto& padsBegin = conv->get_pads_begin();
    const auto& padsEnd = conv->get_pads_end();
    const auto& dilations = conv->get_dilations();
    const auto group = conv->get_group();

    const auto mvDType = mv::DType("Default");
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(conv);
    const auto& opName = conv->get_friendly_name();

    IE_ASSERT(dilations.at(1) == dilations.at(0));
    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmConvOutput = mcmModel.conv(
        mcmData, mcmWeights,
        {
            static_cast<uint16_t>(strides.at(1)),
            static_cast<uint16_t>(strides.at(0))},
        {
            static_cast<uint16_t>(padsBegin.at(1)), static_cast<uint16_t>(padsEnd.at(1)),
            static_cast<uint16_t>(padsBegin.at(0)), static_cast<uint16_t>(padsEnd.at(0))
        },
        static_cast<uint32_t>(dilations.at(1)),
        static_cast<uint32_t>(group),
        mvDType, mvQuantParams, opName);

    registerOutputs(conv, {mcmConvOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<McmBias> bias, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(bias, mcmOutputsMap);

    const auto mcmData = mcmInputs.at(0);
    const auto mcmBias = mcmInputs.at(1);

    const auto mvDType = mv::DType("Default");
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(bias);
    const auto& opName = bias->get_friendly_name();

    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmBiasOutput = mcmModel.bias(
        mcmData, mcmBias,
        mvDType, mvQuantParams, opName);

    registerOutputs(bias, {mcmBiasOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::MaxPool> maxPool, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(maxPool, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());

    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& outputQuantParams = McmOpAttrs::getQuantParams(maxPool);
    const auto& opName = maxPool->get_friendly_name();

    const auto kernelShape = maxPool->get_kernel();
    const auto strides = maxPool->get_strides();
    const auto padsBegin = maxPool->get_pads_begin();
    const auto padsEnd = maxPool->get_pads_begin();

    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmMaxPoolOutput = mcmModel.maxPool(mcmData,
            {static_cast<uint16_t>(kernelShape.at(0)), static_cast<uint16_t>(kernelShape.at(0))},
            {static_cast<uint16_t>(strides.at(0)), static_cast<uint16_t>(strides.at(1))},
            {static_cast<uint16_t>(padsBegin.at(1)), static_cast<uint16_t>(padsEnd.at(1)),
            static_cast<uint16_t>(padsBegin.at(0)), static_cast<uint16_t>(padsEnd.at(0))},
            true, mvDType, outputQuantParams, opName);

    registerOutputs(maxPool, {mcmMaxPoolOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<ngraph::op::v1::AvgPool> avgPool, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(avgPool, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());

    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& outputQuantParams = McmOpAttrs::getQuantParams(avgPool);
    const auto& opName = avgPool->get_friendly_name();

    const auto kernelShape = avgPool->get_kernel();
    const auto strides = avgPool->get_strides();
    const auto padsBegin = avgPool->get_pads_begin();
    const auto padsEnd = avgPool->get_pads_begin();

    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmAvgPoolOutput = mcmModel.averagePool(mcmData,
            {static_cast<uint16_t>(kernelShape.at(0)), static_cast<uint16_t>(kernelShape.at(0))},
            {static_cast<uint16_t>(strides.at(0)), static_cast<uint16_t>(strides.at(1))},
            {static_cast<uint16_t>(padsBegin.at(1)), static_cast<uint16_t>(padsEnd.at(1)),
            static_cast<uint16_t>(padsBegin.at(0)), static_cast<uint16_t>(padsEnd.at(0))},
            true, mvDType, outputQuantParams, opName);

    registerOutputs(avgPool, {mcmAvgPoolOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::ConvolutionIE> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v0::Relu> relu, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(relu, mcmOutputsMap).at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(relu);
    const auto& opName = relu->get_friendly_name();

    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmReluOutput = mcmModel.relu(mcmData, mvDType, inputQuantParams, opName);

    registerOutputs(relu, {mcmReluOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<McmEltwise> eltwise, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(eltwise, mcmOutputsMap);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(eltwise);
    const auto& opName = eltwise->get_friendly_name();
    const auto& opType = eltwise->getOperationType();

    IE_ASSERT(2 == mcmInputs.size());
    IE_ASSERT(McmEltwise::OperationType::SUM == opType);
    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmEltwiseOutput = mcmModel.eltwise(mcmInputs, "Add", mvDType, inputQuantParams, opName);

    registerOutputs(eltwise, {mcmEltwiseOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<ngraph::op::v1::ReduceMean> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v1::Reshape> reshape, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(reshape, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(reshape);
    const auto& opName = reshape->get_friendly_name();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    // TBD:  mv::Shape newShape = cvtShapeToMCM(reshape->get_output_shape(0));
    mv::Shape newShape {1, 1, 1, 1};
    const auto shape_size = reshape->get_output_shape(0).size();
    if (2 == shape_size) {
        newShape[2] = reshape->get_output_shape(0).at(1);
        newShape[3] = reshape->get_output_shape(0).at(0);
    } else {
        IE_ASSERT(2 == shape_size);
    }

    IE_ASSERT(mv::DType("Default") == mvDType);
    auto mcmReshapeOutput = mcmModel.reshape(mcmData, newShape, mvDType, inputQuantParams, opName);
    registerOutputs(reshape, {mcmReshapeOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<McmFC> fc, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(fc, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto mcmWeights = mcmInputs.at(1);

    const auto mvDType = mv::DType("Default");//cvtElemTypeToMCM(fc->get_element_type());
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(fc);
    const auto& opName = fc->get_friendly_name();

    const auto mcmFCOutput = mcmModel.fullyConnected(mcmData, mcmWeights, mvDType, mvQuantParams, opName);

    registerOutputs(fc, {mcmFCOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<ngraph::op::v0::Clamp> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v0::Concat> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v1::Softmax> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v0::LRN> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v0::Convert> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::PowerIE> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v0::PRelu> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v0::RegionYolo> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}
void convert(std::shared_ptr<ngraph::op::v0::ReorgYolo> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
}

void convert(std::shared_ptr<ngraph::op::v0::FakeQuantize> fq, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(fq, mcmOutputsMap);
    IE_ASSERT(5 == mcmInputs.size());
    const auto inputData = mcmInputs.at(0);
    const auto inputMin = mcmInputs.at(1);
    const auto inputMax = mcmInputs.at(2);
    const auto outputMin = mcmInputs.at(3);
    const auto outputMax = mcmInputs.at(4);
    const auto mvDType = mv::DType("Default");// cvtElemTypeToMCM(fq->get_element_type());
    const auto& opName = fq->get_friendly_name();
    const unsigned levels = fq->get_levels();

    const auto mcmFQOutput = mcmModel.fakeQuantize(inputData,
        inputMin, inputMax, outputMin, outputMax, levels, opName);
    registerOutputs(fq, {mcmFQOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<McmScale> scale, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(scale, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto mcmWeights = mcmInputs.at(1);
    const auto mvDType = mv::DType("Default");
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(scale);
    const auto& opName = scale->get_friendly_name();

    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmScaleOutput = mcmModel.scale(mcmData, mcmWeights, mvDType, mvQuantParams, opName);

    registerOutputs(scale, {mcmScaleOutput}, mcmOutputsMap);
}

template <typename T>
void convertDispatch(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    convert(std::dynamic_pointer_cast<T>(node), mcmModel, mcmOutputsMap);
}

#define MAP_ENTRY(__OP__) {__OP__::type_info, convertDispatch<__OP__>}

static const DispatchMap dispatchMap {
    MAP_ENTRY(ngraph::op::Parameter),
    MAP_ENTRY(ngraph::op::Result),
    MAP_ENTRY(ngraph::op::Constant), // crashes on yolo
    MAP_ENTRY(McmConv),
    MAP_ENTRY(McmBias),
    MAP_ENTRY(McmScale),
    MAP_ENTRY(ngraph::op::v1::MaxPool),
    MAP_ENTRY(ngraph::op::v0::Relu),
    MAP_ENTRY(McmEltwise),
    MAP_ENTRY(ngraph::op::v0::FakeQuantize),
    MAP_ENTRY(ngraph::op::v1::AvgPool),
    MAP_ENTRY(ngraph::op::v1::Reshape),
    MAP_ENTRY(McmFC),
#if 0
    MAP_ENTRY(McmDequantize),
    MAP_ENTRY(McmQuantize),
    // ResNet-50
    MAP_ENTRY(ngraph::op::v1::MaxPool),
    MAP_ENTRY(ngraph::op::v0::Relu),
//     MAP_ENTRY(ngraph::op::v1::Add), Eltwise
    MAP_ENTRY(ngraph::op::v1::ReduceMean),
    MAP_ENTRY(ngraph::op::v1::Reshape),
    MAP_ENTRY(ngraph::op::FullyConnected),
    // PT_MobileNet_V2
    MAP_ENTRY(ngraph::op::v0::Clamp),
    // CF_Inception_V1
    MAP_ENTRY(ngraph::op::v0::Concat),
    MAP_ENTRY(ngraph::op::v1::AvgPool),
    MAP_ENTRY(ngraph::op::v1::Softmax),
    MAP_ENTRY(ngraph::op::v0::LRN),
    MAP_ENTRY(ngraph::op::v0::Convert),
    // PT_Inception_V3
    MAP_ENTRY(ngraph::op::PowerIE),
    // CF SqueezeNet_1_1 - nothing
    // TF Yolo tiny v2
    MAP_ENTRY(ngraph::op::v0::PRelu),
    MAP_ENTRY(ngraph::op::v0::RegionYolo),
    // TF Yolo V2
    MAP_ENTRY(ngraph::op::v0::ReorgYolo)
#endif

};

#undef MAP_ENTRY

}  // namespace

bool ConvertToMcmModel::run_on_function(std::shared_ptr<ngraph::Function> func) {
    bool isConvertionFailed = false;
    for (const auto& op : func->get_ordered_ops()) {
        const auto dispatchIt = dispatchMap.find(op->get_type_info());
        if (dispatchIt != dispatchMap.end()) {
            const auto convertor = dispatchIt->second;
            if (convertor != nullptr) {
                try {
                    convertor(op, _mcmModel, _mcmOutputsMap);
                } catch (const std::runtime_error& ex) {
                    std::cout << "NGraph to MCM ,Model Convertion failed: " << ex.what() << std::endl;
                    THROW_IE_EXCEPTION << "NGraph to MCM ,Model Convertion failed: " << ex.what();
                }
            } else {
                std::cout << "Convertor not found for operation: " << op->get_friendly_name() << std::endl;
                isConvertionFailed = true;
                break;
            }
        } else {
            std::cout << "Unsupported operation: " << op->get_friendly_name()
                      << " with name " << op->get_name()
                      << " with type " << op->get_type_name() << std::endl;
            isConvertionFailed = true;
            // break;
        }
    }
    if (isConvertionFailed) {
        IE_ASSERT(false) << "NGraph to MCM ,Model Convertion failed";
        std::cout<< "NGraph to MCM ,Model Convertion failed";
        return false;
    }
    return false;
}

#endif
// clang-format on
