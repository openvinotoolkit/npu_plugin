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

#include "ie_macro.hpp"
#include "debug.h"
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
#include "ngraph/op/clamp.hpp"
#include "ngraph/op/sigmoid.hpp"

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/softmax.hpp"

#include "ngraph/op/prelu.hpp"
#include <ngraph/op/roi_pooling.hpp>

#include "ngraph/op/region_yolo.hpp"

#include "ngraph/op/reorg_yolo.hpp"

#include <ngraph/op/power.hpp>
#include <ngraph_ops/relu_ie.hpp>
#include <ngraph_ops/eltwise.hpp>
#include <ngraph_ops/gather_ie.hpp>
#include <ngraph_ops/power.hpp>
#include <ngraph/op/normalize_l2.hpp>

#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"

#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"

#include <ngraph/op/fake_quantize.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph_ops/crop_ie.hpp>
#include <ngraph_ops/deconvolution_ie.hpp>
#include <ngraph_ops/scaleshift.hpp>

#include <ngraph/op/parameter.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/constant.hpp>

#include <ngraph/op/transpose.hpp>
#include <ngraph/op/squeeze.hpp>
#include <ngraph/op/unsqueeze.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/op/topk.hpp>
#include <ngraph/op/tanh.hpp>
#include <ngraph/op/exp.hpp>
#include <ngraph/op/multiply.hpp>
#include <ngraph/op/elu.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/minimum.hpp>
#include <ngraph/op/hswish.hpp>

#include <ngraph/op/prior_box.hpp>
#include <ngraph/op/prior_box_clustered.hpp>
#include <ngraph/op/detection_output.hpp>

#include <ngraph/op/split.hpp>

#include <ngraph_ops/interp.hpp>
#include <ngraph_ops/prior_box_clustered_ie.hpp>
#include <ngraph_ops/prior_box_ie.hpp>
#include "ngraph_ops/lrn_ie.hpp"
#include <ngraph_ops/normalize_ie.hpp>
#include <ngraph_ops/topk_ie.hpp>
#include <ngraph_ops/proposal_ie.hpp>

#include <ngraph/variant.hpp>

#include <parse_layers_helpers.hpp>
#include <dims_parser.hpp>
#include "ngraph_mcm_frontend/ie_helpers.hpp"

#include <memory>
#include <vector>
#include <map>

#include <include/mcm/tensor/tiling.hpp>
#include <vpu/utils/error.hpp>
#include <converters.hpp>

namespace {

using Callback = void (*)(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
InferenceEngine::DataPtr, bool);
using DispatchMap = std::map<ngraph::NodeTypeInfo, Callback>;

std::vector<mv::Data::TensorIterator> getMcmInputs(std::shared_ptr<ngraph::Node> node, const NodeOutputToMcmMap& mcmOutputsMap) {
    std::vector<mv::Data::TensorIterator> out;
    out.reserve(node->get_input_size());

    for (const auto& input : node->inputs()) {
        try {
            out.push_back(mcmOutputsMap.at(input.get_source_output()));
        } catch (const std::exception &ex) {
            THROW_IE_EXCEPTION << "For operation " << node->get_type_name() << " name " << node->get_friendly_name()
                << "output not found: " << input.get_source_output().get_tensor().get_name()
                << " " << ex.what();
        }
    }

    return out;
}

mv::Shape getWHCN(const ngraph::Shape& shape) {
    size_t dimN, dimZ, dimY, dimX;
    std::vector<size_t> dims = shape;
    vpu::parseDims(dims, dimN, dimZ, dimY, dimX);
    return mv::Shape({dimX, dimY, dimZ, dimN});
}

void cvtPaddingsFromCeilToFloorMode(
    int input_size_ceil, int output_size, int kernel, int stride, int& pad_start, int& pad_end) {
    const auto input_size_floor = mv::Tiling::inferInputSize(output_size, pad_start, pad_end, kernel, stride);

    pad_end = pad_end + (input_size_floor - input_size_ceil);
    pad_end = std::max(pad_end, 0);
}

void registerOutputs(std::shared_ptr<ngraph::Node> node, const std::vector<mv::Data::TensorIterator>& mcmOutputs, NodeOutputToMcmMap& mcmOutputsMap) {
    size_t ind = 0;
    for (const auto& mcmOutput : mcmOutputs) {
        mcmOutputsMap.insert({node->output(ind), mcmOutput});
        ++ind;
    }
}

bool isInputPrecisionSupported(const ie::Precision& inputPrecision) {
    const std::set<ie::Precision> supportedInPrecisions = {ie::Precision::U8, ie::Precision::FP16, ie::Precision::FP32};
    return supportedInPrecisions.find(inputPrecision) != supportedInPrecisions.end();
}

bool isInputLayoutSupported(const ie::Layout& inputLayout) {
    const std::set<ie::Layout> supportedInLayouts = {
        ie::Layout::NHWC, ie::Layout::NCHW, ie::Layout::CHW, ie::Layout::NC, ie::Layout::C};
    return supportedInLayouts.find(inputLayout) != supportedInLayouts.end();
}

bool isOutputPrecisionSupported(const ie::Precision& outputPrecision) {
    std::set<ie::Precision> supportedOutPrecisions = {ie::Precision::U8, ie::Precision::FP16, ie::Precision::FP32};
    return supportedOutPrecisions.find(outputPrecision) != supportedOutPrecisions.end();
}

bool isOutputLayoutSupported(const ie::Layout& outputLayout) {
    std::set<ie::Layout> supportedOutLayouts = {
        ie::Layout::NHWC, ie::Layout::NCHW, ie::Layout::CHW, ie::Layout::NC, ie::Layout::C};
    return supportedOutLayouts.find(outputLayout) != supportedOutLayouts.end();
}

static const mv::QuantizationParams& initialQuantParams() {
    double inf = std::numeric_limits<double>::infinity();
    static mv::QuantizationParams init{{0}, {1}, {-inf}, {inf}};
    return init;
};

void convert(std::shared_ptr<ngraph::op::Parameter> param, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr ieData, bool allowNCHWInput) {
    auto mvShape = getWHCN(param->get_shape());
    // Use data from InputInfo DataPtr
    // const auto mvDType = mv::DType("UInt8"); // Test framework sets fp32, cvtElemTypeToMCM(param->get_element_type());
    bool mvNetworkInput = true;
    const auto& opName = param->get_friendly_name();

    if (param->get_shape().size() > 4 || param->get_shape().size() == 0) {
       THROW_IE_EXCEPTION << "Input shape size is not supported: " << param->get_shape().size();
    }

    const InferenceEngine::Layout inputLayout = ieData->getTensorDesc().getLayout();
    if (!isInputLayoutSupported(inputLayout)) {
        THROW_IE_EXCEPTION << "Input layout is not supported: " << ieData->getTensorDesc().getLayout();
    }

    const InferenceEngine::Precision inputPrecision = ieData->getTensorDesc().getPrecision();
    if (!isInputPrecisionSupported(inputPrecision)) {
        THROW_IE_EXCEPTION << "Input data type is not supported: " << ieData->getTensorDesc().getPrecision();
    }

    const auto mvOrder = [&] {
        if (inputLayout == InferenceEngine::Layout::NCHW && allowNCHWInput) {
            return layoutToOrder(InferenceEngine::Layout::NCHW);
        }
        return layoutToOrder(InferenceEngine::Layout::NHWC);
    }();
    const auto mvDType = cvtElemTypeToMCM(cvtPrecisionToElemType(inputPrecision));
    // MCM Compiler requirements
    // IE_ASSERT(mv::DType("Float16") == mvDType || mv::DType("UInt8") == mvDType);
    // IE_ASSERT(mv::Order("NHWC") == mvOrder);
    const auto mcmOutput = mcmModel.input(opName, mvShape, mvDType, mvOrder, mvNetworkInput);
    mcmOutput->setQuantParams(initialQuantParams());

    registerOutputs(param, {mcmOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::Result> result, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr ieData) {
    const auto mcmInputs = getMcmInputs(result, mcmOutputsMap);
    // IE Output Type may differ from ngraph result output type. MCM to instert convertors.

    const auto outputPrecision = ieData->getTensorDesc().getPrecision();
    if (!isOutputPrecisionSupported(outputPrecision)) {
        THROW_IE_EXCEPTION << "Output data type is not supported: " << outputPrecision;
    }

    const InferenceEngine::Layout outputLayout = ieData->getTensorDesc().getLayout();
    if (!isOutputLayoutSupported(outputLayout)) {
        THROW_IE_EXCEPTION << "Output layout is not supported: " << outputLayout;
    }

    // TODO: kmbPlugin already has a function convert_data_type() for matching IE precision to mcm, but
    // in this case we can't use due to limitations on mcm level (not all precisions are supported).
    // mcmCompiler right now support only 2 types of precisions for output: U8 and FP16
    // for avoid this limitations plugin has a WA: translate FP32 output like a FP16 and convert output blob
    // in getResult() function after the inference.
    mv::DType outputType = mv::DType("Default");
    switch (outputPrecision) {
    case ie::Precision::UNSPECIFIED:
        outputType = mv::DType("Default");
        break;
    case ie::Precision::U8:
        outputType = mv::DType("UInt8");
        break;
    case ie::Precision::FP16:
        outputType = mv::DType("Float16");
        break;
    case ie::Precision::FP32:
        outputType = mv::DType("Float16");
        break;
    default:
        THROW_IE_EXCEPTION << "Data type handling is not implemented" << outputPrecision.name();
    }

    if (result->get_shape().size() > 4 || result->get_shape().size() == 0) {
       THROW_IE_EXCEPTION << "Output shape size is not supported: " << result->get_shape().size();
    }

    // MCM compiler compiles wrong blob when concat layer is the last layer in the network,
    // e.g. person_attributes_recognition_crossroad_0238 person_attributes_recognition_crossroad_0234
    // TODO: remove this workaround when this case will be handled on mcm compiler side
    const auto concat = std::dynamic_pointer_cast<ngraph::op::Concat>(result->input_value(0).get_node_shared_ptr());
    if (nullptr != concat) {
        const auto mcmConcat = mcmModel.maxPool(concat->get_friendly_name() + "_maxpool", mcmInputs.at(0),
            {1, 1}, {1, 1}, {0, 0, 0, 0}, true);
        mcmConcat->setQuantParams(initialQuantParams());
        mcmModel.output("", mcmConcat, outputType);
        return;
    }
    // end of workaround

    // MCM Compiler requirements
    // IE_ASSERT(mv::DType("Float16") == mvDType || mv::DType("UInt8") == mvDType);
    IE_ASSERT(mv::DType("Float16") == outputType || mv::DType("UInt8") == outputType);
    mcmModel.output("", mcmInputs.at(0), outputType);
}

void convert(std::shared_ptr<ngraph::op::Constant> constant, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {

    auto mvShape = cvtShapeToMCM((constant->get_shape().size()) ? (constant->get_shape()) : (ngraph::Shape {1}));

    // IE add constant folding for PriorBox\PriorBox clustered
    // As a result we get 3d const instead of concat for DetectionOut layer
    // Current case unsupported on mcm side. WA expand dims (3d->4d)
    if (mvShape.ndims() == 3) {
        mvShape = mv::Shape::augment_major(mvShape, 4);
    }

    auto mvDType = cvtElemTypeToMCM(constant->get_element_type());
    std::string opName = constant->get_friendly_name();
    // MCM compiler can't work with constant blob with dims {1} and i64 precision
    // TODO: remove these workarounds when this case will be handled on mcm compiler side
    if (mvShape.ndims() == 1) {
        for (auto&& consumerNode : constant->get_users()) {
            if (ngraph::op::GatherIE::type_info == consumerNode->get_type_info()) {
                mvShape = mv::Shape::augment_major(mvShape, 4);
                // int64 precision for indices is not supported by runtime yet
                if (ngraph::element::i64 == constant->get_element_type()) {
                    mvDType == mv::DType("Int32");
                    opName += "_indices_i32";
                }
                break;
            }
            if (ngraph::op::v1::Split::type_info == consumerNode->get_type_info()) {
                mvShape = mv::Shape::augment_major(mvShape, 4);
                if (ngraph::element::i64 == constant->get_element_type()) {
                    mvDType == mv::DType("Int32");
                    opName += "_indices_i32";
                }
            }
        }
    }
    // end of workaround

    const auto mvOrder = mv::Order::getColMajorID(mvShape.ndims()) ; //McmOpAttrs::getOrder(constant);

    mv::Data::TensorIterator mcmOutput;
    if (constant->get_element_type().is_real()) {
        mvDType = mv::DType("Float32");
        mcmOutput = mcmModel.constant(opName, constant->cast_vector<double>(), mvShape, mvDType, mvOrder);
    } else {
        mcmOutput = mcmModel.constantInt(opName, constant->cast_vector<int64_t>(), mvShape, mvDType, mvOrder);
    }
    mcmOutput->setQuantParams(initialQuantParams());

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
    const auto groupSize = conv->get_group();

    const auto& opName = conv->get_friendly_name();

    IE_ASSERT(dilations.at(1) == dilations.at(0));

    const auto inputShape = conv->get_input_shape(0);
    const auto outputShape = conv->get_output_shape(0);
    IE_ASSERT(4 == inputShape.size());
    IE_ASSERT(4 == outputShape.size());
    const auto inputGroupSize = inputShape.at(1);
    const auto outputGroupSize = outputShape.at(1);

    bool isDepthWiseConv = groupSize > 1 && groupSize == inputGroupSize && groupSize == outputGroupSize;

    int padLeft = padsBegin.at(1);
    int padRight = padsEnd.at(1);
    int padTop = padsBegin.at(0);
    int padBottom = padsEnd.at(0);
    const auto filterShape = mcmWeights->getShape();
    IE_ASSERT(4 == filterShape.ndims());
    const auto kernelSizeX = filterShape[0];
    const auto kernelSizeY = filterShape[1];
    const auto kernelStrideX = strides.at(1);
    const auto kernelStrideY = strides.at(0);
    const auto dilationX = dilations.at(1);
    const auto dilationY = dilations.at(0);

    cvtPaddingsFromCeilToFloorMode(mcmData->getShape()[0], outputShape.at(3),
        kernelSizeX * dilationX - (dilationX - 1), kernelStrideX, padLeft, padRight);
    cvtPaddingsFromCeilToFloorMode(mcmData->getShape()[1], outputShape.at(2),
        kernelSizeY * dilationY - (dilationY - 1), kernelStrideY, padTop, padBottom);

    if (isDepthWiseConv) {
        // TODO: Need align API in mcmCompiler
        // mcm expects (1,*,*,*) shape for depthwise weights, but Openvino has a (*,1,*,*)

        auto sourceWeightsOp = mcmModel.getSourceOp(mcmWeights);
        auto constWeightTensor = mcmWeights;
        if (sourceWeightsOp->getOpType() == "FakeQuantize") {
            constWeightTensor = sourceWeightsOp->getInputTensor(0);
            sourceWeightsOp = mcmModel.getSourceOp(constWeightTensor);
        }
        constWeightTensor->set<bool>("is_depthwise_weights", true);
        sourceWeightsOp->set<bool>("is_depthwise_weights", true);
        const std::initializer_list<std::size_t> newWeightsShape = {
            static_cast<std::size_t>(kernelSizeX), static_cast<std::size_t>(kernelSizeY), inputGroupSize, 1lu};

        constWeightTensor->setShape(newWeightsShape);
        mcmWeights->setShape(newWeightsShape);
        sourceWeightsOp->set<mv::Shape>("shape", newWeightsShape);

        const auto mcmConvOutput = mcmModel.depthwiseConv(
            opName, mcmData, mcmWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop), static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX));
        mcmConvOutput->setQuantParams(initialQuantParams());
        registerOutputs(conv, {mcmConvOutput}, mcmOutputsMap);
    } else {
        const auto mcmConvOutput = mcmModel.conv(
            opName, mcmData, mcmWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop), static_cast<uint16_t>(padBottom)},
            static_cast<uint32_t>(dilationX),
            static_cast<uint32_t>(groupSize));
        mcmConvOutput->setQuantParams(initialQuantParams());
        registerOutputs(conv, {mcmConvOutput}, mcmOutputsMap);
    }
}

void convert(std::shared_ptr<McmBias> bias, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(bias, mcmOutputsMap);

    const auto mcmData = mcmInputs.at(0);
    const auto mcmBias = mcmInputs.at(1);

    const auto& opName = bias->get_friendly_name();

    const auto mcmBiasOutput = mcmModel.bias(
        opName, mcmData, mcmBias);
    mcmBiasOutput->setQuantParams(initialQuantParams());

    registerOutputs(bias, {mcmBiasOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::MaxPool> maxPool, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(maxPool, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());

    const auto mcmData = mcmInputs.at(0);
    const auto& opName = maxPool->get_friendly_name();

    const auto kernelShape = maxPool->get_kernel();
    const auto strides = maxPool->get_strides();
    const auto padsBegin = maxPool->get_pads_begin();
    const auto padsEnd = maxPool->get_pads_begin();

    const int kernelSizeX = kernelShape.at(1);
    const int kernelSizeY = kernelShape.at(0);

    const int kernelStrideX = strides.at(1);
    const int kernelStrideY = strides.at(0);

    int padLeft = padsBegin.at(1);
    int padRight = padsEnd.at(1);
    int padTop = padsBegin.at(0);
    int padBottom = padsEnd.at(0);

    auto outputShape = maxPool->get_output_shape(0);

    cvtPaddingsFromCeilToFloorMode(
        mcmData->getShape()[0], outputShape.at(3), kernelSizeX, kernelStrideX, padLeft, padRight);
    cvtPaddingsFromCeilToFloorMode(
        mcmData->getShape()[1], outputShape.at(2), kernelSizeY, kernelStrideY, padTop, padBottom);

    const auto mcmMaxPoolOutput = mcmModel.maxPool(opName, mcmData,
            {static_cast<uint16_t>(kernelSizeX), static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop), static_cast<uint16_t>(padBottom)},
            true);
    mcmMaxPoolOutput->setQuantParams(initialQuantParams());

    registerOutputs(maxPool, {mcmMaxPoolOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<ngraph::op::v1::AvgPool> avgPool, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(avgPool, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());

    const auto mcmData = mcmInputs.at(0);
    const auto& opName = avgPool->get_friendly_name();

    const auto kernelShape = avgPool->get_kernel();
    const auto strides = avgPool->get_strides();
    const auto padsBegin = avgPool->get_pads_begin();
    const auto padsEnd = avgPool->get_pads_begin();

    const int kernelSizeX = kernelShape.at(1);
    const int kernelSizeY = kernelShape.at(0);

    const int kernelStrideX = strides.at(1);
    const int kernelStrideY = strides.at(0);

    int padLeft = padsBegin.at(1);
    int padRight = padsEnd.at(1);
    int padTop = padsBegin.at(0);
    int padBottom = padsEnd.at(0);

    auto outputShape = avgPool->get_output_shape(0);

    cvtPaddingsFromCeilToFloorMode(
        mcmData->getShape()[0], outputShape.at(3), kernelSizeX, kernelStrideX, padLeft, padRight);
    cvtPaddingsFromCeilToFloorMode(
        mcmData->getShape()[1], outputShape.at(2), kernelSizeY, kernelStrideY, padTop, padBottom);

    const auto mcmAvgPoolOutput = mcmModel.averagePool(opName, mcmData,
            {static_cast<uint16_t>(kernelSizeX), static_cast<uint16_t>(kernelSizeY)},
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop), static_cast<uint16_t>(padBottom)},
             avgPool->get_exclude_pad()); //false
    mcmAvgPoolOutput->setQuantParams(initialQuantParams());

    registerOutputs(avgPool, {mcmAvgPoolOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<ngraph::op::v0::Relu> relu, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(relu, mcmOutputsMap).at(0);
    const auto& opName = relu->get_friendly_name();

    const auto mcmReluOutput = mcmModel.relu(opName, mcmData);
    mcmReluOutput->setQuantParams(initialQuantParams());

    registerOutputs(relu, {mcmReluOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Elu> elu, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(elu, mcmOutputsMap).at(0);
    const auto& opName = elu->get_friendly_name();

    auto alpha = elu->get_alpha();

    const auto mcmEluOutput = mcmModel.elu(opName, mcmData, alpha);

    registerOutputs(elu, {mcmEluOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v4::HSwish> hswish, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(hswish, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto mcmOpOutput = mcmModel.hSwish(hswish->get_friendly_name(), mcmInputs.at(0));
    registerOutputs(hswish, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<McmEltwise> eltwise, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(eltwise, mcmOutputsMap);
    const auto& opName = eltwise->get_friendly_name();
    const auto& opType = eltwise->getOperationType();

    IE_ASSERT(2 == mcmInputs.size());
    IE_ASSERT(McmEltwise::OperationType::SUM == opType);
    const auto mcmEltwiseOutput = mcmModel.eltwise(opName, mcmInputs, "Add");
    mcmEltwiseOutput->setQuantParams(initialQuantParams());

    registerOutputs(eltwise, {mcmEltwiseOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::Eltwise> eltwise, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(eltwise, mcmOutputsMap);
    const auto& opName = eltwise->get_friendly_name();
    const auto& opType = eltwise->eltwise_type;

    IE_ASSERT(2 == mcmInputs.size());

    mv::Data::TensorIterator mcmEltwiseOutput;

    if (ELTWISE_TYPE::Sum == opType)
        mcmEltwiseOutput = mcmModel.eltwise(opName, mcmInputs, "Add");
    else if (ELTWISE_TYPE::Prod  == opType)
        mcmEltwiseOutput = mcmModel.eltwise(opName, mcmInputs, "Multiply");
    else
        THROW_IE_EXCEPTION << "Operation " << eltwise->get_type_name() << " " << opName << " has unsupported parameter ";
    mcmEltwiseOutput->setQuantParams(initialQuantParams());

    registerOutputs(eltwise, {mcmEltwiseOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<ngraph::op::v1::Reshape> reshape, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(reshape, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = reshape->get_friendly_name();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    mv::Shape newShape = getWHCN(reshape->get_output_shape(0));

    auto mcmReshapeOutput = mcmModel.reshape(opName, mcmData, newShape);
    mcmReshapeOutput->setQuantParams(initialQuantParams());
    registerOutputs(reshape, {mcmReshapeOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<McmFC> fc, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(fc, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto mcmWeights = mcmInputs.at(1);

    //cvtElemTypeToMCM(fc->get_element_type());
    const auto& opName = fc->get_friendly_name();

    const auto mcmFCOutput = mcmModel.fullyConnected(opName, mcmData, mcmWeights);
    mcmFCOutput->setQuantParams(initialQuantParams());

    registerOutputs(fc, {mcmFCOutput}, mcmOutputsMap);
}
void convert(std::shared_ptr<ngraph::op::v0::FakeQuantize> fq, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(fq, mcmOutputsMap);
    IE_ASSERT(5 == mcmInputs.size());
    const auto inputData = mcmInputs.at(0);
    const auto inputMin = mcmInputs.at(1);
    const auto inputMax = mcmInputs.at(2);
    const auto outputMin = mcmInputs.at(3);
    const auto outputMax = mcmInputs.at(4);
    // cvtElemTypeToMCM(fq->get_element_type());
    const auto& opName = fq->get_friendly_name();
    const unsigned levels = fq->get_levels();

    const auto mcmFQOutput = mcmModel.fakeQuantize(opName, inputData,
        inputMin, inputMax, outputMin, outputMax, levels);
    registerOutputs(fq, {mcmFQOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::PowerIE> power, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(power, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = power->get_friendly_name();
    const float scale = power->scale;
    const float shift = power->shift;

    if (1.0f == power->power) {
        const auto shape = power->get_output_shape(0);
        const size_t weights_size = (1 == shape.size()) ? shape.at(0) : getWHCN(shape)[2];

        std::vector<double> weights(weights_size, scale);
        mv::Shape weightsShape = {weights.size()};
        auto mcmWeights = mcmModel.constant(
            "", weights, weightsShape, mv::DType("Float32"), mv::Order::getColMajorID(1));
        mcmWeights->setQuantParams(initialQuantParams());

        const auto mcmScaleOutput = mcmModel.scale(opName, mcmData, mcmWeights);
        mcmScaleOutput->setQuantParams(initialQuantParams());
        if (0.0f != shift) {
            std::vector<double> biases (weights.size(), shift);
            mv::Shape shiftShape { biases.size() };
            auto shiftData = mcmModel.constant("", biases, shiftShape, mv::DType("Float32"), mv::Order::getColMajorID(1));
            shiftData->setQuantParams(initialQuantParams());
            auto biasOutput = mcmModel.bias(opName + "_bias", mcmScaleOutput, shiftData);
            biasOutput->setQuantParams(initialQuantParams());
            registerOutputs(power, {biasOutput}, mcmOutputsMap);
        } else {
            registerOutputs(power, {mcmScaleOutput}, mcmOutputsMap);
        }
    } else if (-1.0f == power->power) {
            auto reciprocal_result = mcmModel.reciprocal(opName, mcmData);
            reciprocal_result->setQuantParams(initialQuantParams());
            registerOutputs(power, {reciprocal_result}, mcmOutputsMap);
    } else
        THROW_IE_EXCEPTION << "Operation " << power->get_type_name() << " " + opName + " has unsupported power " << power->power;
}

void convert(std::shared_ptr<McmScale> scale, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(scale, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto mcmWeights = mcmInputs.at(1);
    const auto& opName = scale->get_friendly_name();

    const auto mcmScaleOutput = mcmModel.scale(opName, mcmData, mcmWeights);
    mcmScaleOutput->setQuantParams(initialQuantParams());

    registerOutputs(scale, {mcmScaleOutput}, mcmOutputsMap);
}

std::string getDimLabel(size_t dimIndex, ie::Layout ieLayout) {
    std::ostringstream ostr;
    ostr << ieLayout;
    const auto layoutStr = ostr.str();
    IE_ASSERT(dimIndex < layoutStr.size());
    return std::string(1, layoutStr[dimIndex]);
}

void convert(std::shared_ptr<ngraph::op::v0::Concat> concat, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(concat, mcmOutputsMap);
    IE_ASSERT(0 < mcmInputs.size());
    const auto ieLayout = ie::TensorDesc::getLayoutByDims(concat->input(0).get_shape());
    std::string mcmAxis = getDimLabel(concat->get_axis(), ieLayout);
    const auto& opName = concat->get_friendly_name();

    const auto mcmConcatOutput = mcmModel.concat(opName, mcmInputs, mcmAxis);
    mcmConcatOutput->setQuantParams(initialQuantParams());
    registerOutputs(concat, {mcmConcatOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Transpose> permute, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(permute, mcmOutputsMap);
    IE_ASSERT(2 ==  mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = permute->get_friendly_name();

    std::shared_ptr<ngraph::Node> orderNode = permute->input(1).get_source_output().get_node_shared_ptr();
    std::vector<size_t> orderIndices = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(orderNode)->cast_vector<size_t>();

    // 4d NCHW inputs are supported
    std::string newOrder;
    const auto ieLayout = ie::TensorDesc::getLayoutByDims(permute->input(0).get_shape());
    for (size_t i = 0; i < orderIndices.size(); i++) {
        newOrder += getDimLabel(orderIndices[orderIndices.size() - 1 - i], ieLayout);
    }

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    auto mcmPermuteOutput = mcmModel.permute(opName, mcmData, mv::Order(newOrder));

    // Workaround to avoid parsing stage crash:
    // 'ArgumentError: attribute identifer quantParams - Undefined identifier'
    // [Track number: D#2284, D#2237]
    // TBD
    mcmPermuteOutput->set<mv::QuantizationParams>("quantParams", initialQuantParams());

    registerOutputs(permute, {mcmPermuteOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Squeeze> reshape, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(reshape, mcmOutputsMap);
    IE_ASSERT(2 ==  mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = reshape->get_friendly_name();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    mv::Shape newShape = getWHCN(reshape->get_shape());

    auto mcmReshapeOutput = mcmModel.reshape(opName, mcmData, newShape);
    mcmReshapeOutput->setQuantParams(initialQuantParams());
    registerOutputs(reshape, {mcmReshapeOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Unsqueeze> reshape, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(reshape, mcmOutputsMap);
    IE_ASSERT(2 ==  mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = reshape->get_friendly_name();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    mv::Shape newShape = getWHCN(reshape->get_shape());

    auto mcmReshapeOutput = mcmModel.reshape(opName, mcmData, newShape);
    mcmReshapeOutput->setQuantParams(initialQuantParams());
    registerOutputs(reshape, {mcmReshapeOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Softmax> softmax, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(softmax, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = softmax->get_friendly_name();

    std::string order = McmOpAttrs::getOrder(softmax, 0).toString();
    std::string mcmAxis = std::string(1, order.at(softmax->get_axis()));

    auto mcmSoftmaxOutput = mcmModel.softmax(opName, mcmData, mcmAxis);
    mcmSoftmaxOutput->setQuantParams(initialQuantParams());
    registerOutputs(softmax, {mcmSoftmaxOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Clamp> clamp, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(clamp, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = clamp->get_friendly_name();
    const double minValue = clamp->get_min();
    const double maxValue = clamp->get_max();

    auto mcmClampMin = mcmModel.minimum(opName + "clamp-min", mcmData, maxValue);
    mcmClampMin->setQuantParams(initialQuantParams());
    auto mcmClampMax = mcmModel.maximum(opName + "clamp-max", mcmClampMin, minValue);
    mcmClampMax->setQuantParams(initialQuantParams());
    registerOutputs(clamp, {mcmClampMax}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::ReLUIE> relu, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(relu, mcmOutputsMap).at(0);
    const auto& opName = relu->get_friendly_name();
    const float slope = relu->get_slope();

    if (std::fabs(slope) < std::numeric_limits<float>::epsilon()) {
        const auto mcmReluOutput = mcmModel.relu(opName, mcmData);
        mcmReluOutput->setQuantParams(initialQuantParams());
        registerOutputs(relu, {mcmReluOutput}, mcmOutputsMap);
    } else {
        const auto mcmReluOutput = mcmModel.leakyRelu(opName, mcmData, slope);
        mcmReluOutput->setQuantParams(initialQuantParams());
        registerOutputs(relu, {mcmReluOutput}, mcmOutputsMap);
    }
}

void convert(std::shared_ptr<ngraph::op::v0::ROIPooling> roipool, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    auto mcmInputs = getMcmInputs(roipool, mcmOutputsMap);
    IE_ASSERT(2 == mcmInputs.size());
    const auto& opName = roipool->get_friendly_name();
    const double spatial_scale = roipool->get_spatial_scale();
    const std::string method = roipool->get_method();
    const unsigned roi_pooling_method = (method == "bilinear") ? 1 : 0;
    unsigned num_rois = roipool->get_input_shape(0)[0];
    unsigned pooled_h = roipool->get_output_shape(0)[2];
    unsigned pooled_w = roipool->get_output_shape(0)[3];

    const auto roipoolOutput = mcmModel.rOIPooling(opName, mcmInputs, pooled_w, pooled_h,
        spatial_scale, roi_pooling_method, num_rois);
    roipoolOutput->setQuantParams(initialQuantParams());
    registerOutputs(roipool, {roipoolOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::PriorBoxIE> priorbox, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(priorbox, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = priorbox->get_friendly_name();
    const auto attrs = priorbox->get_attrs();
    // min_size         Desired min_size of prior boxes
    // max_size         Desired max_size of prior boxes
    // aspect_ratio     Aspect ratios of prior boxes
    // clip             Clip output to [0,1]
    // flip             Flip aspect ratios
    // step             Distance between prior box centers
    // offset           Box offset relative to top center of image
    // variance         Values to adjust prior boxes with
    // scale_all_sizes  Scale all sizes

    if (mcmInputs.size() != 2)
        THROW_IE_EXCEPTION << opName + " Incorrect number of input edges!";

    if (priorbox->get_input_shape(0).size() != 4 ||
        priorbox->get_input_shape(1).size() != 4)
        THROW_IE_EXCEPTION << opName + " PriorBox supports only 4D blobs!";
    auto data_dims = priorbox->get_input_shape(0);
    auto image_dims = priorbox->get_input_shape(1);
    auto out_dims = priorbox->get_output_shape(0);

    vpu::KmbPlugin::utils::priorBoxParam param(attrs.offset, attrs.step, attrs.min_size, attrs.max_size, attrs.flip, attrs.clip, attrs.scale_all_sizes,
     attrs.fixed_size, attrs.fixed_ratio, attrs.density, attrs.aspect_ratio, attrs.variance, data_dims, image_dims, out_dims);

    auto boxes = vpu::KmbPlugin::utils::computePriorbox(param);
    auto priorboxOutput = mcmModel.constant(opName + "_const", boxes, {boxes.size() / 2, 2, 1, 1}, mv::DType("Float64"), mv::Order("NHWC"));
    priorboxOutput->setQuantParams(initialQuantParams());

    registerOutputs(priorbox, {priorboxOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::PriorBoxClusteredIE> pbc, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    // const auto& opName = pbc->get_friendly_name();
    const auto attrs = pbc->get_attrs();
    // widths         Desired widths of prior boxes
    // heights        Desired heights of prior boxes
    // clip           Clip output to [0,1]
    // step_widths    Distance between prior box centers
    // step_heights   Distance between prior box centers
    // offset         Box offset relative to top center of image
    // variances      Values to adjust prior boxes with
    int img_width = pbc->get_input_shape(1).at(3);
    int img_height = pbc->get_input_shape(1).at(2);
    int layer_width = pbc->get_input_shape(0).at(3);
    int layer_height = pbc->get_input_shape(0).at(2);
    float step_w = attrs.step_widths;
    float step_h = attrs.step_heights;
    // if (std::abs(attr.step_heights - attr.step_widths) < 1e-5) {
    //     res->params["step"] = asString(attr.step_widths);
    if (step_w == 0.f && step_h == 0.f) {
        step_w = static_cast<float>(img_width) / layer_width;
        step_h = static_cast<float>(img_height) / layer_height;
    }
    IE_ASSERT(step_w != 0.f);
    IE_ASSERT(step_h != 0.f);
    IE_ASSERT(attrs.widths.size() == attrs.heights.size());
    int num_priors = attrs.widths.size();
    std::vector<float> variances = attrs.variances;
    if (variances.empty()) {
        variances.push_back(0.1f);
    }
    const auto& dims = pbc->get_output_shape(0);
    IE_ASSERT(dims.size() == 3);
    int size = dims[0] * dims[1] * dims[2];

    vpu::KmbPlugin::utils::priorBoxClusteredParam param{attrs.offset, attrs.clip,
        step_w, step_h, layer_width, layer_height, img_width,
        img_height, num_priors, attrs.widths, attrs.heights, variances, size};

    auto boxes = vpu::KmbPlugin::utils::computePriorboxClustered(param);
    auto priorboxClustered =
            mcmModel.constant("", boxes, {boxes.size() / 2, 2, 1, 1}, mv::DType("Float64"), mv::Order("NHWC"));

    registerOutputs(pbc, {priorboxClustered}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::NormalizeL2> normL2, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(normL2, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 2);
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = normL2->get_friendly_name();

    double eps = normL2->get_eps();
    auto const_axis = std::dynamic_pointer_cast<ngraph::op::Constant> (normL2->input(1).get_source_output().get_node_shared_ptr());
    IE_ASSERT(nullptr != const_axis);
    auto axis = const_axis->cast_vector<size_t>();
    bool across_spatial = !(axis.size() == 1 && axis[0] == 1);

    size_t weightsSize = mcmData->getShape()[2];
    const mv::Shape weightsShape = {1, weightsSize, 1, 1};
    std::vector<double> weightsData (weightsSize, 1.0); // see convert_normalizel2_to_normalize_ie.cpp
    bool channel_shared = false;

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    auto mvWeightsValues = mcmModel.constant("", weightsData, weightsShape, mv::DType("Float32"), mv::Order::getZMajorID(4));

    auto mvNormalizeOutput = mcmModel.normalize(opName, mcmData, mvWeightsValues, eps, across_spatial, channel_shared);
    mvNormalizeOutput->setQuantParams(initialQuantParams());
    registerOutputs(normL2, {mvNormalizeOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Sigmoid> sigmoid, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(sigmoid, mcmOutputsMap).at(0);
    const auto& opName = sigmoid->get_friendly_name();
    const auto mcmSigmoidOutput = mcmModel.sigmoid(opName, mcmData);
    mcmSigmoidOutput->setQuantParams(initialQuantParams());
    registerOutputs(sigmoid, {mcmSigmoidOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::DetectionOutput> detection, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(detection, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 3);
    const auto& opName = detection->get_friendly_name();
    const auto attrs = detection->get_attrs();
    // int num_classes;
    // int background_label_id = 0;
    // int top_k = -1;
    // bool variance_encoded_in_target = false;
    // std::vector<int> keep_top_k = {1};
    // std::string code_type = std::string{"caffe.PriorBoxParameter.CORNER"};
    // bool share_location = true;
    // float nms_threshold;
    // float confidence_threshold = std::numeric_limits<float>::min();
    // bool clip_after_nms = false;
    // bool clip_before_nms = false;
    // bool decrease_label_id = false;
    // bool normalized = false;
    // size_t input_height = 1;
    // size_t input_width = 1;
    // float objectness_score = 0;

    int64_t keep_top_k = attrs.keep_top_k.at(0);

    auto mcmDetectionOutput = mcmModel.detectionOutput(opName, mcmInputs, attrs.num_classes, keep_top_k, attrs.nms_threshold,
        attrs.background_label_id, attrs.top_k, attrs.variance_encoded_in_target, attrs.code_type, attrs.share_location, attrs.confidence_threshold,
        attrs.clip_before_nms, attrs.clip_after_nms, attrs.decrease_label_id, attrs.normalized, attrs.input_height, attrs.input_width, attrs.objectness_score);
    mcmDetectionOutput->setQuantParams(initialQuantParams());

    registerOutputs(detection, {mcmDetectionOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::ReorgYolo> reorg, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(reorg, mcmOutputsMap).at(0);
    const auto& opName = reorg->get_friendly_name();
    const std::size_t stride = reorg->get_strides().at(0);

    for (const auto& s : reorg->get_strides()) {
        IE_ASSERT(stride == s);
    }

    const auto mcmReorgYoloOutput = mcmModel.reorgYolo(opName, mcmData, static_cast<unsigned>(stride));
    mcmReorgYoloOutput->setQuantParams(initialQuantParams());
    registerOutputs(reorg, {mcmReorgYoloOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::RegionYolo> region, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(region, mcmOutputsMap).at(0);
    const auto& opName = region->get_friendly_name();
    const std::size_t coords = region->get_num_coords();
    const std::size_t classes = region->get_num_classes();
    const std::size_t num = region->get_num_regions();
    const bool do_softmax = region->get_do_softmax();
    const std::vector<int64_t> mask = region->get_mask();
    const std::vector<unsigned> mcmMask(mask.begin(), mask.end());

    const auto mcmRegionYoloOutput = mcmModel.regionYolo(
        opName, mcmData,
        static_cast<unsigned>(coords),
        static_cast<unsigned>(classes),
        do_softmax,
        static_cast<unsigned>(num),
        mcmMask);
    mcmRegionYoloOutput->setQuantParams(initialQuantParams());
    registerOutputs(region, {mcmRegionYoloOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::TopK> topk, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(topk, mcmOutputsMap);
    IE_ASSERT(2 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = topk->get_friendly_name();
    uint64_t axis = topk->get_axis();
    std::string mode = ngraph::as_string<ngraph::op::v1::TopK::Mode>(topk->get_mode());
    std::string sort = ngraph::as_string<ngraph::op::v1::TopK::SortType>(topk->get_sort_type());

    auto const_k = std::dynamic_pointer_cast<ngraph::op::Constant> (topk->input(1).get_source_output().get_node_shared_ptr());
    if (!const_k) // can be dynamic. see top_k_to_top_k_ie
        THROW_IE_EXCEPTION << opName + " has non-constant k";
    int32_t k = const_k->cast_vector<int32_t>().at(0); // topk->get_k();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    const auto mcmTopKOutput = mcmModel.topK(opName, mcmData, sort, mode, k, axis);
    mcmTopKOutput->setQuantParams(initialQuantParams());

    auto topKOp = mcmModel.getSourceOp(mcmTopKOutput);
    const auto outputSlots = topKOp->outputSlots();

    if (1 == outputSlots)
        registerOutputs(topk, {mcmTopKOutput}, mcmOutputsMap);
    else if (2 == outputSlots)
        registerOutputs(topk, {mcmTopKOutput, topKOp->getOutputTensor(1)}, mcmOutputsMap);
    else
        THROW_IE_EXCEPTION << opName + " has too many outputs " << outputSlots;
}

void convert(std::shared_ptr<ngraph::op::TopKIE> topk, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(topk, mcmOutputsMap);
    IE_ASSERT(2 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = topk->get_friendly_name();
    uint64_t axis = topk->get_axis();
    std::string mode = ngraph::as_string<ngraph::op::v1::TopK::Mode>(topk->get_mode());
    std::string sort = ngraph::as_string<ngraph::op::v1::TopK::SortType>(topk->get_sort_type());

    auto const_k = std::dynamic_pointer_cast<ngraph::op::Constant> (topk->input(1).get_source_output().get_node_shared_ptr());
    if (!const_k) // can be dynamic. see top_k_to_top_k_ie
        THROW_IE_EXCEPTION << opName + " has non-constant k";
    int32_t k = const_k->cast_vector<int32_t>().at(0); // topk->get_k();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    const auto mcmTopKOutput = mcmModel.topK(opName, mcmData, sort, mode, k, axis);
    mcmTopKOutput->setQuantParams(initialQuantParams());

    auto topKOp = mcmModel.getSourceOp(mcmTopKOutput);
    const auto outputSlots = topKOp->outputSlots();

    if (1 == outputSlots)
        registerOutputs(topk, {mcmTopKOutput}, mcmOutputsMap);
    else if (2 == outputSlots)
        registerOutputs(topk, {mcmTopKOutput, topKOp->getOutputTensor(1)}, mcmOutputsMap);
    else
        THROW_IE_EXCEPTION << opName + " has too many outputs " << outputSlots;
}

const static std::map<std::string, std::string> interpolationMap = {
        {"nearest", "NEAREST"},
        {"cubic", "BICUBIC"},
        {"linear", "BILINEAR"},
};

void convert(std::shared_ptr<ngraph::op::ResampleV2> resample, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(resample, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = resample->get_friendly_name();
    const auto antialias = false;
    const auto& resampleAttrs = resample->get_attrs();
    std::string mode = "nearest";
    if (resampleAttrs.mode != "") {
        mode = resampleAttrs.mode;
    }

    mv::Shape output_shape = getWHCN(resample->get_output_shape(0));
    auto mcmResampleOutput = mcmModel.resample(opName, mcmData, interpolationMap.at(mode), antialias, output_shape);
    mcmResampleOutput->setQuantParams(initialQuantParams());

    registerOutputs(resample, {mcmResampleOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::Interp> interp, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(interp, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = interp->get_friendly_name();
    auto interpAttrs = interp->get_attrs();
    auto factor = interpAttrs.scale_factor;
    auto height = interpAttrs.height;
    auto width = interpAttrs.width;
    auto pad_begin = interpAttrs.pad_beg;
    auto pad_end = interpAttrs.pad_end;
    auto align_corners = interpAttrs.align_corners;

    auto mcmInterpOutput = mcmModel.interp(opName, mcmData, factor, pad_begin, pad_end, height, width, align_corners);
    mcmInterpOutput->setQuantParams(initialQuantParams());
    registerOutputs(interp, {mcmInterpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::DeconvolutionIE> deconvIE, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(deconvIE, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size() || 3u == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto mcmWeights = mcmInputs.at(1);
    const auto& opName = deconvIE->get_friendly_name();

    const auto& strides = deconvIE->get_strides();
    const auto& padsBegin = deconvIE->get_pads_begin();
    const auto& padsEnd = deconvIE->get_pads_end();
    const auto& dilations = deconvIE->get_dilations();
    const size_t& groupSize = deconvIE->get_group();

    const auto ngFilterShape = deconvIE->get_input_shape(1);
    const auto filterShape = mcmWeights->getShape();
    IE_ASSERT(4 == filterShape.ndims());
    IE_ASSERT(4 == ngFilterShape.size());

    size_t kernelSizeX = filterShape[0];
    size_t kernelSizeY = filterShape[1];
    const int kernelStrideX = strides.at(1);
    const int kernelStrideY = strides.at(0);
    const auto dilationX = dilations.at(1);
    const auto dilationY = dilations.at(0);

    IE_ASSERT(2u == padsBegin.size());
    IE_ASSERT(2u == padsEnd.size());
    int padLeft = padsBegin.at(1);
    int padRight = padsEnd.at(1);
    int padTop = padsBegin.at(0);
    int padBottom = padsEnd.at(0);

    if (dilationX != dilationY)
        THROW_IE_EXCEPTION << "Deconvolution supports only equal dilationX and dilationY";

    // TODO: implement cvtPaddingsFromCeilToFloorMode for deconv, existing func does not suit

    mv::Data::TensorIterator mcmDeconv;
    mv::Data::TensorIterator mcmDeconvOnly;

    auto inputShape = deconvIE->get_input_shape(0);
    auto outputShape = deconvIE->get_output_shape(0);
    IE_ASSERT(4 == inputShape.size());
    IE_ASSERT(4 == outputShape.size());
    const auto inputGroupSize = inputShape.at(1);
    const auto outputGroupSize = outputShape.at(1);

    bool isDepthWiseConv = (groupSize > 1) && (groupSize == inputGroupSize) && (groupSize == outputGroupSize);

    if (isDepthWiseConv) {
        IE_ASSERT(2u == mcmInputs.size());
        /* TODO: Need align API in mcmCompiler
           mcm expects (1,*,*,*) shape for depthwise weights, but Openvino has a (*,1,*,*) */
        //auto weights = layer->blobs["weights"];
        //auto weightsData = mcmWeights packBlobToVector<double>(weights, weights->size());

        const mv::Shape mcmShape = {static_cast<uint64_t>(kernelSizeY), static_cast<uint64_t>(kernelSizeX), groupSize, 1lu};
        IE_ASSERT(mcmWeights->getShape() == mcmShape);
        IE_ASSERT(mcmWeights->getDType() == mv::DType("Float32"));

        mcmWeights->setOrder(mv::Order::getZMajorID(mcmShape.ndims())); // TODO
        IE_ASSERT(mv::Order(mv::Order::getZMajorID(mcmShape.ndims())) == mcmWeights->getOrder());

        mcmWeights->set<bool>("is_depthwise_weights", true);

        mcmDeconv = mcmModel.deconv(opName, mcmData, mcmWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX), static_cast<unsigned>(groupSize), true);
    } else {
        const mv::Shape mcmShape = {static_cast<uint64_t>(kernelSizeY), static_cast<uint64_t>(kernelSizeX), inputGroupSize, outputGroupSize};

        const auto ngraphWeights = std::dynamic_pointer_cast<ngraph::op::Constant>(deconvIE->input_value(1).get_node_shared_ptr());
        IE_ASSERT(nullptr != ngraphWeights);
        const std::vector<double> weightsData = ngraphWeights->cast_vector<double>();
        std::vector<double> weightsDataReorder(weightsData.size());

        for (size_t k = 0; k < outputGroupSize; k++)
            for (size_t c = 0; c < inputGroupSize; c++)
                for (size_t h = 0; h < kernelSizeY; h++)
                    for (size_t w = 0; w < kernelSizeX; w++) {
                        size_t src_idx = c * outputGroupSize * kernelSizeY * kernelSizeX +
                                         k * kernelSizeY * kernelSizeX + h * kernelSizeX + w;
                        size_t dst_idx = k * inputGroupSize * kernelSizeY * kernelSizeX +
                                         c * kernelSizeY * kernelSizeX + h * kernelSizeX + w;
                        weightsDataReorder[dst_idx] = weightsData[src_idx];
                    }

        mcmModel.removeOp(mcmModel.getSourceOp(mcmWeights));
        const auto mcmWeightsReordered = mcmModel.constant("", weightsDataReorder, mcmShape, mv::DType("Float32"), mv::Order("NCHW"));

        mcmDeconv = mcmModel.deconv(opName, mcmData, mcmWeightsReordered,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight), static_cast<uint16_t>(padTop),
                static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX), static_cast<unsigned>(groupSize), false);
    }
    mcmDeconv->setQuantParams(initialQuantParams());

    if (3u == mcmInputs.size()) {
        const auto mcmBiases = mcmInputs.at(2);
        IE_ASSERT(1u == deconvIE->get_input_shape(2).size());
        mv::Shape mcmShape = {deconvIE->get_input_shape(2).at(0)};

        IE_ASSERT(mcmBiases->getShape() == mcmShape);
        IE_ASSERT(mcmBiases->getDType() == mv::DType("Float32"));
        IE_ASSERT(mcmBiases->getOrder() == mv::Order(mv::Order::getColMajorID(mcmShape.ndims())));

        mcmDeconvOnly = mcmDeconv;
        mcmDeconv = mcmModel.bias(opName + "_bias", mcmDeconvOnly, mcmBiases);
        mcmDeconv->setQuantParams(initialQuantParams());
    }

    registerOutputs(deconvIE, {mcmDeconv}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::CropIE> crop, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(crop, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const std::vector<int64_t>& axes = crop->axes;     // number of a dimension to crop
    const std::vector<int64_t>& dim = crop->dim;       // starting point for crop in the input blob
    const std::vector<int64_t>& offset = crop->offset; // resulting size of the output blob for the specified axis
    const mv::Shape outShape = getWHCN(crop->get_output_shape(0));
    const std::size_t ndims = outShape.ndims();

    if (ndims == axes.size() && ndims == offset.size() && ndims == dim.size()) {
        mv::Shape mvOffsets(ndims);
        mv::Shape mvOutDims(ndims);
        // fill offsets and out dimensions size with conversion NCHW->WHCN
        for (std::size_t i = 0; i < ndims; ++i) {
            mvOffsets[ndims - 1 - axes[i]] = offset[i];
            mvOutDims[ndims - 1 - axes[i]] = dim[i];
        }
        if (mvOutDims != outShape)
            THROW_IE_EXCEPTION << "Crop layer dim parameter mismatches output shape";
        // mcmModel.crop() is single dimensional and mcmModel.slice() is multdimensional
        auto mcmSlice = mcmModel.slice(crop->get_friendly_name(), mcmInputs.at(0), mvOffsets, outShape);
        mcmSlice->setQuantParams(initialQuantParams());
        registerOutputs(crop, {mcmSlice}, mcmOutputsMap);
    } else {
        THROW_IE_EXCEPTION << "Unsupported Crop layer parameters:"
            << " axes.size() = " << axes.size()
            << ", offset.size() = " << offset.size()
            << ", dims.size() = " << dim.size();
    }
}

void convert(std::shared_ptr<ngraph::op::v0::Exp> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto mcmOpOutput = mcmModel.exp(op->get_friendly_name(), mcmInputs.at(0));
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Tanh> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto mcmOpOutput = mcmModel.tanh(op->get_friendly_name(), mcmInputs.at(0));
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Multiply> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());
    const auto mcmOpOutput = mcmModel.eltwise(op->get_friendly_name(), mcmInputs, "Multiply");
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::NormalizeIE> normalizeIE, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(normalizeIE, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 2);
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = normalizeIE->get_friendly_name();

    const double eps = normalizeIE->get_eps();
    auto const_axis = std::dynamic_pointer_cast<ngraph::op::Constant> (normalizeIE->input(1).get_source_output().get_node_shared_ptr());
    IE_ASSERT(nullptr != const_axis);
    const auto& axis = const_axis->cast_vector<size_t>();
    const bool across_spatial = !(axis.size() == 1 && axis[0] == 1);

    const size_t weightsSize = mcmData->getShape()[2];
    const mv::Shape weightsShape = {1, weightsSize, 1, 1};
    std::vector<double> weightsData (weightsSize, 1.0); // see convert_normalizel2_to_normalize_ie.cpp
    bool channel_shared = false;

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    auto mvWeightsValues = mcmModel.constant("", weightsData, weightsShape, mv::DType("Float32"), mv::Order::getZMajorID(4));

    auto mvNormalizeOutput = mcmModel.normalize(opName, mcmData, mvWeightsValues, eps, across_spatial, channel_shared);
    mvNormalizeOutput->setQuantParams(initialQuantParams());
    registerOutputs(normalizeIE, {mvNormalizeOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::LRN_IE> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(op, mcmOutputsMap).at(0);
    auto mcmNorm = mcmModel.norm(op->get_friendly_name(), mcmData,
        op->get_alpha(), op->get_beta(), op->get_region(), static_cast<unsigned>(op->get_nsize()));
    mcmNorm->setQuantParams(initialQuantParams());

    registerOutputs(op, {mcmNorm}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::ProposalIE> proposalIE, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(proposalIE, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 3u);
    const auto& opName = proposalIE->get_friendly_name();
    const ngraph::op::ProposalAttrs& attrs = proposalIE->get_attrs();
    // size_t base_size;                  // Anchor sizes
    // size_t pre_nms_topn;               // Number of boxes before nms
    // size_t post_nms_topn;              // Number of boxes after nms
    // float nms_thresh = 0.0f;           // Threshold for nms
    // size_t feat_stride = 1;            // Feature stride
    // size_t min_size = 1;               // Minimum box size
    // std::vector<float> ratio;          // Ratios for anchor generation
    // std::vector<float> scale;          // Scales for anchor generation
    // bool clip_before_nms = true;       // Clip before NMs
    // bool clip_after_nms = false;       // Clip after NMs
    // bool normalize = false;            // Normalize boxes to [0,1]
    // float box_size_scale = 1.0f;       // Scale factor for scaling box size
    // float box_coordinate_scale = 1.0f; // Scale factor for scaling box coordiate
    // std::string framework;             // Calculation frameworkrithm to use
    // bool infer_probs = false;

    // ngraph does not have these params
    auto for_deformable = false;
    float pre_nms_thresh = 0.0f;

    std::string framework = attrs.framework;
    if (framework == "") framework = "caffe"; // IE assumes empty is "caffe" like.
    if (framework == "tensorflow" || framework == "caffe") {
        std::transform(framework.begin(), framework.end(), framework.begin(), ::toupper);
    } else {
        THROW_IE_EXCEPTION << "Proposal layer doesn't support framework: \'" << framework << "\'";
    }

    auto mcmProposal = mcmModel.proposal(opName, mcmInputs, attrs.scale, attrs.ratio, attrs.base_size, attrs.pre_nms_topn,
        attrs.post_nms_topn, attrs.nms_thresh, attrs.feat_stride, attrs.min_size, pre_nms_thresh,
        attrs.clip_before_nms, attrs.clip_after_nms,  attrs.normalize, attrs.box_size_scale,
        attrs.box_coordinate_scale, framework, for_deformable);
    mcmProposal->setQuantParams(initialQuantParams());

    registerOutputs(proposalIE, {mcmProposal}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::GatherIE> gatherIE, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(gatherIE, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());
    const auto& opName = gatherIE->get_friendly_name();
    const int64_t axis = gatherIE->get_axis();
    // TODO: Replace Float16 with Default when MCM Compiler is fixed. See ticket #40356
    auto mcmGather = mcmModel.gather(opName, mcmInputs.at(0), mcmInputs.at(1), axis);
    mcmGather->setDType(mv::DType("Float16"));
    mcmGather->setQuantParams(initialQuantParams());

    registerOutputs(gatherIE, {mcmGather}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Maximum> maximum, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(maximum, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());
    const auto& opName = maximum->get_friendly_name();
    auto mcmMax = mcmModel.eltwise(opName, mcmInputs, "Maximum");
    registerOutputs(maximum, {mcmMax}, mcmOutputsMap);
}


void convert(std::shared_ptr<ngraph::op::v1::Minimum> minimum, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(minimum, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());
    const auto& opName = minimum->get_friendly_name();
    auto mcmMin = mcmModel.eltwise(opName, mcmInputs, "Minimum");
    registerOutputs(minimum, {mcmMin}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Split> split, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto& opName = split->get_friendly_name();
    const auto mcmInputs = getMcmInputs(split, mcmOutputsMap);
    assert(mcmInputs.size() == 1);

    // Find axis.
    const auto axis_node = split->input_value(1).get_node_shared_ptr();
    const auto axis_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(axis_node);
    auto axis = axis_node_const->get_data_ptr<int64_t>()[0];

    std::vector<size_t> startCoords(mcmInputs.at(0)->getShape().ndims());
    std::vector<mv::Data::TensorIterator> mcmOutputs;
    auto outDimSize = split->get_output_shape(0).size();
    for (size_t i = 0; i < split->get_output_size(); ++i) {
        mv::Shape beginShape(startCoords);
        mv::Shape sizeShape(getWHCN(split->get_output_shape(i)));
        auto mcmSplit = mcmModel.slice(opName + ":" + std::to_string(i), mcmInputs.at(0), beginShape, sizeShape);
        mcmOutputs.push_back(mcmSplit);
        startCoords[outDimSize - 1 - axis] += split->get_output_shape(i)[axis];
    }
    registerOutputs(split, mcmOutputs, mcmOutputsMap);
}

// TODO: move converters to class ConvertToMcmModel scope to remove references to data

template <typename T>
void convertDispatch(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr /*unused*/, bool /*unused*/) {
    convert(std::dynamic_pointer_cast<T>(node), mcmModel, mcmOutputsMap);
}

// Propagate ieData precision to MCM in order to perform conversion on hardware
template<>
void convertDispatch<ngraph::op::Parameter>(std::shared_ptr<ngraph::Node> node,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap, InferenceEngine::DataPtr ieData, bool allowNCHWInput) {
    convert(std::dynamic_pointer_cast<ngraph::op::Parameter>(node), mcmModel, mcmOutputsMap, ieData, allowNCHWInput);
}

template<>
void convertDispatch<ngraph::op::Result>(std::shared_ptr<ngraph::Node> node,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap, InferenceEngine::DataPtr ieData, bool /*unused*/) {
    convert(std::dynamic_pointer_cast<ngraph::op::Result>(node), mcmModel, mcmOutputsMap, ieData);
}

#define MAP_ENTRY(__OP__) {__OP__::type_info, convertDispatch<__OP__>}

static const DispatchMap dispatchMap {
    MAP_ENTRY(ngraph::op::Parameter),
    MAP_ENTRY(ngraph::op::Result),
    MAP_ENTRY(ngraph::op::Constant),
    MAP_ENTRY(ngraph::op::v0::ROIPooling),
    MAP_ENTRY(McmConv),
    MAP_ENTRY(McmBias),
    MAP_ENTRY(McmScale),
    MAP_ENTRY(ngraph::op::v1::MaxPool),
    MAP_ENTRY(ngraph::op::v0::Relu),
    MAP_ENTRY(McmEltwise),
    MAP_ENTRY(ngraph::op::Eltwise),
    MAP_ENTRY(ngraph::op::v0::FakeQuantize),
    MAP_ENTRY(ngraph::op::v1::AvgPool),
    MAP_ENTRY(ngraph::op::v1::Reshape),
    MAP_ENTRY(McmFC),
    MAP_ENTRY(ngraph::op::v0::Concat),
    MAP_ENTRY(ngraph::op::v1::Transpose),
    MAP_ENTRY(ngraph::op::v0::Squeeze),
    MAP_ENTRY(ngraph::op::v1::Softmax),
    MAP_ENTRY(ngraph::op::v0::Clamp),
    MAP_ENTRY(ngraph::op::ReLUIE),
    MAP_ENTRY(ngraph::op::v0::NormalizeL2),
    MAP_ENTRY(ngraph::op::PriorBoxIE),
    MAP_ENTRY(ngraph::op::v0::Unsqueeze),
    MAP_ENTRY(ngraph::op::PowerIE),
    MAP_ENTRY(ngraph::op::v0::Sigmoid),
    MAP_ENTRY(ngraph::op::PriorBoxClusteredIE),
    MAP_ENTRY(ngraph::op::v0::DetectionOutput),
    MAP_ENTRY(ngraph::op::v0::RegionYolo),
    MAP_ENTRY(ngraph::op::v0::ReorgYolo),
    MAP_ENTRY(ngraph::op::v1::TopK),
    MAP_ENTRY(ngraph::op::TopKIE),
    MAP_ENTRY(ngraph::op::ResampleV2),
    MAP_ENTRY(ngraph::op::Interp),
    MAP_ENTRY(ngraph::op::DeconvolutionIE),
    MAP_ENTRY(ngraph::op::CropIE),
    MAP_ENTRY(ngraph::op::v0::Exp),
    MAP_ENTRY(ngraph::op::v0::Tanh),
    MAP_ENTRY(ngraph::op::v1::Multiply),
    MAP_ENTRY(ngraph::op::LRN_IE),
    MAP_ENTRY(ngraph::op::NormalizeIE),
    MAP_ENTRY(ngraph::op::ProposalIE),
    MAP_ENTRY(ngraph::op::GatherIE),
    MAP_ENTRY(ngraph::op::v0::Elu),
    MAP_ENTRY(ngraph::op::v1::Maximum),
    MAP_ENTRY(ngraph::op::v1::Minimum),
    MAP_ENTRY(ngraph::op::v1::Split),
    MAP_ENTRY(ngraph::op::v4::HSwish)
};

#undef MAP_ENTRY

void ConvertNode(const std::shared_ptr<ngraph::Node> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr ieData, bool allowNCHWInput) {
    const auto dispatchIt = dispatchMap.find(op->get_type_info());
    if (dispatchIt != dispatchMap.end()) {
        const auto convertor = dispatchIt->second;
        if (convertor != nullptr) {
            try {
                convertor(op, mcmModel, mcmOutputsMap, ieData, allowNCHWInput);
            } catch (const std::runtime_error& ex) {
                THROW_IE_EXCEPTION << "Convertor for operation " << op->get_friendly_name()
                                   << " failed due to runtime error " << ex.what();
            }
        } else {
            THROW_IE_EXCEPTION << "Convertor not found for operation: " << op->get_friendly_name();
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported operation: " << op->get_friendly_name() << " with name " << op->get_name()
                           << " with type " << op->get_type_name() << " with C++ type " << typeid(*op.get()).name();
    }
}

}  // namespace

// clang-format on

void ConvertToMcmModel::parseCustom(
    std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(node, mcmOutputsMap);

    auto parser = vpu::CustomLayerParserNGraph(node, mcmInputs);

    const auto customLayer = [&] {
        const auto customLayersForType = _customLayers.find(node->description());
        IE_ASSERT(customLayersForType != _customLayers.end());
        const auto suitableLayers = vpu::getSuitableCustomLayers(customLayersForType->second, node);
        IE_ASSERT(!suitableLayers.empty());
        return vpu::findMatchingCustomLayer(suitableLayers, mcmInputs);
    }();

    int stageIdx = 0;
    for (const auto& kernel : customLayer->kernels()) {
        const auto sortedKernelBindings = [&] {
            auto bindings = std::vector<vpu::CustomKernel::BindingParameter>{};
            bindings.reserve(kernel.arguments().size());

            for (const auto& arg : kernel.arguments()) {
                const auto& binding = kernel.bindings().find(arg.name);
                VPU_THROW_UNLESS(binding != kernel.bindings().end(),
                    "Failed to bind '%s' custom layer. "
                    "Can't find kernel argument '%s' in binding list.",
                    customLayer->layerName(), arg.name);
                bindings.push_back(binding->second);
            }

            return bindings;
        }();

        const auto stage = parser.parseKernelArguments(sortedKernelBindings);

        const auto kernelData = parser.resolveKernelArguments(kernel, stage.arguments);
        const auto stageOutputs = parser.resolveStageOutputs(kernel, *customLayer, stage.outputs);

        const auto layerName = node->get_friendly_name() + "_custom" +
                               (customLayer->kernels().size() > 1 ? (":" + std::to_string(stageIdx)) : (""));
        stageIdx++;

        auto custom = mcmModel.custom(layerName, stage.inputs, kernel.kernelBinary(), kernelData, stageOutputs);

        const auto sourceOp = mcmModel.getSourceOp(custom);
        const auto mcmOutputTensors = sourceOp->getOutputTensor();

        IE_ASSERT(stage.outputs.size() == mcmOutputTensors.size());

        for (size_t i = 0; i < stage.outputs.size(); i++) {
            const auto& output = stage.outputs[i];
            if (output.isBuffer) {
                parser.addBuffer(output.portIndex, mcmOutputTensors[i]);
            } else {
                registerOutputs(node, {custom}, mcmOutputsMap);
            }
        }
    }
}

bool ConvertToMcmModel::run_on_function(std::shared_ptr<ngraph::Function> func) {
    // Ngraph representation and IE CNNNetwork may have inputs and outpus in different order.
    // MCM compiler processes inputs and outputs by add-to-model order, not by their name.
    // Therefore plugin must reorder them manually to follow IE CNNNetwork
    // Also propagate IE input/output precision/layout to MCM, so conversion will be done on
    // hardware.

    // FIXME
    // McmModel hard-codes NHWC layout for all of its inputs
    // Provide an opportunity to use NCHW layout for McmModel inputs
    const auto allowNCHWInput = _config.allowNCHWLayoutForMcmModelInput();

    for (const auto& inputInfo : _networkInputs) {
        bool isFound = false;
        for (const auto& op : func->get_parameters()) {
            if (op->get_friendly_name() == _ioMap.at(inputInfo.first)) {
                ConvertNode(op, _mcmModel, _mcmOutputsMap, inputInfo.second->getInputData(), allowNCHWInput);
                isFound = true;
            }
        }
        if (!isFound) THROW_IE_EXCEPTION << "Input not found: " << inputInfo.first;
    }

    if (!_config.customLayers().empty()) {
        _customLayers = vpu::CustomLayer::loadFromFile(_config.customLayers());
    }

    for (const auto& op : func->get_ordered_ops()) {
        if (ngraph::op::Constant::type_info == op->get_type_info()) {
            ConvertNode(op, _mcmModel, _mcmOutputsMap, nullptr, false);
        }
    }

    for (const auto& op : func->get_ordered_ops()) {
        if (ngraph::op::Parameter::type_info == op->get_type_info()) continue;
        if (ngraph::op::Result::type_info == op->get_type_info()) continue;
        if (ngraph::op::Constant::type_info == op->get_type_info()) continue;

        const auto customLayersForType = _customLayers.find(op->description());

        if (customLayersForType != _customLayers.end()) {
            const auto suitableLayers = getSuitableCustomLayers(customLayersForType->second, op);
            if (!suitableLayers.empty()) {
                parseCustom(op, _mcmModel, _mcmOutputsMap);
                continue;
            }
        }

        ConvertNode(op, _mcmModel, _mcmOutputsMap, nullptr, false);
    }

    for (const auto& outputInfo : _networkOutputs) {
        bool isFound = false;
        for (const auto& op : func->get_results()) {
            if (op->get_friendly_name() == _ioMap.at(outputInfo.first)) {
                ConvertNode(op, _mcmModel, _mcmOutputsMap, outputInfo.second, false);
                isFound = true;
            }
        }
        if (!isFound) THROW_IE_EXCEPTION << "Output not found: " << outputInfo.first;
    }

    return false;
}
