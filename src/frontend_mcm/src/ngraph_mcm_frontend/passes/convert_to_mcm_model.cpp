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
#include <ngraph_ops/power.hpp>
#include <ngraph/op/normalize_l2.hpp>

#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"

#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"

#include <ngraph/op/fake_quantize.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph_ops/scaleshift.hpp>

#include <ngraph/op/parameter.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/constant.hpp>

#include <ngraph/op/transpose.hpp>
#include <ngraph/op/squeeze.hpp>
#include <ngraph/op/unsqueeze.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/op/topk.hpp>

#include <ngraph/op/prior_box.hpp>
#include <ngraph/op/prior_box_clustered.hpp>
#include <ngraph/op/detection_output.hpp>

#include <ngraph_ops/interp.hpp>
#include <ngraph_ops/prior_box_clustered_ie.hpp>
#include <ngraph_ops/prior_box_ie.hpp>
#include <ngraph_ops/normalize_ie.hpp>
#include <ngraph_ops/topk_ie.hpp>

#include <parse_layers_helpers.hpp>
#include <dims_parser.hpp>
#include "ngraph_mcm_frontend/ie_helpers.hpp"

#include <memory>
#include <vector>
#include <map>

#include <include/mcm/tensor/tiling.hpp>

namespace {

using Callback = void (*)(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
InferenceEngine::DataPtr);
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


void registerOutputs(std::shared_ptr<ngraph::Node> node, std::vector<mv::Data::TensorIterator> mcmOutputs, NodeOutputToMcmMap& mcmOutputsMap) {
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

void convert(std::shared_ptr<ngraph::op::Parameter> param, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr ieData) {
    auto mvShape = getWHCN(param->get_shape());
    // Use data from InputInfo DataPtr
    // const auto mvDType = mv::DType("UInt8"); // Test framework sets fp32, cvtElemTypeToMCM(param->get_element_type());
    // const auto mvOrder = mv::Order("NHWC"); // TBD how to get layout from ngraph function?
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(param);
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

    // TBD
    // const auto mvOrder = (inputLayout == ie::Layout::NCHW /*&& _config.allowNCHWLayoutForMcmModelInput()*/) ? mv::Order("NCHW") : mv::Order("NHWC");
    const auto mvOrder = mv::Order("NHWC");
    const auto mvDType = cvtElemTypeToMCM(cvtPrecisionToElemType(inputPrecision));
    // MCM Compiler requirements
    // IE_ASSERT(mv::DType("Float16") == mvDType || mv::DType("UInt8") == mvDType);
    // IE_ASSERT(mv::Order("NHWC") == mvOrder);
    const auto mcmOutput = mcmModel.input(mvShape, mvDType, mvOrder, mvQuantParams, mvNetworkInput, opName);

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

    // MCM Compiler requirements
    // IE_ASSERT(mv::DType("Float16") == mvDType || mv::DType("UInt8") == mvDType);
    IE_ASSERT(mv::DType("Float16") == outputType || mv::DType("UInt8") == outputType);
    mcmModel.output(mcmInputs.at(0), outputType, {{}, {}, {}, {}});
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
    const auto mvOrder = mv::Order::getColMajorID(mvShape.ndims()) ; //McmOpAttrs::getOrder(constant);
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(constant);
    const auto& opName = constant->get_friendly_name();

    mv::Data::TensorIterator mcmOutput;
    if (constant->get_element_type().is_real()) {
        mvDType = mv::DType("Float32");
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
    const auto groupSize = conv->get_group();

    const auto mvDType = mv::DType("Default");
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(conv);
    const auto& opName = conv->get_friendly_name();

    IE_ASSERT(dilations.at(1) == dilations.at(0));
    IE_ASSERT(mv::DType("Default") == mvDType);

    auto inputShape = conv->get_input_shape(0);
    auto outputShape = conv->get_output_shape(0);
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
    const auto kernelStrideX = strides.at(0);
    const auto kernelStrideY = strides.at(1);
    const auto dilationX = dilations.at(0);
    const auto dilationY = dilations.at(1);

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
            mcmData, mcmWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop), static_cast<uint16_t>(padBottom)},
            static_cast<unsigned>(dilationX),
            mvDType, mvQuantParams, opName);
        registerOutputs(conv, {mcmConvOutput}, mcmOutputsMap);
    } else {
        const auto mcmConvOutput = mcmModel.conv(
            mcmData, mcmWeights,
            {static_cast<uint16_t>(kernelStrideX), static_cast<uint16_t>(kernelStrideY)},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop), static_cast<uint16_t>(padBottom)},
            static_cast<uint32_t>(dilationX),
            static_cast<uint32_t>(groupSize),
            mvDType, mvQuantParams, opName);
        registerOutputs(conv, {mcmConvOutput}, mcmOutputsMap);
    }
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

    int padLeft = padsBegin.at(1);
    int padRight = padsEnd.at(1);
    int padTop = padsBegin.at(0);
    int padBottom = padsEnd.at(0);

    auto outputShape = maxPool->get_output_shape(0);

    cvtPaddingsFromCeilToFloorMode(
        mcmData->getShape()[0], outputShape.at(3), kernelShape.at(0), strides.at(0), padLeft, padRight);
    cvtPaddingsFromCeilToFloorMode(
        mcmData->getShape()[1], outputShape.at(2), kernelShape.at(1), strides.at(1), padTop, padBottom);

    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmMaxPoolOutput = mcmModel.maxPool(mcmData,
            {static_cast<uint16_t>(kernelShape.at(0)), static_cast<uint16_t>(kernelShape.at(1))},
            {static_cast<uint16_t>(strides.at(0)), static_cast<uint16_t>(strides.at(1))},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop), static_cast<uint16_t>(padBottom)},
            true,
            mvDType, outputQuantParams, opName);

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

    int padLeft = padsBegin.at(1);
    int padRight = padsEnd.at(1);
    int padTop = padsBegin.at(0);
    int padBottom = padsEnd.at(0);

    auto outputShape = avgPool->get_output_shape(0);

    cvtPaddingsFromCeilToFloorMode(
        mcmData->getShape()[0], outputShape.at(3), kernelShape.at(0), strides.at(0), padLeft, padRight);
    cvtPaddingsFromCeilToFloorMode(
        mcmData->getShape()[1], outputShape.at(2), kernelShape.at(1), strides.at(1), padTop, padBottom);

    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmAvgPoolOutput = mcmModel.averagePool(mcmData,
            {static_cast<uint16_t>(kernelShape.at(0)), static_cast<uint16_t>(kernelShape.at(1))},
            {static_cast<uint16_t>(strides.at(0)), static_cast<uint16_t>(strides.at(1))},
            {static_cast<uint16_t>(padLeft), static_cast<uint16_t>(padRight),
             static_cast<uint16_t>(padTop), static_cast<uint16_t>(padBottom)},
             avgPool->get_exclude_pad(), //false,
             mvDType, outputQuantParams, opName);

    registerOutputs(avgPool, {mcmAvgPoolOutput}, mcmOutputsMap);
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

void convert(std::shared_ptr<ngraph::op::Eltwise> eltwise, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(eltwise, mcmOutputsMap);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(eltwise);
    const auto& opName = eltwise->get_friendly_name();
    const auto& opType = eltwise->eltwise_type;

    IE_ASSERT(2 == mcmInputs.size());
    IE_ASSERT(ELTWISE_TYPE::Sum == opType);
    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmEltwiseOutput = mcmModel.eltwise(mcmInputs, "Add", mvDType, inputQuantParams, opName + "_FUF");

    registerOutputs(eltwise, {mcmEltwiseOutput}, mcmOutputsMap);
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

    mv::Shape newShape = getWHCN(reshape->get_output_shape(0));

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

void convert(std::shared_ptr<ngraph::op::PowerIE> power, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(power, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(power);
    const auto& opName = power->get_friendly_name();
    const float scale = power->scale;
    const float shift = power->shift;

    if (1.0f != power->power)
        THROW_IE_EXCEPTION << opName + " unsupported power " << power->power;

    const auto shape = power->get_output_shape(0);
    const size_t weights_size = (1 == shape.size()) ? shape.at(0) : getWHCN(shape)[2];

    std::vector<double> weights(weights_size, scale);
    mv::Shape weightsShape = {weights.size()};
    auto mcmWeights = mcmModel.constant(
        weights, weightsShape, mv::DType("Float32"), mv::Order::getColMajorID(1), makeQuantParams());

    IE_ASSERT(mv::DType("Default") == mvDType);
    const auto mcmScaleOutput = mcmModel.scale(mcmData, mcmWeights, mvDType, mvQuantParams, opName);
    if (0.0f != shift) {
        std::vector<double> biases (weights.size(), shift);
        mv::Shape shiftShape { biases.size() };
        auto shiftData = mcmModel.constant(biases, shiftShape, mv::DType("Float32"), mv::Order::getColMajorID(1), makeQuantParams());
        auto biasOutput = mcmModel.bias(mcmScaleOutput, shiftData, mv::DType("Default"), mvQuantParams, opName + "_bias");
        registerOutputs(power, {biasOutput}, mcmOutputsMap);
    } else {
        registerOutputs(power, {mcmScaleOutput}, mcmOutputsMap);
    }
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
    const auto mvDType = mv::DType("Default");
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(concat);
    const auto& opName = concat->get_friendly_name();

    const auto mcmConcatOutput = mcmModel.concat(mcmInputs, mcmAxis, mvDType, mvQuantParams, opName);
    registerOutputs(concat, {mcmConcatOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Transpose> permute, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(permute, mcmOutputsMap);
    IE_ASSERT(2 ==  mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(permute);
    const auto& opName = permute->get_friendly_name();

    std::shared_ptr<ngraph::Node> orderNode = permute->input(1).get_source_output().get_node_shared_ptr();
    std::vector<size_t> orderIndices = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(orderNode)->cast_vector<size_t>();

    std::string oldOrder = "NHWC"; // McmOpAttrs::getOrder(permute, 0).toString();
    std::string newOrder;
    for (size_t i = 0; i < orderIndices.size(); i++) {
        newOrder += oldOrder[orderIndices.size() - 1 - i];
    }

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    auto mcmPermuteOutput = mcmModel.permute(mcmData, mv::Order(newOrder), mvDType, mvQuantParams, opName);

    // Workaround to avoid parsing stage crash:
    // 'ArgumentError: attribute identifer quantParams - Undefined identifier'
    // [Track number: D#2284, D#2237]
    // TBD
    mcmPermuteOutput->set<mv::QuantizationParams>("quantParams", mvQuantParams);

    registerOutputs(permute, {mcmPermuteOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Squeeze> reshape, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(reshape, mcmOutputsMap);
    IE_ASSERT(2 ==  mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(reshape);
    const auto& opName = reshape->get_friendly_name();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    mv::Shape newShape = getWHCN(reshape->get_shape());

    auto mcmReshapeOutput = mcmModel.reshape(mcmData, newShape, mvDType, inputQuantParams, opName);
    registerOutputs(reshape, {mcmReshapeOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Unsqueeze> reshape, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(reshape, mcmOutputsMap);
    IE_ASSERT(2 ==  mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(reshape);
    const auto& opName = reshape->get_friendly_name();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    mv::Shape newShape = getWHCN(reshape->get_shape());

    auto mcmReshapeOutput = mcmModel.reshape(mcmData, newShape, mvDType, inputQuantParams, opName);
    registerOutputs(reshape, {mcmReshapeOutput}, mcmOutputsMap);
}


void convert(std::shared_ptr<ngraph::op::v1::Softmax> softmax, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(softmax, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(softmax);
    const auto& opName = softmax->get_friendly_name();

    std::string order = McmOpAttrs::getOrder(softmax, 0).toString();
    std::string mcmAxis = std::string(1, order.at(softmax->get_axis()));

    auto mcmSoftmaxOutput = mcmModel.softmax(mcmData, mcmAxis, mvDType, inputQuantParams, opName);
    registerOutputs(softmax, {mcmSoftmaxOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Clamp> clamp, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(clamp, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(clamp);
    const auto& opName = clamp->get_friendly_name();
    const double minValue = clamp->get_min();
    const double maxValue = clamp->get_max();

    auto mcmClampMin = mcmModel.minimum(mcmData, maxValue, mvDType, inputQuantParams, opName + "clamp-min");
    auto mcmClampMax = mcmModel.maximum(mcmClampMin, minValue, mvDType, inputQuantParams, opName+ "clamp-max");
    registerOutputs(clamp, {mcmClampMax}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::ReLUIE> relu, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(relu, mcmOutputsMap).at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(relu);
    const auto& opName = relu->get_friendly_name();
    const float slope = relu->get_slope();

    IE_ASSERT(mv::DType("Default") == mvDType);
    if (std::fabs(slope) < std::numeric_limits<float>::epsilon()) {
        const auto mcmReluOutput = mcmModel.relu(mcmData, mvDType, inputQuantParams, opName);
        registerOutputs(relu, {mcmReluOutput}, mcmOutputsMap);
    } else {
        const auto mcmReluOutput = mcmModel.leakyRelu(mcmData, slope, mvDType, inputQuantParams, opName);
        registerOutputs(relu, {mcmReluOutput}, mcmOutputsMap);
    }
}

void convert(std::shared_ptr<ngraph::op::v0::ROIPooling> roipool, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    auto mcmInputs = getMcmInputs(roipool, mcmOutputsMap);
    IE_ASSERT(2 == mcmInputs.size());
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(roipool);
    const auto& opName = roipool->get_friendly_name();
    const double spatial_scale = roipool->get_spatial_scale();
    const std::string method = roipool->get_method();
    const unsigned roi_pooling_method = (method == "bilinear") ? 1 : 0;
    unsigned num_rois = roipool->get_input_shape(0)[0];
    unsigned pooled_h = roipool->get_output_shape(0)[2];
    unsigned pooled_w = roipool->get_output_shape(0)[3];

    const auto roipoolOutput = mcmModel.rOIPooling(mcmInputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois,
        mvDType, inputQuantParams, opName);
    registerOutputs(roipool, {roipoolOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::PriorBoxIE> priorbox, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(priorbox, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(priorbox);
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
    auto priorboxOutput = mcmModel.constant(boxes, {boxes.size() / 2, 2, 1, 1}, mv::DType("Float64"), mv::Order("NHWC"),
        inputQuantParams, opName + "_const");

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
   // auto priorboxClustered = mcmModel.constant(boxes, {boxes.size() / 2, 2, 1, 1}, mv::DType("Float64"), mv::Order("NHWC"), opName + "_const");
    auto priorboxClustered =
            mcmModel.constant(boxes, {boxes.size() / 2, 2, 1, 1}, mv::DType("Float64"), mv::Order("NHWC"));

    registerOutputs(pbc, {priorboxClustered}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::NormalizeL2> normL2, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(normL2, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 2);
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(normL2);
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

    auto mvWeightsValues = mcmModel.constant(weightsData, weightsShape, mv::DType("Float32"), mv::Order::getZMajorID(4));

    auto mvNormalizeOutput = mcmModel.normalize(mcmData, mvWeightsValues, eps, across_spatial, channel_shared,
        mvDType, inputQuantParams, opName);
    registerOutputs(normL2, {mvNormalizeOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Sigmoid> sigmoid, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(sigmoid, mcmOutputsMap).at(0);
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(sigmoid); 
    // TBD auto inputQuantParams = inputs[0]->getMcmNode()->get<mv::QuantizationParams>("quantParams");
    const auto& opName = sigmoid->get_friendly_name();
    const auto mcmSigmoidOutput = mcmModel.sigmoid(mcmData, mv::DType("Default"), inputQuantParams, opName);
    registerOutputs(sigmoid, {mcmSigmoidOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::DetectionOutput> detection, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(detection, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 3);
    const auto& quantParams = McmOpAttrs::getQuantParams(detection);
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

    auto mcmDetectionOutput = mcmModel.detectionOutput(mcmInputs, attrs.num_classes, keep_top_k, attrs.nms_threshold,
        attrs.background_label_id, attrs.top_k, attrs.variance_encoded_in_target, attrs.code_type, attrs.share_location, attrs.confidence_threshold,
        attrs.clip_before_nms, attrs.clip_after_nms, attrs.decrease_label_id, attrs.normalized, attrs.input_height, attrs.input_width, attrs.objectness_score,
        mv::DType("Default"), quantParams, opName);

    registerOutputs(detection, {mcmDetectionOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::ReorgYolo> reorg, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(reorg, mcmOutputsMap).at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(reorg);
    const auto& opName = reorg->get_friendly_name();
    const std::size_t stride = reorg->get_strides().at(0);

    for (auto&& s : reorg->get_strides()) {
        IE_ASSERT(stride == s);
    }

    const auto mcmReorgYoloOutput = mcmModel.reorgYolo(mcmData, static_cast<unsigned>(stride), mvDType, inputQuantParams, opName);
    registerOutputs(reorg, {mcmReorgYoloOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::RegionYolo> region, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(region, mcmOutputsMap).at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(region);
    const auto& opName = region->get_friendly_name();
    const std::size_t coords = region->get_num_coords();
    const std::size_t classes = region->get_num_classes();
    const std::size_t num = region->get_num_regions();
    const bool do_softmax = region->get_do_softmax();
    const std::vector<int64_t> mask = region->get_mask();
    const std::vector<unsigned> mcmMask(mask.begin(), mask.end());

    const auto mcmRegionYoloOutput = mcmModel.regionYolo(
        mcmData,
        static_cast<unsigned>(coords),
        static_cast<unsigned>(classes),
        do_softmax,
        static_cast<unsigned>(num),
        mcmMask,
        mvDType, inputQuantParams, opName);
    registerOutputs(region, {mcmRegionYoloOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::TopK> topk, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(topk, mcmOutputsMap);
    IE_ASSERT(2 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& inputQuantParams = makeQuantParams();
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

    const auto mcmTopKOutput = mcmModel.topK(mcmData, sort, mode, k, axis, mv::DType("Default"), makeQuantParams(), opName);

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
    const auto& inputQuantParams = makeQuantParams();
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

    const auto mcmTopKOutput = mcmModel.topK(mcmData, sort, mode, k, axis, mv::DType("Default"), makeQuantParams(), opName);

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
    const auto& inputQuantParams = makeQuantParams();
    const auto& opName = resample->get_friendly_name();
    auto antialias = false;
    auto resampleAttrs = resample->get_attrs();
    std::string mode = "nearest";
    if (resampleAttrs.mode != "") {
        mode = resampleAttrs.mode;
    }

    mv::Shape output_shape = getWHCN(resample->get_output_shape(0));
    auto mcmResampleOutput = mcmModel.resample(mcmData, interpolationMap.at(mode), antialias,
                                              output_shape, mv::DType("Default"), inputQuantParams, opName);

    registerOutputs(resample, {mcmResampleOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::Interp> interp, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(interp, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& inputQuantParams = makeQuantParams();
    const auto& opName = interp->get_friendly_name();
    auto interpAttrs = interp->get_attrs();
    auto factor = interpAttrs.scale_factor;
    auto height = interpAttrs.height;
    auto width = interpAttrs.width;
    auto pad_begin = interpAttrs.pad_beg;
    auto pad_end = interpAttrs.pad_end;
    auto align_corners = interpAttrs.align_corners;

    auto mcmInterpOutput = mcmModel.interp(mcmData, factor, pad_begin, pad_end, height, width, align_corners,
                                     mv::DType("Default"), inputQuantParams, opName);
    registerOutputs(interp, {mcmInterpOutput}, mcmOutputsMap);
}


void convert(std::shared_ptr<ngraph::op::NormalizeIE> normalizeIE, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(normalizeIE, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 2);
    const auto mcmData = mcmInputs.at(0);
    const auto mvDType = mv::DType("Default");
    const auto& inputQuantParams = McmOpAttrs::getQuantParams(normalizeIE);
    const auto& opName = normalizeIE->get_friendly_name();

    double eps = normalizeIE->get_eps();
    auto const_axis = std::dynamic_pointer_cast<ngraph::op::Constant> (normalizeIE->input(1).get_source_output().get_node_shared_ptr());
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

    auto mvWeightsValues = mcmModel.constant(weightsData, weightsShape, mv::DType("Float32"), mv::Order::getZMajorID(4));

    auto mvNormalizeOutput = mcmModel.normalize(mcmData, mvWeightsValues, eps, across_spatial, channel_shared,
                                                mvDType, inputQuantParams, opName);
    registerOutputs(normalizeIE, {mvNormalizeOutput}, mcmOutputsMap);
}

// TODO: move converters to class ConvertToMcmModel scope to remove references to data

template <typename T>
void convertDispatch(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr /*unused*/) {
    convert(std::dynamic_pointer_cast<T>(node), mcmModel, mcmOutputsMap);
}

// Propagate ieData precision to MCM in order to perform conversion on hardware
template<>
void convertDispatch<ngraph::op::Parameter>(std::shared_ptr<ngraph::Node> node,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap, InferenceEngine::DataPtr ieData) {
    convert(std::dynamic_pointer_cast<ngraph::op::Parameter>(node), mcmModel, mcmOutputsMap, ieData);
}

template<>
void convertDispatch<ngraph::op::Result>(std::shared_ptr<ngraph::Node> node,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap, InferenceEngine::DataPtr ieData) {
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
    MAP_ENTRY(ngraph::op::NormalizeIE),
};

#undef MAP_ENTRY

void ConvertNode(const std::shared_ptr<ngraph::Node> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr ieData) {
    const auto dispatchIt = dispatchMap.find(op->get_type_info());
    if (dispatchIt != dispatchMap.end()) {
        const auto convertor = dispatchIt->second;
        if (convertor != nullptr) {
            try {
                convertor(op, mcmModel, mcmOutputsMap, ieData);
            } catch (const std::runtime_error& ex) {
                THROW_IE_EXCEPTION << "Convertor for operation " << op->get_friendly_name()
                    << " failed due to runtime error " << ex.what();
            }
        } else {
            THROW_IE_EXCEPTION << "Convertor not found for operation: " << op->get_friendly_name();
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported operation: " << op->get_friendly_name()
                    << " with name " << op->get_name()
                    << " with type " << op->get_type_name()
                    << " with C++ type " << typeid(*op.get()).name();
    }
}

}  // namespace

bool ConvertToMcmModel::run_on_function(std::shared_ptr<ngraph::Function> func) {
    // Ngraph representation and IE CNNNetwork may have inputs and outpus in different order.
    // MCM compiler processes inputs and outputs by add-to-model order, not by their name.
    // Therefore plugin must reorder them manually to follow IE CNNNetwork
    // Also propagate IE input/output precision/layout to MCM, so conversion will be done on hardware.

    for (auto&& inputInfo : _networkInputs) {
        bool isFound = false;
        for (const auto& op : func->get_parameters()) {
            if (op->get_friendly_name() == _ioMap.at(inputInfo.first)) {
                ConvertNode(op, _mcmModel, _mcmOutputsMap, inputInfo.second->getInputData());
                isFound = true;
            }
        }
        if (!isFound)
            THROW_IE_EXCEPTION << "Input not found: " << inputInfo.first;
    }

    for (const auto& op : func->get_ordered_ops()) {
        if (ngraph::op::Parameter::type_info == op->get_type_info())
            continue;
        if (ngraph::op::Result::type_info == op->get_type_info())
            continue;
        ConvertNode(op, _mcmModel, _mcmOutputsMap, nullptr);
    }

    for (auto&& outputInfo : _networkOutputs) {
        bool isFound = false;
        for (const auto& op : func->get_results()) {
            if (op->get_friendly_name() == _ioMap.at(outputInfo.first)) {
                ConvertNode(op, _mcmModel, _mcmOutputsMap, outputInfo.second);
                isFound = true;
            }
        }
        if (!isFound)
            THROW_IE_EXCEPTION << "Ouput not found: " << outputInfo.first;
    }

    return false;
}

// clang-format on
