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
#include "ngraph/op/region_yolo.hpp"

#include "ngraph/op/reorg_yolo.hpp"

#include <ngraph_ops/power.hpp>
#include <ngraph_ops/relu_ie.hpp>

#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"

#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"

#include <ngraph/op/fused/fake_quantize.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph_ops/scaleshift.hpp>

#include <ngraph/op/parameter.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/constant.hpp>

#include <ngraph/op/transpose.hpp>
#include <ngraph/op/fused/squeeze.hpp>
#include <ngraph/op/softmax.hpp>

#include <memory>
#include <vector>
#include <map>

#include <include/mcm/tensor/tiling.hpp>

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

void convert(std::shared_ptr<ngraph::op::Parameter> param, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mvShape = cvtShapeToMCM(param->get_shape());
    const auto mvDType = mv::DType("UInt8"); // cvtElemTypeToMCM(param->get_element_type());
    const auto mvOrder = mv::Order("NHWC");// McmOpAttrs::getOrder(param);
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(param);
    const auto& opName = param->get_friendly_name();

    // MCM Compiler requirements
    IE_ASSERT(mv::DType("UInt8") == mvDType);
    IE_ASSERT(mv::Order("NHWC") == mvOrder);

	bool mvNetworkInput = true;
    const auto mcmOutput = mcmModel.input(mvShape, mvDType, mvOrder, mvQuantParams, mvNetworkInput, opName);

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

void convert(std::shared_ptr<ngraph::op::v0::Concat> concat, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(concat, mcmOutputsMap);
    IE_ASSERT(0 < mcmInputs.size());
    const std::string order = McmOpAttrs::getOrder(concat, 0).toString();
    std::string mcmAxis = std::string(1, order.at(concat->get_axis()));
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

    std::shared_ptr<ngraph::Node> orderNode = permute->get_inputs().at(1).get_output().get_node();
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

    // TBD:  mv::Shape newShape = cvtShapeToMCM(reshape->get_output_shape(0));
    mv::Shape newShape {1, 1, 1, 1};
    const auto shape_size = reshape->get_output_shape(0).size();
    if (2 == shape_size) {
        newShape[2] = reshape->get_output_shape(0).at(1);
        newShape[3] = reshape->get_output_shape(0).at(0);
    } else {
        IE_ASSERT(2 == shape_size);
    }

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
    MAP_ENTRY(ngraph::op::v0::Concat),
    MAP_ENTRY(ngraph::op::v1::Transpose),
    MAP_ENTRY(ngraph::op::v0::Squeeze),
    MAP_ENTRY(ngraph::op::v1::Softmax),
    MAP_ENTRY(ngraph::op::v0::Clamp),
    MAP_ENTRY(ngraph::op::ReLUIE),
    MAP_ENTRY(ngraph::op::v0::RegionYolo),
    MAP_ENTRY(ngraph::op::v0::ReorgYolo)
#if 0
//     MAP_ENTRY(ngraph::op::v1::Add), Eltwise
    MAP_ENTRY(ngraph::op::v1::ReduceMean),
    MAP_ENTRY(ngraph::op::FullyConnected),
    // PT_MobileNet_V2

    // CF_Inception_V1
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
