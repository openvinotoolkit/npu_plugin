//
// Copyright 2020-2021 Intel Corporation.
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

#include "debug.h"
#include "ngraph_mcm_frontend/passes/convert_to_mcm_model.hpp"
#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"
#include "ngraph_mcm_frontend/ops/mcm_fc.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/interpolate.hpp"

#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include <legacy/ngraph_ops/fully_connected.hpp>
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
#include <ngraph/op/psroi_pooling.hpp>

#include "ngraph/op/region_yolo.hpp"

#include "ngraph/op/reorg_yolo.hpp"

#include "ngraph/op/ctc_greedy_decoder.hpp"

#include <ngraph/op/power.hpp>
#include <legacy/ngraph_ops/relu_ie.hpp>
#include <legacy/ngraph_ops/eltwise.hpp>
#include <legacy/ngraph_ops/gather_ie.hpp>
#include <legacy/ngraph_ops/power.hpp>
#include <ngraph/op/normalize_l2.hpp>

#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"

#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"

#include <ngraph/op/fake_quantize.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <legacy/ngraph_ops/crop_ie.hpp>
#include <ngraph_ops/deconvolution_ie.hpp>
#include <legacy/ngraph_ops/scaleshift.hpp>

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
#include <ngraph/op/softplus.hpp>
#include <ngraph/op/pad.hpp>
#include <ngraph/op/mish.hpp>
#include <ngraph/op/floor.hpp>
#include <ngraph/op/round.hpp>
#include <ngraph/op/ceiling.hpp>
#include <ngraph/op/erf.hpp>
#include <ngraph/op/gelu.hpp>
#include <ngraph/op/ctc_greedy_decoder_seq_len.hpp>
#include <ngraph/op/log.hpp>
#include <ngraph/op/reverse_sequence.hpp>

#include <ngraph/op/prior_box.hpp>
#include <ngraph/op/prior_box_clustered.hpp>
#include <ngraph/op/detection_output.hpp>

#include <ngraph/op/split.hpp>
#include <ngraph/op/variadic_split.hpp>
#include <ngraph/op/strided_slice.hpp>
#include <ngraph/slice_plan.hpp>
#include <ngraph/op/mvn.hpp>
#include <ngraph/op/space_to_depth.hpp>
#include <ngraph/op/squared_difference.hpp>
#include <ngraph/op/depth_to_space.hpp>

#include <legacy/ngraph_ops/interp.hpp>
#include <legacy/ngraph_ops/prior_box_clustered_ie.hpp>
#include <legacy/ngraph_ops/prior_box_ie.hpp>
#include <legacy/ngraph_ops/lrn_ie.hpp>
#include <legacy/ngraph_ops/normalize_ie.hpp>
#include <legacy/ngraph_ops/topk_ie.hpp>
#include <legacy/ngraph_ops/proposal_ie.hpp>
#include <legacy/ngraph_ops/tile_ie.hpp>
#include <legacy/ngraph_ops/swish_ie.hpp>
#include <legacy/ngraph_ops/pad_ie.hpp>

#include <ngraph/variant.hpp>

#include <legacy/ngraph_ops/tile_ie.hpp>
#include <parse_layers_helpers.hpp>
#include <dims_parser.hpp>
#include "ngraph_mcm_frontend/ie_helpers.hpp"

#include <memory>
#include <vector>
#include <map>
#include <algorithm>

#include <include/mcm/tensor/tiling.hpp>
#include <converters.hpp>
#include <custom_layer/custom_layer.hpp>

namespace mv {
    namespace op_conversion {
        bool isConversionSupported(const DType& inDType, const DType& outDType, const std::string& opName, std::string& errMsg);
    }
}

namespace {

using Callback = void (*)(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
InferenceEngine::DataPtr, bool, bool);
using DispatchMap = std::map<ngraph::NodeTypeInfo, Callback>;

std::vector<mv::Data::TensorIterator> getMcmInputs(std::shared_ptr<ngraph::Node> node, const NodeOutputToMcmMap& mcmOutputsMap) {
    std::vector<mv::Data::TensorIterator> out;
    out.reserve(node->get_input_size());

    for (const auto& input : node->inputs()) {
        try {
            out.push_back(mcmOutputsMap.at(input.get_source_output()));
        } catch (const std::exception &ex) {
            IE_THROW() << "For operation " << node->get_type_name() << " name " << node->get_friendly_name()
                << "output not found: " << input.get_source_output().get_tensor().get_name()
                << " " << ex.what();
        }
    }

    return out;
}

// 'memory order' being row-column-channel for CHW and channel-row-column for HWC
// i.e. the order which is used to represent data in RAM
mv::Shape getMemoryOrder(const ngraph::Shape& shape) {
    size_t dimN, dimZ, dimY, dimX, dimD;
    std::vector<size_t> dims = shape;
    vpu::parseDims(dims, dimN, dimZ, dimY, dimX, dimD);
    mv::Shape result;
    if (dims.size() == 5) {
        result = mv::Shape({dimX, dimY, dimD, dimZ, dimN});
    } else {
        result = mv::Shape({dimX, dimY, dimZ, dimN});
    }
    return result;
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
    const std::set<ie::Precision> supportedInPrecisions = {ie::Precision::BF16, ie::Precision::U8, ie::Precision::FP16, ie::Precision::FP32};
    return supportedInPrecisions.find(inputPrecision) != supportedInPrecisions.end();
}

bool isInputLayoutSupported(const ie::Layout& inputLayout) {
    const std::set<ie::Layout> supportedInLayouts = {
        ie::Layout::NHWC, ie::Layout::NCHW, ie::Layout::CHW, ie::Layout::NC, ie::Layout::C, ie::Layout::NCDHW};
    return supportedInLayouts.find(inputLayout) != supportedInLayouts.end();
}

bool isOutputPrecisionSupported(const ie::Precision& outputPrecision) {
    std::set<ie::Precision> supportedOutPrecisions = {ie::Precision::BF16, ie::Precision::U8, ie::Precision::FP16, ie::Precision::FP32, ie::Precision::I32};
    return supportedOutPrecisions.find(outputPrecision) != supportedOutPrecisions.end();
}

bool isOutputLayoutSupported(const ie::Layout& outputLayout) {
    std::set<ie::Layout> supportedOutLayouts = {
        ie::Layout::NHWC, ie::Layout::NCHW, ie::Layout::CHW, ie::Layout::NC, ie::Layout::C, ie::Layout::NCDHW};
    return supportedOutLayouts.find(outputLayout) != supportedOutLayouts.end();
}

static const mv::QuantizationParams& initialQuantParams() {
    double inf = std::numeric_limits<double>::infinity();
    static mv::QuantizationParams init{{0}, {1}, {-inf}, {inf}};
    return init;
};

void convert(std::shared_ptr<ngraph::op::Parameter> param, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr ieData, bool allowNCHWInput) {
    auto mvShape = getMemoryOrder(param->get_shape());
    // Use data from InputInfo DataPtr
    bool mvNetworkInput = true;
    const auto& opName = param->get_friendly_name();

    if (param->get_shape().size() > 5 || param->get_shape().size() == 0) {
       IE_THROW() << "Input shape size is not supported: " << param->get_shape().size();
    }

    const InferenceEngine::Layout inputLayout = ieData->getTensorDesc().getLayout();
    if (!isInputLayoutSupported(inputLayout)) {
        IE_THROW() << "Input layout is not supported: " << ieData->getTensorDesc().getLayout();
    }

    const auto mvOrder = [&] {
        if ((inputLayout == InferenceEngine::Layout::NCHW || inputLayout == InferenceEngine::Layout::CHW)
            && allowNCHWInput) {
            return layoutToOrder(InferenceEngine::Layout::NCHW);
        } else if (inputLayout == InferenceEngine::Layout::NCDHW) {
            return layoutToOrder(InferenceEngine::Layout::NCDHW);
        }
        return layoutToOrder(InferenceEngine::Layout::NHWC);
    }();
    // MCM Compiler requirements
    // IE_ASSERT(mv::Order("NHWC") == mvOrder);
    const auto mvDType = cvtElemTypeToMCM(param->get_element_type());
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
        IE_THROW() << "Output data type is not supported: " << outputPrecision;
    }

    const InferenceEngine::Layout outputLayout = ieData->getTensorDesc().getLayout();
    if (!isOutputLayoutSupported(outputLayout)) {
        IE_THROW() << "Output layout is not supported: " << outputLayout;
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
        outputType = mv::DType("Float32");
        break;
    case ie::Precision::I32:
        outputType = mv::DType("Int32");
        break;
    case ie::Precision::BF16:
        outputType = mv::DType("BFloat16");
        break;
    default:
        IE_THROW() << "Data type handling is not implemented" << outputPrecision.name();
    }

    if (result->get_shape().size() > 5 || result->get_shape().size() == 0) {
       IE_THROW() << "Output shape size is not supported: " << result->get_shape().size();
    }

    // MCM Compiler requirements
    mcmModel.output(result->get_friendly_name(), mcmInputs.at(0), outputType);
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
            if (ngraph::op::GatherIE::type_info == consumerNode->get_type_info() ||
                ngraph::op::v1::Split::type_info == consumerNode->get_type_info() ||
                ngraph::op::v1::StridedSlice::type_info == consumerNode->get_type_info()) {
                mvShape = mv::Shape::augment_major(mvShape, 4);
                // int64 precision for indices is not supported by runtime yet
                if (ngraph::element::i64 == constant->get_element_type()) {
                    mvDType = mv::DType("Int32");
                    opName += "_indices_i32";
                }
                break;
            }
        }
    }
    // end of workaround

    const auto mvOrder = mv::Order::getColMajorID(mvShape.ndims()) ; //McmOpAttrs::getOrder(constant);

    mv::Data::TensorIterator mcmOutput;
    if (constant->get_element_type().is_real()) {

        if (mvDType.isDoubleType() || mvDType == mv::DType("Float16"))
        {
            //legacy
            mvDType = mv::DType("Float32");
            mcmOutput = mcmModel.constant(opName, constant->cast_vector<double>(), mvShape, mvDType, mvOrder);
        }
        else
        {
            //BF16
            mcmOutput = mcmModel.constant(opName, constant->cast_vector<double>(), mvShape, mvDType, mvOrder);
        }

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

void convert(std::shared_ptr<ngraph::op::v0::PRelu> prelu, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(prelu, mcmOutputsMap);
    const auto& opName = prelu->get_friendly_name();

    const auto mcmData = mcmInputs.at(0);
    const auto mcmSlope = mcmInputs.at(1);
    const auto mcmPReluOutput = mcmModel.prelu(opName, mcmData, mcmSlope);

    mcmPReluOutput->setQuantParams(initialQuantParams());

    registerOutputs(prelu, {mcmPReluOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Elu> elu, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmData = getMcmInputs(elu, mcmOutputsMap).at(0);
    const auto& opName = elu->get_friendly_name();

    auto alpha = elu->get_alpha();

    const auto mcmEluOutput = mcmModel.elu(opName, mcmData, alpha);
    mcmEluOutput->setQuantParams(initialQuantParams());

    registerOutputs(elu, {mcmEluOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v4::HSwish> hswish, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(hswish, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto mcmOpOutput = mcmModel.hSwish(hswish->get_friendly_name(), mcmInputs.at(0));
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(hswish, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v4::SoftPlus> softplus, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(softplus, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto mcmOpOutput = mcmModel.softPlus(softplus->get_friendly_name(), mcmInputs.at(0));
    registerOutputs(softplus, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v4::Mish> mish, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(mish, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto& opName = mish->get_friendly_name();
    const auto& opInput = mcmInputs.at(0);
    const auto mcmOpOutput = mcmModel.mish(opName, opInput);
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(mish, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Floor> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto& opName = op->get_friendly_name();
    const auto& opInput = mcmInputs.at(0);
    const auto mcmOpOutput = mcmModel.floor(opName, opInput);
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v5::Round> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const static std::map<ngraph::op::v5::Round::RoundMode, std::string> roundMode = {
            {ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN,        "half_to_even"},
            {ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO, "half_away_from_zero"}
    };

    const auto round_mode = op->get_mode();
    const auto roundModeIter = roundMode.find(round_mode);
    if (roundModeIter == roundMode.end()) {
        IE_THROW() << "Convertor for operation " << op->get_friendly_name()
                           << " failed due to unsupported mode " << static_cast<int>(round_mode);

    }
    const auto& mode = roundModeIter->second;

    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto& opName = op->get_friendly_name();
    const auto& opInput = mcmInputs.at(0);
    const auto mcmOpOutput = mcmModel.round(opName, opInput, mode);
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Ceiling> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto& opName = op->get_friendly_name();
    const auto& opInput = mcmInputs.at(0);
    const auto mcmOpOutput = mcmModel.ceiling(opName, opInput);
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Erf> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto& opName = op->get_friendly_name();
    const auto& opInput = mcmInputs.at(0);
    const auto mcmOpOutput = mcmModel.erf(opName, opInput);
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Gelu> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto& opName = op->get_friendly_name();
    const auto& opInput = mcmInputs.at(0);
    const auto mcmOpOutput = mcmModel.gelu(opName, opInput);
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::Log> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto& opName = op->get_friendly_name();
    const auto& opInput = mcmInputs.at(0);
    const auto mcmOpOutput = mcmModel.log(opName, opInput);
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

// TODO: Replace SwishIE with v4::Swish -- to process
//       beta parameter as 2nd (optional) input tensor
// #-46185: Swish expects Beta parameter as attribute
void convert(std::shared_ptr<ngraph::op::SwishIE> swish_ie, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(swish_ie, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    auto beta = swish_ie->get_alpha();
    const auto& opName = swish_ie->get_friendly_name();
    const auto& opInput = mcmInputs.at(0);
    const auto mcmOpOutput = mcmModel.swish(opName, opInput, beta);
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(swish_ie, {mcmOpOutput}, mcmOutputsMap);
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
    auto mcmInputs = getMcmInputs(eltwise, mcmOutputsMap);
    const auto& opName = eltwise->get_friendly_name();
    const auto& opType = eltwise->eltwise_type;

    IE_ASSERT(2 == mcmInputs.size());
    if (mcmInputs[0]->isPopulated() && mcmInputs[1]->isPopulated()){
        THROW_IE_EXCEPTION << "At least one input is unpopulated for Eltwise op ";
    }else if (mcmInputs[0]->isPopulated() && (!mcmInputs[1]->isPopulated())){
        /// swap them to ensure the input 0 is unpopulated, 
        /// it's a compiler assumption.
        std::swap(mcmInputs[0], mcmInputs[1]);
    }   
    
    /// extend 1d weights to 4d to adapt compiler
    auto weights_shape= mcmInputs[1]->getShape();
    if (weights_shape.ndims()==1){
        mcmInputs[1]->setShape(mv::Shape({1,1,weights_shape[0],1}));
        mcmInputs[1]->setOrder(mv::Order::getZMajorID(4));
    }

    mv::Data::TensorIterator mcmEltwiseOutput;

    if (ELTWISE_TYPE::Sum == opType)
        mcmEltwiseOutput = mcmModel.eltwise(opName, mcmInputs, "Add");
    else if (ELTWISE_TYPE::Prod  == opType)
        mcmEltwiseOutput = mcmModel.eltwise(opName, mcmInputs, "Multiply");
    else
        IE_THROW() << "Operation " << eltwise->get_type_name() << " " << opName << " has unsupported parameter ";
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

    mv::Shape newShape = getMemoryOrder(reshape->get_output_shape(0));

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

void convert(std::shared_ptr<ngraph::op::v0::Convert> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    const auto mcmData = mcmInputs.at(0);
    const auto inDType = mcmData->getDType();

    const auto& opName = op->get_friendly_name();
    const auto outType = op->get_convert_element_type();
    const auto outDType = cvtElemTypeToMCM(outType);

    std::string errMsg;
    if (!mv::op_conversion::isConversionSupported(inDType, outDType, opName, errMsg)) {
        IE_THROW() << errMsg;
    }

    // E#9602: Convert layer does not support floating-point to U8 (and back)
    using typename mv::DType;
    if ((inDType == DType("UInt8") && (outDType == DType("Float16") || outDType == DType("Float32"))) ||
        (outDType == DType("UInt8") && (inDType == DType("Float16") || inDType == DType("Float32"))))
    {
        IE_THROW() << "Convert layer does not support FP<->U8 cases"
                   << " (" << opName << ")"
                   <<  ": inDType=" <<  inDType.toString()
                   << ", outDType=" << outDType.toString();
    }

    const auto mcmConvertOutput = mcmModel.conversion(opName, mcmData, outDType);
    mcmConvertOutput->setQuantParams(initialQuantParams());

    registerOutputs(op, {mcmConvertOutput}, mcmOutputsMap);
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
        const size_t weights_size = (1 == shape.size()) ? shape.at(0) : getMemoryOrder(shape)[2];

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
        IE_THROW() << "Operation " << power->get_type_name() << " " + opName + " has unsupported power " << power->power;
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

void convert(std::shared_ptr<ngraph::op::v1::Transpose> permute, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    bool allowPermuteND) {
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

    mv::Data::TensorIterator mcmPermuteOutput{};
    if (allowPermuteND) {
        std::vector<int64_t> permNDOrder(orderIndices.begin(), orderIndices.end());
        mcmPermuteOutput = mcmModel.permuteND(opName, mcmData, permNDOrder);
    } else {
        mcmPermuteOutput = mcmModel.permute(opName, mcmData, mv::Order(newOrder));
    }

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

    mv::Shape newShape = getMemoryOrder(reshape->get_shape());

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

    mv::Shape newShape = getMemoryOrder(reshape->get_shape());

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

void convert(std::shared_ptr<ngraph::op::v0::PSROIPooling> psroipool, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    auto mcmInputs = getMcmInputs(psroipool, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());
    const auto& opName = psroipool->get_friendly_name();
    const std::size_t output_dim = psroipool->get_output_dim();
    const std::size_t group_size = psroipool->get_group_size();
    const float spatial_scale  = psroipool->get_spatial_scale();
    const int spatial_bins_x = psroipool->get_spatial_bins_x();
    const int spatial_bins_y = psroipool->get_spatial_bins_y();
    const std::string mode = psroipool->get_mode();
    IE_ASSERT(4u == psroipool->get_output_shape(0).size());
    const std::size_t pooled_h = psroipool->get_output_shape(0)[2];
    const std::size_t pooled_w = psroipool->get_output_shape(0)[3];

    const auto roipoolOutput = mcmModel.pSROIPooling(opName, mcmInputs, output_dim, group_size, spatial_scale, pooled_h,
        pooled_w, spatial_bins_x, spatial_bins_y, mode);
    roipoolOutput->setQuantParams(initialQuantParams());

    registerOutputs(psroipool, {roipoolOutput}, mcmOutputsMap);
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
        IE_THROW() << opName + " Incorrect number of input edges!";

    if (priorbox->get_input_shape(0).size() != 4 ||
        priorbox->get_input_shape(1).size() != 4)
        IE_THROW() << opName + " PriorBox supports only 4D blobs!";
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
        IE_THROW() << opName + " has non-constant k";
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
        IE_THROW() << opName + " has too many outputs " << outputSlots;
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
        IE_THROW() << opName + " has non-constant k";
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
        IE_THROW() << opName + " has too many outputs " << outputSlots;
}

const static std::map<std::string, std::string> interpolationMap = {
        {"nearest", "NEAREST"},
        {"cubic", "BICUBIC"},
        {"linear", "BILINEAR"},
        {"linear_onnx", "LINEAR_ONNX"},
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

    mv::Shape output_shape = getMemoryOrder(resample->get_output_shape(0));
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

const static std::map<ngraph::op::v4::Interpolate::InterpolateMode, std::string> interpolateMode = {
        {ngraph::op::v4::Interpolate::InterpolateMode::nearest,     "nearest"},
        {ngraph::op::v4::Interpolate::InterpolateMode::cubic,       "cubic"},
        {ngraph::op::v4::Interpolate::InterpolateMode::linear,      "linear"},
        {ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx, "linear_onnx"},
};

const static std::map<ngraph::op::v4::Interpolate::CoordinateTransformMode, std::string> coordMode = {
        {ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,           "half_pixel"},
        {ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,   "pytorch_half_pixel"},
        {ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,           "asymmetric"},
        {ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn, "tf_half_pixel_for_nn"},
        {ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,        "align_corners"},
};

const static std::map<ngraph::op::v4::Interpolate::NearestMode, std::string> nearestMode = {
        {ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor, "round_prefer_floor"},
        {ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,  "round_prefer_ceil"},
        {ngraph::op::v4::Interpolate::NearestMode::floor,              "floor"},
        {ngraph::op::v4::Interpolate::NearestMode::ceil,               "ceil"},
        {ngraph::op::v4::Interpolate::NearestMode::simple,             "simple"},
};

void convert(std::shared_ptr<ngraph::op::v4::Interpolate> interpolate, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(interpolate, mcmOutputsMap);
    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = interpolate->get_friendly_name();
    const auto antialias = false;
    const auto& interpolateAttrs = interpolate->get_attrs();

    const auto interpolateModeIter = interpolateMode.find(interpolateAttrs.mode);
    if (interpolateModeIter == interpolateMode.end())
        IE_THROW() << "interpolateMode map doesn't contain reqested interpolate mode";
    const std::string mode  = interpolateModeIter->second;


    const auto coordModeIter = coordMode.find(interpolateAttrs.coordinate_transformation_mode);
    if (coordModeIter == coordMode.end())
        IE_THROW() << "coordMode map doesn't contain reqested coordinate transformation mode";
    const std::string coord  = coordModeIter->second;

    const auto nearestModeIter = nearestMode.find(interpolateAttrs.nearest_mode);
    if (nearestModeIter == nearestMode.end())
        IE_THROW() << "nearestMode map doesn't contain reqested nearest mode";
    const std::string near  = nearestModeIter->second;

    const auto align_corners = (coord == "align_corners");

    mv::Shape output_shape = getMemoryOrder(interpolate->get_output_shape(0));
    auto mcmInterpolateOutput = mcmModel.interpolate(opName, mcmData, output_shape, mode, near, coord, align_corners, antialias);
    mcmInterpolateOutput->setQuantParams(initialQuantParams());

    registerOutputs(interpolate, {mcmInterpolateOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::ReverseSequence> reverse, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(reverse, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 2);

    const auto& opName = reverse->get_friendly_name();
    const auto seqAxis = reverse->get_sequence_axis();
    const auto batchAxis = reverse->get_batch_axis();

    mcmInputs.at(1)->setDType(mv::DType("Float16"));

    auto mvReverseSequenceOutput = mcmModel.reverseSequence(opName, mcmInputs.at(0), mcmInputs.at(1), seqAxis, batchAxis);
    mvReverseSequenceOutput->setQuantParams(initialQuantParams());

    registerOutputs(reverse, {mvReverseSequenceOutput}, mcmOutputsMap);
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
        IE_THROW() << "Deconvolution supports only equal dilationX and dilationY";

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
    const mv::Shape outShape = getMemoryOrder(crop->get_output_shape(0));
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
            IE_THROW() << "Crop layer dim parameter mismatches output shape";
        // mcmModel.crop() is single dimensional and mcmModel.slice() is multdimensional
        auto mcmSlice = mcmModel.slice(crop->get_friendly_name(), mcmInputs.at(0), mvOffsets, outShape);
        mcmSlice->setQuantParams(initialQuantParams());
        registerOutputs(crop, {mcmSlice}, mcmOutputsMap);
    } else {
        IE_THROW() << "Unsupported Crop layer parameters:"
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
    const auto opName = op->get_friendly_name();
    mv::Data::TensorIterator mcmOpOutput;
    if (1u == op->input(1).get_shape().size())
        mcmOpOutput = mcmModel.scale(opName, mcmInputs.at(0), mcmInputs.at(1));
    else
        mcmOpOutput = mcmModel.eltwise(opName, mcmInputs, "Multiply");
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::NormalizeIE> normalizeIE, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(normalizeIE, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 2);
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = normalizeIE->get_friendly_name();

    auto weights_node = std::dynamic_pointer_cast<ngraph::op::Constant> (normalizeIE->input(1).get_source_output().get_node_shared_ptr());
    IE_ASSERT(nullptr != weights_node);
    const auto weights_shape = weights_node->get_shape();
    std::vector<double> weights = weights_node->cast_vector<double>();
    mv::Shape weights_shape_4d = (weights_shape.size() == 4) ? weights_shape : mv::Shape {1, weights_shape[0], 1, 1};

    const bool channel_shared = normalizeIE->get_channel_shared();
    const bool across_spatial = normalizeIE->get_across_spatial();
    const double eps = normalizeIE->get_eps();

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    auto mvWeightsValues = mcmModel.constant("", weights, weights_shape_4d, mv::DType("Float32"), mv::Order::getZMajorID(4));

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
        IE_THROW() << "Proposal layer doesn't support framework: \'" << framework << "\'";
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

    auto data= mcmInputs.at(0);
    auto index= mcmInputs.at(1);
    if (index->getDType().toString() == "Float16"){
        auto index_i32= mcmModel.conversion(opName+"/Conversion", index, mv::DType("Int32"));
        index_i32->setQuantParams(initialQuantParams());
        index = index_i32;
    }

    // TODO: Replace Float16 with Default when MCM Compiler is fixed. See ticket #40356
    auto mcmGather = mcmModel.gather(opName, data, index, axis);
    mcmGather->setDType(mv::DType("Float16"));
    mcmGather->setQuantParams(initialQuantParams());

    registerOutputs(gatherIE, {mcmGather}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Maximum> maximum, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(maximum, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());
    const auto& opName = maximum->get_friendly_name();
    auto mcmMax = mcmModel.eltwise(opName, mcmInputs, "Maximum");
    mcmMax->setQuantParams(initialQuantParams());
    registerOutputs(maximum, {mcmMax}, mcmOutputsMap);
}


void convert(std::shared_ptr<ngraph::op::v1::Minimum> minimum, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(minimum, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());
    const auto& opName = minimum->get_friendly_name();
    auto mcmMin = mcmModel.eltwise(opName, mcmInputs, "Minimum");
    mcmMin->setQuantParams(initialQuantParams());
    registerOutputs(minimum, {mcmMin}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Split> split, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto& opName = split->get_friendly_name();
    const auto mcmInputs = getMcmInputs(split, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 2u);

    // Find axis.
    const auto axis_node = split->input_value(1).get_node_shared_ptr();
    const auto axis_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(axis_node);
    auto axis = axis_node_const->get_data_ptr<int64_t>()[0];

    for (size_t i = 1; i < mcmInputs.size(); ++i) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    std::vector<size_t> startCoords(mcmInputs.at(0)->getShape().ndims());
    std::vector<mv::Data::TensorIterator> mcmOutputs;
    auto outDimSize = split->get_output_shape(0).size();
    for (size_t i = 0; i < split->get_output_size(); ++i) {
        mv::Shape beginShape(startCoords);
        mv::Shape sizeShape(getMemoryOrder(split->get_output_shape(i)));
        auto mcmSplit = mcmModel.slice(opName + ":" + std::to_string(i), mcmInputs.at(0), beginShape, sizeShape);
        mcmSplit->setQuantParams(initialQuantParams());
        mcmOutputs.push_back(mcmSplit);
        startCoords[outDimSize - 1 - axis] += split->get_output_shape(i)[axis];
    }
    registerOutputs(split, mcmOutputs, mcmOutputsMap);
}

template <typename Vec>
mv::Shape vector_canonicalize(const Vec& vec) {
    const auto negative_value = std::find_if(begin(vec), end(vec),
        [](decltype(vec[0]) value) { return value < 0; });
    IE_ASSERT(negative_value == vec.end());

    auto shape = std::vector<size_t>(vec.size());
    std::copy(begin(vec), end(vec), begin(shape));

    return mv::Shape{shape};
};

void convert(std::shared_ptr<ngraph::op::v1::StridedSlice> stridedSlice, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto& opName = stridedSlice->get_friendly_name();
    const auto mcmInputs = getMcmInputs(stridedSlice, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 4u);

    const auto get_input_data = [&](int index) {
        auto node = stridedSlice->input_value(index).get_node_shared_ptr();
        IE_ASSERT(node.get() != nullptr);
        auto node_const = ngraph::as_type_ptr<ngraph::op::Constant>(node);
        IE_ASSERT(node_const.get() != nullptr);
        return node_const->cast_vector<int64_t>();
    };

    auto mask_to_axis_set = [](const std::vector<int64_t>& mask) {
        ngraph::AxisSet axis_set{};
        for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
            if (mask[i] == 1) {
                axis_set.emplace(i);
            }
        }
        return axis_set;
    };

    auto plan = make_slice_plan(stridedSlice->get_input_shape(0),
                                get_input_data(1),
                                get_input_data(2),
                                get_input_data(3),
                                mask_to_axis_set(stridedSlice->get_begin_mask()),
                                mask_to_axis_set(stridedSlice->get_end_mask()),
                                mask_to_axis_set(stridedSlice->get_new_axis_mask()),
                                mask_to_axis_set(stridedSlice->get_shrink_axis_mask()),
                                mask_to_axis_set(stridedSlice->get_ellipsis_mask()));

    std::reverse(begin(plan.reshape_out_shape), end(plan.reshape_out_shape));

    // Remove unused constant inputs.
    for (size_t i = 1; i < mcmInputs.size(); ++i) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    auto beginShape = vector_canonicalize(plan.begins);
    auto endShape = vector_canonicalize(plan.ends);
    auto strideShape = vector_canonicalize(plan.strides);
    auto outShape = vector_canonicalize(plan.reshape_out_shape);

    auto mcmStridedSlice = mcmModel.stridedSlice(opName, mcmInputs.at(0), beginShape, endShape, strideShape, outShape);

    // TODO Add suport for negative strides
    // TODO Add Reverse for plan.reverse_axes dims

    registerOutputs(stridedSlice, {mcmStridedSlice}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::TileIE> tileIE, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(tileIE, mcmOutputsMap);
    IE_ASSERT(1u == mcmInputs.size());
    const auto& opName = tileIE->get_friendly_name();
    const int64_t axis = tileIE->axis;
    const int64_t tiles = tileIE->tiles;
    auto mcmTile = mcmModel.tile(opName, mcmInputs.at(0), axis, tiles);

    registerOutputs(tileIE, {mcmTile}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::VariadicSplit> variadicSplit,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto& opName = variadicSplit->get_friendly_name();
    const auto mcmInputs = getMcmInputs(variadicSplit, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 3u);

    for (size_t i = 1; i < mcmInputs.size(); i++) {
        mcmModel.removeOp(mcmModel.getSourceOp(mcmInputs.at(i)));
    }

    // Find axis.
    const auto axis_node = variadicSplit->input_value(1).get_node_shared_ptr();
    const auto axis_node_const = ngraph::as_type_ptr<ngraph::op::Constant>(axis_node);
    auto axis = axis_node_const->get_data_ptr<int64_t>()[0];

    std::vector<size_t> startCoords(mcmInputs.at(0)->getShape().ndims());
    std::vector<mv::Data::TensorIterator> mcmOutputs;
    auto outDimSize = variadicSplit->get_output_shape(0).size();
    for (size_t i = 0; i < variadicSplit->get_output_size(); ++i) {
        mv::Shape beginShape(startCoords);
        mv::Shape sizeShape(getMemoryOrder(variadicSplit->get_output_shape(i)));
        auto mcmSplit = mcmModel.slice(opName + ":" + std::to_string(i), mcmInputs.at(0), beginShape, sizeShape);
        mcmSplit->setQuantParams(initialQuantParams());
        mcmOutputs.push_back(mcmSplit);
        startCoords[outDimSize - 1 - axis] += variadicSplit->get_output_shape(i)[axis];
    }
    registerOutputs(variadicSplit, mcmOutputs, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::CTCGreedyDecoder> CTCGreedyDecoder, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(CTCGreedyDecoder, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());

    auto mcmCTCGreedyDecoder = mcmModel.cTCDecoder(CTCGreedyDecoder->get_friendly_name(),
                                                   mcmInputs.at(0), mcmInputs.at(1),
                                                   CTCGreedyDecoder->get_ctc_merge_repeated());

    registerOutputs(CTCGreedyDecoder, {mcmCTCGreedyDecoder}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v1::Pad> pad, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(pad, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());

    const auto mcmData = mcmInputs.at(0);
    const auto &opName = pad->get_friendly_name();

    const auto padsBegin = pad->get_pads_begin();
    const auto padsEnd = pad->get_pads_end();
    const auto mode = pad->get_pad_mode();
    std::string padMode;

    switch (mode) {
        case ngraph::op::PadMode::CONSTANT:
            padMode = "constant";
            break;
        case ngraph::op::PadMode::EDGE:
            padMode = "edge";
            break;
        case ngraph::op::PadMode::REFLECT:
            padMode = "reflect";
            break;
        case ngraph::op::PadMode::SYMMETRIC:
            padMode = "symmetric";
            break;
        default:
            IE_THROW() << "Invalid border mode " << mode << " in layer ";
    }

    const auto padValue = 0.0; //pad->get_pad_value(); //in pad_ie.hpp

    uint16_t pad0_begin = static_cast<uint16_t>(padsBegin.at(0));
    uint16_t pad1_begin = static_cast<uint16_t>(padsBegin.at(1));
    uint16_t pad2_begin = static_cast<uint16_t>(padsBegin.at(2));
    uint16_t pad3_begin = static_cast<uint16_t>(padsBegin.at(3));

    uint16_t pad0_end = static_cast<uint16_t>(padsEnd.at(0));
    uint16_t pad1_end = static_cast<uint16_t>(padsEnd.at(1));
    uint16_t pad2_end = static_cast<uint16_t>(padsEnd.at(2));
    uint16_t pad3_end = static_cast<uint16_t>(padsEnd.at(3));

    const auto mcmPadOutput = mcmModel.pad(opName, mcmData,
                                           {pad0_begin, pad1_begin, pad2_begin, pad3_begin},
                                           {pad0_end, pad1_end, pad2_end, pad3_end},
                                           padMode, padValue);

    mcmPadOutput->setQuantParams(initialQuantParams());
    registerOutputs(pad, {mcmPadOutput}, mcmOutputsMap);

}

void convert(std::shared_ptr<ngraph::op::PadIE> pad, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(pad, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());

    const auto mcmData = mcmInputs.at(0);
    const auto &opName = pad->get_friendly_name();

    const auto padsBegin = pad->get_pads_begin();
    const auto padsEnd = pad->get_pads_end();
    const auto mode = pad->get_pad_mode();
    std::string padMode;

    switch (mode) {
        case ngraph::op::PadMode::CONSTANT:
            padMode = "constant";
            break;
        case ngraph::op::PadMode::EDGE:
            padMode = "edge";
            break;
        case ngraph::op::PadMode::REFLECT:
            padMode = "reflect";
            break;
        case ngraph::op::PadMode::SYMMETRIC:
            padMode = "symmetric";
            break;
        default:
            IE_THROW() << "Invalid border mode " << mode << " in layer ";
    }

    const auto padValue = pad->get_pad_value();

    uint16_t pad0_begin = static_cast<uint16_t>(padsBegin.at(0));
    uint16_t pad1_begin = static_cast<uint16_t>(padsBegin.at(1));
    uint16_t pad2_begin = static_cast<uint16_t>(padsBegin.at(2));
    uint16_t pad3_begin = static_cast<uint16_t>(padsBegin.at(3));

    uint16_t pad0_end = static_cast<uint16_t>(padsEnd.at(0));
    uint16_t pad1_end = static_cast<uint16_t>(padsEnd.at(1));
    uint16_t pad2_end = static_cast<uint16_t>(padsEnd.at(2));
    uint16_t pad3_end = static_cast<uint16_t>(padsEnd.at(3));

    const auto mcmPadOutput = mcmModel.pad(opName, mcmData,
                                           {pad0_begin, pad1_begin, pad2_begin, pad3_begin},
                                           {pad0_end, pad1_end, pad2_end, pad3_end},
                                           padMode, padValue);

    mcmPadOutput->setQuantParams(initialQuantParams());
    registerOutputs(pad, {mcmPadOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::MVN> MVN, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(MVN, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = MVN->get_friendly_name();

    auto mcmMVN = mcmModel.mVN(opName, mcmData, MVN->get_across_channels(), MVN->get_normalize_variance(), MVN->get_eps());

    registerOutputs(MVN, {mcmMVN}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::SpaceToDepth> SpaceToDepth, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(SpaceToDepth, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = SpaceToDepth->get_friendly_name();

    std::string mode;
    switch (SpaceToDepth->get_mode()) {
        case ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
            mode = "blocks_first";
            break;
        case ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
            mode = "depth_first";
            break;
        default:
            IE_THROW() << "Invalid mode " << mode << " in SpaceToDepth layer ";;
    }

    auto mcmSpaceToDepth = mcmModel.spaceToDepth(opName, mcmData, SpaceToDepth->get_block_size(), mode);

    registerOutputs(SpaceToDepth, {mcmSpaceToDepth}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen> CTCGreedyDecoderSeqLen,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(CTCGreedyDecoderSeqLen, mcmOutputsMap);
    IE_ASSERT(mcmInputs.size() == 2u || mcmInputs.size() == 3u);

    auto ctcOutput0 = mcmModel.cTCGreedyDecoderSeqLen(
        CTCGreedyDecoderSeqLen->get_friendly_name(),
        mcmInputs.at(0), mcmInputs.at(1), mcmInputs.at(2),
        CTCGreedyDecoderSeqLen->get_merge_repeated());
    ctcOutput0->setQuantParams(initialQuantParams());

    auto ctcOp = mcmModel.getSourceOp(ctcOutput0);
    const auto outputSlots = ctcOp->outputSlots();
    IE_ASSERT(2 == outputSlots);
    auto ctcOutput1 = ctcOp->getOutputTensor(1);
    ctcOutput1->setQuantParams(initialQuantParams());
    registerOutputs(CTCGreedyDecoderSeqLen, {ctcOutput0, ctcOutput1}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::SquaredDifference> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(op, mcmOutputsMap);
    IE_ASSERT(2u == mcmInputs.size());
    const auto opName = op->get_friendly_name();
    mv::Data::TensorIterator mcmOpOutput;
    mcmOpOutput = mcmModel.eltwise(opName, mcmInputs, "SqDiff");
    mcmOpOutput->setQuantParams(initialQuantParams());
    registerOutputs(op, {mcmOpOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::v0::DepthToSpace> DepthToSpace, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(DepthToSpace, mcmOutputsMap);
    IE_ASSERT(1 == mcmInputs.size());
    const auto mcmData = mcmInputs.at(0);
    const auto& opName = DepthToSpace->get_friendly_name();

    std::string mode;
    switch (DepthToSpace->get_mode()) {
        case ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
            mode = "blocks_first";
            break;
        case ngraph::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
            mode = "depth_first";
            break;
        default:
            THROW_IE_EXCEPTION << "Invalid mode " << mode << " in DepthToSpace layer ";;
    }

    auto mcmDepthToSpace = mcmModel.depthToSpace(opName, mcmData, DepthToSpace->get_block_size(), mode);

    registerOutputs(DepthToSpace, {mcmDepthToSpace}, mcmOutputsMap);
}

// TODO: move converters to class ConvertToMcmModel scope to remove references to data

template <typename T>
void convertDispatch(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr /*unused*/, bool /*unused*/, bool /*unused*/) {
    convert(std::dynamic_pointer_cast<T>(node), mcmModel, mcmOutputsMap);
}

// Propagate ieData precision to MCM in order to perform conversion on hardware
template<>
void convertDispatch<ngraph::op::Parameter>(std::shared_ptr<ngraph::Node> node,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap, InferenceEngine::DataPtr ieData, bool allowNCHWInput,
    bool /*unused*/) {
    convert(std::dynamic_pointer_cast<ngraph::op::Parameter>(node), mcmModel, mcmOutputsMap, ieData, allowNCHWInput);
}

template<>
void convertDispatch<ngraph::op::Result>(std::shared_ptr<ngraph::Node> node,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap, InferenceEngine::DataPtr ieData, bool /*unused*/, bool /*unused*/) {
    convert(std::dynamic_pointer_cast<ngraph::op::Result>(node), mcmModel, mcmOutputsMap, ieData);
}

template<>
void convertDispatch<ngraph::op::Transpose>(std::shared_ptr<ngraph::Node> node,
    mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap, InferenceEngine::DataPtr /*unused*/, bool /*unused*/,
    bool allowPermuteND) {
    convert(std::dynamic_pointer_cast<ngraph::op::Transpose>(node), mcmModel, mcmOutputsMap, allowPermuteND);
}

#define MAP_ENTRY(__OP__) {__OP__::type_info, convertDispatch<__OP__>}

static const DispatchMap dispatchMap {
    MAP_ENTRY(ngraph::op::Parameter),
    MAP_ENTRY(ngraph::op::Result),
    MAP_ENTRY(ngraph::op::Constant),
    MAP_ENTRY(ngraph::op::v0::ROIPooling),
    MAP_ENTRY(ngraph::op::v0::PSROIPooling),
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
    MAP_ENTRY(ngraph::op::v1::StridedSlice),
    MAP_ENTRY(ngraph::op::v4::HSwish),
    MAP_ENTRY(ngraph::op::SwishIE),
    MAP_ENTRY(ngraph::op::v4::Mish),
    MAP_ENTRY(ngraph::op::v0::Floor),
    MAP_ENTRY(ngraph::op::v5::Round),
    MAP_ENTRY(ngraph::op::v0::Erf),
    MAP_ENTRY(ngraph::op::v0::Gelu),
    MAP_ENTRY(ngraph::op::v0::Log),
    MAP_ENTRY(ngraph::op::TileIE),
    MAP_ENTRY(ngraph::op::v1::VariadicSplit),
    MAP_ENTRY(ngraph::op::CTCGreedyDecoder),
    MAP_ENTRY(ngraph::op::v4::SoftPlus),
    MAP_ENTRY(ngraph::op::v1::Pad),
    MAP_ENTRY(ngraph::op::PadIE),
    MAP_ENTRY(ngraph::op::v4::Interpolate),
    MAP_ENTRY(ngraph::op::v0::MVN),
    MAP_ENTRY(ngraph::op::v0::Ceiling),
    MAP_ENTRY(ngraph::op::v0::PRelu),
    MAP_ENTRY(ngraph::op::v0::SpaceToDepth),
    MAP_ENTRY(ngraph::op::v6::CTCGreedyDecoderSeqLen),
    MAP_ENTRY(ngraph::op::v0::Convert),
    MAP_ENTRY(ngraph::op::v0::SquaredDifference),
    MAP_ENTRY(ngraph::op::v0::DepthToSpace),
    MAP_ENTRY(ngraph::op::v0::ReverseSequence)
};

#undef MAP_ENTRY

void ConvertNode(const std::shared_ptr<ngraph::Node> op, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
    InferenceEngine::DataPtr ieData, bool allowNCHWInput, bool allowPermuteND) {
    const auto dispatchIt = dispatchMap.find(op->get_type_info());
    if (dispatchIt != dispatchMap.end()) {
        const auto convertor = dispatchIt->second;
        if (convertor != nullptr) {
            try {
                convertor(op, mcmModel, mcmOutputsMap, ieData, allowNCHWInput, allowPermuteND);
            } catch (const std::runtime_error& ex) {
                IE_THROW() << "Convertor for operation " << op->get_friendly_name()
                                   << " failed due to runtime error " << ex.what();
            }
        } else {
            IE_THROW() << "Convertor not found for operation: " << op->get_friendly_name();
        }
    } else {
        IE_THROW() << "Unsupported operation: " << op->get_friendly_name() << " with name " << op->get_name()
                           << " with type " << op->get_type_name() << " with C++ type " << typeid(*op.get()).name();
    }
}

}  // namespace

// clang-format on

void ConvertToMcmModel::parseCustom(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel,
                                    NodeOutputToMcmMap& mcmOutputsMap) {
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
        const auto stage = parser.parseKernelArguments(kernel->bindings());

        const auto kernelData = parser.resolveKernelArguments(*kernel, stage.arguments);
        const auto stageOutputs = parser.resolveStageOutputs(*customLayer, stage.outputs);

        vpu::OperationFactory opFactory{stageIdx,     mcmModel,     kernelData,
                                        stage.inputs, stageOutputs, node->get_friendly_name()};

        kernel->accept(opFactory);
        auto custom = opFactory.result();

        stageIdx++;

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
    const auto allowU8InputForFp16Models = _config.allowU8InputForFp16Models();
    const auto allowPermuteND = _config.allowPermuteND();
    for (const auto& inputInfo : _networkInputs) {
        bool isFound = false;
        for (const auto& op : func->get_parameters()) {
            if (op->get_friendly_name() == _ioMap.at(inputInfo.first)) {
                ConvertNode(op, _mcmModel, _mcmOutputsMap, inputInfo.second->getInputData(), allowNCHWInput,
                            allowPermuteND);
                isFound = true;
            }
        }
        if (!isFound)
            IE_THROW() << "Input not found: " << inputInfo.first;
    }

    if (!_config.customLayers().empty()) {
        _customLayers = vpu::CustomLayer::loadFromFile(_config.customLayers());
    }

    for (const auto& op : func->get_ordered_ops()) {
        if (ngraph::op::Constant::type_info == op->get_type_info()) {
            ConvertNode(op, _mcmModel, _mcmOutputsMap, nullptr, false, allowPermuteND);
        }
    }

    for (const auto& op : func->get_ordered_ops()) {
        if (ngraph::op::Parameter::type_info == op->get_type_info())
            continue;
        if (ngraph::op::Result::type_info == op->get_type_info())
            continue;
        if (ngraph::op::Constant::type_info == op->get_type_info())
            continue;

        const auto customLayersForType = _customLayers.find(op->description());

        if (customLayersForType != _customLayers.end()) {
            const auto suitableLayers = getSuitableCustomLayers(customLayersForType->second, op);
            if (!suitableLayers.empty()) {
                parseCustom(op, _mcmModel, _mcmOutputsMap);
                continue;
            }
        }

        ConvertNode(op, _mcmModel, _mcmOutputsMap, nullptr, false, allowPermuteND);
    }

    for (const auto& outputInfo : _networkOutputs) {
        bool isFound = false;
        for (const auto& op : func->get_results()) {
            if (op->get_friendly_name() == _ioMap.at(outputInfo.first)) {
                ConvertNode(op, _mcmModel, _mcmOutputsMap, outputInfo.second, false, allowPermuteND);
                isFound = true;
            }
        }
        if (!isFound)
            IE_THROW() << "Output not found: " << outputInfo.first;
    }

    for (const auto& inputInfo : _networkInputs) {
        const auto inputData = inputInfo.second->getInputData();
        const auto inputPrecision = inputData->getTensorDesc().getPrecision();
        if (!isInputPrecisionSupported(inputPrecision)) {
            IE_THROW() << "Input data type is not supported: " << inputData->getTensorDesc().getPrecision();
        }
        mv::DType dType = cvtElemTypeToMCM(cvtPrecisionToElemType(inputPrecision));
        if (allowU8InputForFp16Models)
            dType = mv::DType("Float16");
        if (*_needConvertInputPrecision)
            dType = mv::DType("UInt8");

        for (const auto& mcmInput : _mcmModel.getNetworkInputs()) {
            if (mcmInput->getName() == _ioMap.at(inputInfo.first)) {
                mcmInput->set<mv::DType>("dType", dType);
                mcmInput->getOutputTensor(0)->set<mv::DType>("dType", dType);
                break;
            }
        }
    }

    return false;
}
