//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/network_description.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/schema.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ie_data.h>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>

#include <algorithm>

using namespace vpux;
using namespace InferenceEngine;

namespace {

const uint32_t DIM_N = 0, DIM_C = 1, DIM_H = 2, DIM_W = 3, DIM_D = 4;

static const std::unordered_map<DimsOrder, std::vector<float>> orderMapping = {
        {DimsOrder::NCHW, {DIM_N, DIM_C, DIM_H, DIM_W}},
        {DimsOrder::NHWC, {DIM_N, DIM_H, DIM_W, DIM_C}},
        {DimsOrder::NCDHW, {DIM_N, DIM_C, DIM_D, DIM_H, DIM_W}},
        {DimsOrder::NDHWC, {DIM_N, DIM_D, DIM_H, DIM_W, DIM_C}},
        {DimsOrder::C, {DIM_C}},
        {DimsOrder::CHW, {DIM_C, DIM_H, DIM_W}},
        {DimsOrder::NC, {DIM_N, DIM_C}},
};

InferenceEngine::Precision extractPrecisionFromDType(MVCNN::DType dtype) {
    static const EnumMap<MVCNN::DType, Precision> dataTypeMapping = {
            {MVCNN::DType_FP32, Precision::FP32}, {MVCNN::DType_FP16, Precision::FP16},
            {MVCNN::DType_U64, Precision::U64},   {MVCNN::DType_U32, Precision::U32},
            {MVCNN::DType_U16, Precision::U16},   {MVCNN::DType_U8, Precision::U8},
            {MVCNN::DType_I64, Precision::I64},   {MVCNN::DType_I32, Precision::I32},
            {MVCNN::DType_I16, Precision::I16},   {MVCNN::DType_I8, Precision::I8},
            {MVCNN::DType_BIN, Precision::BIN},
    };

    return dataTypeMapping.at(dtype);
}

DimsOrder extractLayoutFromStrides(const llvm::ArrayRef<float>& inStrides) {
    const std::size_t MAX_DIM_COUNT = 5;
    const std::size_t /*DIM_X = 0, DIM_N = 1,*/ DIM_C = 2, DIM_H = 3, DIM_W = 4;

    IE_ASSERT(inStrides.size() == MAX_DIM_COUNT)
            << "extractLayoutFromStrides works only with " << MAX_DIM_COUNT << " elements in strides parameter";

    DimsOrder tensorLayout = DimsOrder::NCHW;
    auto maxStrideVal = *std::max_element(inStrides.begin() + DIM_C, inStrides.end());
    if (maxStrideVal == inStrides[DIM_H]) {
        if (std::max(inStrides[DIM_W], inStrides[DIM_C]) == inStrides[DIM_W]) {
            tensorLayout = DimsOrder::NHWC;
        }
    } else if (maxStrideVal == inStrides[DIM_C]) {
        if (std::max(inStrides[DIM_W], inStrides[DIM_H]) == inStrides[DIM_H]) {
            tensorLayout = DimsOrder::NCHW;
        }
    } else {
        // width-major
        IE_THROW() << "getIOLayout: W-major layout is not supported";
    }

    return tensorLayout;
}

Data deserializeTensor(const MVCNN::TensorReference* tensor,
                       DimsOrder (*backupStridesToLayoutConvertor)(const llvm::ArrayRef<float>&)) {
    const auto* dims = tensor->dimensions();

    SizeVector dataDims;
    dataDims.resize(dims->size());
    std::copy_n(dims->data(), dims->size(), dataDims.data());

    DimsOrder dimsOrder;
    auto order = tensor->order();
    if (order != 0 || dataDims.empty()) {
        dimsOrder = DimsOrder::fromCode(order);
    } else {
        // if `order` filed doesn't present in blob let's try to guess layout by strides using
        // backupStridesToLayoutConvertor method
        const auto* strides = tensor->strides();
        const llvm::ArrayRef<float> stridesArray = makeArrayRef(tensor->strides()->data(), strides->size());

        dimsOrder = backupStridesToLayoutConvertor(stridesArray);
    }

    VPUX_THROW_UNLESS(dimsOrder.numDims() == dims->size(), "DimsOrder {0} doesn't match to dims {1}", dimsOrder,
                      dataDims);

    const auto dataLayout = dimsOrder.numDims() <= 5 ? dimsOrder.toIE() : InferenceEngine::Layout::ANY;
    const auto dataPrecision = extractPrecisionFromDType(tensor->data_dtype());

    TensorDesc dataDesc(dataPrecision, dataDims, dataLayout);

    return Data(tensor->name()->str(), dataDesc);
}

using TensorReferenceVector = flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>>;

NetworkIOVector deserializeIOVector(const TensorReferenceVector* tensors,
                                    DimsOrder (*backupStridesToLayoutConvertor)(const llvm::ArrayRef<float>&)) {
    NetworkIOVector out;

    if (tensors == nullptr) {
        return out;
    }

    for (auto ind : irange(tensors->size())) {
        const auto* tensor = tensors->Get(ind);

        const auto ieData = deserializeTensor(tensor, backupStridesToLayoutConvertor);

        out.emplace_back(ieData.getName(), std::make_shared<Data>(ieData));
    }

    return out;
}

const EnumMap<MVCNN::OVNodeType, ov::element::Type_t> mapElementTypeIE = {
        {MVCNN::OVNodeType::OVNodeType_UNDEFINED, ov::element::Type_t::undefined},
        {MVCNN::OVNodeType::OVNodeType_DYNAMIC, ov::element::Type_t::dynamic},
        {MVCNN::OVNodeType::OVNodeType_BOOLEAN, ov::element::Type_t::boolean},
        {MVCNN::OVNodeType::OVNodeType_BF16, ov::element::Type_t::bf16},
        {MVCNN::OVNodeType::OVNodeType_F16, ov::element::Type_t::f16},
        {MVCNN::OVNodeType::OVNodeType_F32, ov::element::Type_t::f32},
        {MVCNN::OVNodeType::OVNodeType_F64, ov::element::Type_t::f64},
        {MVCNN::OVNodeType::OVNodeType_I4, ov::element::Type_t::i4},
        {MVCNN::OVNodeType::OVNodeType_I8, ov::element::Type_t::i8},
        {MVCNN::OVNodeType::OVNodeType_I16, ov::element::Type_t::i16},
        {MVCNN::OVNodeType::OVNodeType_I32, ov::element::Type_t::i32},
        {MVCNN::OVNodeType::OVNodeType_I64, ov::element::Type_t::i64},
        {MVCNN::OVNodeType::OVNodeType_U1, ov::element::Type_t::u1},
        {MVCNN::OVNodeType::OVNodeType_U4, ov::element::Type_t::u4},
        {MVCNN::OVNodeType::OVNodeType_U8, ov::element::Type_t::u8},
        {MVCNN::OVNodeType::OVNodeType_U16, ov::element::Type_t::u16},
        {MVCNN::OVNodeType::OVNodeType_U32, ov::element::Type_t::u32},
        {MVCNN::OVNodeType::OVNodeType_U64, ov::element::Type_t::u64},
};

std::vector<OVRawNode> deserializeOVNodes(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::OVNode>>* mvcnnOVNode) {
    // Check for the existence of a field in a blob. In older versions of the blob, this field may not exist
    if (mvcnnOVNode == nullptr) {
        return {};
    }
    std::vector<OVRawNode> nodes;

    for (auto ind : irange(mvcnnOVNode->size())) {
        if (const auto* node = mvcnnOVNode->Get(ind)) {
            const auto nodeType = mapElementTypeIE.at(node->type());
            const auto nodeFriendlyName = node->friendly_name()->str();

            const auto nodeShape = [&node]() {
                ov::Shape retShape;
                for (auto iter = node->shape()->cbegin(); iter != node->shape()->cend(); ++iter) {
                    retShape.push_back(*iter);
                }
                return retShape;
            }();

            const auto tensorNames = [&node]() {
                std::unordered_set<std::string> retTensorNames;
                for (auto iter = node->tensor_names()->cbegin(); iter != node->tensor_names()->cend(); ++iter) {
                    retTensorNames.insert(iter->str());
                }
                return retTensorNames;
            }();

            const auto inputName = node->input_name()->str();
            nodes.push_back({nodeFriendlyName, nodeType, nodeShape, tensorNames, inputName});
        }
    }
    return nodes;
}
}  // namespace

vpux::VPUIP::NetworkDescription::NetworkDescription(std::vector<char> blob): _compiledNetwork(std::move(blob)) {
    OV_ITT_TASK_CHAIN(NETWORK_DESCRIPTION, itt::domains::VPUXPlugin, "NetworkDescription::NetworkDescription",
                      "VerifyGraphFileBuffer");
    VPUX_THROW_UNLESS(!_compiledNetwork.empty(), "Got NULL pointer");

    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(_compiledNetwork.data()), _compiledNetwork.size(),
                                   /*max_depth=*/128, /*max_tables=*/UINT32_MAX);
    VPUX_THROW_UNLESS(MVCNN::VerifyGraphFileBuffer(verifier), "Got invalid VPUIP blob - network description");

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "GetGraphFile");
    const auto* graphFile = MVCNN::GetGraphFile(_compiledNetwork.data());
    const auto* header = graphFile->header();

    if (header->identifier() != nullptr) {
        _name = header->identifier()->str();
    }

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeOVNodes");
    const auto ovParams = header->ov_parameters();
    if (ovParams != nullptr) {
        _ovParameters = deserializeOVNodes(ovParams);
    }
    const auto ovResults = header->ov_results();
    if (ovResults != nullptr) {
        _ovResults = deserializeOVNodes(ovResults);
    }

    OV_ITT_TASK_NEXT(NETWORK_DESCRIPTION, "deserializeIOVector");
    _deviceInputs = deserializeIOVector(header->net_input(), extractLayoutFromStrides);
    _deviceOutputs = deserializeIOVector(header->net_output(), extractLayoutFromStrides);
    _deviceProfilingOutputs = deserializeIOVector(header->profiling_output(), extractLayoutFromStrides);

    VPUX_THROW_UNLESS(!_deviceOutputs.empty(), "VPUIP blob does not contain device outputs");
}
