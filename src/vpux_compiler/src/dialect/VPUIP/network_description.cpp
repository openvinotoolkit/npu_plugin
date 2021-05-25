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

#include "vpux/compiler/dialect/VPUIP/network_description.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"

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
            {MVCNN::DType_U64, Precision::U64},   {MVCNN::DType_U16, Precision::U16},
            {MVCNN::DType_U8, Precision::U8},     {MVCNN::DType_I64, Precision::I64},
            {MVCNN::DType_I32, Precision::I32},   {MVCNN::DType_I16, Precision::I16},
            {MVCNN::DType_I8, Precision::I8},     {MVCNN::DType_BIN, Precision::BIN},
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

DimsOrder orderVectorToLayout(const llvm::ArrayRef<float>& inStrides) {
    std::function<bool(const std::pair<DimsOrder, llvm::ArrayRef<float>>&)> mapSearchPredicate =
            [inStrides](const std::pair<DimsOrder, llvm::ArrayRef<float>>& orderPair) -> bool {
        size_t orderSize = inStrides.size();
        size_t pairSize = orderPair.second.size();
        return (orderSize == pairSize) && std::equal(inStrides.begin(), inStrides.end(), orderPair.second.begin());
    };
    std::unordered_map<DimsOrder, std::vector<float>>::const_iterator mapIter =
            std::find_if(orderMapping.begin(), orderMapping.end(), mapSearchPredicate);
    if (mapIter == orderMapping.end()) {
        IE_THROW() << "orderToLayout: failed to convert input order";
    }
    return mapIter->first;
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

DataMap deserializeDataMap(const TensorReferenceVector* tensors,
                           DimsOrder (*backupStridesToLayoutConvertor)(const llvm::ArrayRef<float>&)) {
    DataMap out;

    for (auto ind : irange(tensors->size())) {
        const auto* tensor = tensors->Get(ind);

        const auto ieData = deserializeTensor(tensor, backupStridesToLayoutConvertor);

        out.emplace(ieData.getName(), std::make_shared<Data>(ieData));
    }

    return out;
}

}  // namespace

vpux::VPUIP::NetworkDescription::NetworkDescription(std::vector<char> blob): _compiledNetwork(std::move(blob)) {
    VPUX_THROW_UNLESS(!_compiledNetwork.empty(), "Got NULL pointer");

    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(_compiledNetwork.data()), _compiledNetwork.size());
    VPUX_THROW_UNLESS(MVCNN::VerifyGraphFileBuffer(verifier), "Got invalid VPUIP blob");

    const auto* graphFile = MVCNN::GetGraphFile(_compiledNetwork.data());
    const auto* header = graphFile->header();

    if (header->identifier() != nullptr) {
        _name = header->identifier()->str();
    }

    _networkInputs = deserializeDataMap(header->in_tensor_desc(), orderVectorToLayout);
    _networkOutputs = deserializeDataMap(header->out_tensor_desc(), orderVectorToLayout);

    _deviceInputs = deserializeDataMap(header->net_input(), extractLayoutFromStrides);
    _deviceOutputs = deserializeDataMap(header->net_output(), extractLayoutFromStrides);

    VPUX_THROW_UNLESS(!_networkOutputs.empty(), "VPUIP blob does not contain network outputs");
    VPUX_THROW_UNLESS(!_deviceOutputs.empty(), "VPUIP blob does not contain device outputs");
}
