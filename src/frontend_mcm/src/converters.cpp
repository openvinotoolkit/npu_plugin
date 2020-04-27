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

#include <flatbuffers/flatbuffers.h>
#include <ie_layouts.h>
#include <schema/graphfile/graphfile_generated.h>

#include <ie_precision.hpp>
#include <map>

const size_t DIM_N = 0, DIM_C = 1, DIM_H = 2, DIM_W = 3, DIM_D = 4;

static const std::map<InferenceEngine::Layout, std::vector<uint32_t>> orderMapping = {
    {InferenceEngine::Layout::NCHW, {DIM_N, DIM_C, DIM_H, DIM_W}},
    {InferenceEngine::Layout::NHWC, {DIM_N, DIM_H, DIM_W, DIM_C}},
    {InferenceEngine::Layout::NCDHW, {DIM_N, DIM_C, DIM_D, DIM_H, DIM_W}},
    {InferenceEngine::Layout::NDHWC, {DIM_N, DIM_D, DIM_H, DIM_W, DIM_C}},
    {InferenceEngine::Layout::C, {DIM_C}},
    {InferenceEngine::Layout::CHW, {DIM_C, DIM_H, DIM_W}},
    {InferenceEngine::Layout::NC, {DIM_N, DIM_C}},
};

static const std::map<InferenceEngine::Precision, MVCNN::DType> dataTypeMapping = {
    {InferenceEngine::Precision::FP32, MVCNN::DType::DType_FP32},
    {InferenceEngine::Precision::FP16, MVCNN::DType::DType_FP16},
    {InferenceEngine::Precision::U64, MVCNN::DType::DType_U64},
    {InferenceEngine::Precision::U16, MVCNN::DType::DType_U16},
    {InferenceEngine::Precision::U8, MVCNN::DType::DType_U8},
    {InferenceEngine::Precision::I64, MVCNN::DType::DType_I64},
    {InferenceEngine::Precision::I32, MVCNN::DType::DType_I32},
    {InferenceEngine::Precision::I16, MVCNN::DType::DType_I16},
    {InferenceEngine::Precision::I8, MVCNN::DType::DType_I8},
    {InferenceEngine::Precision::BIN, MVCNN::DType::DType_BIN},
};

InferenceEngine::Layout orderToLayout(const std::vector<uint32_t>& tensorOrder) {
    std::function<bool(const std::pair<InferenceEngine::Layout, std::vector<uint32_t>>&)> mapSearchPredicate =
        [tensorOrder](const std::pair<InferenceEngine::Layout, std::vector<uint32_t>>& orderPair) -> bool {
        size_t orderSize = tensorOrder.size();
        size_t pairSize = orderPair.second.size();
        return (orderSize == pairSize) && std::equal(tensorOrder.begin(), tensorOrder.end(), orderPair.second.begin());
    };
    std::map<InferenceEngine::Layout, std::vector<uint32_t>>::const_iterator mapIter =
        std::find_if(orderMapping.begin(), orderMapping.end(), mapSearchPredicate);
    if (mapIter == orderMapping.end()) {
        THROW_IE_EXCEPTION << "orderToLayout: failed to convert input order";
    }
    return mapIter->first;
}

InferenceEngine::Precision DTypeToPrecision(const MVCNN::DType& dtype) {
    std::function<bool(const std::pair<InferenceEngine::Precision, MVCNN::DType>&)> mapSearchPredicate =
        [dtype](const std::pair<InferenceEngine::Precision, MVCNN::DType>& dataTypePair) -> bool {
        return dtype == dataTypePair.second;
    };
    std::map<InferenceEngine::Precision, MVCNN::DType>::const_iterator mapIter =
        std::find_if(dataTypeMapping.begin(), dataTypeMapping.end(), mapSearchPredicate);
    if (mapIter == dataTypeMapping.end()) {
        THROW_IE_EXCEPTION << "DTypeToPrecision: failed to convert dtype: " << dtype;
    }
    return mapIter->first;
}

#ifdef ENABLE_MCM_COMPILER
std::vector<uint32_t> layoutToOrder(const InferenceEngine::Layout& tensorLayout) {
    return orderMapping.at(tensorLayout);
}

MVCNN::DType precisionToDType(const InferenceEngine::Precision& tensorPrecision) {
    return dataTypeMapping.at(tensorPrecision);
}
#endif
