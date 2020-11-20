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

#include "vpux/compiler/dialect/VPUIP/network_description.hpp"

#include "vpux/compiler/core/dims_order.hpp"
#include "vpux/compiler/dialect/VPUIP/schema.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ie_data.h>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>

#include <algorithm>

using namespace vpux;
using namespace InferenceEngine;

namespace {

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

Data deserializeTensor(const MVCNN::TensorReference* tensor) {
    const auto* dims = tensor->dimensions();

    SizeVector dataDims;
    dataDims.resize(dims->size());
    std::copy_n(dims->data(), dims->size(), dataDims.data());

    const auto dimsOrder = DimsOrder::fromCode(tensor->order());
    VPUX_THROW_UNLESS(dimsOrder.numDims() == dims->size(), "DimsOrder {0} doesn't match to dims {1}", dimsOrder,
                      dataDims);

    const auto dataLayout = dimsOrder.toIE();
    const auto dataPrecision = extractPrecisionFromDType(tensor->data_dtype());

    TensorDesc dataDesc(dataPrecision, dataDims, dataLayout);

    return Data(tensor->name()->str(), dataDesc);
}

using TensorReferenceVector = flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>>;

DataMap deserializeDataMap(const TensorReferenceVector* tensors) {
    DataMap out;

    for (auto ind : irange(tensors->size())) {
        const auto* tensor = tensors->Get(ind);

        const auto ieData = deserializeTensor(tensor);

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

    _networkInputs = deserializeDataMap(header->in_tensor_desc());
    _networkOutputs = deserializeDataMap(header->out_tensor_desc());

    _deviceInputs = deserializeDataMap(header->net_input());
    _deviceOutputs = deserializeDataMap(header->net_output());

    VPUX_THROW_UNLESS(!_networkInputs.empty(), "VPUIP blob does not contain network inputs");
    VPUX_THROW_UNLESS(!_networkOutputs.empty(), "VPUIP blob does not contain network outputs");

    VPUX_THROW_UNLESS(!_deviceInputs.empty(), "VPUIP blob does not contain device inputs");
    VPUX_THROW_UNLESS(!_deviceOutputs.empty(), "VPUIP blob does not contain device outputs");
}
