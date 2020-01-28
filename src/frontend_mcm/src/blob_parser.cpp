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

#include "blob_parser.hpp"

#ifdef ENABLE_MCM_COMPILER
#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile/graphfile_generated.h>
#endif

#include <cassert>
#include <ie_input_info.hpp>
#include <memory>
#include <string>

namespace vpu {
namespace MCMAdapter {
#ifdef ENABLE_MCM_COMPILER
static InferenceEngine::Precision convertmvType(const MVCNN::DType& mvDataType) {
    InferenceEngine::Precision iePrecision = InferenceEngine::Precision::UNSPECIFIED;
    switch (mvDataType) {
    case MVCNN::DType::DType_NOT_SET:
        iePrecision = InferenceEngine::Precision::UNSPECIFIED;
        break;
    case MVCNN::DType::DType_FP32:
        iePrecision = InferenceEngine::Precision::FP32;
        break;
    case MVCNN::DType::DType_FP16:
        iePrecision = InferenceEngine::Precision::FP16;
        break;
    case MVCNN::DType::DType_U16:
        iePrecision = InferenceEngine::Precision::U16;
        break;
    case MVCNN::DType::DType_U8:
        iePrecision = InferenceEngine::Precision::U8;
        break;
    case MVCNN::DType::DType_I32:
        iePrecision = InferenceEngine::Precision::I32;
        break;
    case MVCNN::DType::DType_I16:
        iePrecision = InferenceEngine::Precision::I16;
        break;
    case MVCNN::DType::DType_I8:
        iePrecision = InferenceEngine::Precision::I8;
        break;
    case MVCNN::DType::DType_BIN:
        iePrecision = InferenceEngine::Precision::BIN;
        break;
    default:
        iePrecision = InferenceEngine::Precision::CUSTOM;
    }
    return iePrecision;
}

static InferenceEngine::SizeVector convertmvDims(const flatbuffers::Vector<uint32_t>* mvcnnDims) {
    InferenceEngine::SizeVector ieDims(mvcnnDims->begin(), mvcnnDims->end());
    return ieDims;
}

static InferenceEngine::Layout getLayoutFrommv(const flatbuffers::Vector<uint32_t>* mvcnnDims) {
    InferenceEngine::Layout ieLayout = InferenceEngine::Layout::ANY;

    switch (mvcnnDims->size()) {
    case 1:
        ieLayout = InferenceEngine::Layout::C;
        break;
    case 2:
        ieLayout = InferenceEngine::Layout::NC;
        break;
    case 3:
        ieLayout = InferenceEngine::Layout::CHW;
        break;
    case 4:
        ieLayout = InferenceEngine::Layout::NCHW;
        break;
    default:
        ieLayout = InferenceEngine::Layout::ANY;
    }
    return ieLayout;
}

static InferenceEngine::Data deserializeTensor(const MVCNN::TensorReference& tensor) {
    auto dimsPtr = tensor.dimensions();
    InferenceEngine::SizeVector dataDims = convertmvDims(dimsPtr);

    auto dataNamePtr = tensor.name();
    std::string dataName = "";
    if (dataNamePtr != nullptr) {
        dataName = dataNamePtr->c_str();
    }

    InferenceEngine::Layout dataLayout = getLayoutFrommv(dimsPtr);
    auto dataType = tensor.data_dtype();
    InferenceEngine::Precision dataPrecision = convertmvType(dataType);

    InferenceEngine::TensorDesc ieDesc(dataPrecision, dataDims, dataLayout);
    InferenceEngine::Data ieData(dataName, ieDesc);

    return ieData;
}

#endif

void getNetworkInputs(const void* data, InferenceEngine::InputsDataMap& networkInputs) {
    IE_ASSERT(nullptr != data);
#ifdef ENABLE_MCM_COMPILER
    const MVCNN::GraphFile* file = MVCNN::GetGraphFile(data);
    auto header = file->header();
    auto inputs = header->net_input();

    auto processTensor = [&](const MVCNN::TensorReference& tensor) {
        InferenceEngine::Data ieData = deserializeTensor(tensor);

        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(ieData));
        networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    };

    for (auto tensor : *inputs) {
        processTensor(*tensor);
    }
#else
    THROW_IE_EXCEPTION << "MCM Compiler is disabled";
#endif
}

void getNetworkOutputs(const void* data, InferenceEngine::OutputsDataMap& networkOutputs) {
    IE_ASSERT(nullptr != data);
#ifdef ENABLE_MCM_COMPILER
    const MVCNN::GraphFile* file = MVCNN::GetGraphFile(data);
    auto header = file->header();
    auto outputs = header->net_output();

    auto processTensor = [&](const MVCNN::TensorReference& tensor) {
        InferenceEngine::Data ieData = deserializeTensor(tensor);
        networkOutputs[ieData.getName()] = std::make_shared<InferenceEngine::Data>(ieData);
    };

    for (auto tensor : *outputs) {
        processTensor(*tensor);
    }
#else
    THROW_IE_EXCEPTION << "MCM Compiler is disabled";
#endif
}

}  // namespace MCMAdapter
}  // namespace vpu
