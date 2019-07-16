//
// Copyright 2019 Intel Corporation.
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

#include "kmb_blob_parser.hpp"

#include <string>
#include <vector>
#include <cassert>
#include <memory>

#include <vpu/utils/extra.hpp>

#ifdef ENABLE_MCM_COMPILER

namespace vpu {
namespace KmbPlugin {

inline InferenceEngine::Precision convertMVCNNType(const MVCNN::DType& mvcnnDataType) {
    InferenceEngine::Precision iePrecision = InferenceEngine::Precision::UNSPECIFIED;
    switch (mvcnnDataType) {
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

inline InferenceEngine::SizeVector convertMVCNNDims(const flatbuffers::Vector<uint32_t>* mvcnnDims) {
    InferenceEngine::SizeVector ieDims(mvcnnDims->begin(), mvcnnDims->end());
    return ieDims;
}

InferenceEngine::Layout getLayoutFromMVCNN(const flatbuffers::Vector<uint32_t>* mvcnnDims) {
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

InferenceEngine::Data deserializeTensor(const MVCNN::TensorReference& tensorRef) {
    auto dataDimsPtr = tensorRef.dimensions();
    InferenceEngine::SizeVector dataDims = convertMVCNNDims(dataDimsPtr);

    auto dataNamePtr = tensorRef.name();
    std::string dataName = "";
    if (dataNamePtr != nullptr) {
        dataName = dataNamePtr->c_str();
    }

    InferenceEngine::Layout dataLayout = getLayoutFromMVCNN(dataDimsPtr);
    auto dataType = tensorRef.data_dtype();
    InferenceEngine::Precision dataPrecision = convertMVCNNType(dataType);

    InferenceEngine::TensorDesc ieDesc(dataPrecision, dataDims, dataLayout);
    InferenceEngine::Data ieData(dataName, ieDesc);

    return ieData;
}

KmbBlob::KmbBlob(const void* data) {
//  FlatBuffer blob parser to get information about network inputs/outputs
//  from flatbuffer blob file loaded by ImportNetwork method

    assert(nullptr != data);

    const MVCNN::GraphFile* file = MVCNN::GetGraphFile(data);
    auto header = file->header();
    auto inputs = header->net_input();
    auto outputs = header->net_output();

    auto processTensor = [&](const MVCNN::TensorReference &tensor, bool isInput) {
        InferenceEngine::Data ieData = deserializeTensor(tensor);

        if (isInput) {
            InferenceEngine::InputInfo inputInfo;
            inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(ieData));
            _networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);

        } else {
            _networkOutputs[ieData.getName()] = std::make_shared<InferenceEngine::Data>(ieData);
        }
    };

    for (auto tensor : *inputs) {
        processTensor(*tensor, true);
    }

    for (auto tensor : *outputs) {
        processTensor(*tensor, false);
    }

    _blobHeader = header;
}

}  // namespace KmbPlugin
}  // namespace vpu
#endif
