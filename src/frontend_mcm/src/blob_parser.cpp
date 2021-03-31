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

#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile/graphfile_generated.h>

#include <cassert>
#include <ie_input_info.hpp>
#include <memory>
#include <string>

#include "converters.hpp"

#ifndef UNUSED
#define UNUSED(var) (void)var
#endif

namespace vpu {
namespace MCMAdapter {
// FIXME: inconsistency with how we extract layout info from meta data
// we need a single way how to extract layout from compiled network
static InferenceEngine::Layout extractLayoutFromStrides(const std::vector<float>& strides) {
    const size_t NCHW_DIM_COUNT = 5;
    const size_t NCDHW_DIM_COUNT = 6;
    IE_ASSERT(strides.size() == NCHW_DIM_COUNT || strides.size() == NCDHW_DIM_COUNT)
            << " extractLayoutFromStrides works only with 5 or 6 elements in strides parameter";

    InferenceEngine::Layout tensorLayout = InferenceEngine::Layout::NCHW;
    if (strides.size() == NCHW_DIM_COUNT) {
        /// size_t DIM_BYTE_SIZE = 0;
        /// size_t DIM_N = 1;
        size_t DIM_C = 2;
        size_t DIM_H = 3;
        size_t DIM_W = 4;
        auto maxStrideVal = *std::max_element(strides.begin() + DIM_C, strides.end());
        if (maxStrideVal == strides[DIM_H]) {
            if (std::max(strides[DIM_W], strides[DIM_C]) == strides[DIM_W]) {
                tensorLayout = InferenceEngine::Layout::NHWC;
            }
        } else if (maxStrideVal == strides[DIM_C]) {
            if (std::max(strides[DIM_W], strides[DIM_H]) == strides[DIM_H]) {
                tensorLayout = InferenceEngine::Layout::NCHW;
            }
        } else {
            // width-major
            IE_THROW() << "getIOLayout: W-major layout is not supported";
        }
    } else {
        /// size_t DIM_BYTE_SIZE = 0;
        /// size_t DIM_N = 1;
        size_t DIM_C = 2;
        size_t DIM_D = 3;
        /// size_t DIM_H = 4;
        /// size_t DIM_W = 5;
        auto maxStrideVal = *std::max_element(strides.begin() + DIM_C, strides.end());
        if (maxStrideVal == strides[DIM_C]) {
            tensorLayout = InferenceEngine::Layout::NCDHW;
        } else if (maxStrideVal == strides[DIM_D]) {
            tensorLayout = InferenceEngine::Layout::NDHWC;
        } else {
            // width-major
            IE_THROW() << "getIOLayout: only NCDHW and NDHWC layouts are supported";
        }
    }

    return tensorLayout;
}

static InferenceEngine::Data deserializeTensor(const std::unique_ptr<MVCNN::TensorReferenceT>& tensor) {
    auto dimsPtr = tensor->dimensions;
    InferenceEngine::SizeVector dataDims;
    std::copy(tensor->dimensions.begin(), tensor->dimensions.end(), std::back_inserter(dataDims));

    InferenceEngine::Layout dataLayout = extractLayoutFromStrides(tensor->strides);
    InferenceEngine::Precision dataPrecision = MvcnnDTypeToPrecision(tensor->data_dtype);

    InferenceEngine::TensorDesc ieDesc(dataPrecision, dataDims, dataLayout);

    auto eraseSubStr = [](std::string& str, std::string strToRemove, bool removeAllAfterSubstr = false) {
        std::size_t pos = str.find(strToRemove);
        if (pos != std::string::npos) {
            if (removeAllAfterSubstr) {
                str.erase(pos);
            } else {
                str.erase(pos, strToRemove.size());
            }
        }
    };

    std::string name = tensor->name;
    // FIXME: For some reason, the compiler adds Precision prefix for all its outputs
    // remove once it fixed
    eraseSubStr(name, "Precision");
    // FIXME: frontend_mcm adds REMOVE_ME postfix to make output name unique
    // remove once the compiler able to handle output which name equal to one of network operations
    eraseSubStr(name, "REMOVE_ME", true);
    InferenceEngine::Data ieData(name, ieDesc);

    return ieData;
}

void getNetworkInputs(const void* data, InferenceEngine::InputsDataMap& networkInputs) {
    IE_ASSERT(nullptr != data);

    const auto* graphFilePtr = MVCNN::GetGraphFile(data);
    MVCNN::GraphFileT graphFileInstance;
    graphFilePtr->UnPackTo(&graphFileInstance);

    auto& inputs = graphFileInstance.header->net_input;

    auto processTensor = [&](const std::unique_ptr<MVCNN::TensorReferenceT>& tensor) {
        InferenceEngine::Data ieData = deserializeTensor(tensor);

        InferenceEngine::InputInfo inputInfo;
        inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(ieData));
        networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    };

    for (const auto& tensor : inputs) {
        processTensor(tensor);
    }
}

void getNetworkOutputs(const void* data, InferenceEngine::OutputsDataMap& networkOutputs) {
    IE_ASSERT(nullptr != data);
    const auto* graphFilePtr = MVCNN::GetGraphFile(data);
    MVCNN::GraphFileT graphFileInstance;
    graphFilePtr->UnPackTo(&graphFileInstance);

    const auto& outputs = graphFileInstance.header->net_output;
    auto processTensor = [&](const std::unique_ptr<MVCNN::TensorReferenceT>& tensor) {
        InferenceEngine::Data ieData = deserializeTensor(tensor);
        networkOutputs[ieData.getName()] = std::make_shared<InferenceEngine::Data>(ieData);
    };

    for (const auto& tensor : outputs) {
        processTensor(tensor);
    }
}

}  // namespace MCMAdapter
}  // namespace vpu
