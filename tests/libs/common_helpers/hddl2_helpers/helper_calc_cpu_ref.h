//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "ie_core.hpp"
#include "vpux/utils/IE/blob.hpp"
#include <blob_factory.hpp>
#include "models/model_loader.h"

namespace IE = InferenceEngine;

namespace ReferenceHelper {
    IE::Blob::Ptr CalcCpuReferenceSingleOutput(const std::string &modelPath, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input = true, const IE::PreProcessInfo* preprocInfo = nullptr);
    IE::Blob::Ptr CalcCpuReferenceSingleOutput(IE::CNNNetwork& network, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input = true, const IE::PreProcessInfo* preprocInfo = nullptr);
    IE::BlobMap CalcCpuReferenceMultipleOutput(const std::string& modelPath, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input = true, const IE::PreProcessInfo* preprocInfo = nullptr);
    IE::BlobMap CalcCpuReferenceMultipleOutput(IE::CNNNetwork& network, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input = true, const IE::PreProcessInfo* preprocInfo = nullptr);
}

namespace {
inline IE::BlobMap CalcCpuReferenceCommon(IE::CNNNetwork& network, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input, const IE::PreProcessInfo* preprocInfo) {
    IE::Core ie;
    if (forceU8Input) {
        IE::InputsDataMap inputInfo = network.getInputsInfo();
        for (auto& item : inputInfo) {
            auto inputData = item.second;
            inputData->setPrecision(IE::Precision::U8);
        }
    }

    IE::ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "CPU");
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();
    const auto blockDescNetwork = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getBlockingDesc();
    IE::Blob::Ptr correctInputBlob = nullptr;
    if (preprocInfo == nullptr && inputBlob->getTensorDesc().getBlockingDesc() != blockDescNetwork) {
        IE::TensorDesc correctTensorDesc(inputBlob->getTensorDesc().getPrecision(), inputBlob->getTensorDesc().getDims(), blockDescNetwork);
        correctInputBlob = vpux::toLayout(IE::as<IE::MemoryBlob>(inputBlob), correctTensorDesc.getLayout());
    } else {
        correctInputBlob = inputBlob;
    }

    const auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    if (preprocInfo != nullptr) {
        inferRequest.SetBlob(inputBlobName, correctInputBlob, *preprocInfo);
    } else {
        inferRequest.SetBlob(inputBlobName, correctInputBlob);
    }

    const auto outputInfo = executableNetwork.GetOutputsInfo();
    for (const auto& output : outputInfo) {
        const auto outputBlobName = output.first;
        auto output_blob = make_blob_with_precision(output.second->getTensorDesc());
        output_blob->allocate();
        inferRequest.SetBlob(outputBlobName, output_blob);
    }

    inferRequest.Infer();

    IE::BlobMap outputBlobs;
    for (const auto& output : outputInfo) {
        const auto outputBlobName = output.first;
        auto outputBlob = inferRequest.GetBlob(outputBlobName);
        outputBlobs[outputBlobName] = outputBlob;
    }

    return outputBlobs;
}
}

inline IE::Blob::Ptr ReferenceHelper::CalcCpuReferenceSingleOutput(const std::string& modelPath, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input, const IE::PreProcessInfo* preprocInfo) {
    std::cout << "Calculating reference on CPU (single output)..." << std::endl;

    IE::Core ie;
    const std::string modelFullPath = ModelLoader_Helper::getTestModelsPath() + modelPath + ".xml";
    auto network = ie.ReadNetwork(modelFullPath);

    IE::OutputsDataMap outputsInfo = network.getOutputsInfo();
    const size_t NUM_OUTPUTS = 1;
    if (outputsInfo.size() != NUM_OUTPUTS) {
        IE_THROW() << "Number of outputs isn't equal to 1";
    }

    return CalcCpuReferenceCommon(network, inputBlob, forceU8Input, preprocInfo).begin()->second;
}

inline IE::Blob::Ptr ReferenceHelper::CalcCpuReferenceSingleOutput(IE::CNNNetwork& network, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input, const IE::PreProcessInfo* preprocInfo) {
    std::cout << "Calculating reference on CPU (single output)..." << std::endl;

    IE::OutputsDataMap outputsInfo = network.getOutputsInfo();
    const size_t NUM_OUTPUTS = 1;
    if (outputsInfo.size() != NUM_OUTPUTS) {
        IE_THROW() << "Number of outputs isn't equal to 1";
    }

    return CalcCpuReferenceCommon(network, inputBlob, forceU8Input, preprocInfo).begin()->second;
}

inline IE::BlobMap ReferenceHelper::CalcCpuReferenceMultipleOutput(const std::string& modelPath, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input, const IE::PreProcessInfo* preprocInfo) {
    std::cout << "Calculating reference on CPU (multiple output)..." << std::endl;

    IE::Core ie;
    const std::string modelFullPath = ModelLoader_Helper::getTestModelsPath() + modelPath + ".xml";
    auto network = ie.ReadNetwork(modelFullPath);

    return CalcCpuReferenceCommon(network, inputBlob, forceU8Input, preprocInfo);
}

inline IE::BlobMap ReferenceHelper::CalcCpuReferenceMultipleOutput(IE::CNNNetwork& network, const IE::Blob::Ptr& inputBlob,
    const bool forceU8Input, const IE::PreProcessInfo* preprocInfo) {
    std::cout << "Calculating reference on CPU (multiple output)..." << std::endl;

    return CalcCpuReferenceCommon(network, inputBlob, forceU8Input, preprocInfo);
}
