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

#include "helper_calc_cpu_ref.h"

#include "vpux/utils/IE/blob.hpp"

#include <blob_factory.hpp>
#include "models/model_loader.h"

namespace {
IE::BlobMap CalcCpuReferenceCommon(IE::CNNNetwork& network, const IE::Blob::Ptr& input_blob,
    const IE::PreProcessInfo* preproc_info) {
    IE::Core ie;
    IE::InputsDataMap input_info = network.getInputsInfo();
    for (auto& item : input_info) {
        auto input_data = item.second;
        input_data->setPrecision(IE::Precision::U8);
    }

    IE::ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "CPU");
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    auto block_desc_network = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getBlockingDesc();
    IE::Blob::Ptr correct_input_blob = nullptr;
    if (preproc_info == nullptr && input_blob->getTensorDesc().getBlockingDesc() != block_desc_network) {
        IE::TensorDesc correct_tensor_desc(input_blob->getTensorDesc().getPrecision(), input_blob->getTensorDesc().getDims(), block_desc_network);
        correct_input_blob = vpux::toLayout(IE::as<IE::MemoryBlob>(input_blob), correct_tensor_desc.getLayout());
    } else {
        correct_input_blob = input_blob;
    }

    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    if (preproc_info != nullptr) {
        inferRequest.SetBlob(inputBlobName, correct_input_blob, *preproc_info);
    } else {
        inferRequest.SetBlob(inputBlobName, correct_input_blob);
    }

    IE::ConstOutputsDataMap output_info = executableNetwork.GetOutputsInfo();
    for (const auto& output : output_info) {
        const auto outputBlobName = output.first;
        auto output_blob = make_blob_with_precision(output.second->getTensorDesc());
        output_blob->allocate();
        inferRequest.SetBlob(outputBlobName, output_blob);
    }

    inferRequest.Infer();

    IE::BlobMap outputBlobs;
    for (const auto& output : output_info) {
        const auto outputBlobName = output.first;
        auto output_blob = inferRequest.GetBlob(outputBlobName);
        outputBlobs[outputBlobName] = output_blob;
    }

    return outputBlobs;
}
}

IE::Blob::Ptr ReferenceHelper::CalcCpuReferenceSingleOutput(const std::string& modelPath, const IE::Blob::Ptr& inputBlob,
    const IE::PreProcessInfo* preproc_info) {
    std::cout << "Calculating reference on CPU (single output)..." << std::endl;

    IE::Core ie;
    const std::string modelFullPath = ModelLoader_Helper::getTestModelsPath() + modelPath + ".xml";
    auto network = ie.ReadNetwork(modelFullPath);

    IE::OutputsDataMap outputs_info = network.getOutputsInfo();
    const size_t NUM_OUTPUTS = 1;
    if (outputs_info.size() != NUM_OUTPUTS) {
        IE_THROW() << "Number of outputs isn't equal to 1";
    }

    return CalcCpuReferenceCommon(network, inputBlob, preproc_info).begin()->second;
}

IE::BlobMap ReferenceHelper::CalcCpuReferenceMultipleOutput(const std::string& modelPath, const IE::Blob::Ptr& input_blob,
    const IE::PreProcessInfo* preproc_info) {
    std::cout << "Calculating reference on CPU (multiple output)..." << std::endl;

    IE::Core ie;
    const std::string modelFullPath = ModelLoader_Helper::getTestModelsPath() + modelPath + ".xml";
    auto network = ie.ReadNetwork(modelFullPath);

    return CalcCpuReferenceCommon(network, input_blob, preproc_info);
}
