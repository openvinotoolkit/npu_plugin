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

#include "helper_calc_cpu_ref.h"

#include <vpu/utils/ie_helpers.hpp>
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
        correct_input_blob = make_blob_with_precision(correct_tensor_desc);
        correct_input_blob->allocate();
        vpu::copyBlob(input_blob, correct_input_blob);
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
