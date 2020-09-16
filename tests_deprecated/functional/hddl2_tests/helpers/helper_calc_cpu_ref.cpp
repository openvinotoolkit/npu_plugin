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

#include <ie_utils.hpp>
#include <blob_factory.hpp>

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

    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    if (preproc_info != nullptr) {
        inferRequest.SetBlob(inputBlobName, input_blob, *preproc_info);
    } else {
        inferRequest.SetBlob(inputBlobName, input_blob);
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

IE::Blob::Ptr ReferenceHelper::CalcCpuReferenceSingleOutput(const std::string &model_path, const IE::Blob::Ptr& input_blob,
    const IE::PreProcessInfo* preproc_info) {
    std::cout << "Calculating reference on CPU (single output)..." << std::endl;

    IE::Core ie;
    auto network = ie.ReadNetwork(model_path);

    IE::OutputsDataMap outputs_info = network.getOutputsInfo();
    const size_t NUM_OUTPUTS = 1;
    if (outputs_info.size() != NUM_OUTPUTS) {
        THROW_IE_EXCEPTION << "Number of outputs isn't equal to 1";
    }

    return CalcCpuReferenceCommon(network, input_blob, preproc_info).begin()->second;
}

IE::BlobMap ReferenceHelper::CalcCpuReferenceMultipleOutput(const std::string& model_path, const IE::Blob::Ptr& input_blob,
    const IE::PreProcessInfo* preproc_info) {
    std::cout << "Calculating reference on CPU (multiple output)..." << std::endl;

    IE::Core ie;
    auto network = ie.ReadNetwork(model_path);

    return CalcCpuReferenceCommon(network, input_blob, preproc_info);
}
