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

#include <blob_factory.hpp>

IE::Blob::Ptr ReferenceHelper::CalcCpuReference(IE::CNNNetwork& network, const IE::Blob::Ptr& input_blob) {
    std::cout << "Calculating reference on CPU..." << std::endl;
    IE::Core ie;
    IE::ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "CPU");
    IE::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    auto inputBlobName = executableNetwork.GetInputsInfo().begin()->first;
    inferRequest.SetBlob(inputBlobName, input_blob);

    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    IE::Blob::Ptr output_blob;
    output_blob = make_blob_with_precision(executableNetwork.GetOutputsInfo().begin()->second->getTensorDesc());
    output_blob->allocate();
    inferRequest.SetBlob(outputBlobName, output_blob);

    inferRequest.Infer();
    output_blob = inferRequest.GetBlob(outputBlobName);

    return output_blob;
}
