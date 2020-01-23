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

#pragma once

#include <cpp/ie_cnn_net_reader.h>
#include <ie_api.h>
#include "test_model_path.hpp"

//------------------------------------------------------------------------------
//      PrecompiledResNet_Helper parameters
//------------------------------------------------------------------------------
struct modelBlobInfo {
    std::string graphName, graphPath, inputPath, outputPath;
};

//------------------------------------------------------------------------------
//      PrecompiledResNet_Helper Declaration
//------------------------------------------------------------------------------
namespace PrecompiledResNet_Helper {
    static const modelBlobInfo resnet =
            {
                    .graphName = "resnet",
                    .graphPath = ModelsPath() + "/KMB_models/BLOBS/resnet/resnet.blob",
                    .inputPath = ModelsPath() + "/KMB_models/BLOBS/resnet/input.dat",
                    .outputPath = ModelsPath() + "/KMB_models/BLOBS/resnet/output.dat",
            };
};
