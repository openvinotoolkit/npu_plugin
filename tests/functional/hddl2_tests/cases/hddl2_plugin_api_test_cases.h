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

#pragma once

#include "gtest/gtest.h"
#include <string>
#include <ie_core.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include "test_model_path.hpp"

//------------------------------------------------------------------------------
//      class HDDL2_Plugin_API
//------------------------------------------------------------------------------
class HDDL2_Plugin_API : public ::testing::Test {
public:
    std::string                     device_name = "HDDL2";
    InferenceEngine::Core           ie;
    InferenceEngine::CNNNetwork     network;
    InferenceEngine::ExecutableNetwork  executableNetwork;
    InferenceEngine::InferRequest       inferRequest;

    void LoadModel();

private:
    std::string                     _modelName = "googlenet/bvlc_googlenet_fp16";
    InferenceEngine::CNNNetReader   _netReader;
};

struct modelBlobsInfo {
    std::string _graphPath, _inputPath, _outputPath;
};

const static std::vector<modelBlobsInfo> pathToPreCompiledGraph = {
        {
                ._graphPath = "/KMB_models/BLOBS/resnet/resnet.blob",
                ._inputPath = "/KMB_models/BLOBS/resnet/input.dat",
                ._outputPath = "/KMB_models/BLOBS/resnet/output.dat",
        },
};

class InferWithPath: public HDDL2_Plugin_API,
                     public testing::WithParamInterface<modelBlobsInfo> {
};

