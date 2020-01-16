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

#include <Inference.h>
#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <test_model_path.hpp>

#include "helper_ie_core.h"

//------------------------------------------------------------------------------
//      class HDDL2_Core_API_Tests Declaration
//------------------------------------------------------------------------------
class HDDL2_Core_API_Tests : public ::testing::Test,
                             public IE_Core_Helper {
public:
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest inferRequest;
};
