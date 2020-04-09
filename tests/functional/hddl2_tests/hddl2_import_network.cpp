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

#include <Inference.h>

#include <fstream>

#include "core_api.h"
#include "gtest/gtest.h"
#include "hddl2_helpers/helper_workload_context.h"
#include "helper_remote_context.h"
#include "models/precompiled_resnet.h"

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class HDDL2_ImportNetwork_Tests : public CoreAPI_Tests {
public:
    modelBlobInfo blobInfo = PrecompiledResNet_Helper::resnet50_dpu;
    void SetUp() override;
    InferenceEngine::ParamMap params;

private:
    WorkloadContext_Helper workloadContextHelper;
};

void HDDL2_ImportNetwork_Tests::SetUp() {
    WorkloadID workloadId = workloadContextHelper.getWorkloadId();
    params = Remote_Context_Helper::wrapWorkloadIdToMap(workloadId);
}

//------------------------------------------------------------------------------
// [Track number: S#28523]
TEST_F(HDDL2_ImportNetwork_Tests, DISABLED_CanFindPlugin) {
    ASSERT_NO_THROW(ie.ImportNetwork(blobInfo.graphPath, pluginName));
}

// [Track number: S#28523]
TEST_F(HDDL2_ImportNetwork_Tests, DISABLED_CanCreateExecutableNetwork) {
    std::map<std::string, std::string> config = {};

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName));
}

// [Track number: S#28523]
TEST_F(HDDL2_ImportNetwork_Tests, DISABLED_CanCreateExecutableNetworkWithConfig) {
    std::map<std::string, std::string> config = {};

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName, config));
}

// [Track number: S#28523]
TEST_F(HDDL2_ImportNetwork_Tests, DISABLED_CanCreateInferRequest) {
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(blobInfo.graphPath, pluginName));

    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());
}

// [Track number: S#28523]
TEST_F(HDDL2_ImportNetwork_Tests, DISABLED_CanCreateExecutableNetworkWithStream) {
    const std::map<std::string, std::string> config = {};

    std::filebuf blobFile;
    if (!blobFile.open(blobInfo.graphPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << blobInfo.graphPath;
    }
    std::istream tmp_stream(&blobFile);

    InferenceEngine::RemoteContext::Ptr remoteContextPtr = ie.CreateContext(pluginName, params);

    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(tmp_stream, remoteContextPtr, config));
    blobFile.close();
}
