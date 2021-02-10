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

#include <gtest/gtest.h>
#include <skip_conditions.h>

#include "video_workload_device.h"
#include "hddl2_params.hpp"
#include "helper_remote_context.h"
#include "helper_remote_allocator.h"

using namespace vpux::HDDL2;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class VideoWorkloadDevice_UnitTests : public ::testing::Test {
public:
    std::shared_ptr<VideoWorkloadDevice> device = nullptr;

protected:
    vpu::HDDL2Plugin::RemoteContext_Helper::Ptr _remoteContextHelperPtr = nullptr;
    void SetUp() override;
};
void VideoWorkloadDevice_UnitTests::SetUp() {
    if (canWorkWithDevice()) {
        _remoteContextHelperPtr = std::make_shared<vpu::HDDL2Plugin::RemoteContext_Helper>();
        const IE::ParamMap _deviceParams = _remoteContextHelperPtr->wrapWorkloadIdToMap( _remoteContextHelperPtr->getWorkloadId());
        device = std::make_shared<VideoWorkloadDevice>(_deviceParams);
    }
}


//------------------------------------------------------------------------------
/** video workload allocator only suitable for memory wrapping. For other case it useless */
TEST_F(VideoWorkloadDevice_UnitTests, constructor_EmptyAllocatorParams) {
    SKIP_IF_NO_DEVICE();

    IE::ParamMap emptyParamMap = {{}};
    std::shared_ptr<vpux::Allocator> allocator = nullptr;

    ASSERT_ANY_THROW(allocator = device->getAllocator(emptyParamMap));
}

TEST_F(VideoWorkloadDevice_UnitTests, constructor_CorrectAllocatorParams) {
    SKIP_IF_NO_DEVICE();

    RemoteMemory_Helper _remoteMemoryHelper;
    const HddlUnite::RemoteMemory::Ptr remoteMemory =
        _remoteMemoryHelper.allocateRemoteMemory(_remoteContextHelperPtr->getWorkloadId(), 1);

    IE::ParamMap correctParamMap = {{IE::HDDL2_PARAM_KEY(REMOTE_MEMORY), remoteMemory}};
    std::shared_ptr<vpux::Allocator> allocator = nullptr;


    ASSERT_NO_THROW(allocator = device->getAllocator(correctParamMap));
    ASSERT_NE(allocator, nullptr);
}

TEST_F(VideoWorkloadDevice_UnitTests, constructor_IncorrectAllocatorParams) {
    SKIP_IF_NO_DEVICE();

    IE::ParamMap incorrectParamMap = {{"VPUX_INCORRECT_PARAM_NAME", 66}};
    std::shared_ptr<vpux::Allocator> allocator = nullptr;


    ASSERT_ANY_THROW(allocator = device->getAllocator(incorrectParamMap));
}
