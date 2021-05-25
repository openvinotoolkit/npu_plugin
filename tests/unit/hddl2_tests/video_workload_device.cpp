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

#include <gtest/gtest.h>
#include <skip_conditions.h>

#include "video_workload_device.h"
#include "vpux/vpux_plugin_params.hpp"
#include "helper_remote_context.h"
#include "helper_remote_allocator.h"

using namespace vpux::hddl2;
using namespace vpu;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
class VideoWorkloadDevice_UnitTests : public ::testing::Test {
public:
    std::shared_ptr<VideoWorkloadDevice> device = nullptr;

protected:
    RemoteContext_Helper::Ptr _remoteContextHelperPtr = nullptr;
    void SetUp() override;
};
void VideoWorkloadDevice_UnitTests::SetUp() {
    if (canWorkWithDevice()) {
        _remoteContextHelperPtr = std::make_shared<RemoteContext_Helper>();
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
    const auto remoteMemoryFD =
        _remoteMemoryHelper.allocateRemoteMemory(_remoteContextHelperPtr->getWorkloadId(), 1);

    IE::ParamMap correctParamMap = {{IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD}};
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
