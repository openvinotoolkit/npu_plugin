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

#include "mvnc.h"
#include "mvnc_tests_common.hpp"

//  ***********************************************  //
//               PCIE TESTS                          //

class MvncPCIEOpenDevice : public MvncTestCommon {
protected:
    virtual ~MvncPCIEOpenDevice() {}
};

TEST_F(MvncPCIEOpenDevice, OpenAndClose) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    std::string actDeviceName;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_PCIE, watchdogInterval, _firmwarePath));

    actDeviceName = deviceHandle->private_data->dev_addr;
    ASSERT_TRUE(isMyriadPCIeDevice(actDeviceName));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

/**
 * @brief Open device twice one run after another. It should check, that link to device closed correctly
 * @note Mostly this test important for PCIE and connect to booted option, as in that cases XLinkReset have another behavior
 */
TEST_F(MvncPCIEOpenDevice, OpenDeviceWithOneXLinkInitializion) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    std::string actDeviceName;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_PCIE, watchdogInterval, _firmwarePath));

    actDeviceName = deviceHandle->private_data->dev_addr;
    ASSERT_TRUE(isMyriadPCIeDevice(actDeviceName));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));

    // Second open
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_PCIE, watchdogInterval, _firmwarePath));

    actDeviceName = deviceHandle->private_data->dev_addr;
    ASSERT_TRUE(isMyriadPCIeDevice(actDeviceName));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

/**
 * @brief Try to open PCIe device twice. Second open should return error
 */
TEST_F(MvncPCIEOpenDevice, OpenSameDeviceTwice) {
    // PCIe device would be determined as booted
    //#-18685
    // ASSERT_EQ(getAmountOfBootedDevices(), 1);

    ncDeviceHandle_t *deviceHandle1 = nullptr;
    ncDeviceHandle_t *deviceHandle2 = nullptr;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle1, ANY_PLATFORM, NC_PCIE, watchdogInterval, _firmwarePath));

    // Till we don't have multiple device support, this function would try to open same device
    ASSERT_ERROR(ncDeviceOpen(&deviceHandle2, ANY_PLATFORM, NC_PCIE, watchdogInterval, _firmwarePath));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle1));
}
