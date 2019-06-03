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
//              Open Device TESTS                    //

class MvncOpenUSBDevice : public MvncTestCommon {
public:
    int available_devices = 0;
protected:
    virtual ~MvncOpenUSBDevice() {}
    void SetUp() override {
        MvncTestCommon::SetUp();
        available_devices = getAmountOfNotBootedDevices();
        ASSERT_TRUE(available_devices > 0);
    }
};

/**
* @brief Open any device with custom firmware path as ncDeviceOpen argument
*/
TEST_F(MvncOpenUSBDevice, WithCustomFirmware) {
    ncDeviceHandle_t *deviceHandle = nullptr;

    // Use custom firmware dir path as parameter for ncDeviceOpen
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));

}

/**
* @brief Open all available devices and close them
*/
TEST_F(MvncOpenUSBDevice, AllAvailableDevices) {
    ncDeviceHandle_t * deviceHandle[MAX_DEVICES] = {nullptr};

    for (int index = 0; index < available_devices; ++index) {
        ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle[index], ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));
    }
    for (int index = 0; index < available_devices; ++index) {
        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle[index]));
    }
}

/**
* @brief Open all available devices in parallel threads and close them
*/
TEST_F(MvncOpenUSBDevice, AllAvailableMultiThreads) {
    std::thread requests[MAX_DEVICES];
    ncDeviceHandle_t * deviceHandle[MAX_DEVICES] = {nullptr};
    ncStatus_t rc[MAX_DEVICES];

    for (int i = 0; i < available_devices; ++i) {
        requests[i] = std::thread([i, &rc, &deviceHandle, this]() {
            rc[i] = ncDeviceOpen(&deviceHandle[i], ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath);
        });
    }

    for (int i = 0; i < available_devices; ++i) {
        requests[i].join();
        ASSERT_NO_ERROR(rc[i]);
    }

    for (int i = 0; i < available_devices; ++i) {
        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle[i]));
    }
}

/**
* @brief Open any device with invalid firmware path
*/
TEST_F(MvncOpenUSBDevice, WithInvalidFirmwarePath) {
    const char invalidPath[MAX_PATH] = "./InvalidPath/";

    // Use custom firmware dir path as parameter for ncDeviceOpen
    ncDeviceHandle_t *deviceHandle = nullptr;
    ASSERT_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_USB, watchdogInterval, invalidPath));

    ASSERT_EQ(deviceHandle, nullptr);
}

//  ***********************************************  //
//        Specific device (open and etc) TESTS       //

class MvncDevicePlatform : public MvncOpenUSBDevice {
public:
    long available_myriadX = 0;
    long available_myriad2 = 0;

protected:
    ~MvncDevicePlatform() override = default;
    void SetUp() override {
        MvncOpenUSBDevice::SetUp();
        available_myriadX = getAmountOfMyriadXDevices();
        available_myriad2 = getAmountOfMyriad2Devices();
        ASSERT_TRUE(available_myriadX > 0);
        ASSERT_TRUE(available_myriad2 > 0);
    }
};

/**
* @brief Open Myriad2 device and close it
*/
TEST_F(MvncDevicePlatform, OpenAndCloseMyriad2) {
    ncDeviceHandle_t *deviceHandle = nullptr;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, MYRIAD_2, NC_USB, watchdogInterval, _firmwarePath));

    char deviceName[MAX_DEV_NAME];
    unsigned int size = MAX_DEV_NAME;
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME, deviceName, &size));

    EXPECT_TRUE(strstr(deviceName, MYRIAD_2_NAME_STR) != nullptr);

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));

}

/**
* @brief Open MyriadX device and close it
*/
TEST_F(MvncDevicePlatform, OpenAndCloseMyriadX) {
    ncDeviceHandle_t *deviceHandle = nullptr;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, MYRIAD_X, NC_USB, watchdogInterval, _firmwarePath));

    char deviceName[MAX_DEV_NAME];
    unsigned int size = MAX_DEV_NAME;
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME, deviceName, &size));

    EXPECT_TRUE(strstr(deviceName, MYRIAD_X_NAME_STR) != nullptr);

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}
