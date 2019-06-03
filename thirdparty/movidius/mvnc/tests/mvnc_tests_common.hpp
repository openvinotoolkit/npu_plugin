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

#if (defined(_WIN32) || defined(_WIN64))
#include "windows.h"
#endif

#include <thread>
#include <gtest/gtest.h>
#include <fstream>

#include "XLink.h"
#include "mvnc.h"
#include "mvnc_ext.h"
#include "mvLog.h"
#include "usb_boot.h"
#include "ncPrivateTypes.h"

#define ASSERT_NO_ERROR(call)   ASSERT_EQ(call, 0)
#define ASSERT_ERROR(call)      ASSERT_TRUE(call)

#define MYRIAD_X_NAME_STR "ma2480"
#define MYRIAD_2_NAME_STR "ma2450"

#if (defined(_WIN32) || defined(_WIN64))
#define PCIE_NAME_STR     "mxlink"
#else
#define PCIE_NAME_STR     "mxlk"
#endif

const int MAX_DEVICES = 32;
const int MAX_DEV_NAME = 20;

#ifndef MAX_PATH
const int MAX_PATH = 255;
#endif

// Without this initialization find device on windows could not work
#if (defined(_WIN32) || defined(_WIN64) )
extern "C" void initialize_usb_boot();
#else
#define initialize_usb_boot()
#endif

class MvncTestCommon : public ::testing::Test {
public:
#if !(defined(_WIN32) || defined(_WIN64))
    // On linux we should use custom path to firmware due to another searching mechanism for library
    const char  _firmwarePath[MAX_PATH] = "./lib/";
#else
    // Search firmware in default folder 
    const char* _firmwarePath = nullptr;
#endif
    const int watchdogInterval = 1000;
protected:
    mvLog_t     ncLogLevel = MVLOG_INFO;
    virtual ~MvncTestCommon() {
    }
    void SetUp() override {
        initialize_usb_boot();
        ASSERT_NO_ERROR(setLogLevel(ncLogLevel));
    }

  void TearDown() override {
      ncDeviceResetAll();
  }

public:
    int setLogLevel(const mvLog_t logLevel) {
        ncStatus_t status = ncGlobalSetOption(NC_RW_LOG_LEVEL, &logLevel,
                                              sizeof(logLevel));
        if (status != NC_OK) {
            fprintf(stderr,
                    "WARNING: failed to set log level: %d with error: %d\n",
                    ncLogLevel, status);
            return -1;
        }
        ncLogLevel = logLevel;
        return 0;
    }

    /*
     * @brief Get amount of all currently connected Myriad devices
     */
    int getAmountOfDevices() {
        int amount = 0;
        deviceDesc_t deviceDesc = {};
        for (; amount < MAX_DEVICES; ++amount) {
            if (XLinkGetDeviceName(amount, &deviceDesc, 0, X_LINK_ANY_PROTOCOL))
                break;
        }
        return amount;
    }

    /**
     * @brief Boot any not booted device
     */
    virtual void bootOneDevice() {
        ASSERT_NO_ERROR(ncDeviceLoadFirmware(ANY_PLATFORM, _firmwarePath));
    }

    /**
     * @brief Get list of all currently connected Myriad devices
     */
    static std::vector<std::string> getDevicesList() {
        std::vector < std::string > devName;
        deviceDesc_t tempDeviceDesc = {};
        for (int i = 0; i < MAX_DEVICES; ++i) {
            if (XLinkGetDeviceName(i, &tempDeviceDesc, 0, X_LINK_ANY_PROTOCOL))
                break;
            devName.emplace_back(tempDeviceDesc.name);
        }
        return devName;
    }

    static bool isMyriadXDevice(const std::string &deviceName) {
        return (deviceName.find(MYRIAD_X_NAME_STR) != std::string::npos);
    }

    static bool isMyriad2Device(const std::string &deviceName) {
        return (deviceName.find(MYRIAD_2_NAME_STR) != std::string::npos);
    }

    static bool isMyriadPCIeDevice(std::string& deviceName) {
        return deviceName.find(std::string(PCIE_NAME_STR)) != std::string::npos;
    }

    static bool isMyriadBootedDevice(const std::string &deviceName) {
        return (!isMyriad2Device(deviceName) && !isMyriadXDevice(deviceName));
    }

    static long getAmountOfMyriadXDevices() {
        auto devName = getDevicesList();
        return count_if(devName.begin(), devName.end(), isMyriadXDevice);
    }

    static long getAmountOfMyriad2Devices() {
        auto devName = getDevicesList();
        return count_if(devName.begin(), devName.end(), isMyriad2Device);
    }

    static long getAmountOfBootedDevices() {
        auto devName = getDevicesList();
        return count_if(devName.begin(), devName.end(), isMyriadBootedDevice);
    }

    static long getAmountOfNotBootedDevices() {
        return (getAmountOfMyriadXDevices() + getAmountOfMyriad2Devices());
    }

    static int readBINFile(const std::string& fileName, std::vector<char>& buf) {
        std::ifstream file(fileName, std::ios_base::binary | std::ios_base::ate);
        if (!file.is_open()) {
            std::cout << "Can't open file!" << std::endl;
            return -1;
        }
        buf.resize(static_cast<unsigned int>(file.tellg()));
        file.seekg(0);
        file.read(buf.data(), buf.size());
        return 0;
    }
};
