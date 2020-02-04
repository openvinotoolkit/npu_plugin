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

#include <hddl2_helpers/helper_device_name.h>

#include <fstream>

#include "gtest/gtest.h"

//------------------------------------------------------------------------------
//      class HDDL2_Vpu_SMM_Driver_Tests Declaration
//------------------------------------------------------------------------------
class HDDL2_Vpu_SMM_Driver_Tests : public ::testing::Test {};

//------------------------------------------------------------------------------
//      class HDDL2_Vpu_SMM_Driver_Tests Initiations
//------------------------------------------------------------------------------
TEST_F(HDDL2_Vpu_SMM_Driver_Tests, getVpusmmDriver) {
    if (DeviceName::isEmulator()) {
        SKIP() << "SMM driver is not required for real device";
    }

    bool isVPUSMMDriverFound = false;
    std::ifstream modulesLoaded("/proc/modules");
    std::string line;
    while (std::getline(modulesLoaded, line)) {
        if (line.find("vpusmm_driver") != std::string::npos) {
            isVPUSMMDriverFound = true;
            std::cout << " [INFO] - Driver found: " << line << std::endl;
            break;
        }
    }
    ASSERT_TRUE(isVPUSMMDriverFound);
}
