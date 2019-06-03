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

#include "ncCommPrivate.h"
#include "mvnc_tests_common.hpp"
#include <fstream>

class MvncUtilsTest : public ::testing::Test {
public:
    void TearDown() override {
        std::remove(mvcmdExpectedPath.c_str());
    }

protected:
    std::string mvcmdExpectedPath = "";
    // FIXME: seems it is not going to work on Windows
    const std::string tmpDir = "/tmp";
};

TEST_F(MvncUtilsTest, CanGetSpecialFWIfUniversalIsNotPresent) {
    mvcmdExpectedPath = tmpDir + "/MvNCAPI-ma2480.mvcmd";

    std::ofstream mvcmd;
    mvcmd.open(mvcmdExpectedPath, std::ios::out);

    char mvcmdFilePath[MAX_PATH] = "";
    strcpy(mvcmdFilePath, tmpDir.c_str());

    const char *dummyDevAddr2480 = "0-ma2480";

    ASSERT_EQ(NC_OK, getFirmwarePath(mvcmdFilePath, dummyDevAddr2480));
    ASSERT_STRCASEEQ(mvcmdExpectedPath.c_str(), mvcmdFilePath);
}

TEST_F(MvncUtilsTest, CanGetUniversalFWIfItExists) {
    mvcmdExpectedPath = tmpDir + "/MvNCAPI-ma2x8x.mvcmd";

    std::ofstream mvcmd;
    mvcmd.open(mvcmdExpectedPath, std::ios::out);

    char mvcmdFilePath[MAX_PATH] = "";
    strcpy(mvcmdFilePath, tmpDir.c_str());

    const char *dummyDevAddr2480 = "0-ma2480";

    ASSERT_EQ(NC_OK, getFirmwarePath(mvcmdFilePath, dummyDevAddr2480));
    ASSERT_STRCASEEQ(mvcmdExpectedPath.c_str(), mvcmdFilePath);
}
