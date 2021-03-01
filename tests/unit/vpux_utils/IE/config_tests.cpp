//
// Copyright Intel Corporation.
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

#include "vpux/utils/IE/config.hpp"

#include <gtest/gtest.h>

using namespace vpux;

struct SimpleOption : OptionBase<SimpleOption, bool> {
    static StringRef key() {
        return "PUBLIC_OPT";
    }

    static SmallVector<StringRef> deprecatedKeys() {
        return {"DEPRECATED_OPT"};
    }
};

struct PrivateOption : OptionBase<PrivateOption, int64_t> {
    static StringRef key() {
        return "PRIVATE_OPT";
    }

    static int64_t defaultValue() {
        return 42;
    }

    static void validateValue(int64_t val) {
        VPUX_THROW_UNLESS(val >= 0, "Got negative value");
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }
};

class MLIR_ConfigTests : public testing::Test {
public:
    Config conf;

    void SetUp() override {
        testing::Test::SetUp();

        Logger::global().setLevel(LogLevel::Warning);

        conf.desc().addOption<SimpleOption>();
        conf.desc().addOption<PrivateOption>();
    }
};

TEST_F(MLIR_ConfigTests, GetSupported) {
    const auto publicOpts = conf.desc().getSupported(false);
    EXPECT_EQ(publicOpts, std::vector<std::string>({"PUBLIC_OPT"}));

    auto allOpts = conf.desc().getSupported(true);
    std::sort(allOpts.begin(), allOpts.end());
    EXPECT_EQ(allOpts, std::vector<std::string>({"PRIVATE_OPT", "PUBLIC_OPT"}));
}

TEST_F(MLIR_ConfigTests, UpdateAndValidate) {
    EXPECT_NO_THROW(conf.update({{"PUBLIC_OPT", "YES"}}));
    EXPECT_ANY_THROW(conf.update({{"PUBLIC_OPT", "1"}}));

    EXPECT_NO_THROW(conf.update({{"PRIVATE_OPT", "15"}}));
    EXPECT_ANY_THROW(conf.update({{"PRIVATE_OPT", "aaa"}}));
    EXPECT_ANY_THROW(conf.update({{"PRIVATE_OPT", "-1"}}));
}

TEST_F(MLIR_ConfigTests, UpdateAndHas) {
    EXPECT_FALSE(conf.has<SimpleOption>());
    EXPECT_FALSE(conf.has<PrivateOption>());

    conf.update({{"PUBLIC_OPT", "YES"}});
    EXPECT_TRUE(conf.has<SimpleOption>());

    conf.update({{"PRIVATE_OPT", "5"}}, OptionMode::RunTime);
    EXPECT_TRUE(conf.has<PrivateOption>());

}

TEST_F(MLIR_ConfigTests, Get) {
    conf.update({{"PUBLIC_OPT", "YES"}, {"PRIVATE_OPT", "5"}});

    EXPECT_TRUE(conf.get<SimpleOption>());
    EXPECT_EQ(conf.get<PrivateOption>(), 5);
}

TEST_F(MLIR_ConfigTests, GetNonExist) {
    EXPECT_ANY_THROW(conf.get<SimpleOption>());
}

TEST_F(MLIR_ConfigTests, GetDefault) {
    EXPECT_EQ(conf.get<PrivateOption>(), 42);
}

TEST_F(MLIR_ConfigTests, Deprecated) {
    EXPECT_FALSE(conf.has<SimpleOption>());

    EXPECT_NO_THROW(conf.update({{"DEPRECATED_OPT", "NO"}}));
    EXPECT_TRUE(conf.has<SimpleOption>());
    EXPECT_FALSE(conf.get<SimpleOption>());

    EXPECT_ANY_THROW(conf.update({{"DEPRECATED_OPT", "1"}}));
}

TEST_F(MLIR_ConfigTests, Unsupported) {
    EXPECT_ANY_THROW(conf.update({{"UNSUPPORTED_OPT", "1"}}));
}
