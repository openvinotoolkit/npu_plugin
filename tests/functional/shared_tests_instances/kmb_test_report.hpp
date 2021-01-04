// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <array>
#include <map>
#include <gtest/gtest.h>

namespace LayerTestsUtils {

enum class KmbTestStage {
    RUN=0, COMPILED, IMPORTED, INFERRED, VALIDATED, SKIPPED_EXCEPTION, LAST_VALUE
};

class KmbTestReport {
private:
    std::map<std::string, std::array<int, static_cast<int>(KmbTestStage::LAST_VALUE)>> counters;
public:
    static const std::array<std::string, static_cast<int>(KmbTestStage::LAST_VALUE)> stages;
public:
    explicit KmbTestReport();
    void run(const testing::TestInfo* testInfo);
    void compiled(const testing::TestInfo* testInfo);
    void imported(const testing::TestInfo* testInfo);
    void inferred(const testing::TestInfo* testInfo);
    void validated(const testing::TestInfo* testInfo);
    void skipped(const testing::TestInfo* testInfo);

    const std::map<std::string, std::array<int, static_cast<int>(KmbTestStage::LAST_VALUE)>>& getCounters() const {
        return counters;
    }

    static KmbTestReport& getInstance() {
        static KmbTestReport instance;
        return instance;
    }
};

class KmbTestReportEnvironment : public testing::Environment {
public:
    ~KmbTestReportEnvironment() override {}

    void TearDown() override;
};

}  // namespace LayerTestsUtils
