// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_test_report.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

namespace LayerTestsUtils {

const std::array<std::string, static_cast<int>(KmbTestStage::LAST_VALUE)> KmbTestReport::stages = 
    {"RUN", "COMPILED", "IMPORTED", "INFERRED", "VALIDATED", "SKIPPED_EXCEPTION"};

std::string testName(const testing::TestInfo* testInfo) {
    const std::string name(testInfo->test_case_name());
    auto npos = name.find("/");

    return (npos != std::string::npos) ? name.substr(npos + 1) : name;
}

KmbTestReport::KmbTestReport() {
}

void KmbTestReport::run(const testing::TestInfo* testInfo) {
    std::cout << "TestReportProgress: " << testName(testInfo) << " run" << std::endl;
    ++counters[testName(testInfo)][static_cast<int>(KmbTestStage::RUN)];
}

void KmbTestReport::compiled(const testing::TestInfo* testInfo) {
    std::cout << "TestReportProgress: "<< testName(testInfo) << " compiled" << std::endl;
    ++counters[testName(testInfo)][static_cast<int>(KmbTestStage::COMPILED)];
}

void KmbTestReport::imported(const testing::TestInfo* testInfo) {
    std::cout << "TestReportProgress: "<< testName(testInfo) << " imported" << std::endl;
    ++counters[testName(testInfo)][static_cast<int>(KmbTestStage::IMPORTED)];
}

void KmbTestReport::inferred(const testing::TestInfo* testInfo) {
    std::cout << "TestReportProgress: "<< testName(testInfo) << " inferred" << std::endl;
    ++counters[testName(testInfo)][static_cast<int>(KmbTestStage::INFERRED)];
}

void KmbTestReport::validated(const testing::TestInfo* testInfo) {
    std::cout << "TestReportProgress: "<< testName(testInfo) << " validated" << std::endl;
    ++counters[testName(testInfo)][static_cast<int>(KmbTestStage::VALIDATED)];
}

void KmbTestReport::skipped(const testing::TestInfo* testInfo) {
    std::cout << "TestReportProgress: "<< testName(testInfo) << " skipped due to exception" << std::endl;
    ++counters[testName(testInfo)][static_cast<int>(KmbTestStage::SKIPPED_EXCEPTION)];
}

void KmbTestReportEnvironment::TearDown() {
    std::cout << "TestReportResult: " << std::endl;   
    const auto& counters = KmbTestReport::getInstance().getCounters();

    std::array<int, static_cast<int>(KmbTestStage::LAST_VALUE)> totals = {};
    for (auto const& cit : counters)
    {
        std::cout << cit.first << ": ";
        for (int it = static_cast<int>(KmbTestStage::RUN); it < static_cast<int>(KmbTestStage::LAST_VALUE); ++it) {
            totals [it] += cit.second[it];
            std::cout << KmbTestReport::stages[it] << " - " << cit.second[it] << "; ";
        }
        std::cout << std::endl;
    }    

    std::cout << "KmbTotalCases: ";
    for (int it = static_cast<int>(KmbTestStage::RUN); it < static_cast<int>(KmbTestStage::LAST_VALUE); ++it) {
        std::cout << KmbTestReport::stages[it] << " - " << totals[it] << "; ";
    }
    std::cout << std::endl;

};

}  // namespace LayerTestsUtils
