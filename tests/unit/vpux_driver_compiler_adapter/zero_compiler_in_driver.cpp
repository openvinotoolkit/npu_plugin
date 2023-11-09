//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "zero_compiler_in_driver.h"

namespace vpux {
namespace driverCompilerAdapter {

namespace ie = InferenceEngine;

class ZeroCompilerAdapterTests : public ::testing::Test {
public:
    ie::InputsDataMap createSingleInputDataMap(const std::string inputName = "inputName1",
                                               const ie::Precision precision = ie::Precision::U8,
                                               const ie::Layout layout = ie::Layout::NCHW);
    ie::OutputsDataMap createSingleOutputDataMap(const std::string inputName = "outputName1",
                                                 const ie::Precision precision = ie::Precision::FP32,
                                                 const ie::Layout layout = ie::Layout::NC);
    ie::InputsDataMap createTwoInputDataMap(const std::string inputName1 = "inputName1",
                                            const std::string inputName2 = "inputName2",
                                            const ie::Precision precision1 = ie::Precision::U8,
                                            const ie::Precision precision2 = ie::Precision::U8,
                                            const ie::Layout layout1 = ie::Layout::NCHW,
                                            const ie::Layout layout2 = ie::Layout::NCHW);
    ie::OutputsDataMap createTwoOutputDataMap(const std::string outputName1 = "outputName1",
                                              const std::string outputName2 = "outputName2",
                                              const ie::Precision precision1 = ie::Precision::U8,
                                              const ie::Precision precision2 = ie::Precision::U8,
                                              const ie::Layout layout1 = ie::Layout::NCHW,
                                              const ie::Layout layout2 = ie::Layout::NCHW);
};

ie::InputsDataMap ZeroCompilerAdapterTests::createSingleInputDataMap(const std::string name,
                                                                     const ie::Precision precision,
                                                                     const ie::Layout layout) {
    const ie::InputInfo::Ptr inputInfoPtr = std::make_shared<ie::InputInfo>();
    const ie::DataPtr inputDataPtr = std::make_shared<ie::Data>(name, precision, layout);
    inputInfoPtr->setInputData(inputDataPtr);

    const ie::InputsDataMap inputsInfo{{name, inputInfoPtr}};
    return inputsInfo;
}

ie::OutputsDataMap ZeroCompilerAdapterTests::createSingleOutputDataMap(const std::string name,
                                                                       const ie::Precision precision,
                                                                       const ie::Layout layout) {
    const ie::DataPtr outputDataPtr = std::make_shared<ie::Data>(name, precision, layout);
    const ie::OutputsDataMap outputsInfo{{name, outputDataPtr}};
    return outputsInfo;
}

ie::InputsDataMap ZeroCompilerAdapterTests::createTwoInputDataMap(const std::string inputName1,
                                                                  const std::string inputName2,
                                                                  const ie::Precision precision1,
                                                                  const ie::Precision precision2,
                                                                  const ie::Layout layout1, const ie::Layout layout2) {
    const ie::DataPtr inputDataPtr = std::make_shared<ie::Data>(inputName1, precision1, layout1);
    const ie::DataPtr inputDataPtr2 = std::make_shared<ie::Data>(inputName2, precision2, layout2);

    const ie::InputInfo::Ptr inputInfoPtr = std::make_shared<ie::InputInfo>();
    const ie::InputInfo::Ptr inputInfoPtr2 = std::make_shared<ie::InputInfo>();

    inputInfoPtr->setInputData(inputDataPtr);
    inputInfoPtr2->setInputData(inputDataPtr2);
    const ie::InputsDataMap inputsInfo{{inputName1, inputInfoPtr}, {inputName2, inputInfoPtr2}};
    return inputsInfo;
}

ie::OutputsDataMap ZeroCompilerAdapterTests::createTwoOutputDataMap(
        const std::string outputName1, const std::string outputName2, const ie::Precision precision1,
        const ie::Precision precision2, const ie::Layout layout1, const ie::Layout layout2) {
    const ie::DataPtr outputDataPtr = std::make_shared<ie::Data>(outputName1, precision1, layout1);
    const ie::DataPtr outputDataPtr2 = std::make_shared<ie::Data>(outputName2, precision2, layout2);

    const ie::OutputsDataMap outputsInfo{{outputName1, outputDataPtr}, {outputName2, outputDataPtr2}};
    return outputsInfo;
}

TEST_F(ZeroCompilerAdapterTests, SingleIONetwork_ipU8opFP32) {
    const auto inputsInfo = createSingleInputDataMap();
    const auto outputsInfo = createSingleOutputDataMap();
    const std::string ioInfo =
            LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::serializeIOInfo(inputsInfo, outputsInfo);

    const std::string expectedStr = "--inputs_precisions=\"inputName1:U8\" --inputs_layouts=\"inputName1:NCHW\""
                                    " --outputs_precisions=\"outputName1:FP32\" --outputs_layouts=\"outputName1:NC\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

TEST_F(ZeroCompilerAdapterTests, TwoIONetwork_ipU8U8opFP32FP32) {
    const std::string inputName1 = "inputName1";
    const std::string inputName2 = "inputName2";
    const ie::Precision inputsPrecision = ie::Precision::U8;
    const ie::Layout inputsLayout = ie::Layout::NCHW;
    const auto inputsInfo =
            createTwoInputDataMap(inputName1, inputName2, inputsPrecision, inputsPrecision, inputsLayout, inputsLayout);

    const std::string outputName1 = "outputName1";
    const std::string outputName2 = "outputName2";
    const ie::Precision outputsPrecision = ie::Precision::FP32;
    const ie::Layout outputsLayout = ie::Layout::NC;

    const auto outputsInfo = createTwoOutputDataMap(outputName1, outputName2, outputsPrecision, outputsPrecision,
                                                    outputsLayout, outputsLayout);

    const std::string ioInfo =
            LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::serializeIOInfo(inputsInfo, outputsInfo);

    const std::string expectedStr =
            "--inputs_precisions=\"inputName1:U8 inputName2:U8\" --inputs_layouts=\"inputName1:NCHW inputName2:NCHW\""
            " --outputs_precisions=\"outputName1:FP32 outputName2:FP32\" --outputs_layouts=\"outputName1:NC"
            " outputName2:NC\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

TEST_F(ZeroCompilerAdapterTests, OneInputTwoOuputsNetwork_ipU8opFP16FP32) {
    const auto inputsInfo = createSingleInputDataMap();
    const std::string outputName = "outputName1";
    const std::string outputName2 = "outputName2";
    const ie::Precision outputsPrecision1 = ie::Precision::FP32;
    const ie::Layout outputsLayout1 = ie::Layout::NC;
    const ie::Precision outputsPrecision2 = ie::Precision::FP16;
    const ie::Layout outputsLayout2 = ie::Layout::NCHW;

    const auto outputsInfo = createTwoOutputDataMap(outputName, outputName2, outputsPrecision1, outputsPrecision2,
                                                    outputsLayout1, outputsLayout2);

    const std::string ioInfo =
            LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::serializeIOInfo(inputsInfo, outputsInfo);

    const std::string expectedStr = "--inputs_precisions=\"inputName1:U8\" --inputs_layouts=\"inputName1:NCHW\""
                                    " --outputs_precisions=\"outputName1:FP32 outputName2:FP16\""
                                    " --outputs_layouts=\"outputName1:NC outputName2:NCHW\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

}  // namespace driverCompilerAdapter
}  // namespace vpux
