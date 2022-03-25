//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include <gtest/gtest.h>
#include "zero_compiler_in_driver.h"

namespace vpux {
namespace driverCompilerAdapter {

namespace IE = InferenceEngine;

class ZeroCompilerAdapterTests: public ::testing::Test {
    public:
        IE::InputsDataMap createSingleInputDataMap(const std::string inputName = "inputName1", const IE::Precision precision = IE::Precision::U8, const IE::Layout layout = IE::Layout::NCHW);
        IE::OutputsDataMap createSingleOutputDataMap(const std::string inputName = "outputName1", const IE::Precision precision = IE::Precision::FP32, const IE::Layout layout = IE::Layout::NC);

        IE::InputsDataMap createTwoInputDataMap(const std::string inputName1 = "inputName1", const std::string inputName2 = "inputName2",
                                          const IE::Precision precision1 = IE::Precision::U8, const IE::Precision precision2 = IE::Precision::U8,
                                          const IE::Layout layout1 = IE::Layout::NCHW, const IE::Layout layout2 = IE::Layout::NCHW);
        IE::OutputsDataMap createTwoOutputDataMap(const std::string outputName1 = "outputName1", const std::string outputName2 = "outputName2",
                                    const IE::Precision precision1 = IE::Precision::U8, const IE::Precision precision2 = IE::Precision::U8,
                                    const IE::Layout layout1 = IE::Layout::NCHW, const IE::Layout layout2 = IE::Layout::NCHW);
};


IE::InputsDataMap ZeroCompilerAdapterTests::createSingleInputDataMap(const std::string name, const IE::Precision precision, const IE::Layout layout) {
    const IE::InputInfo::Ptr inputInfoPtr = std::make_shared<IE::InputInfo>();
    const IE::DataPtr inputDataPtr = std::make_shared<IE::Data>(name, IE::Precision::U8);
    inputInfoPtr->setInputData(inputDataPtr);

    const IE::InputsDataMap inputsInfo {{name, inputInfoPtr}};
    return inputsInfo;
}

IE::OutputsDataMap ZeroCompilerAdapterTests::createSingleOutputDataMap(const std::string name, const IE::Precision precision, const IE::Layout layout) {
    const IE::DataPtr outputDataPtr = std::make_shared<IE::Data>(name, precision, layout);
    const IE::OutputsDataMap outputsInfo {{name, outputDataPtr}};
    return outputsInfo;
}

IE::InputsDataMap ZeroCompilerAdapterTests::createTwoInputDataMap(const std::string inputName1, const std::string inputName2,
                                                                    const IE::Precision precision1, const IE::Precision precision2,
                                                                    const IE::Layout layout1, const IE::Layout layout2) {
    const IE::DataPtr inputDataPtr = std::make_shared<IE::Data>(inputName1, precision1, layout1);
    const IE::DataPtr inputDataPtr2 = std::make_shared<IE::Data>(inputName2, precision2, layout2);

    const IE::InputInfo::Ptr inputInfoPtr = std::make_shared<IE::InputInfo>();
    const IE::InputInfo::Ptr inputInfoPtr2 = std::make_shared<IE::InputInfo>();

    inputInfoPtr->setInputData(inputDataPtr);
    inputInfoPtr2->setInputData(inputDataPtr2);
    const IE::InputsDataMap inputsInfo {{inputName1, inputInfoPtr}, {inputName2, inputInfoPtr2}};
    return inputsInfo;
}

IE::OutputsDataMap ZeroCompilerAdapterTests::createTwoOutputDataMap(const std::string outputName1, const std::string outputName2,
                                                                    const IE::Precision precision1, const IE::Precision precision2,
                                                                    const IE::Layout layout1, const IE::Layout layout2) {
    const IE::DataPtr outputDataPtr = std::make_shared<IE::Data>(outputName1, precision1, layout1);
    const IE::DataPtr outputDataPtr2 = std::make_shared<IE::Data>(outputName2, precision2, layout2);

    const IE::OutputsDataMap outputsInfo {{outputName1, outputDataPtr}, {outputName2, outputDataPtr2}};
    return outputsInfo;
}

TEST_F(ZeroCompilerAdapterTests, SingleIONetwork_ipU8opFP32) {
    const auto inputsInfo = createSingleInputDataMap();
    const auto outputsInfo = createSingleOutputDataMap();
    const std::string ioInfo = LevelZeroCompilerInDriver::serializeIOInfo(inputsInfo, outputsInfo);

    const std::string expectedStr = "--inputs_precisions=\"inputName1:U8\" --inputs_layouts=\"inputName1:NCHW\" --outputs_precisions=\"outputName1:FP32\" --outputs_layouts=\"outputName1:NC\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

TEST_F(ZeroCompilerAdapterTests, TwoIONetwork_ipU8U8opFP32FP32) {
    const std::string inputName1 = "inputName1";
    const std::string inputName2 = "inputName2";
    const IE::Precision inputsPrecision = IE::Precision::U8;
    const IE::Layout inputsLayout = IE::Layout::NCHW;
    
    const auto inputsInfo = createTwoInputDataMap(inputName1, inputName2, inputsPrecision, inputsPrecision, inputsLayout, inputsLayout);

    const std::string outputName1 = "outputName1";
    const std::string outputName2 = "outputName2";
    const IE::Precision outputsPrecision = IE::Precision::FP32;
    const IE::Layout outputsLayout = IE::Layout::NC;

    const auto outputsInfo = createTwoOutputDataMap(outputName1, outputName2, outputsPrecision, outputsPrecision, outputsLayout, outputsLayout);

    const std::string ioInfo = LevelZeroCompilerInDriver::serializeIOInfo(inputsInfo, outputsInfo);

    const std::string expectedStr = "--inputs_precisions=\"inputName1:U8 inputName2:U8\" --inputs_layouts=\"inputName1:NCHW inputName2:NCHW\" --outputs_precisions=\"outputName1:FP32 outputName2:FP32\" --outputs_layouts=\"outputName1:NC outputName2:NC\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

TEST_F(ZeroCompilerAdapterTests, OneInputTwoOuputsNetwork_ipU8opFP16FP32) {
    const auto inputsInfo = createSingleInputDataMap();
    
    const std::string outputName = "outputName1";
    const std::string outputName2 = "outputName2";

    const IE::Precision outputsPrecision1 = IE::Precision::FP32;
    const IE::Layout outputsLayout1 = IE::Layout::NC;
    const IE::Precision outputsPrecision2 = IE::Precision::FP16;
    const IE::Layout outputsLayout2 = IE::Layout::NCHW;

    const auto outputsInfo = createTwoOutputDataMap(outputName, outputName2, outputsPrecision1, outputsPrecision2, outputsLayout1, outputsLayout2);

    const std::string ioInfo = LevelZeroCompilerInDriver::serializeIOInfo(inputsInfo, outputsInfo);

    const std::string expectedStr = "--inputs_precisions=\"inputName1:U8\" --inputs_layouts=\"inputName1:NCHW\" --outputs_precisions=\"outputName1:FP32 outputName2:FP16\" --outputs_layouts=\"outputName1:NC outputName2:NCHW\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

}
}
