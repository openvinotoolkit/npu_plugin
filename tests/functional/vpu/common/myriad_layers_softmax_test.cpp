// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND (1.e-3f)

typedef struct {
    int axis;
    SizeVector sizes;
} SoftmaxAxisSizes;

void PrintTo(const SoftmaxAxisSizes& p, std::ostream* os) {
    *os << "axis=" << p.axis << ", sizes=" << testing::PrintToString(p.sizes);
}

typedef myriadLayerTestBaseWithParam<SoftmaxAxisSizes> myriadLayersTestsSoftMaxParams_nightly;

class myriadLayersTestsSoftMax_nightly: public myriadLayersTestsSoftMaxParams_nightly {
protected:
    SoftmaxAxisSizes _testingInput;

    void SetUp() override {
        myriadLayersTestsSoftMaxParams_nightly::SetUp();
        _testingInput = GetParam();
    }
};

TEST_P(myriadLayersTestsSoftMax_nightly, TestsSoftMax)
{
    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPU_CONFIG_VALUE(NCHW);//std::get<6>(p);
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    SetInputTensors({_testingInput.sizes});
    SetOutputTensors({_testingInput.sizes});

    std::map<std::string, std::string> params;
    params["axis"] = std::to_string(_testingInput.axis);

    ASSERT_NO_FATAL_FAILURE(
        NetworkInit("Softmax", &params, 0, 0, nullptr, Precision::FP16, Precision::FP16)
    );

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(
        ref_softMax(_inputMap.begin()->second, _refBlob, _testingInput.axis)
    );

    Compare(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<SoftmaxAxisSizes> s_softMaxTensors = {
        {0, {  10,   91}},
        {1, {  10,   91}},
        {0, {5000}},
        {1, {   1, 1000, 1, 1}},
        {1, {   1, 1024, 7, 7}},
        {3, {   1,    7, 7, 1024}},
        {2, {   1,    1, 32, 32}},
        {0, {   8,   16, 16}},
        {1, {   4,   16,  8}},
        {2, {   3,    2, 16}},
        {0, {2268,   21}},
        {1, {  10,   10, 10, 10, 16, 16}},
        {5, {  10,   10, 10, 10, 16, 16}},
// TODO: CVS-16146
//        {5, {   9,   10, 11, 12, 13,  5, 6}},
//        {6, {   9,   10, 11, 12, 13,  5, 6}},
//        {0, {   9,   10, 11, 12, 13,  5, 6}},
};

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayersTestsSoftMax_nightly,
    ::testing::ValuesIn(s_softMaxTensors)
);
