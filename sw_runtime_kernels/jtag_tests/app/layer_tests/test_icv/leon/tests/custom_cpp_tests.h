//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "custom_cpp_test_base.h"

using namespace icv_tests;

extern uint64_t cmxParamContainer[];

namespace
{

#define ICV_TEST_SUITE_NAME CustomCpp

// Binary representations of some floats
#define F_ONE      0x3F800000
#define F_TWO      0x40000000
#define F_FOUR     0x40800000

// Derived class for OpenCL tests, that use single input tensor and single output tensor.
// Parametrized by input type for "Convert u8->fp16" layer.
template <class InType, class TestType = SingleTest>
class CustomCppTests: public CustomCppTestBase
{
public:
    explicit CustomCppTests() = default;
    virtual ~CustomCppTests() = default;
protected:
    void initData() override {
        sw_params::BaseKernelParams emptyParamData;
        m_params = {
            nullptr,
            emptyParamData,
            0,
            0xFFFFFFFF,
            0,
            MAX_LOCAL_PARAMS
        };
        initTestCase();
        const Dims& inputDims = m_currentTest->inputDims;
        const Dims& outputDims = m_currentTest->outputDims;
        const StorageOrder& storageOrder = m_currentTest->storageOrder;

        const TensorDims dims3In(inputDims.begin()[0], inputDims.begin()[1], inputDims.begin()[2], 1);
        const TensorDims dims3Out(outputDims.begin()[0], outputDims.begin()[1], outputDims.begin()[2], 1);

        m_inputTensor.init(storageOrder, dims3In);
        m_outputTensor.init(storageOrder, dims3Out);
        m_referenceOutputTensor.init(storageOrder, dims3Out);

        allocBuffer(m_inputTensor);
        allocBuffer(m_outputTensor);
        allocBuffer(m_referenceOutputTensor);
    }

    void formatTestParams(char* str, int maxLength) const override
    {
        const auto& d = m_outputTensor.tensorDims();
        const auto& l = m_outputTensor.tensorLimits();

        const char* layout_text = layoutString(m_currentTest->storageOrder);

        snprintf_append(str, maxLength, "H W C = %u %u %u (%u %u %u), %s",
                        d.height, d.width, d.channels, l.height, l.width, l.channels, layout_text);
    }

    void initParserRunner() override
    {
        initMyriadResources();

        static_assert(std::is_base_of<Op, CustomCpp>());
        CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
        OpTensor inBuff;
        OpTensor outBuff;
        m_inputTensor.exportToBuffer(inBuff);
        m_outputTensor.exportToBuffer(outBuff);

        customCppOp->addInputBuffer(inBuff, m_requiredTensorLocation);
        customCppOp->addOutputBuffer(outBuff, m_requiredTensorLocation);
        customCppOp->ops = *getParams();
    }

    void resetOutputData() override
    {
        resetTensorBuffer(m_outputTensor);
    }
    bool checkResult() override
    {
        mvTensorAssert(m_test_threshold >= 0.0f, "Threshold is incorrect!");
        m_outputTensor.confirmBufferData();
        m_referenceOutputTensor.confirmBufferData();

        if (m_save_to_file)
        {
            saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(), this->suiteName());
        }

        bool result = true;
        int counter = m_num_of_debug_output;
        m_outputTensor.forEach(false, [&](const MemoryDims& indices)
        {
            const float value = f16Tof32(m_outputTensor.at(indices));
            const float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
            const float abs_diff = fabs(value - gt_value);
            const bool is_below_threshold = bool(abs_diff <= m_test_threshold);
            if (counter > 0 && !is_below_threshold) {
                --counter;
                nnLog(MVLOG_DEBUG, "Given %f, expected %f", value, gt_value);
            }
            result &= is_below_threshold;
        });
        return result;
    }
protected:
    // A pointer to current test case. It is set by initTestCase() in derived class.
    // Whole set of tests is stored in derived class.
    const TestType* m_currentTest;

    Tensor<InType> m_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceOutputTensor;
    sw_params::Location m_requiredTensorLocation = sw_params::Location::DDR;

    //  FIXME: Temporarily is located on CMX due to problem of ACT_SHAVE cache invalidation
    uint64_t * paramContainer = cmxParamContainer;
    float m_test_threshold = 0.0f;

    // Debug-specific
    bool m_save_to_file = false;
    int m_num_of_debug_output = 0;
};

} // anonymous namespace
