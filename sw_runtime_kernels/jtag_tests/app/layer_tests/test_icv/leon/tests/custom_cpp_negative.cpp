//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_negative.3720xx.text.xdat"

#include "param_negative.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Negative)) {
    const std::initializer_list<SingleTest> negative_tests_list = {
            {{2, 2, 2}, {2, 2, 2}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1, 2, 19}, {1, 2, 19}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1, 50, 1}, {1, 50, 1}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, {sw_params::Location::NN_CMX}},
    };

    class CustomCppNegativeTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppNegativeTest(): m_testsLoop(negative_tests_list, "test") {
        }
        virtual ~CustomCppNegativeTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppNegativeTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;

            m_negativeParams = reinterpret_cast<sw_params::NegativeParams*>(paramContainer);
            *m_negativeParams = sw_params::NegativeParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::NegativeParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_negativeParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_single_shave_negative_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.0f;
        }

        void generateInputData() override {
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }

        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = -1 * (val);
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(),
                                 "icv-input.bin");
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "icv-output.bin");
                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                                 m_referenceOutputTensor.bufferSize(), "icv-ref.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float input = f16Tof32(m_inputTensor.at(indices));
                float abs_diff = 0.0f;

                if (isnan(value) && isnan(gt_value)) {
                    abs_diff = 0.0f;
                } else {
                    abs_diff = fabsf(value - gt_value);
                }

                bool differ = !bool(abs_diff <= m_test_threshold);
                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] in = %f out = %f ref =  %f abs_diff = %f\n", ti.height, ti.width,
                           ti.channels, input, value, gt_value, abs_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        sw_params::NegativeParams* m_negativeParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppNegativeTest)
}
