//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.mish_fp16.3720xx.text.xdat"

#include "param_mish.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Mish)) {
    static constexpr std::initializer_list<SingleTest> mish_test_list {
            {{1, 1, 10}, {1, 1, 10}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1, 20, 2}, {1, 20, 2}, orderZYX, {sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, {sw_params::Location::NN_CMX}}
#endif
            };

    class CustomCppMishTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppMishTest(): m_testsLoop(mish_test_list, "test") {
        }
        virtual ~CustomCppMishTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppMishTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            m_mishParams = reinterpret_cast<sw_params::MishParams*>(paramContainer);
            *m_mishParams = sw_params::MishParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::MishParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_mishParams);

            m_params.kernel = reinterpret_cast<uint32_t>(sk_mish_fp16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.01f;
        }

        void generateInputData() override {
            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }
        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float in = f16Tof32(m_inputTensor.at(indices));
                float val = log1pf(exp((double)in));
                float ref = val * -2.0f;
                ref = 1.0f + exp((double)ref);
                ref = 2.0f / ref - 1.0f;
                ref = in * ref;
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }
        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value,
                           abs_diff);
                }
            });
            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        sw_params::MishParams* m_mishParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppMishTest)
}  // namespace )
