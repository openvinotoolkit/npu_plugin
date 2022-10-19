//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.hardsigmoid_fp16.3720xx.text.xdat"

#include "param_hardsigmoid.h"

struct HardSigmoidCustomParams {
    float alpha;
    float beta;
    int32_t layerParams[MAX_LOCAL_PARAMS];
};

struct HardSigmoidTest {
    Dims inputDims;
    Dims outputDims;
    StorageOrder storageOrder;
    const char* kernelName;
    HardSigmoidCustomParams customLayerParams;
};

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, HardSigmoid)) {    
    static constexpr std::initializer_list<HardSigmoidTest> hardsigmoid_test_list{
            {{1, 1, 20}, {1, 1, 20}, orderZYX, FPE("hardsigmoid_fp16.elf"), {0.2f, 0.5f, {sw_params::Location::NN_CMX}}},
            {{100, 1, 1}, {100, 1, 1}, orderZYX, FPE("hardsigmoid_fp16.elf"), {0.2f, 0.5f, {sw_params::Location::NN_CMX}}},
            {{1, 111, 1}, {1, 111, 1}, orderZYX, FPE("hardsigmoid_fp16.elf"), {0.2f, 0.5f, {sw_params::Location::NN_CMX}}},
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("hardsigmoid_fp16.elf"), {0.2f, 0.5f, {sw_params::Location::NN_CMX}}}
    };

    class CustomCppHardSigmoidTest : public CustomCppTests<fp16, HardSigmoidTest> {
    public:
        explicit CustomCppHardSigmoidTest(): m_testsLoop(hardsigmoid_test_list, "test") {
        }
        virtual ~CustomCppHardSigmoidTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppHardSigmoidTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

            CustomCppTests<fp16, HardSigmoidTest>::initData();
            const HardSigmoidTest* test = m_currentTest;

            m_hardsigmoidParams = reinterpret_cast<sw_params::HardSigmoidParams*>(paramContainer);
            *m_hardsigmoidParams = sw_params::HardSigmoidParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::HardSigmoidParams);
            m_alpha = test->customLayerParams.alpha;
            m_beta = test->customLayerParams.beta;
            m_hardsigmoidParams->alpha = m_alpha;
            m_hardsigmoidParams->beta = m_beta;
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_hardsigmoidParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_hardsigmoid_fp16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.0005f;
        }

        void generateInputData() override {
            rand_seed();

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
            float alpha = m_alpha;
            float beta = m_beta;

            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = val * alpha + beta;
                ref = std::min(1.0f, ref);
                ref = std::max(0.0f, ref);
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

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
        ListIterator<HardSigmoidTest> m_testsLoop;

        float m_alpha;
        float m_beta;

        sw_params::HardSigmoidParams* m_hardsigmoidParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppHardSigmoidTest)
}  // namespace )
