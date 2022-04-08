//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "mvSubspaces.h"

enum TrigonometricOpType : int32_t {
    SIN,
    SINH,
    COSH,
    TAN,
};

typedef struct {
    TrigonometricOpType type;
    void* kernel;
} TrigonometricOpInfo;

__attribute__((aligned(1024)))
#include "sk.cosh_fp16.3720xx.text.xdat"
#include "sk.sin_fp16.3720xx.text.xdat"
#include "sk.sinh_fp16.3720xx.text.xdat"
#include "sk.tan_fp16.3720xx.text.xdat"

#include "param_trigonometric.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Trigonometric)) {
    static constexpr std::initializer_list<SingleTest> test_list{
            {{1, 1, 7}, {1, 1, 7}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1, 1, 20}, {1, 1, 20}, orderZYX, {sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, {sw_params::Location::NN_CMX}}
#endif
    };

    static constexpr std::initializer_list<TrigonometricOpInfo> kernel_list{
            {TrigonometricOpType::SIN, sk_sin_fp16_3720xx_text},
            {TrigonometricOpType::SINH, sk_sinh_fp16_3720xx_text},
            {TrigonometricOpType::COSH, sk_cosh_fp16_3720xx_text},
            {TrigonometricOpType::TAN, sk_tan_fp16_3720xx_text},
    };
    class CustomCppTrigonometricTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppTrigonometricTest(): m_testsLoop(test_list, "test"), m_opInfoLoop(kernel_list, "kernel") {
        }

        virtual ~CustomCppTrigonometricTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppTrigonometricTest";
        }
        void userLoops() override {
            addLoop(m_opInfoLoop);
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            m_trigonometricParams = reinterpret_cast<sw_params::TrigonometricParams*>(paramContainer);
            *m_trigonometricParams = sw_params::TrigonometricParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::TrigonometricParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_trigonometricParams);

            m_params.kernel = reinterpret_cast<uint32_t>(m_opInfoLoop.value().kernel);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.008f;
        }

        void generateInputData() override {
            rand_seed();
            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }
        void generateReferenceData() override {
            std::function<float(const float&)> reference;

            switch (m_opInfoLoop.value().type) {
            case TrigonometricOpType::SIN:
                reference = [](const float& a) {
                    return sinf(a);
                };
                break;
            case TrigonometricOpType::SINH:
                reference = [](const float& a) {
                    return sinhf(a);
                };
                break;
            case TrigonometricOpType::COSH:
                reference = [](const float& a) {
                    return coshf(a);
                };
                break;
            case TrigonometricOpType::TAN:
                reference = [](const float& a) {
                    return tanf(a);
                };
                break;
            default:
                assert(0);  // unimp
            }
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = reference(val);
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
        ListIterator<TrigonometricOpInfo> m_opInfoLoop;
        sw_params::TrigonometricParams* m_trigonometricParams;
    };

// Trigonomatric kernel hangs on HW, need to investigate
#ifdef CONFIG_MOVISIM_RUN
    ICV_TESTS_REGISTER_SUITE(CustomCppTrigonometricTest)
#endif
}  // namespace )
