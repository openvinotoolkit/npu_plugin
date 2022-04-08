//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <Fp16Convert.h>
#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.erf_fp16.3720xx.text.xdat"

#include "param_erf.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Erf)) {
    const bool save_to_file = false;

    static constexpr std::initializer_list<SingleTest> erf_test_list{
            {{20, 1, 1}, {20, 1, 1}, orderZYX, {sw_params::Location::NN_CMX}},
            {{2, 2, 2}, {2, 2, 2}, orderZYX, {sw_params::Location::NN_CMX}},
            {{2, 2, 8}, {2, 2, 8}, orderZYX, {sw_params::Location::NN_CMX}},
            {{2, 2, 12}, {2, 2, 12}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1, 1, 20}, {1, 1, 20}, orderZYX, {sw_params::Location::NN_CMX}},
            {{4, 2, 12}, {4, 2, 12}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, {sw_params::Location::NN_CMX}}};

    class CustomCppErfTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppErfTest(): m_testsLoop(erf_test_list, "test") {
        }
        virtual ~CustomCppErfTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppErfTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_erfParams = reinterpret_cast<sw_params::ErfParams*>(paramContainer);
            *m_erfParams = sw_params::ErfParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::ErfParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_erfParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_erf_fp16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void generateInputData() override {
            rand_seed();

            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 1000) / 100 - 5.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }

        // currently exp in math library has errors, use this instead
        float local_exp(float x) {
            float x_abs;
            if (x < 0) {
                x_abs = -x;
                return 1.0 / exp(x_abs);
            } else {
                return exp(x);
            }
        }

        // currently erf in math library has errors, use this instead
        float local_erf(float x) {
            float a1 = 0.254829592;
            float a2 = -0.284496736;
            float a3 = 1.421413741;
            float a4 = -1.453152027;
            float a5 = 1.061405429;
            float p = 0.3275911;

            float sign = 1.0f;
            if (x < 0) {
                sign = -1.0f;
            }

            x = fabs(x);

            // formula 7.1.26 from Handbook of Mathematical Functions
            float t = 1.0 / (1.0 + p * x);

            float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * local_exp((-x) * x);

            return sign * y;
        }

        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = local_erf(val);  // replace erf in math library
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(),
                                 "inMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                                 m_referenceOutputTensor.bufferSize(), "refOutMyriad.bin");
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

        sw_params::ErfParams* m_erfParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppErfTest)
}  // namespace )
