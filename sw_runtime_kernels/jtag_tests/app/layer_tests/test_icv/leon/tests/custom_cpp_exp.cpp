//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.exp_fp16.3720xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_exp.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Exp)) {
    static constexpr std::initializer_list<SingleTest> exp_test_list {
        {{1, 1, 7}, {1, 1, 7}, orderZYX, FPE("exp_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1, 1, 20}, {1, 1, 20}, orderZYX, FPE("exp_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("exp_fp16.elf"), {sw_params::Location::NN_CMX}}};

    class CustomCppExpTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppExpTest(): m_testsLoop(exp_test_list, "test") {
        }
        virtual ~CustomCppExpTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppExpTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_expParams = reinterpret_cast<sw_params::ExpParams*>(paramContainer);
            *m_expParams = sw_params::ExpParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::ExpParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_expParams);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_exp_fp16_3720xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(exp_fp16));
#endif
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            // The kernel of implementing Exp with SAU.EXP2 is based on formula: exp( x ) ≈ 2 ^ (x * 0x3dc5) , and
            // the absolute diff will be greater than 0.02 in a few cases. So the threshold is set to 0.05 to be safe.
            m_test_threshold = 0.05f;
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
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = exp((double)val);
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

        sw_params::ExpParams* m_expParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppExpTest)
}  // namespace )
