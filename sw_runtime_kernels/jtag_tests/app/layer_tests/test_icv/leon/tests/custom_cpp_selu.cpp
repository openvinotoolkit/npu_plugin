//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.selu_fp16.3720xx.text.xdat"

#include "param_selu.h"
#define USE_SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed

struct SeluCustomParams {
    float alpha;
    float lambda;
    int32_t layerParams[MAX_LOCAL_PARAMS];
};

struct SeluTest {
    Dims inputDims;
    Dims outputDims;
    StorageOrder storageOrder;
    const char* kernelName;
    SeluCustomParams customLayerParams;
};

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Selu)) {
    static constexpr std::initializer_list<SeluTest> selu_test_list{
        {{1, 1, 20}, {1, 1, 20}, orderZYX, FPE("selu_fp16.elf"), {0.2f, 0.5f, {sw_params::Location::NN_CMX}}},
        {{100, 1, 1}, {100, 1, 1}, orderZYX, FPE("selu_fp16.elf"), {0.2f, 0.5f, {sw_params::Location::NN_CMX}}},
        {{1, 111, 1}, {1, 111, 1}, orderZYX, FPE("selu_fp16.elf"), {0.2f, 0.5f, {sw_params::Location::NN_CMX}}},
        {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("selu_fp16.elf"), {0.2f, 0.5f, {sw_params::Location::NN_CMX}}}
    };

    class CustomCppSeluTest : public CustomCppTests<fp16, SeluTest> {
    public:
        explicit CustomCppSeluTest(): m_testsLoop(selu_test_list, "test") {
        }
        virtual ~CustomCppSeluTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppSeluTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

            CustomCppTests<fp16, SeluTest>::initData();
            const SeluTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_seluParams = reinterpret_cast<sw_params::SeluParams*>(paramContainer);
            *m_seluParams = sw_params::SeluParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::SeluParams);
            m_alpha = test->customLayerParams.alpha;
            m_lambda = test->customLayerParams.lambda;
            m_seluParams->alpha = m_alpha;
            m_seluParams->lambda = m_lambda;
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_seluParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_selu_fp16_3720xx_text);
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
                float tmp = float(rand() % 200) / 10 - 10.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
            generateReferenceData();
        }
        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float min = std::min(val, 0.0f);
                float max = std::max(val, 0.0f);
                float ref = m_lambda * (max + m_alpha * (exp((double)min) - 1.0f));
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
        ListIterator<SeluTest> m_testsLoop;

        float m_alpha;
        float m_lambda;

        sw_params::SeluParams* m_seluParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppSeluTest)
}
