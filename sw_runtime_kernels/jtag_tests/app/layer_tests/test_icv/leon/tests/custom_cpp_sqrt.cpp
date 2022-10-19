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

__attribute__((aligned(1024)))
#include "sk.sqrt_fp16.3720xx.text.xdat"

#include "param_sqrt.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Sqrt)) {
    const bool save_to_file = false;

    static constexpr std::initializer_list<SingleTest> sqrt_test_list{
            {{2, 2, 2}, {2, 2, 2}, orderZYX, FPE("sqrt_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1, 1, 20}, {1, 1, 20}, orderZYX, FPE("sqrt_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("sqrt_fp16.elf"), {sw_params::Location::NN_CMX}}};

    class CustomCppSqrtTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppSqrtTest(): m_testsLoop(sqrt_test_list, "test") {
        }
        virtual ~CustomCppSqrtTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppSqrtTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_sqrtParams = reinterpret_cast<sw_params::SqrtParams*>(paramContainer);
            *m_sqrtParams = sw_params::SqrtParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::SqrtParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_sqrtParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_sqrt_fp16_3720xx_text);
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
                float tmp = float(rand() % 1000) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }

        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = sqrtf(val);
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

        sw_params::SqrtParams* m_sqrtParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppSqrtTest)
}  // namespace )
