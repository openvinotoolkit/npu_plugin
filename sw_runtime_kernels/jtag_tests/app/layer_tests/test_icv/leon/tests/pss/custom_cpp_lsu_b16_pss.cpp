//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <nn_cache.h>
#include <cmath>

__attribute__((aligned(1024)))
#include "sk.lsu_b16.3720xx.text.xdat"

#include "pss/param_lsu_b16.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, B16_pss)) {
    static constexpr std::initializer_list<SingleTest> b16_test_list{
            {{1, 1, 8}, {1, 1, 8}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1, 2, 8}, {1, 2, 8}, orderZYX, {sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
            {{1, 1, 1000}, {1, 1, 1000}, orderZYX, {sw_params::Location::NN_CMX}},
#endif
    };

    class CustomCppB16Test : public CustomCppTests<fp16> {
    public:
        explicit CustomCppB16Test(): m_testsLoop(b16_test_list, "test") {
        }
        virtual ~CustomCppB16Test() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppB16Testpss";
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
            m_lsuB16Params = reinterpret_cast<sw_params::LsuB16Params*>(paramContainer);
            *m_lsuB16Params = sw_params::LsuB16Params();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::LsuB16Params);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_lsuB16Params);

            m_params.kernel = reinterpret_cast<uint32_t>(sk_lsu_b16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 1.0f;
        }

        void generateInputData() override {
            rand_seed();

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 1000) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }

        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                m_referenceOutputTensor.at(indices) = f32Tof16(val);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
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
                float ulp_diff = ulp::absdiff_fp32(value, gt_value);

                bool differ = !bool(ulp_diff <= m_test_threshold);
                threshold_test_failed |= differ;

                float input = f16Tof32(m_inputTensor.at(indices));

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] input: %f value: %f ref_value: %f ulp: %f\n", ti.height, ti.width,
                           ti.channels, input, value, gt_value, ulp_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        sw_params::LsuB16Params* m_lsuB16Params;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppB16Test)
}  // namespace )
