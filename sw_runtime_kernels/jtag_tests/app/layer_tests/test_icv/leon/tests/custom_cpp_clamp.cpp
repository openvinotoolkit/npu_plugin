//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include <custom_cpp_tests.h>
#include <numeric>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "param_clamp.h"
#include "sk.clamp_fp16.3720xx.text.xdat"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Clamp)) {
    struct ClampTest {
        std::initializer_list<int32_t> inputDims;
        std::initializer_list<int32_t> outputDims;
        StorageOrder storageOrder;
        float min;
        float max;
        CustomParams customLayerParams;
    };

    const bool save_to_file = false;

    static constexpr std::initializer_list<ClampTest> clamp_test_list{
            // 1D input Tensor
            {{256}, {256}, orderC, 10.0f, 50.0f, sw_params::Location::NN_CMX},
            {{9}, {9}, orderC, 2.0f, 5.0f, sw_params::Location::NN_CMX},
            {{30}, {30}, orderC, -5.0f, 10.0f, sw_params::Location::NN_CMX},

    // 2D input Tensor
#ifdef CONFIG_RUN_LARGE_TESTS
            {{30, 57}, {30, 57}, orderNC, -13.0f, 30.0f, sw_params::Location::NN_CMX},
            {{48, 80}, {48, 80}, orderNC, 0.0f, 80.0f, sw_params::Location::NN_CMX},
#endif
            {{5, 73}, {5, 73}, orderNC, -5.0f, 40.0f, sw_params::Location::NN_CMX},

            // 3D input Tensor
            {{1, 1, 1}, {1, 1, 1}, orderCHW, 10.0f, 50.0f, sw_params::Location::NN_CMX},
            {{2, 2, 2}, {2, 2, 2}, orderCHW, 10.0f, 50.0f, sw_params::Location::NN_CMX},
            {{2, 2, 2}, {2, 2, 2}, orderCHW, -2.0f, 2.0f, sw_params::Location::NN_CMX},
            {{2, 4, 6}, {2, 4, 6}, orderCHW, 15.0f, 48.0f, sw_params::Location::NN_CMX},
#ifdef CONFIG_RUN_LARGE_TESTS
            {{256, 1, 1}, {256, 1, 1}, orderCHW, 10.0f, 50.0f, sw_params::Location::NN_CMX},
#endif

            // 4D input Tensor
            {{2, 4, 6, 9}, {2, 4, 6, 9}, orderNCHW, 15.0f, 48.0f, sw_params::Location::NN_CMX},
            {{3, 5, 7, 13}, {3, 5, 7, 13}, orderNCHW, 15.0f, 48.0f, sw_params::Location::NN_CMX},
#ifdef CONFIG_RUN_LARGE_TESTS
            {{7, 16, 7, 25}, {7, 16, 7, 25}, orderNCHW, 5.0f, 20.0f, sw_params::Location::NN_CMX},
#endif
    };

    class CustomCppClampTest : public CustomCppTests<fp16, ClampTest> {
    public:
        explicit CustomCppClampTest(): m_testsLoop(clamp_test_list, "test") {
        }
        virtual ~CustomCppClampTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppClampTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};
            initTestCase();

            std::vector<int32_t> inputDims = m_testsLoop.value().inputDims;
            std::vector<int32_t> outputDims = m_testsLoop.value().outputDims;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            MemoryDims md_input_dims(inputDims.data(), inputDims.size());
            MemoryDims md_output_dims(outputDims.data(), outputDims.size());

            m_inputTensor.init(maskOrder(storageOrder, inputDims.size()), md_input_dims, md_input_dims);
            m_outputTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);
            m_referenceOutputTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);

            allocBuffer(m_inputTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            m_min = m_testsLoop.value().min;
            m_max = m_testsLoop.value().max;
            m_clampParams = reinterpret_cast<sw_params::ClampParams*>(paramContainer);
            *m_clampParams = sw_params::ClampParams();
            m_clampParams->min = m_min;
            m_clampParams->max = m_max;
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::ClampParams);
            m_requiredTensorLocation =
                    static_cast<sw_params::Location>(m_currentTest->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_clampParams);
            m_params.kernel = reinterpret_cast<uint32_t>(sk_clamp_fp16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.0f;
        }

        void formatTestParams(char* str, int maxLength) const override {
            char inSizes_str[100];
            char outSizes_str[100];

            snprintf_append(str, maxLength, "input: %s, output: %s", m_inputTensor.dimsToStringNCHW(inSizes_str),
                            m_outputTensor.dimsToStringNCHW(outSizes_str));
        }

        void generateInputData() override {
            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 1000) / 10 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }

        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = std::min(m_clampParams->max, std::max(m_clampParams->min, val));
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

            mvTensorAssert(m_outputTensor.storageOrder() == m_inputTensor.storageOrder());
            mvTensorAssert(m_outputTensor.storageOrder() == m_referenceOutputTensor.storageOrder());

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float input = f16Tof32(m_inputTensor.at(indices));

                float abs_diff = fabs(value - gt_value);
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
        ListIterator<ClampTest> m_testsLoop;
        float m_min;
        float m_max;

        sw_params::ClampParams* m_clampParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppClampTest)
}  // namespace
