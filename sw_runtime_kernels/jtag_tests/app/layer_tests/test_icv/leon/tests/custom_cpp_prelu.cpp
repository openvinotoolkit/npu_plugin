//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "param_prelu.h"
#include "sk.prelu_fp16.3720xx.text.xdat"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, PRelu)) {
    static constexpr std::initializer_list<SingleTest> prelu_test_list{
            {{7, 1, 1}, {7, 1, 1}, orderZYX, {sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, {sw_params::Location::NN_CMX}},
#endif
            {{12, 2, 2}, {12, 2, 2}, orderZYX, {sw_params::Location::NN_CMX}},
            {{7, 2, 20}, {7, 2, 20}, orderZYX, {sw_params::Location::NN_CMX}},
    };

    class CustomCppPReluTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppPReluTest(): m_testsLoop(prelu_test_list, "test") {
        }
        virtual ~CustomCppPReluTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppPReluTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};
            CustomCppTests<fp16>::initData();

            const t_D8StorageOrder& storageOrder = m_currentTest->storageOrder;
            const Dims& inputDims = m_currentTest->inputDims;

            MemoryDims inputMemDims(inputDims.begin(), inputDims.size());

            int32_t map[MaxTensorDims];
            orderToIndices(storageOrder, map);
            m_axis = map[Map::C];
            int dimsSize = inputDims.size();
            mvTensorAssert((m_axis >= 0) && (m_axis < dimsSize), "StorageOrder must contain channels axis");

            const int numChannels = inputMemDims.dims[m_axis];

            const Dims channelsDims = {numChannels};
            const MemoryDims wbdims(channelsDims.begin(), channelsDims.size());
            const t_D8StorageOrder wborder = maskOrder(FULL_ORDER, 1);

            m_preluParams = reinterpret_cast<sw_params::PReluParams*>(paramContainer);
            *m_preluParams = sw_params::PReluParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::PReluParams);
            m_requiredTensorLocation =
                    static_cast<sw_params::Location>(m_currentTest->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_preluParams);
            m_params.kernel = reinterpret_cast<uint32_t>(sk_prelu_fp16_3720xx_text);

            m_negSlopeTensor.init(wborder, wbdims);

            allocBuffer(m_negSlopeTensor);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.005f;
        }

        void initParserRunner() override {
            initMyriadResources();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);

            OpTensor inputBuff;
            OpTensor negSlopeBuff;
            OpTensor outputBuff;
            m_inputTensor.exportToBuffer(inputBuff);
            m_negSlopeTensor.exportToBuffer(negSlopeBuff);
            m_outputTensor.exportToBuffer(outputBuff);

            customCppOp->addInputBuffer(inputBuff, m_requiredTensorLocation);
            customCppOp->addInputBuffer(negSlopeBuff, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        void formatTestParams(char* str, int maxLength) const override {
            const auto& id = m_inputTensor.tensorDims();
            const auto& il = m_inputTensor.tensorLimits();
            const auto& nd = m_negSlopeTensor.tensorDims();
            const auto& nl = m_negSlopeTensor.tensorLimits();
            const auto& od = m_outputTensor.tensorDims();
            const auto& ol = m_outputTensor.tensorLimits();

            snprintf_append(
                    str, maxLength,
                    "input: %ux%ux%u (%ux%ux%u) negative slope: %ux%ux%u (%ux%ux%u) => output: %ux%ux%u (%ux%ux%u)",
                    id.channels, id.height, id.width, il.channels, il.height, il.width, nd.channels, nd.height,
                    nd.width, nl.channels, nl.height, nl.width, od.channels, od.height, od.width, ol.channels,
                    ol.height, ol.width);
        }

        void generateInputData() override {
            rand_seed();

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });

            const float scale = 1.0f / (float)m_negSlopeTensor.memoryLimits().dims[0];

            m_negSlopeTensor.forEach(false, [&](const MemoryDims& indices) {
                m_negSlopeTensor.at(indices) = f32Tof16(-0.5f - ((float)indices.dims[0] * scale));
            });
        }
        void generateReferenceData() override {
            auto negSlope = m_negSlopeTensor.data();

            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                const int channels = indices.dims[m_axis];

                float val = f16Tof32(m_inputTensor.at(indices));
                float maxVal = std::max(val, 0.0f);
                float minVal = std::min(val, 0.0f);

                float weight = f16Tof32(negSlope[channels]);
                float ref = f16Tof32(f32Tof16(maxVal + f16Tof32(f32Tof16(weight * minVal))));

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
                    printf("DIFF HWC [%d:%d:%d] output %f ref %f diff %f\n", ti.height, ti.width, ti.channels, value,
                           gt_value, abs_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        int m_axis = 0;

        Tensor<fp16> m_negSlopeTensor;
        sw_params::PReluParams* m_preluParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppPReluTest)
}  // namespace )
