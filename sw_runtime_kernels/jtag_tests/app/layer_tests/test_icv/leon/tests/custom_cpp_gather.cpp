//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_gather.3720xx.text.xdat"

#include "param_gather.h"

struct GatherTest {
    Dims inputDims;
    Dims indicesDims;
    Dims outputDims;
    StorageOrder storageOrder;
    CustomParams customLayerParams;
};

static inline StorageOrder maskOrder(StorageOrder fullOrder, int nOrd) {
    return static_cast<StorageOrder>(fullOrder & (0xffffffffu >> ((MAX_DIMS - nOrd) * HEX_DIGIT_BITS)));
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Gather)) {
    const bool saveToFile = false;

    static constexpr std::initializer_list<GatherTest> gather_test_list {
            {{4, 3, 2}, {2}, {2, 3, 2}, orderCHW, {0/*axis*/, 0/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{4, 3, 2}, {2}, {4, 2, 2}, orderCHW, {1/*axis*/, 0/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{4, 3, 2}, {2}, {4, 3, 2}, orderCHW, {2/*axis*/, 0/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{4, 3, 2}, {2, 3}, {4, 2, 3, 2}, orderNCHW, {1/*axis*/, 0/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{4, 3, 2}, {2}, {2, 3, 2}, orderHWC, {0/*axis*/, 0/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{4, 3, 2}, {2}, {4, 2, 2}, orderHWC, {1/*axis*/, 0/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            // For some reason fails on both movisim and HW, needs investigation
            // {{4, 3, 2}, {2}, {4, 3, 2}, orderHWC, {2/*axis*/, 0/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{4, 3, 2}, {2, 3}, {4, 2, 3, 2}, orderNHWC, {1/*axis*/, 0/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{5, 2}, {3, 2}, {3, 2}, orderNCHW, {0/*axis*/, 1/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{5, 2, 2}, {3, 2, 2}, {3, 2, 2}, orderNCHW, {2/*axis*/, 2/*batch dims*/, sw_params::Location::NN_CMX/*mem type*/}},
    };

    class CustomCppGatherTest : public CustomCppTests<fp16, GatherTest> {
    public:
        explicit CustomCppGatherTest(): m_testsLoop(gather_test_list, "test") {
        }
        virtual ~CustomCppGatherTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppGatherTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            initTestCase();
            const Dims& inputDims = m_currentTest->inputDims;
            const Dims& indicesDims = m_currentTest->indicesDims;
            const Dims& outputDims = m_currentTest->outputDims;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            int32_t numInputDims = inputDims.size();
            int32_t numIndicesDims = indicesDims.size();
            int32_t numOutputDims = outputDims.size();

            const StorageOrder inputOrder = maskOrder(storageOrder, numInputDims);
            const StorageOrder indicesOrder = maskOrder(storageOrder, numIndicesDims);
            const StorageOrder outputOrder = maskOrder(storageOrder, numOutputDims);

            const MemoryDims inputMemDims(inputDims.begin(), numInputDims);
            const MemoryDims indicesMemDims(indicesDims.begin(), numIndicesDims);
            const MemoryDims outputMemDims(outputDims.begin(), numOutputDims);

            m_inputTensor.init(inputOrder, inputMemDims);
            m_indicesTensor.init(indicesOrder, indicesMemDims);
            m_outputTensor.init(outputOrder, outputMemDims);
            m_referenceOutputTensor.init(outputOrder, outputMemDims);

            allocBuffer(m_inputTensor);
            allocBuffer(m_indicesTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(m_currentTest->storageOrder), ind);
            m_axis = (int64_t)ind[m_currentTest->customLayerParams.layerParams[0]];
            m_batchDims = (int64_t)m_currentTest->customLayerParams.layerParams[1];

            m_gatherParams = reinterpret_cast<sw_params::GatherParams*>(paramContainer);
            *m_gatherParams = sw_params::GatherParams();
            m_gatherParams->axis = m_axis;
            m_gatherParams->batchDims = m_batchDims;
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::GatherParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(m_currentTest->customLayerParams.layerParams[2]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_gatherParams);

            m_params.kernel = reinterpret_cast<uint32_t>(sk_single_shave_gather_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void initParserRunner() override {
            initMyriadResources();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);

            OpTensor inputBuff;
            m_inputTensor.exportToBuffer(inputBuff);
            customCppOp->addInputBuffer(inputBuff, m_requiredTensorLocation);

            OpTensor indicesBuff;
            m_indicesTensor.exportToBuffer(indicesBuff);
            customCppOp->addInputBuffer(indicesBuff, m_requiredTensorLocation);

            OpTensor outputBuff;
            m_outputTensor.exportToBuffer(outputBuff);
            customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);

            customCppOp->ops = *getParams();
        }

        void generateInputData() override {
            std::default_random_engine gen(123);

            std::uniform_real_distribution<float> uniformReal(-10.0, 10.0);
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float fp32Val = uniformReal(gen);
                m_inputTensor.at(indices) = f32Tof16(fp32Val);
            });

            const int maxIndices = m_inputTensor.memoryDims().dims[m_axis];
            std::uniform_int_distribution<int> uniformInt(0, maxIndices - 1);
            m_indicesTensor.forEach(false, [&](const MemoryDims& indices) {
                m_indicesTensor.at(indices) = uniformInt(gen);
            });
        }

        void generateReferenceData() override {
            int numInputDims = m_inputTensor.ndims();
            int numIndicesDims = m_indicesTensor.ndims();
            int numOutputDims = m_outputTensor.ndims();
            m_referenceOutputTensor.forEach(false, [&](const MemoryDims& outIdx) {
                int32_t indicesDims[MAX_DIMS];
                for (int i = 0; i < numIndicesDims; i++) {
                    if (i < numIndicesDims - m_batchDims) {
                        indicesDims[i] = outIdx.dims[m_axis + i];
                    } else {
                        indicesDims[i] = outIdx.dims[numOutputDims - numIndicesDims + i];
                    }
                }
                MemoryDims indicesIdx(indicesDims, numIndicesDims);
                int inputDimIdx = m_indicesTensor.at(indicesIdx);

                int32_t inputDims[MAX_DIMS];
                for (int i = 0; i < numInputDims; i++) {
                    if (i < m_axis) {
                        inputDims[i] = outIdx.dims[i];
                    } else if (i == m_axis) {
                        inputDims[i] = inputDimIdx;
                    } else {
                        inputDims[i] = outIdx.dims[i + numIndicesDims - m_batchDims - 1];
                    }
                }
                MemoryDims inputIdx(inputDims, numInputDims);

                m_referenceOutputTensor.at(outIdx) = m_inputTensor.at(inputIdx);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            if (saveToFile) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(),
                        "inMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                        "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                        m_referenceOutputTensor.bufferSize(), "refOutMyriad.bin");
            }

            bool thresholdTestFailed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gtValue = f16Tof32(m_referenceOutputTensor.at(indices));
                float absDiff = fabs(value - gtValue);
                bool differ = !bool(absDiff <= m_test_threshold);

                thresholdTestFailed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gtValue,
                           absDiff);
                }
            });

            return !thresholdTestFailed;
        }

    private:
        ListIterator<GatherTest> m_testsLoop;

        Tensor<int32_t> m_indicesTensor;
        int64_t m_axis;
        int64_t m_batchDims;

        sw_params::GatherParams* m_gatherParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppGatherTest)
} // namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Gather))
