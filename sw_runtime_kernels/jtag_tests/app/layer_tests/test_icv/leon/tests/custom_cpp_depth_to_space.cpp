//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include <iostream>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_depth_to_space.3720xx.text.xdat"

#include "param_depth_to_space.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, DepthToSpace)) {
    const bool saveToFile = false;

    static constexpr std::initializer_list<SingleTest> depth_to_space_test_list {
            /*mode blocks_first: 0, depth_first: 1*/
            {{1, 1, 12, 2}, {2, 2, 3, 2}, orderNCHW, {2/*block size*/, 0/*mode*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{1, 1, 12, 2}, {2, 2, 3, 2}, orderNCHW, {2/*block size*/, 1/*mode*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{1, 1, 12, 2}, {2, 2, 3, 2}, orderNHWC, {2/*block size*/, 0/*mode*/, sw_params::Location::NN_CMX/*mem type*/}},
            {{1, 1, 12, 2}, {2, 2, 3, 2}, orderNHWC, {2/*block size*/, 1/*mode*/, sw_params::Location::NN_CMX/*mem type*/}},
    };

    class CustomCppDepthToSpaceTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppDepthToSpaceTest(): m_testsLoop(depth_to_space_test_list, "test") {
        }
        virtual ~CustomCppDepthToSpaceTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppDepthToSpaceTest";
        }

        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            initTestCase();
            const Dims& inputDims = m_currentTest->inputDims;
            const Dims& outputDims = m_currentTest->outputDims;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            int32_t numInputDims = inputDims.size();
            int32_t numOutputDims = outputDims.size();
            const TensorDims inputTensorDims(inputDims.begin(), numInputDims);
            const TensorDims outputTensorDims(outputDims.begin(), numOutputDims);

            m_inputTensor.init(storageOrder, inputTensorDims);
            m_outputTensor.init(storageOrder, outputTensorDims);
            m_referenceOutputTensor.init(storageOrder, outputTensorDims);

            allocBuffer(m_inputTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            const SingleTest* test = m_currentTest;
            m_depthtospaceParams = reinterpret_cast<sw_params::DepthToSpaceParams*>(paramContainer);
            *m_depthtospaceParams = sw_params::DepthToSpaceParams();
            m_depthtospaceParams->blockSize = (int64_t)test->customLayerParams.layerParams[0];
            m_depthtospaceParams->mode = (int64_t)test->customLayerParams.layerParams[1];

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::DepthToSpaceParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[2]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_depthtospaceParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_single_shave_depth_to_space_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void generateInputData() override {
            std::default_random_engine gen(123);

            std::uniform_real_distribution<float> uniformReal(-10.0, 10.0);
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float fp32Val = uniformReal(gen);
                m_inputTensor.at(indices) = f32Tof16(fp32Val);
            });
        }

        void generateReferenceData() override {
            int32_t blockSize = (int32_t)m_depthtospaceParams->blockSize;
            int32_t mode = (int32_t)m_depthtospaceParams->mode;
            const auto inputDims = m_inputTensor.memoryDims();
            NDOrder inputOrder = m_inputTensor.storageOrder();

            int32_t k = m_inputTensor.ndims() - 2;
            int32_t inputReshapeDims[MAX_DTS_DIMS] = {0};
            int32_t orders[MAX_DTS_DIMS] = {0};

            if ((inputOrder == orderNCHW || inputOrder == orderCHW) && mode == 0) {
                for (int i = 0; i < k; i++) {
                    inputReshapeDims[i] = inputDims.dims[i];
                    inputReshapeDims[i + k + 1] = blockSize;
                }
                inputReshapeDims[k] = inputDims.dims[k] / (blockSize * blockSize);
                inputReshapeDims[2 * k + 1] = inputDims.dims[k + 1];

                for (int i = 0; i < k; i++) {
                    orders[2 * i] = k + 1 + i; // 2 * k + 1 - (k - i)
                    orders[2 * i + 1] = i; // 2 * k + 1 - ((k - i) + k + 1)
                }
                orders[2 * k] = k;
                orders[2 * k + 1] = 2 * k + 1;
            } else if ((inputOrder == orderNCHW || inputOrder == orderCHW) && mode == 1) {
                for (int i = 0; i < k; i++) {
                    inputReshapeDims[i] = inputDims.dims[i];
                    inputReshapeDims[i + k] = blockSize;
                }
                inputReshapeDims[2 * k] = inputDims.dims[k] / (blockSize * blockSize);
                inputReshapeDims[2 * k + 1] = inputDims.dims[k + 1];

                for (int i = 0; i < k; i++) {
                    orders[2 * i] = k + i; // 2 * k + 1 - (k + 1 - i)
                    orders[2 * i + 1] = i; // 2 * k + 1 - ((k - i) + k + 1)
                }
                orders[2 * k] = 2 * k;
                orders[2 * k + 1] = 2 * k + 1;
            } else if ((inputOrder == orderNHWC || inputOrder == orderHWC) && mode == 0) {
                for (int i = 0; i < k; i++) {
                    inputReshapeDims[i + k + 1] = inputDims.dims[i + 1];
                    inputReshapeDims[i + 1] = blockSize;
                }
                inputReshapeDims[0] = inputDims.dims[0] / (blockSize * blockSize);
                inputReshapeDims[2 * k + 1] = inputDims.dims[k + 1];

                for (int i = 0; i < k; i++) {
                    orders[2 * i + 1] = i + 1; // 2 * k + 1 - (k + k - i)
                    orders[2 * i + 2] = k + i + 1; // 2 * k + 1 - (k - i)
                }
                orders[0] = 0;
                orders[2 * k + 1] = 2 * k + 1;
            } else if ((inputOrder == orderNHWC || inputOrder == orderHWC) && mode == 1) {
                for (int i = 0; i < k; i++) {
                    inputReshapeDims[i + k + 1] = inputDims.dims[i + 1];
                    inputReshapeDims[i] = blockSize;
                }
                inputReshapeDims[k] = inputDims.dims[0] / (blockSize * blockSize);
                inputReshapeDims[2 * k + 1] = inputDims.dims[k + 1];

                for (int i = 0; i < k; i++) {
                    orders[2 * i] = k + i; // 2 * k + 1 - (k + 1 - i)
                    orders[2 * i + 1] = i; // 2 * k + 1 - ((k - i) + k + 1)
                }
                orders[2 * k] = 2 * k;
                orders[2 * k + 1] = 2 * k + 1;
            } else {
                printf("Unsupported DepthToSpace layout or mode.\n");
                return;
            }

            int32_t outputReshapeDims[MAX_DTS_DIMS];
            int32_t numReshapeDims = 2 * k + 2;
            for(int i = 0; i < numReshapeDims; i++) {
                int32_t idx = orders[i];
                outputReshapeDims[i] = inputReshapeDims[idx];
            }

            int32_t numValues = 1;
            for (int i = 0; i < m_inputTensor.ndims(); i++) {
                numValues *= inputDims.dims[i];
            }

            for (int inIdx = 0; inIdx < numValues; inIdx++) {
                int32_t inputReshapeCoords[MAX_DTS_DIMS] = {0};
                subspace::getCoord(inIdx, inputReshapeDims, numReshapeDims, inputReshapeCoords);

                int32_t outputReshapeCoords[MAX_DTS_DIMS] = {0};
                for (int j = 0; j < numReshapeDims; j++) {
                    int32_t idx = orders[j];
                    outputReshapeCoords[j] = inputReshapeCoords[idx];
                }

                int32_t outIdx = 0;
                int32_t sizeCount = 1;
                for (int j = 0; j < numReshapeDims; j++) {
                    outIdx += sizeCount * outputReshapeCoords[j];
                    sizeCount *= outputReshapeDims[j];
                }

                m_referenceOutputTensor.at(MemoryDims(outIdx)) = m_inputTensor.at(MemoryDims(inIdx));
            }
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
        ListIterator<SingleTest> m_testsLoop;

        sw_params::DepthToSpaceParams* m_depthtospaceParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppDepthToSpaceTest)
}; // namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, DepthToSpace))
