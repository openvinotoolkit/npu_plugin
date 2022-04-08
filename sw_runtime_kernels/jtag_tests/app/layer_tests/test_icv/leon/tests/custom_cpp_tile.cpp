//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_tile.3720xx.text.xdat"

#include "param_tile.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Tile)) {
    struct TileTest {
        Dims inputDims;
        Dims repeatsDims;
        Dims outputDims;
        StorageOrder storageOrder;
        CustomParams customLayerParams;
    };
    static constexpr std::initializer_list<TileTest> tile_test_list{
            {{2, 3, 2}, {5}, {2, 3, 10}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            {{2, 1, 4}, {2, 2}, {2, 2, 8}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            {{10, 12}, {1, 2, 2}, {1, 20, 24}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            {{1, 3, 2}, {5, 2}, {1, 15, 4}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            {{4, 2}, {1, 2, 5}, {1, 8, 10}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            {{2, 2, 2, 2}, {2, 2, 1}, {2, 4, 4, 2}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            {{3, 1, 7, 2}, {1, 1, 2}, {3, 1, 7, 4}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            // OV tests
            {{4, 3, 2}, {1, 3, 2}, {4, 9, 4}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            {{4, 3, 2, 1}, {3, 2, 1, 5}, {12, 6, 2, 5}, FULL_ORDER, {sw_params::Location::NN_CMX}},
            {{4, 3, 2, 5}, {1, 3, 2, 1}, {4, 9, 4, 5}, FULL_ORDER, {sw_params::Location::NN_CMX}},
    };

    class CustomCppTileTest : public CustomCppTests<fp16, TileTest> {
    public:
        explicit CustomCppTileTest(): m_testsLoop(tile_test_list, "test") {
        }
        virtual ~CustomCppTileTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppTileTest";
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

            std::vector<int32_t> repeatsDims;
            repeatsValues = m_testsLoop.value().repeatsDims;
            repeatsDims.push_back(m_testsLoop.value().repeatsDims.size());
            inputDims.insert(inputDims.begin(), outputDims.size() - inputDims.size(), 1);

            const MemoryDims inputMemDims(inputDims.data(), inputDims.size());
            const MemoryDims repeatsMemDims(repeatsDims.data(), repeatsDims.size());
            const MemoryDims outputMemDims(outputDims.data(), outputDims.size());

            m_inputTensor.init(maskOrder(storageOrder, inputDims.size()), inputMemDims, inputMemDims);
            m_repeatsTensor.init(maskOrder(storageOrder, repeatsDims.size()), repeatsMemDims, repeatsMemDims);
            m_outputTensor.init(maskOrder(storageOrder, outputDims.size()), outputMemDims, outputMemDims);
            m_referenceOutputTensor.init(maskOrder(storageOrder, outputDims.size()), outputMemDims, outputMemDims);

            allocBuffer(m_inputTensor);
            allocBuffer(m_repeatsTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            const TileTest* test = m_currentTest;
            m_tileParams = reinterpret_cast<sw_params::TileParams*>(paramContainer);
            *m_tileParams = sw_params::TileParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::TileParams);

            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_tileParams);

            m_params.kernel = reinterpret_cast<uint32_t>(sk_single_shave_tile_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.002f;
        }

        void initParserRunner() override {
            initMyriadResources();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inputBuff;
            OpTensor repeatsBuff;
            OpTensor outputBuff;

            m_inputTensor.exportToBuffer(inputBuff);
            m_repeatsTensor.exportToBuffer(repeatsBuff);
            m_outputTensor.exportToBuffer(outputBuff);

            customCppOp->addInputBuffer(inputBuff, m_requiredTensorLocation);
            customCppOp->addInputBuffer(repeatsBuff, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        void generateInputData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float index = float(m_inputTensor.index(indices));
                m_inputTensor.at(indices) = f32Tof16(index);
            });
            int repeats_index = 0;
            m_repeatsTensor.forEach(false, [&](const MemoryDims& indices) {
                m_repeatsTensor.at(indices) = repeatsValues[repeats_index++];
            });
        }

        void generateReferenceData() override {
            MemoryDims ref = m_inputTensor.toMemory(m_inputTensor.tensorDims());
            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                MemoryDims ti = m_outputTensor.toMemory(m_outputTensor.toTensor(indices));
                for (int i = 0; i < m_inputTensor.ndims(); i++) {
                    ti.dims[i] %= ref.dims[i];
                }
                m_referenceOutputTensor.at(indices) = m_inputTensor.at(ti);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");
            }
            bool threshold_test_failed = false;
            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = bool(!(abs_diff <= m_test_threshold));

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
        ListIterator<TileTest> m_testsLoop;
        std::vector<int32_t> repeatsValues;
        Tensor<int32_t> m_repeatsTensor;

        sw_params::TileParams* m_tileParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppTileTest)
}  // namespace
