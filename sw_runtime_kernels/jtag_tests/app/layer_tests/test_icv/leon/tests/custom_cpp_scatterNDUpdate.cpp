//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_test_base.h>
#include <custom_cpp_tests.h>
#include <numeric>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_scatterNDUpdate.3720xx.text.xdat"

#include "param_scatterNDUpdate.h"

namespace {

struct ScatterNDUpdateTestParams {
    std::initializer_list<int32_t> inputDims;
    std::initializer_list<int32_t> indicesDims;
    Dims outputDims;
    StorageOrder storageOrder;
    CustomParams customLayerParams;
};

enum ScatterMode { F16, I32 };

const std::initializer_list<ScatterNDUpdateTestParams> scatterNDUpdate_test_list = {
//      input,             indices
//      1D input Tensor
        {{1}, {1, 1}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
        {{8}, {1, 4}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
        {{100}, {1, 6}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
        {{1000}, {1, 3, 4}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
        {{2560}, {1, 6, 6}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
#endif
//      2D input Tensor
        {{2, 2}, {1, 2}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
        {{4, 2}, {1, 3}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
        {{100, 70}, {2, 9}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
        {{1000, 5}, {2, 2}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
#endif
//      3D input Tensor
        {{3, 3, 3}, {3, 2, 2}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
        {{8, 3, 3}, {2, 2}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
//      4D input Tensor
        {{7, 5, 3, 2}, {2, 2}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
        {{24, 8, 8, 8}, {3, 2, 1}, {}, FULL_ORDER, {sw_params::Location::NN_CMX}},
#endif
};

template <class SrcType, ScatterMode sMode>
class CustomCppScatterNDUpdateTest : public CustomCppTests<fp16, ScatterNDUpdateTestParams> {
public:
    explicit CustomCppScatterNDUpdateTest(): m_testsLoop(scatterNDUpdate_test_list, "test") {
    }
    virtual ~CustomCppScatterNDUpdateTest() {
    }

protected:
    const char* suiteName() const override {
        return "CustomCppScatterNDUpdateTest";
    }

    void userLoops() override {
        addLoop(m_testsLoop);
    }

    void calcUpdatesDims(std::vector<int32_t>& inputDims, std::vector<int32_t>& indicesDims,
                            std::vector<int32_t>& updatesDims) {
        // considering innermost to outermost way of working with shapes,
        // updates shape must be input.dims[:input.ndims - indices.dims[0]] + indices.dims[1:]

        for (size_t i = 0; i < inputDims.size() - indicesDims[0]; i++)
            updatesDims.push_back(inputDims[i]);

        for (size_t i = 1; i < indicesDims.size(); i++)
            updatesDims.push_back(indicesDims[i]);
    }

    void initData() override {
        sw_params::BaseKernelParams emptyParamData;
        m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

        initTestCase();
        std::vector<int32_t> inputDims = m_testsLoop.value().inputDims;
        std::vector<int32_t> indicesDims = m_testsLoop.value().indicesDims;
        std::vector<int32_t> updatesDims;
        const StorageOrder& storageOrder = m_currentTest->storageOrder;

        // construct updates dims
        calcUpdatesDims(inputDims, indicesDims, updatesDims);

        // construct output dims
        const auto& outputDims = inputDims;

        MemoryDims md_input_dims(inputDims.data(), inputDims.size());
        MemoryDims md_indices_dims(indicesDims.data(), indicesDims.size());
        MemoryDims md_updates_dims(updatesDims.data(), updatesDims.size());
        MemoryDims md_output_dims(outputDims.data(), outputDims.size());

        m_inputTensor.init(maskOrder(storageOrder, inputDims.size()), md_input_dims, md_input_dims);
        m_indicesTensor.init(maskOrder(storageOrder, indicesDims.size()), md_indices_dims, md_indices_dims);
        m_updatesTensor.init(maskOrder(storageOrder, updatesDims.size()), md_updates_dims, md_updates_dims);
        m_outputTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);
        m_referenceTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);

        allocBuffer(m_inputTensor);
        allocBuffer(m_outputTensor);
        allocBuffer(m_indicesTensor);
        allocBuffer(m_updatesTensor);
        allocBuffer(m_referenceTensor);

        const ScatterNDUpdateTestParams* test = m_currentTest;
        m_scatterNDUpdateParams = reinterpret_cast<sw_params::ScatterNDUpdateParams*>(paramContainer);
        *m_scatterNDUpdateParams = sw_params::ScatterNDUpdateParams();
        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
        m_params.paramDataLen = sizeof(sw_params::ScatterNDUpdateParams);

        m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_scatterNDUpdateParams);

        m_params.kernel = reinterpret_cast<uint32_t>(sk_single_shave_scatterNDUpdate_3720xx_text);
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.07f;
    }

    void formatTestParams(char* str, int maxLength) const override {
        char inSizes_str[100];
        char outSizes_str[100];
        char idxSizes_str[100];
        char updatesSizes_str[100];

        snprintf_append(str, maxLength, "input: %s, output: %s, indices: %s, updates: %s",
                        m_inputTensor.dimsToStringNCHW(inSizes_str), m_outputTensor.dimsToStringNCHW(outSizes_str),
                        m_indicesTensor.dimsToStringNCHW(idxSizes_str),
                        m_updatesTensor.dimsToStringNCHW(updatesSizes_str));
    }

    void initParserRunner() override {
        initMyriadResources();

        static_assert(std::is_base_of<Op, CustomCpp>());
        CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
        OpTensor inputBuff;
        OpTensor indicesBuff;
        OpTensor updatesBuff;
        OpTensor outputBuff;
        m_inputTensor.exportToBuffer(inputBuff);
        m_indicesTensor.exportToBuffer(indicesBuff);
        m_updatesTensor.exportToBuffer(updatesBuff);
        m_outputTensor.exportToBuffer(outputBuff);

        customCppOp->addInputBuffer(inputBuff, m_requiredTensorLocation);
        customCppOp->addInputBuffer(indicesBuff, m_requiredTensorLocation);
        customCppOp->addInputBuffer(updatesBuff, m_requiredTensorLocation);
        customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);
        customCppOp->ops = *getParams();
    }

    void resetOutputData() override {
        resetTensorBuffer(m_outputTensor);
    }

    void generateInputData() override {
        int i = 100;
        m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            if (i > 200.0)
                i = 100;
            float val = (float)(i++);
            if (sMode == F16) {
                m_inputTensor.at(indices) = f32Tof16(val);
            } else {
                m_inputTensor.at(indices) = (int)(val);
            }
        });
        const auto& test = m_testsLoop.value();
        m_indicesTensor.forEach(false, [&](const MemoryDims& indices) {
            const auto dim = test.inputDims.size() - indices.dims[Dim::W] - 1;
            const auto maxValue = test.inputDims.begin()[dim];
            m_indicesTensor.at(indices) = rand() % maxValue;
        });

        i = 10;
        m_updatesTensor.forEach(false, [&](const MemoryDims& indices) {
            if (i > 99.0)
                i = 10;
            float val = (float)(i++);
            if (sMode == F16) {
                m_updatesTensor.at(indices) = f32Tof16(val);
            } else {
                m_updatesTensor.at(indices) = (int)(val);
            }
        });
    }

    void generateReferenceData() override {
        const auto& test_value = m_testsLoop.value();

        const auto idx = m_indicesTensor.data();
        const auto updates = m_updatesTensor.data();
        const auto src = m_inputTensor.data();
        auto dst = m_referenceTensor.data();

        std::vector<int32_t> input_dims(test_value.inputDims.begin(), test_value.inputDims.end());

        std::vector<int32_t> indices_dims(test_value.indicesDims.begin(), test_value.indicesDims.end());

        int last_idx_dim = indices_dims.front();
        int update_chunk_size = input_dims.size() - last_idx_dim;
        if (update_chunk_size < 0)
            update_chunk_size = 0;
        const auto update_el_number = subspace::getTotal(m_inputTensor.memoryDims().dims, update_chunk_size);

        for (int i = 0; i < subspace::getTotal(m_inputTensor.memoryDims().dims, input_dims.size()); i++)
            dst[i] = src[i];

        const auto input_data_dim_pading = [&] {
            std::vector<size_t> padding(input_dims.size(), 1);
            for (int32_t i = input_dims.size() - 1; i != 0; --i) {
                padding[i - 1] = padding[i] * input_dims[input_dims.size() - 1 - i];
            };
            return padding;
        }();

        const auto num_of_updates =
                subspace::getTotal(&m_indicesTensor.memoryDims().dims[1], indices_dims.size() - 1);
        for (int32_t i = 0; i != num_of_updates; ++i) {
            const auto indices_coord = idx + i * last_idx_dim;
            std::vector<int32_t> coord(indices_coord, indices_coord + last_idx_dim);
            const auto out_index = std::inner_product(coord.begin(), coord.end(), input_data_dim_pading.begin(), 0);
            const int32_t update_mem_size = update_el_number;
            for (int32_t j = 0; j < update_mem_size; j++) {
                dst[out_index + j] = updates[i * update_el_number + j];
            }
        }
    }

    bool checkResult() override {
        m_outputTensor.confirmBufferData();
        m_referenceTensor.confirmBufferData();

        // save output data
        if (m_save_to_file) {
            saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                "outMyriad.bin");
        }

        bool threshold_test_failed = false;

       if (sMode == F16) {
            m_outputTensor.forEach(true, [&](const MemoryDims& indices) {
                float gt_value = f16Tof32(m_referenceTensor.at(indices, true));
                float value = f16Tof32(m_outputTensor.at(indices, true));

                bool differ = bool(!(gt_value == value));
                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                        printf("DIFF HWC [%d:%d:%d] %f %f\n", ti.height, ti.width, ti.channels, value, gt_value);
                }
            });
        } else {
            m_outputTensor.forEach(true, [&](const MemoryDims& indices) {
                int32_t gt_value = m_referenceTensor.at(indices, true);
                int32_t value = m_outputTensor.at(indices, true);

                bool differ = bool(!(gt_value == value));
                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                        printf("DIFF HWC [%d:%d:%d] %ld %ld\n", ti.height, ti.width, ti.channels, value, gt_value);
                }
            });
        }
        return !threshold_test_failed;
    }

private:
    Tensor<SrcType> m_inputTensor;
    Tensor<SrcType> m_outputTensor;
    Tensor<SrcType> m_referenceTensor;
    Tensor<int32_t> m_indicesTensor;
    Tensor<SrcType> m_updatesTensor;

    ListIterator<ScatterNDUpdateTestParams> m_testsLoop;
    sw_params::ScatterNDUpdateParams* m_scatterNDUpdateParams;
};

}  // namespace

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, ScatterNDUpdateF16)) {
    typedef CustomCppScatterNDUpdateTest<fp16, ScatterMode::F16> scatter_tests;
    ICV_TESTS_REGISTER_SUITE(scatter_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, ScatterNDUpdateI32)) {
    typedef CustomCppScatterNDUpdateTest<int32_t, ScatterMode::I32> scatter_tests;
    ICV_TESTS_REGISTER_SUITE(scatter_tests)
}
