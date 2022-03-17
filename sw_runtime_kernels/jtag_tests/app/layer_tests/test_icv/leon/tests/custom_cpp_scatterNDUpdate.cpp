//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <custom_cpp_tests.h>
#include <custom_cpp_test_base.h>
#include <cmath>
#include <numeric>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.singleShaveScatterNDUpdate.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_scatterNDUpdate.h"

namespace
{
// #define USE_MANUAL_DATA
const bool save_to_file = false;

struct ScatterNDUpdateTestParams
{
    std::initializer_list<int32_t> inputDims;
    std::initializer_list<int32_t> indicesDims;
};

const std::initializer_list<ScatterNDUpdateTestParams> scatterNDUpdate_test_list =
    {
        //      3D input Tensor
        { {3, 3, 3},       {3, 2, 2} },
        { {8, 3, 3},       {2, 2}    },
        // { {100, 50, 10},   {3, 2, 2} },
        // { {1000, 90, 2},   {3, 2}    },
        // { {1024, 100, 2},  {3, 2}    },
    };

template <class SrcType>
class CustomCppScatterNDUpdateTest : public CustomCppTestBase<ScatterNDUpdateTestParams>
{
 public:
    explicit CustomCppScatterNDUpdateTest(): m_testsLoop(scatterNDUpdate_test_list, "test") {}
    virtual ~CustomCppScatterNDUpdateTest() {}

 protected:
    const char* suiteName() const override
    {
        static const std::string name = std::string{ICV_TESTS_STRINGIFY(ICV_TEST_SUITE_NAME)}
            + "_" + TypeNameTrait<SrcType>::name()
            + "_" + TypeNameTrait<int32_t>::name();
        return name.c_str();
    }

    void userLoops() override
    {
        addLoop(m_testsLoop);
    }

    void calcUpdatesDims(std::vector<int32_t> &inputDims, std::vector<int32_t> &indicesDims, std::vector<int32_t> &updatesDims){
        // considering innermost to outermost way of working with shapes,
        // updates shape must be input.dims[:input.ndims - indices.dims[0]] + indices.dims[1:]

        for(size_t i = 0; i < inputDims.size() - indicesDims[0]; i++)
            updatesDims.push_back(inputDims[i]);

        for(size_t i = 1; i < indicesDims.size(); i++)
            updatesDims.push_back(indicesDims[i]);
    }

    void initData() override
    {
        m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};
        initElfBuffer();
        initTestCase();

        std::vector<int32_t>  inputDims   = m_testsLoop.value().inputDims;
        std::vector<int32_t>  indicesDims = m_testsLoop.value().indicesDims;
        std::vector<int32_t>  updatesDims;

        // construct updates dims
        calcUpdatesDims(inputDims, indicesDims, updatesDims);

        // construct output dims
        const auto& outputDims = inputDims;

        MemoryDims md_input_dims(inputDims.data(), inputDims.size());
        MemoryDims md_indices_dims(indicesDims.data(), indicesDims.size());
        MemoryDims md_updates_dims(updatesDims.data(), updatesDims.size());
        MemoryDims md_output_dims(outputDims.data(), outputDims.size());

        m_inputTensor.init(maskOrder(FULL_ORDER, inputDims.size()), md_input_dims, md_input_dims);
        m_indicesTensor.init(maskOrder(FULL_ORDER, indicesDims.size()), md_indices_dims, md_indices_dims);
        m_updatesTensor.init(maskOrder(FULL_ORDER, updatesDims.size()), md_updates_dims, md_updates_dims);
        m_outputTensor.init(maskOrder(FULL_ORDER, outputDims.size()), md_output_dims, md_output_dims);
        m_referenceTensor.init(maskOrder(FULL_ORDER, outputDims.size()), md_output_dims, md_output_dims);

        allocBuffer(m_inputTensor);
        allocBuffer(m_outputTensor);
        allocBuffer(m_indicesTensor);
        allocBuffer(m_updatesTensor);
        allocBuffer(m_referenceTensor);

        m_scatterNDUpdateParams = reinterpret_cast<sw_params::ScatterNDUpdateParams *>(paramContainer);
        *m_scatterNDUpdateParams = sw_params::ScatterNDUpdateParams();
        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
        m_params.paramDataLen = sizeof(sw_params::ScatterNDUpdateParams);

        m_requiredTensorLocation =
                static_cast<sw_params::Location>(sw_params::Location::NN_CMX);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_scatterNDUpdateParams);

#ifdef CONFIG_TARGET_SOC_3720
        m_params.kernel = reinterpret_cast<uint64_t>(sk_singleShaveScatterNDUpdate_3010xx_text);
#else
        m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(singleShaveScatterNDUpdate));
#endif
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.07f;
    }

    void formatTestParams(char* str, int maxLength) const override
    {
        char inSizes_str[100];
        char outSizes_str[100];
        char idxSizes_str[100];
        char updatesSizes_str[100];

        snprintf_append(str, maxLength, "input: %s, output: %s, indices: %s, updates: %s",
                        m_inputTensor.dimsToStringNCHW(inSizes_str),
                        m_outputTensor.dimsToStringNCHW(outSizes_str),
                        m_indicesTensor.dimsToStringNCHW(idxSizes_str),
                        m_updatesTensor.dimsToStringNCHW(updatesSizes_str));
    }

    void initParserRunner() override
    {
        initMyriadResources();
        initDebugInfo();

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

    void resetOutputData() override
    {
        resetTensorBuffer(m_outputTensor);
    }

    void generateReferenceData() override
    {
        const auto& test_value = m_testsLoop.value();

        const auto idx     = m_indicesTensor.data();
        const auto updates = m_updatesTensor.data();
        const auto src     = m_inputTensor.data();
              auto dst     = m_referenceTensor.data();

        std::vector<int32_t> input_dims(test_value.inputDims.begin(),
                                        test_value.inputDims.end());

        std::vector<int32_t> indices_dims(test_value.indicesDims.begin(),
                                          test_value.indicesDims.end());

        int last_idx_dim = indices_dims.front();
        int update_chunk_size = input_dims.size() - last_idx_dim;
        if(update_chunk_size < 0)
            update_chunk_size = 0;
        const auto update_el_number = subspace::getTotal(m_inputTensor.memoryDims().dims, update_chunk_size);

        for(int i=0; i<subspace::getTotal(m_inputTensor.memoryDims().dims, input_dims.size()); i++)
            dst[i] = src[i];

        const auto input_data_dim_pading = [&] {
            std::vector<size_t> padding(input_dims.size(), 1);
            for (int32_t i = input_dims.size() - 1; i != 0; --i) {
                padding[i - 1] = padding[i] * input_dims[input_dims.size() - 1 - i];
            };
            return padding;
        }();

        const auto num_of_updates = subspace::getTotal(&m_indicesTensor.memoryDims().dims[1], indices_dims.size()-1);
        for (int32_t i = 0; i != num_of_updates; ++i) {
            const auto indices_coord = idx + i * last_idx_dim;
            std::vector<int32_t> coord (indices_coord, indices_coord + last_idx_dim);
            const auto out_index = std::inner_product(coord.begin(), coord.end(), input_data_dim_pading.begin(), 0);
            const int32_t update_mem_size = update_el_number;
            for(int32_t j = 0; j < update_mem_size; j++){
                dst[out_index + j] = updates[i * update_el_number + j];
            }
        }
    }

 protected:
    Tensor<SrcType> m_inputTensor;
    Tensor<SrcType> m_outputTensor;
    Tensor<SrcType> m_referenceTensor;
    Tensor<int32_t> m_indicesTensor;
    Tensor<SrcType> m_updatesTensor;

    ListIterator<ScatterNDUpdateTestParams> m_testsLoop;
    sw_params::ScatterNDUpdateParams *m_scatterNDUpdateParams;

    sw_params::Location m_requiredTensorLocation = sw_params::Location::DDR;

    //  FIXME: Temporarily is located on CMX due to problem of ACT_SHAVE cache invalidation
    uint64_t * paramContainer = cmxParamContainer;
    float m_test_threshold = 0.0f;

    // Debug-specific
    bool m_save_to_file = false;
    int m_num_of_debug_output = 0;
};


class ScatterNDUpdateTestFP16 : public CustomCppScatterNDUpdateTest<fp16> {
    const char *suiteName() const override { return ICV_TESTS_STRINGIFY(ICV_TEST_SUITE_NAME) "ScatterNDUpdate"; }
    void generateInputData() override
    {
#if defined (USE_MANUAL_DATA)
        counter_expected = 0;
        int k=0;
        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
        {
            float val = (float)manual_input_val[k++];
            m_inputTensor.at(indices) = f32Tof16(val);
        });
        k=0;
        m_indicesTensor.forEach(false, [&](const MemoryDims& indices)
        {
            int32_t val = manual_indices_val[k++];
            m_indicesTensor.at(indices) = val;
        });
        k=0;
        m_updatesTensor.forEach(false, [&](const MemoryDims& indices)
        {
            float val = (float)manual_updates_val[k++];
            m_updatesTensor.at(indices) = f32Tof16(val);
        });
#else
        int i = 100;
        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
        {
            if (i > 200.0) i = 100;
            float val = (float)(i++);
            m_inputTensor.at(indices) = f32Tof16(val);
        });
        const auto &test = m_testsLoop.value();
        m_indicesTensor.forEach(false, [&](const MemoryDims& indices)
        {
            const auto dim = test.inputDims.size() - indices.dims[Dim::W] - 1;
            const auto maxValue = test.inputDims.begin()[dim];
            m_indicesTensor.at(indices) = rand() % maxValue;
        });

        i = 10;
        m_updatesTensor.forEach(false, [&](const MemoryDims& indices)
        {
            if (i > 99.0) i = 10;
            float val = (float)(i++);
            m_updatesTensor.at(indices) = f32Tof16(val);
        });
#endif
    }

    bool checkResult() override
    {
        m_outputTensor.confirmBufferData();
        m_referenceTensor.confirmBufferData();

        // save output data
        if (save_to_file)
        {
            saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(), "outMyriad.bin");
        }

        bool threshold_test_failed = false;
        m_outputTensor.forEach(true, [&](const MemoryDims& indices)
        {
#if defined (USE_MANUAL_DATA)
            float gt_value = manual_expected_val[counter_expected++];
#else
            float gt_value = f16Tof32(m_referenceTensor.at(indices, true));
#endif
            float value = f16Tof32(m_outputTensor.at(indices, true));
            float input = f16Tof32(m_inputTensor.at(indices));

            bool differ = bool(!(gt_value == value));

            if (differ || true)
            {
                const TensorDims ti = m_outputTensor.toTensor(indices);
                threshold_test_failed |= differ;
                if (GlobalData::doPrintDiffs || true)
                    printf("DIFF CHW [%" PRId32":%" PRId32":%" PRId32"] %f %f %f\n", ti.channels, ti.height, ti.width, input, value, gt_value);
            }
        });

        return !threshold_test_failed;
    }
};

}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, ScatterNDUpdate)) {
    ICV_TESTS_REGISTER_SUITE(ScatterNDUpdateTestFP16)
}
