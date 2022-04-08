//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "param_scatter_update.h"
#include "sk.single_shave_scatter_update.3720xx.text.xdat"
#define NAMESPACE ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, ScatterUpdate))

struct ScatterUpdateTest {
    Dims inputDims;
    Dims indicesDims;
    Dims updatesDims;
    Dims outputDims;
    StorageOrder storageOrder;
    CustomParams customLayerParams;
};

static inline StorageOrder maskOrder(StorageOrder fullOrder, int nOrd) {
    return static_cast<StorageOrder>(fullOrder & (0xffffffffu >> ((MAX_DIMS - nOrd) * HEX_DIGIT_BITS)));
}

namespace NAMESPACE {
const bool saveToFile = false;
static constexpr std::initializer_list<ScatterUpdateTest> scatter_update_test_list{
        {{8, 6, 23, 7}, {1}, {8, 6, 23, 1}, {8, 6, 23, 7}, orderNCHW, {0 /*axis*/, sw_params::Location::NN_CMX}},
        {{2, 8}, {3, 4, 5}, {2, 3, 4, 5}, {2, 8}, orderNCHW, {0 /*axis*/, sw_params::Location::NN_CMX}},
        {{5, 4, 12},
         {2, 3},
         {5, 4, 2, 3},
         {5, 4, 12},
         orderNCHW,
         {0 /*axis*/, sw_params::Location::NN_CMX}}};  // inshape, indicesshape, updateshape, outputshape

class CustomCppScatterUpdateTest : public CustomCppTests<fp16, ScatterUpdateTest> {
public:
    explicit CustomCppScatterUpdateTest(): m_testsLoop(scatter_update_test_list, "test") {
    }
    virtual ~CustomCppScatterUpdateTest() {
    }

protected:
    const char* suiteName() const override {
        return "CustomCppScatterUpdateTest";
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
        const Dims& updatesDims = m_currentTest->updatesDims;
        const Dims& outputDims = m_currentTest->outputDims;
        const StorageOrder& storageOrder = m_currentTest->storageOrder;

        int32_t numInputDims = inputDims.size();
        int32_t numIndicesDims = indicesDims.size();
        int32_t numUpdatesDims = updatesDims.size();
        int32_t numOutputDims = outputDims.size();

        const StorageOrder inputOrder = maskOrder(storageOrder, numInputDims);
        const StorageOrder indicesOrder = maskOrder(storageOrder, numIndicesDims);
        const StorageOrder updatesOrder = maskOrder(storageOrder, numUpdatesDims);
        const StorageOrder outputOrder = maskOrder(storageOrder, numOutputDims);

        const MemoryDims inputMemDims(inputDims.begin(), numInputDims);
        const MemoryDims indicesMemDims(indicesDims.begin(), numIndicesDims);
        const MemoryDims updatesMemDims(updatesDims.begin(), numUpdatesDims);
        const MemoryDims outputMemDims(outputDims.begin(), numOutputDims);

        m_inputTensor.init(inputOrder, inputMemDims);
        m_indicesTensor.init(indicesOrder, indicesMemDims);
        m_updatesTensor.init(updatesOrder, updatesMemDims);
        m_outputTensor.init(outputOrder, outputMemDims);
        m_referenceOutputTensor.init(outputOrder, outputMemDims);

        allocBuffer(m_inputTensor);
        allocBuffer(m_indicesTensor);
        allocBuffer(m_updatesTensor);
        allocBuffer(m_outputTensor);
        allocBuffer(m_referenceOutputTensor);

        int32_t ind[subspace::MAX_DIMS] = {0};
        subspace::orderToIndices((t_D8StorageOrder)(m_currentTest->storageOrder), ind);
        m_axis = (int64_t)ind[m_currentTest->customLayerParams.layerParams[0]];

        m_ScatterUpdateParams = reinterpret_cast<sw_params::ScatterUpdateParams*>(paramContainer);
        *m_ScatterUpdateParams = sw_params::ScatterUpdateParams();

        m_ScatterUpdateParams->axis = m_axis;

        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
        m_params.paramDataLen = sizeof(sw_params::ScatterUpdateParams);

        m_requiredTensorLocation = static_cast<sw_params::Location>(m_currentTest->customLayerParams.layerParams[1]);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_ScatterUpdateParams);

        m_params.kernel = reinterpret_cast<uint32_t>(sk_single_shave_scatter_update_3720xx_text);
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.0f;
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

        OpTensor updatesBuff;
        m_updatesTensor.exportToBuffer(updatesBuff);
        customCppOp->addInputBuffer(updatesBuff, m_requiredTensorLocation);

        OpTensor outputBuff;
        m_outputTensor.exportToBuffer(outputBuff);
        customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);

        customCppOp->ops = *getParams();
    }

    void generateInputData() override {
        int32_t numInputDims = m_inputTensor.ndims();
        std::default_random_engine generator(123);

        std::uniform_real_distribution<float> uniformReal(-20, 20);
        m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            float fp32Val = uniformReal(generator);
            m_inputTensor.at(indices) = f32Tof16(fp32Val);
        });

        m_updatesTensor.forEach(false, [&](const MemoryDims& indices) {
            float fp32Val = uniformReal(generator);
            m_updatesTensor.at(indices) = f32Tof16(fp32Val);
        });

        const int maxIndices = m_inputTensor.memoryDims().dims[numInputDims - 1];

        std::uniform_int_distribution<int> uniformInt(0, maxIndices - 1);
        m_indicesTensor.forEach(false, [&](const MemoryDims& indices) {
            m_indicesTensor.at(indices) = uniformInt(generator);
        });
    }

    void generateReferenceData() override {
        mvTensorAssert(m_inputTensor.storageOrder() == m_referenceOutputTensor.storageOrder());

        const int numIndicesDims = m_indicesTensor.ndims();
        const int numInputDims = m_inputTensor.ndims();
        const int numUpdatesDims = m_updatesTensor.ndims();
        const int numOutputDims = m_outputTensor.ndims();
        const int numReferenceDims = m_referenceOutputTensor.ndims();

        mvTensorAssert(numInputDims == numOutputDims);
        mvTensorAssert(numInputDims == numReferenceDims);
        mvTensorAssert(numUpdatesDims == numIndicesDims + numInputDims - 1);

        m_inputTensor.forEach(false, [&](const MemoryDims& i) {
            m_referenceOutputTensor.at(i) = m_inputTensor.at(i);
        });

        m_updatesTensor.forEach(false, [&](const MemoryDims& udims) {
            MemoryDims odims;
            MemoryDims idims(udims.dims + (numUpdatesDims - numIndicesDims), numIndicesDims);

            const int n = m_indicesTensor.at(idims);
            mvTensorAssert(0 <= n && n < m_outputTensor.memoryDims().dims[numOutputDims - 1]);

            odims.dims[numOutputDims - 1] = n;

            for (int i = 0; i < numOutputDims - 1; i++) {
                odims.dims[i] = udims.dims[i];
            }

            m_referenceOutputTensor.at(odims) = m_updatesTensor.at(udims);
        });
    }

    virtual bool checkResult() override {
        m_outputTensor.confirmBufferData();

        if (saveToFile) {
            saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(), "inMyriad.bin");
            saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                             "outMyriad.bin");
            saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                             m_referenceOutputTensor.bufferSize(), "refOutMyriad.bin");
        }

        bool threshold_test_failed = false;

        m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
            float value = f16Tof32(m_outputTensor.at(indices));
            float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
            float abs_diff = fabs(value - gt_value);
            bool differ = !bool(abs_diff <= m_test_threshold);

            threshold_test_failed |= differ;

            if (differ && GlobalData::doPrintDiffs) {
                const TensorDims ti = m_outputTensor.toTensor(indices);
                printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value, abs_diff);
            }
        });

        return !threshold_test_failed;
    }

private:
    ListIterator<ScatterUpdateTest> m_testsLoop;

    Tensor<int32_t> m_indicesTensor;
    Tensor<fp16> m_updatesTensor;

    int64_t m_axis;

    sw_params::ScatterUpdateParams* m_ScatterUpdateParams;
};

// Scatter kernel hangs on HW, need to investigate
#ifdef CONFIG_MOVISIM_RUN
ICV_TESTS_REGISTER_SUITE(CustomCppScatterUpdateTest)
#endif

}  // namespace NAMESPACE
