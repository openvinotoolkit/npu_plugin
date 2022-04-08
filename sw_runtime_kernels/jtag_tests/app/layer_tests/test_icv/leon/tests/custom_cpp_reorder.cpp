//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <cmath>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.reorder_fp16.3720xx.text.xdat"

#include "param_reorder.h"

#define REORDER_TEST_NAMESPACE ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Reorder))

namespace REORDER_TEST_NAMESPACE {

using Parent = CustomCppTests<fp16>;

static constexpr std::initializer_list<SingleTest> reorder_test_list
{
    {{2, 3, 1}, {3, 2, 1}, orderCHW, {{1, 0, 2, sw_params::Location::NN_CMX}}},
    {{19, 1, 16}, {1, 16, 19}, orderCHW, {{1, 2, 0, sw_params::Location::NN_CMX}}},
    {{4, 6, 19}, {19, 4, 6}, orderCHW, {{2, 0, 1, sw_params::Location::NN_CMX}}},
#ifdef CONFIG_RUN_LARGE_TESTS
    {{32, 1, 256}, {1, 256, 32}, orderCHW, {{1, 2, 0, sw_params::Location::NN_CMX}}},
#endif
    {{4, 64, 32}, {32, 4, 64}, orderCHW, {{2, 0, 1, sw_params::Location::NN_CMX}}},
};

class CustomCppReorderTest : public Parent
{
public:
    explicit CustomCppReorderTest(): m_testsLoop(reorder_test_list, "test")
        {}
    virtual ~CustomCppReorderTest()
        {}
protected:
    const char* suiteName() const override
        { return "CustomCppReorderTest"; }
    void userLoops() override
        { addLoop(m_testsLoop); }
    void initData() override
        {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            Parent::initData();
            const SingleTest* test = m_currentTest;

            checkTestConsistency();

            m_reorderParams = reinterpret_cast<sw_params::ReorderParams*>(paramContainer);
            *m_reorderParams = sw_params::ReorderParams();

            const int ndims = m_inputTensor.ndims();

            for (int i = 0; i < ndims; ++i)
                m_reorderParams->perm[i] = (int64_t)test->customLayerParams.layerParams[i];

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::ReorderParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[ndims]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_reorderParams);
        }
    void formatTestParams(char* str, int maxLength) const override
        {
            const auto& id = m_inputTensor.tensorDims();
            const auto& il = m_inputTensor.tensorLimits();
            const auto& od = m_outputTensor.tensorDims();
            const auto& ol = m_outputTensor.tensorLimits();

            snprintf_append(str, maxLength, "%ux%ux%u (%ux%ux%u) => %ux%ux%u (%ux%ux%u)",
                            id.channels, id.height, id.width, il.channels, il.height, il.width,
                            od.channels, od.height, od.width, ol.channels, ol.height, ol.width);
        }
    void initTestCase() override
        {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.0005f;

            //    const StorageOrder& storageOrder = m_currentTest->storageOrder;
            //    const auto& dimIn = m_currentTest->inDim;
            //    const TensorDims dims3In(dimIn.width, dimIn.height, dimIn.channels, 1);
            //    m_inputTensor.init(storageOrder, dims3In);
            //    allocBuffer(m_inputTensor);
        }
    void generateInputData() override
        {
            m_params.kernel = reinterpret_cast<uint32_t>(sk_reorder_fp16_3720xx_text);

            rand_seed();

            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }
    void generateReferenceData() override
        {
            const SingleTest* test = m_currentTest;
            const int ndims = m_inputTensor.ndims();
            m_inputTensor.forEach(false, [&](const MemoryDims& in)
            {
                MemoryDims out;
                permuteArray(in.dims, test->customLayerParams.layerParams, out.dims, ndims);
                m_referenceOutputTensor.at(out) = m_inputTensor.at(in);
            });
        }
    bool checkResult() override
        {
            m_outputTensor.confirmBufferData();

            //            // save output data
            //            if (save_to_file)
            //            {
            //                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()),
            //                m_outputTensor.bufferSize(), "outMyriad.bin");
            //            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                const float value = f16Tof32(m_outputTensor.at(indices));
                const float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                const float abs_diff = fabs(value - gt_value);
                const bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs)
                {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.channels, ti.height, ti.width, value, gt_value, abs_diff);
                }
            });

            return !threshold_test_failed;
        }
private:
    void checkTestConsistency()
        {
            const SingleTest* test = m_currentTest;
            const int ndims = m_inputTensor.ndims();

            const auto inDims = m_inputTensor.memoryDims();
            const auto outDims = m_outputTensor.memoryDims();

            int32_t tmpDims[MAX_ND_DIMS] = {};
            permuteArray(inDims.dims, test->customLayerParams.layerParams, tmpDims, ndims);

            for (int i = 0; i < ndims; ++i)
                mvTensorAssert(outDims.dims[i] == tmpDims[i], "Reorder test: dims/permutation mismatch");
        }
private:
    ListIterator<SingleTest> m_testsLoop;

    sw_params::ReorderParams* m_reorderParams;
};

ICV_TESTS_REGISTER_SUITE(CustomCppReorderTest)

}  // namespace REORDER_TEST_NAMESPACE
