// {% copyright %}

#include <custom_cpp_tests.h>
#include <cmath>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
extern void*(shvNN0_reorder_fp16);
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_reorder.h"

#define REORDER_TEST_NAMESPACE ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Reorder))

namespace REORDER_TEST_NAMESPACE {

using Parent = CustomCppTests<fp16>;

//struct SingleTest {
//    Dimensions inDim;
//    Dimensions outDim;
////    StorageOrder storageOrder;
////    const char* kernelName;
//    CustomParams customLayerParams;
//};

static constexpr std::initializer_list<SingleTest> reorder_test_list{
//        {{1, 1, 20}, {1, 1, 20}, orderZYX, FPE("reorder_fp16.elf"), {sw_params::Location::NN_CMX}},
//        {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("reorder_fp16.elf"), {sw_params::Location::NN_CMX}}
        {{32, 1, 256}, {1, 256, 32}, orderCHW, FPE("reorder_fp16.elf"), {{1, 2, 0, sw_params::Location::NN_CMX}}},
        {{4, 64, 32}, {32, 4, 64}, orderCHW, FPE("reorder_fp16.elf"), {{2, 0, 1, sw_params::Location::NN_CMX}}},
};

class CustomCppReorderTest : public Parent {
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
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            paramContainer.resize(((int)sizeof(sw_params::ReorderParams) + 7) / 8);
            Parent::initData();
            const SingleTest* test = m_currentTest;
//            int32_t ind[subspace::MAX_DIMS] = {0};
//            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);

            checkTestConsistency();

            m_reorderParams = reinterpret_cast<sw_params::ReorderParams*>(paramContainer.data());
            *m_reorderParams = sw_params::ReorderParams();

            const int ndims = m_inputTensor.ndims();

            for (int i = 0; i < ndims; ++i)
                m_reorderParams->perm[i] = test->customLayerParams.layerParams[i];

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer.data());
            m_params.paramDataLen = paramContainer.size() * sizeof(uint64_t);
//            printf("# m_inputTensor.ndims() = %d\n", ndims);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[ndims]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_reorderParams);
        }
    void formatTestParams(char* str, int maxLength) const override
        {
            const auto& id = m_inputTensor.tensorDims();
            const auto& il = m_inputTensor.tensorLimits();
            const auto& od = m_outputTensor.tensorDims();
            const auto& ol = m_outputTensor.tensorLimits();

            snprintf_append(str, maxLength, "%u %u %u (%u %u %u) => %u %u %u (%u %u %u)",
                            id.height, id.width, id.channels, il.height, il.width, il.channels,
                            od.height, od.width, od.channels, ol.height, ol.width, ol.channels);
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
            const auto customData = false;  // m_testLoop.value().customData;

        #ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(&shvNN0_reorder_fp16);
        #else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(reorder_fp16));
        #endif

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
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs)
                {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value, abs_diff);
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

    std::vector<uint64_t> paramContainer;
    sw_params::ReorderParams* m_reorderParams;
};

ICV_TESTS_REGISTER_SUITE(CustomCppReorderTest)

}  // namespace REORDER_TEST_NAMESPACE
