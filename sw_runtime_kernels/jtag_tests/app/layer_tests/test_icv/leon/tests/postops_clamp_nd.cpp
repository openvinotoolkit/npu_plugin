// {% copyright %}

//#define ICV_TESTS_CALL_MVTENSOR_ONCE      (1) /* uncomment it only for debug purposes */
//#define ICV_TESTS_GENERATE_DATA_PER_SHAVE (1) /* use old SHAVE loop behaviour, if defined */
//#define ICV_TEST_DO_CHECK_TENSOR_INDICES  (1) /* do check range of tensor indices upon addressing */

#define ICV_TEST_SUITE_NAME Postops_ND_Clamp

#include "icv_test_suite.h"

#include "PostOps.h"

using namespace icv_tests;

namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
{

//#define ALL_DIMENSIONS_SET
//#define ALL_ORDERS_SET
//#define ALL_STRIDES_SET
//#define ALL_PARAMS_SET

const float test_threshold = 0;
const bool save_to_file = false;

typedef t_ClampLayerParams Params;

typedef std::initializer_list<int32_t> Dims;
typedef std::initializer_list<int32_t> Gaps;
typedef t_MvTensorStorageOrder StorageOrder;

struct Geometry
{
    StorageOrder order;
    Dims         dims;
};

const std::initializer_list<Geometry> geometries_list =
{
#ifdef CONFIG_TARGET_SOC_3720

    { 0x00004321, { 5, 2, 3, 4 }},
    { 0x00054321, { 2, 3, 4, 5, 6 }},

#else // CONFIG_TARGET_SOC_3720

    { 0x1, { 500000 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x21, { 500000, 1 }},
    { 0x321, { 100000, 3, 5 }},
  #endif
    { 0x4321, { 50000, 3, 5, 2 }},

    { 0x00654321, { 6, 7, 2, 3, 4, 5 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00654321, { 2, 3, 4, 5, 6, 7 }},
    { 0x00654321, { 3, 4, 5, 6, 7, 2 }},
    { 0x00654321, { 4, 5, 6, 7, 2, 3 }},
    { 0x00654321, { 5, 6, 7, 2, 3, 4 }},
    { 0x00654321, { 7, 2, 3, 4, 5, 6 }},
    { 0x00654321, { 7, 6, 2, 3, 4, 5 }},
    { 0x00654321, { 7, 6, 5, 2, 3, 4 }},
    { 0x00654321, { 7, 6, 5, 4, 2, 3 }},
    { 0x00654321, { 7, 6, 5, 4, 3, 2 }},
  #endif

    { 0x00054321, { 4, 5, 6, 3, 2 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00054321, { 2, 3, 4, 5, 6 }},
    { 0x00054321, { 3, 4, 5, 6, 2 }},
    { 0x00054321, { 5, 6, 4, 3, 2 }},
    { 0x00054321, { 6, 5, 4, 3, 2 }},
  #endif

    { 0x00004321, { 2, 3, 4, 5 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00004321, { 3, 4, 5, 2 }},
    { 0x00004321, { 4, 5, 2, 3 }},
    { 0x00004321, { 5, 2, 3, 4 }},
  #endif

  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000321, {  16,  15, 4200 }},
  #endif
    { 0x00000321, {  16,  39, 1680 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000321, {  48,  28,  768 }},
  #endif
    { 0x00000321, {  40,  83,  320 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000321, { 136,  88,   88 }},
    { 0x00000321, { 192, 346,   16 }},
  #endif
    { 0x00000321, { 464, 286,    8 }},

#if defined(ALL_ORDERS_SET)
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000231, {  16,  15, 4200 }},
  #endif
    { 0x00000231, {  16,  39, 1680 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000231, {  48,  28,  768 }},
  #endif
    { 0x00000231, {  40,  83,  320 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000231, { 136,  88,   88 }},
    { 0x00000231, { 192, 346,   16 }},
  #endif
    { 0x00000231, { 464, 286,    8 }},

  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000213, {  16,  15, 4200 }},
  #endif
    { 0x00000213, {  16,  39, 1680 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000213, {  48,  28,  768 }},
  #endif
    { 0x00000213, {  40,  83,  320 }},
  #if defined(ALL_DIMENSIONS_SET)
    { 0x00000213, { 136,  88,   88 }},
    { 0x00000213, { 192, 346,   16 }},
  #endif
    { 0x00000213, { 464, 286,    8 }},
#endif // ALL_ORDERS_SET

#endif // CONFIG_TARGET_SOC_3720
};

const std::initializer_list<Gaps> gaps_list =
{
    {0, 0, 0, 0, 0, 0, 0, 0},

#ifdef ALL_STRIDES_SET
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 4, 2, 3, 1, 3, 5, 1},
    {2, 3, 4, 2, 3, 5, 1, 1},
    {1, 2, 3, 2, 1, 3, 2, 1},
#endif
};

const std::initializer_list<Params> params_list =
{
  #if defined(ALL_PARAMS_SET)
    { -10.0f, 5.0f },
  #endif
    { -5.0f, 10.0f },
};

class Tests: public TestSuite
{
public:
    explicit Tests()
        : m_geometriesLoop(geometries_list, "dim")
        , m_inputGaps(gaps_list, "istride")
        , m_outputGaps(gaps_list, "ostride")
        , m_paramsLoop(params_list, "param")
        , m_params(nullptr)
        {}
    virtual ~Tests()
        { m_params = nullptr; }
protected:
    const char* suiteName() const override
        { return ICV_TESTS_STRINGIFY(ICV_TEST_SUITE_NAME); }
    void userLoops() override
        {
            addLoop(m_geometriesLoop);
            addLoop(m_inputGaps);
            addLoop(m_outputGaps);
            addLoop(m_paramsLoop);
        }
    void initData() override
        {
            const auto& geometry = m_geometriesLoop.value();
            const auto& dims = geometry.dims;
            const auto storageOrder = geometry.order;

            const int ndims = dims.size();
            MemoryDims mdims(dims.begin(), ndims);
            MemoryDims inputGap(m_inputGaps.value().begin(), ndims);
            MemoryDims outputGap(m_outputGaps.value().begin(), ndims);
            MemoryDims inputLimits = mdims + inputGap;
            MemoryDims outputLimits = mdims + outputGap;

            m_inputTensor.init(storageOrder, mdims, inputLimits);
            m_outputTensor.init(storageOrder, mdims, outputLimits);
            m_referenceOutput.init(storageOrder, mdims, outputLimits);

            m_params = allocData<Params>();
            allocBuffer(m_inputTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutput);
        }
    void formatTestParams(char* str, int maxLength) const override
        {
            char input_str[64], output_str[64], layout_str[64];
            snprintf_append(str, maxLength, "%s => %s, %s",
                            m_inputTensor.dimsToStringNCHW(input_str),
                            m_outputTensor.dimsToStringNCHW(output_str),
                            layoutString8(m_inputTensor.storageOrder(), layout_str));
        }
    t_MvTensorOpType opType() const override
        { return kClamp; }
    void initParserRunner() override
        {
            initMyriadResources();
            initDebugInfo();

            PostOps* postOp = static_cast<PostOps*>(m_op);

            m_inputTensor.exportToBuffer(postOp->input);
            m_outputTensor.exportToBuffer(postOp->output);

            postOp->hasWeights = false;
            postOp->hasBiases = false;

            postOp->clampParams = m_paramsLoop.value();

            postOp->forceKernel = PostOps::ForceND;

#ifdef CONFIG_TARGET_SOC_3720
            postOp->executeInTestingSystem = true;
#endif // CONFIG_TARGET_SOC_3720
        }
    void generateData() override
        {
            // params
            *m_params = m_paramsLoop.value();

            // input
            resetTensorBuffer(m_referenceOutput, 0xab);
            resetTensorBuffer(m_inputTensor, 0xac);

            const int ndims = m_inputTensor.ndims();
            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                int tmp = 1;
                for (int i = 0; i < ndims; ++i)
                    tmp *= (3 + (indices.dims[i] % 13));
                int tmp2 = (tmp % 33) - 14;
                fp16 val = f32Tof16((float)tmp2);

                m_inputTensor.at(indices) = val;
            });

            // reference output
            calcReferenceOutput();
        }
    void resetOutputData() override
        { resetTensorBuffer(m_outputTensor); }
    bool checkResult() override
        {
            m_outputTensor.confirmBufferData();

            // save output data
            if (save_to_file)
            {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(), "outMyriad.bin");
            }

            // no need to remap memory indices between tensors
            mvTensorAssert(m_outputTensor.storageOrder() == m_inputTensor.storageOrder());
            mvTensorAssert(m_outputTensor.storageOrder() == m_referenceOutput.storageOrder());

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutput.at(indices));

                float abs_diff = fabs(value - gt_value);
                bool differ = bool(!(abs_diff <= test_threshold));

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs)
                {
                    char indices_str[64];
                    printf("DIFF [%s] %f %f %f %f\n",
                           m_outputTensor.indicesToString(indices, indices_str),
                           f16Tof32(m_inputTensor.at(indices)), value, gt_value, abs_diff);
                }
            });

            return !threshold_test_failed;
        }
    void calcReferenceOutput()
        {
            // no need to remap memory indices between tensors
            mvTensorAssert(m_inputTensor.storageOrder() == m_referenceOutput.storageOrder());

            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = std::min(m_params->max, std::max(m_params->min, val));
                m_referenceOutput.at(indices) = f32Tof16(ref);
            });
        }
protected:
    ListIterator<Geometry> m_geometriesLoop;
    ListIterator<Gaps> m_inputGaps;
    ListIterator<Gaps> m_outputGaps;
    ListIterator<Params> m_paramsLoop;
    Tensor<fp16> m_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceOutput;
    Params* m_params;
};

ICV_TESTS_REGISTER_SUITE(Tests)

} // namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
