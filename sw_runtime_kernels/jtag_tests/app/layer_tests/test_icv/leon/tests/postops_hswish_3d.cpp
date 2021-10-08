// {% copyright %}

//#define ICV_TESTS_CALL_MVTENSOR_ONCE      (1) /* uncomment it only for debug purposes */
//#define ICV_TESTS_GENERATE_DATA_PER_SHAVE (1) /* use old SHAVE loop behaviour, if defined */
//#define ICV_TEST_DO_CHECK_TENSOR_INDICES  (1) /* do check range of tensor indices upon addressing */

#define ICV_TEST_SUITE_NAME Postops_3D_HSwish

#include "icv_test_suite.h"

#include "PostOps.h"

#include <random>

using namespace icv_tests;

namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
{

//#define ALL_DIMENSIONS_SET
//#define ALL_ORDERS_SET
//#define ALL_STRIDES_SET  /* -- ++ */
//#define FULL_STRIDES_SET /* -+ +- */
//#define ALL_PARAMS_SET
#define USE_SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed

const float test_rel_threshold = 0.002f;
const bool save_to_file = false;

struct Dimensions
{
    int width;
    int height;
    int channels;
};

struct Strides
{
    bool hasInput;
    bool hasOutput;
};

const std::initializer_list<Dimensions> dimensions_list =
{
      { 1, 8,    8 },
//    { 464, 286,    8 },
//    {  40,  83,  320 },
//    {  16,  39, 1680 },
  #if defined(ALL_DIMENSIONS_SET)
    { 192, 346,   16 },
    { 136,  88,   88 },
    {  48,  28,  768 },
    {  16,  15, 4200 },
  #endif
};

const std::initializer_list<t_MvTensorStorageOrder> orders_list =
{
    orderNYXZ,
  #if defined(ALL_ORDERS_SET)
    orderNYZX,
    orderNYXZ,
    orderNZYX,
  #endif
};

const std::initializer_list<Strides> strides_list =
{
    { false, false },
  #if defined(FULL_STRIDES_SET)
    { false, true  },
    { true,  false },
  #endif
  #if defined(ALL_STRIDES_SET) || defined(FULL_STRIDES_SET)
    { true,  true  },
  #endif
};

class HSwishTests: public TestSuite
{
public:
    explicit HSwishTests()
        : m_dimensionsLoop(dimensions_list, "dim")
        , m_ordersLoop(orders_list, "order")
        , m_stridesLoop(strides_list, "stride")
        {}
    virtual ~HSwishTests()
        {}
protected:
    const char* suiteName() const override
        { return ICV_TESTS_STRINGIFY(ICV_TEST_SUITE_NAME); }
    void userLoops() override
        {
            addLoop(m_dimensionsLoop);
            addLoop(m_ordersLoop);
            addLoop(m_stridesLoop);
        }
    void initData() override
        {
            const auto& dim = m_dimensionsLoop.value();
            const auto storageOrder = m_ordersLoop.value();
            const auto& strides = m_stridesLoop.value();

            TensorDims dims3(dim.width, dim.height, dim.channels, 1);
            TensorAlign align0(0, 0, 0, 0);
            TensorAlign align3input((strides.hasInput ? 16 : 0), 0, 0, 0);
            TensorAlign align3output((strides.hasOutput ? 16 : 0), 0, 0, 0);

            m_inputTensor.init(storageOrder, dims3, align3input);
            m_outputTensor.init(storageOrder, dims3, align3output);
            m_referenceOutput.init(storageOrder, dims3, align0);

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
        { return kHSwish; }
    void initParserRunner() override
        {
            initMyriadResources();
            initDebugInfo();

            PostOps* postOp = static_cast<PostOps*>(m_op);

            m_inputTensor.exportToBuffer(postOp->input);
            m_outputTensor.exportToBuffer(postOp->output);

            postOp->hasWeights = false;
            postOp->hasBiases = false;

            postOp->forceKernel = PostOps::Force3D;
        }
    void generateData() override
        {
#if defined(USE_SEED_VALUE)
            auto seedValue = USE_SEED_VALUE;
#else
            u64 systemTicks;
            DrvTimerGetSystemTicks64(&systemTicks);
            auto seedValue = static_cast<unsigned int>(systemTicks);
#endif
            std::mt19937 generator(seedValue);
            m_inputTensor.forEach(false, [this, &generator](const MemoryDims& indices)
            {
                // We generate the random value between -8.f and 8f and the kernel do x * relu6(x+3) / 6
                // So the minimum resolution is 2^(-7) = 0.00781f and the kernel may calculate 0 output value
                // if input value is less than this resolution. In such cases, relative difference would be 1.
                const float precisionLimitations = 0.00781f;
                float fp32Value = 0.f;
                do {
                    fp32Value = float(generator()) / generator.max() * 16.f - 8.f;
                } while (fabs(fp32Value) < precisionLimitations && fp32Value != 0.f);
                m_inputTensor.at(indices) = f32Tof16(fp32Value);
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
            float max_abs_diff = 0.0;
            float max_rel_diff = 0.0;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutput.at(indices));

                float abs_diff = fabs(value - gt_value);
                float rel_diff = gt_value != 0.0 ? fabs(abs_diff / gt_value) : abs_diff;
                max_abs_diff = std::max(max_abs_diff, abs_diff);
                max_rel_diff = std::max(max_rel_diff, rel_diff);

                float abs_threshold = (fabs(gt_value) * test_rel_threshold);
                bool differ = bool(!(abs_diff <= abs_threshold));

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs)
                {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f abs_diff: %f rel_diff: %f\n",
                           ti.height, ti.width, ti.channels, value, gt_value, abs_diff, rel_diff);
                }
            });

            if (GlobalData::doPrintDiffMax)
            {
                printf("MAX DIFF ABS=%f REL=%f\n", max_abs_diff, max_rel_diff);
            }

            return !threshold_test_failed;
        }
    void calcReferenceOutput()
        {
            // no need to remap memory indices between tensors
            mvTensorAssert(m_inputTensor.storageOrder() == m_referenceOutput.storageOrder());

            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = val * std::min(6.f, std::max(0.f, val + 3.f)) / 6.f;
                m_referenceOutput.at(indices) = f32Tof16(ref);
            });
        }
protected:
    ListIterator<Dimensions> m_dimensionsLoop;
    ListIterator<t_MvTensorStorageOrder> m_ordersLoop;
    ListIterator<Strides> m_stridesLoop;
    Tensor<fp16> m_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceOutput;
};

//ICV_TESTS_REGISTER_SUITE(HSwishTests)

} // namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
