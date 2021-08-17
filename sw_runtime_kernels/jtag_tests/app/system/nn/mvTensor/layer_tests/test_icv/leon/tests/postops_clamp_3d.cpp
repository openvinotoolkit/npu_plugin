// {% copyright %}

//#define ICV_TESTS_CALL_MVTENSOR_ONCE      (1) /* uncomment it only for debug purposes */
//#define ICV_TESTS_GENERATE_DATA_PER_SHAVE (1) /* use old SHAVE loop behaviour, if defined */
//#define ICV_TEST_DO_CHECK_TENSOR_INDICES  (1) /* do check range of tensor indices upon addressing */

#define ICV_TEST_SUITE_NAME Postops_3D_Clamp

#include "icv_test_suite.h"

#include "PostOps.h"

using namespace icv_tests;

namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
{

//#define ALL_DIMENSIONS_SET
//#define ALL_ORDERS_SET
//#define ALL_STRIDES_SET  /* -- ++ */
//#define FULL_STRIDES_SET /* -+ +- */
//#define ALL_PARAMS_SET

const float test_threshold = 0;
const bool save_to_file = false;

typedef t_ClampLayerParams Params;

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
#ifdef CONFIG_TARGET_SOC_3720

    {5,      5,    3 },
    {3,      4,    7 },

#else // CONFIG_TARGET_SOC_3720

    {5,      5,    3 },
    { 128, 128,    8 },
    { 136,  88,   88 },
  #if defined(ALL_DIMENSIONS_SET)
    { 192, 346,   16 },
    { 136,  88,   88 },
    {  40,  83,  320 },
    {  48,  28,  768 },
    {  16,  39, 1680 },
    {  16,  15, 4200 },
  #endif

#endif // CONFIG_TARGET_SOC_3720
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
        : m_dimensionsLoop(dimensions_list, "dim")
        , m_ordersLoop(orders_list, "order")
        , m_stridesLoop(strides_list, "stride")
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
            addLoop(m_dimensionsLoop);
            addLoop(m_ordersLoop);
            addLoop(m_stridesLoop);
            addLoop(m_paramsLoop);
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

            m_params = allocData<Params>();
            allocBuffer(m_inputTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutput);
        }
    void formatTestParams(char* str, int maxLength) const override
        {
            const auto storageOrder = m_ordersLoop.value();
            const auto& strides = m_stridesLoop.value();

            const auto& id = m_inputTensor.tensorDims();
            const auto& il = m_inputTensor.tensorLimits();
            const auto& ol = m_outputTensor.tensorLimits();

            const char* layout_text = layoutString(storageOrder);
            const char* stride_text = strideString(strides.hasInput, strides.hasOutput);

            snprintf_append(str, maxLength, "H W C = %u %u %u (%u %u %u => %u %u %u), %s, %s (%f %f)",
                            id.height, id.width, id.channels,
                            il.height, il.width, il.channels, ol.height, ol.width, ol.channels,
                            layout_text, stride_text,
                            m_params->min, m_params->max);
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

            postOp->forceKernel = PostOps::Force3D;

#ifdef CONFIG_TARGET_SOC_3720
            postOp->executeInTestingSystem = true;
#endif // CONFIG_TARGET_SOC_3720
        }
    void generateData() override
        {
            // params
            *m_params = m_paramsLoop.value();

            // input
            auto magic = [](int c, int h, int w) { return (1 + (c % 11)) * (1 + (h % 13)) * (1 + (w % 17)); };

            const auto il = m_inputTensor.tensorLimits();
            const float scale = 20.0f / magic(il.channels/2, il.height/2, il.width/2);
            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                const TensorDims ti = m_inputTensor.toTensor(indices);

                int index = m_inputTensor.index(indices);

                int cval = ti.channels - il.channels/2;
                int wval = ti.width    - il.width/2;
                int hval = ti.height   - il.height/2;

                float tmp = scale * magic(cval, hval, wval);
                if (index % 3) tmp = -tmp;
                fp16 val = f32Tof16(tmp);

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
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value, abs_diff);
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
    ListIterator<Dimensions> m_dimensionsLoop;
    ListIterator<t_MvTensorStorageOrder> m_ordersLoop;
    ListIterator<Strides> m_stridesLoop;
    ListIterator<Params> m_paramsLoop;
    Tensor<fp16> m_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceOutput;
    Params* m_params;
};

ICV_TESTS_REGISTER_SUITE(Tests)

} // namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
