// {% copyright %}

#include "icv_test_suite.h"

#include "Permute.h"

using namespace icv_tests;

//#define ICV_TESTS_CALL_MVTENSOR_ONCE      (1) /* uncomment it only for debug purposes */
//#define ICV_TESTS_GENERATE_DATA_PER_SHAVE (1) /* use old SHAVE loop behaviour, if defined */
//#define ICV_TEST_DO_CHECK_TENSOR_INDICES  (1) /* do check range of tensor indices upon addressing */

namespace
{

#define ICV_TEST_SUITE_NAME Permute

#ifndef CONFIG_TARGET_SOC_3720
#define ALL_STRIDES_SET
#endif

//#define ALL_TESTS

const bool save_to_file = false;
const char * outputName = "outMyriad.bin";

typedef std::initializer_list<int32_t> Strides;
typedef std::initializer_list<int32_t> LogicalDims;
typedef std::initializer_list<int32_t> LogicalPerm;

struct Permutation
{
    LogicalDims dims;
    LogicalPerm perm;
    uint32_t inOrder;
    uint32_t outOrder;
    bool allow_permute_nd;
    const char * description;
};

const std::initializer_list<Permutation> test_list =
{
    {{ 3,  4,  5,    1}, { 1, 0,  2, 3}, 0x4321, 0x4321, true, ""},
#ifndef CONFIG_TARGET_SOC_3720
    {{ 16,  1,    1000, 1}, { 0, 1, 2, 3}, 0x4321, 0x4213, true, "Convert order NCHW to NHWC"},
    {{ 1,   2000, 32,   1}, { 0, 1, 2, 3}, 0x4213, 0x4321, true, "Convert order  NHWC to NCHW"},

    {{ 64,  128,  3,    1}, { 1, 0,  2, 3}, 0x4213, 0x4213, true, ""},
    {{ 64,  128,  3,    1}, { 1, 0,  2, 3}, 0x4231, 0x4231, true, ""},

    {{ 64,  128,  3,    1}, { 1, 0,  2, 3}, 0x4321, 0x4321, false, ""},
    {{ 64,  128,  3,    1}, { 1, 0,  2, 3}, 0x4321, 0x4321, false, ""},

    {{ 64,  128,  3,    1}, { 1, 0,  2, 3}, 0x4321, 0x4321, true, ""},
    {{ 256, 64,   3,    1}, { 0, 1,  2, 3}, 0x4321, 0x4321, true, ""},
    {{ 256, 128,  3,    1}, { 2, 0,  1, 3}, 0x4321, 0x4321, true, ""},
    {{ 16,  1,    1000, 1}, { 2, 0,  1, 3}, 0x4321, 0x4321, true, "NCHW -> NHWC"},
    {{ 1,   2000, 32,   1}, { 1, 2,  0, 3}, 0x4213, 0x4213, true, "NHWC -> NCHW"},

    {{ 1,   2000, 32,   1}, { 1, 2,  0, 3}, 0x4213, 0x4213, false, ""},
    {{ 64,  128,  3,    1}, { 1, 0,  2, 3}, 0x4321, 0x4321, false, ""},
    {{ 256, 64,   3,    1}, { 0, 1,  2, 3}, 0x4321, 0x4321, false, ""},
    {{ 256, 128,  3,    1}, { 2, 0,  1, 3}, 0x4321, 0x4321, false, ""},
    {{ 16,  1,    1000, 1}, { 2, 0,  1, 3}, 0x4321, 0x4321, false, "NCHW -> NHWC"},

#ifdef ALL_TESTS
    {{ 256, 1,    512,  1},  { 1, 0,  3, 2}, 0x4321, 0x4321, true,   ""},
    {{ 7,  7,    153600, 1}, { 2, 0,  1, 3}, 0x4321, 0x4321, true,  "NCHW -> NHWC Rfcn"},
    {{ 153600,  7,   7, 1},  { 2, 0,  1, 3}, 0x4321, 0x4321, true,  "NCHW -> NHWC Rfcn"},
    {{ 7,  7,    153601, 1}, { 2, 0,  1, 3}, 0x4321, 0x4321, true,  "NCHW -> NHWC Rfcn"},
    {{ 153601,  7, 7, 1},    { 0, 2,  1, 3}, 0x4321, 0x4321, true,  "NCHW -> NHWC Rfcn"},

    {{ 7,  7,    26000, 1},  { 2, 0,  1, 3}, 0x4321, 0x4321, true,  "NCHW -> NHWC Rfcn"},

// To enable after Nd tensors will be enabled
//    {{ 33,  45,  20,  27, 10},  { 1, 0,  3, 4, 2}, 0x54213, 0x54213, true, "5D"},
//    {{ 33,  45,  20,  27, 10},  { 4, 2,  1, 0, 3}, 0x54213, 0x54213, true, "5D"},
#endif
#endif
};

const std::initializer_list<Strides> strides_list =
{
    {0, 0, 0, 0, 0, 0, 0, 0},
#ifdef ALL_STRIDES_SET
    {1, 2, 3, 2, 1, 3, 2, 1},
#endif  //ALL_STRIDES_SET
};

template<class T>
struct TypeConverter { using type = T; };

template<class T>
T write(typename TypeConverter<T>::type src);

template<class T>
typename TypeConverter<T>::type read(T src);

template<class T>
class PermuteTests: public TestSuite
{
public:
    explicit PermuteTests()
        : m_testsLoop(test_list, "test")
        , m_stridesLoop(strides_list, "strides")
        {}
    virtual ~PermuteTests()
        {}
protected:
    const char* suiteName() const override
    {
        static const auto name = std::string{ICV_TESTS_STRINGIFY(ICV_TEST_SUITE_NAME)}
                                 + "_" + TypeNameTrait<T>::name();
        return name.c_str();
    }
    void userLoops() override
    {
        addLoop(m_testsLoop);
        addLoop(m_stridesLoop);
    }
    void initData() override
    {
        const auto& test = m_testsLoop.value();
        m_dims = test.dims.size();
        uint32_t inOrder = test.inOrder;
        uint32_t outOrder = test.outOrder;
        MemoryDims md_inDims(test.dims.begin(), m_dims);
        int32_t outTensorDims[MaxTensorDims];
        subspace::permuteArray(test.dims.begin(), test.perm.begin(), outTensorDims, m_dims);

        const MemoryDims strides(&*m_stridesLoop.value().begin(), m_dims);
        m_hasStrides = strides.isNonZero(m_dims);

        MemoryDims md_outDims(outTensorDims, m_dims);
        MemoryDims md_limits;
        md_limits = md_inDims + strides;

        TensorDims td_inDims  = md_inDims.toTensor(m_dims, the_same);
        TensorDims td_inLimits = md_limits.toTensor(m_dims, the_same);
        m_inputTensor.init (inOrder, td_inDims, td_inLimits);

        md_limits = md_outDims + strides;
        TensorDims td_outDims  = md_outDims.toTensor(m_dims, the_same);
        TensorDims td_outLimits = md_limits.toTensor(m_dims, the_same);
        m_outputTensor.init(outOrder, td_outDims, td_outLimits);

        auto perm = test.perm.begin();
        for (int i = 0; i < m_dims; i++) {
            m_logicalPermutation[i] = perm[i];
        }

        int32_t tmp[MaxTensorDims];
        int32_t orderPerm[MaxTensorDims];
        int32_t orderInd[MaxTensorDims];
        subspace::orderToPermutation(outOrder, orderPerm);
        subspace::orderToIndices(inOrder, orderInd);
        /*
        orderPerm - (Op) permutation from logicalDims to memoryDims (logicalDims --> (orderPerm) --> memoryDims)
                    or memoryDims = Op(logicalDims)
        orderInd  - (Oi) inverse permutation from logicalDims to memoryDims i.e.
                    permutation from memoryDims to logicalDims (memoryDims --> (orderInd) --> logicalDims)
                    or logicalDims = Oi(memoryDims)
        m_logicalPermutation - (P) permutation of logical dims (logOutDims = P(logInDims)):
        m_storagePermutation - (Ps) corresponding permutation of memory dims (memOutDims = Ps(memInDims)),
        then:
        logOutDims = P(logInDims)) ->
        Oi(memOutDims) = P(Oi(memInDims)) -> applying Op permutation (inverse for Oi) to both equality sides ->
        memOutDims = Op(P(Oi(memInDims)))
        then, storagePermutation can be found as:
        Ps = Op(P(Oi))
        */
        subspace::permuteArray(orderInd, m_logicalPermutation, tmp, m_dims);
        subspace::permuteArray(tmp, orderPerm, m_storagePermutation, m_dims);
        m_allowPermuteND = test.allow_permute_nd;

        allocBuffer(m_inputTensor);
        allocBuffer(m_outputTensor);
    }
    void formatTestParams(char* str, int maxLength) const override
    {
        const char* stride_text = (m_hasStrides ? "strides" : "no strides");
        const char* delimiter = ",";
        const char* curDelimiter = "";
        MemoryDims td_Dims  = m_inputTensor.tensorDims().toMemory(m_dims, the_same);
        MemoryDims td_Steps  = m_inputTensor.tensorSteps().toMemory(m_dims, the_same);
        snprintf_append(str, maxLength, "Logical: {");
        for (int i = 0; i < m_dims; i++) {
            snprintf_append(str, maxLength, "%s%d(%d)", curDelimiter, td_Dims.dims[i], td_Steps.dims[i]);
            curDelimiter = delimiter;
        }
        td_Dims  = m_outputTensor.tensorDims().toMemory(m_dims, the_same);
        td_Steps  = m_outputTensor.tensorSteps().toMemory(m_dims, the_same);
        snprintf_append(str, maxLength, "} 0x%lx to {", m_inputTensor.storageOrder());
        curDelimiter = "";
        for (int i = 0; i < m_dims; i++) {
            snprintf_append(str, maxLength, "%s%d(%d)", curDelimiter, td_Dims.dims[i], td_Steps.dims[i]);
            curDelimiter = delimiter;
        }
        snprintf_append(str, maxLength, "} 0x%lx by {", m_outputTensor.storageOrder());
        curDelimiter = "";
        for (int i = 0; i < m_dims; i++) {
            snprintf_append(str, maxLength, "%s%d", curDelimiter, m_logicalPermutation[i]);
            curDelimiter = delimiter;
        }
        snprintf_append(str, maxLength, "}, %s", stride_text);
    }
    t_MvTensorOpType opType() const override
    {
        return kPermute;
    }
    void initParserRunner() override
    {
        initMyriadResources();
        initDebugInfo();

        Permute * PermOp = static_cast<Permute*>(m_op);
        m_inputTensor.exportToBuffer(PermOp->input);
        m_outputTensor.exportToBuffer(PermOp->output);
        for (int i = 0; i < m_dims; i++) {
            PermOp->ops.order[i] = m_logicalPermutation[i];
        }
        PermOp->ops.allow_permute_nd = m_allowPermuteND;
#ifdef CONFIG_TARGET_SOC_3720
        PermOp->executeInTestingSystem = true;
#endif
    }
    void generateData() override
    {
        int i = 0;
        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
        {
            m_inputTensor.at(indices) = write<T>(i++);
        });
    }
    void resetOutputData() override
    {
        resetTensorBuffer(m_outputTensor);
    }
    bool checkResult() override
    {
        m_outputTensor.confirmBufferData();

        // save output data
        if (save_to_file)
        {
            saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()),
                             m_outputTensor.bufferSize(), outputName);
        }

        bool threshold_test_failed = false;

        MemoryDims firstErrIndices;
        int countErr = 0;
        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
        {
            const auto gt_value = m_inputTensor.at(indices);
            const auto value = getPermutedOutput(indices);

            bool differ = bool(!(gt_value == value));
            if(!threshold_test_failed && differ)
            {
                firstErrIndices = indices;
            }
            if(differ)
            {
                ++countErr;
            }
            threshold_test_failed |= differ;

            if (differ && GlobalData::doPrintDiffs)
            {
                const TensorDims ti = m_outputTensor.toTensor(indices);
                printf("DIFF HWC [%d:%d:%d] %f %f\n", ti.height, ti.width, ti.channels,
                       static_cast<float>(read<T>(value)),
                       static_cast<float>(read<T>(gt_value)));
            }
        });

        if (threshold_test_failed)
        {
            printf("Total errs: %d, the first on (%ld, %ld, %ld, %ld)\n",
                    countErr, firstErrIndices.dims[0], firstErrIndices.dims[1],
                    firstErrIndices.dims[2], firstErrIndices.dims[3]);
        }

        return !threshold_test_failed;
    }
    T getPermutedOutput(const MemoryDims& indices)
    {
        MemoryDims nind;
        permuteArray(indices.dims, m_storagePermutation, nind.dims, m_dims);

        T out_val = m_outputTensor.at(nind);

        return out_val;
    }
protected:
    ListIterator<Permutation> m_testsLoop;
    ListIterator<Strides> m_stridesLoop;
    t_MvTensorStorageOrder m_storageOrder;
    bool m_hasStrides;
    int m_dims;
    int32_t m_logicalPermutation[MaxTensorDims];
    int32_t m_storagePermutation[MaxTensorDims];
    Tensor<T> m_inputTensor;
    Tensor<T> m_outputTensor;
    bool m_allowPermuteND;
private:
    const int32_t the_same[MaxTensorDims] = { 0, 1, 2, 3, 4, 5, 6, 7, };
};

//
// Specialize templates for fp16 data type
//

template<>
struct TypeConverter<fp16> { using type = float; };

template<>
fp16 write<fp16>(typename TypeConverter<fp16>::type src) { return f32Tof16(src); }

template<>
typename TypeConverter<fp16>::type read<fp16>(fp16 src) { return f16Tof32(src); }

//
// Specialize templates for int32_t data type
//

template<>
int32_t write<int32_t>(typename TypeConverter<int32_t>::type src) { return src; }

template<>
typename TypeConverter<int32_t>::type read<int32_t>(int32_t src) { return src; }

//
// Specialize templates for uint8_t data type
//

template<>
uint8_t write<uint8_t>(typename TypeConverter<uint8_t>::type src) { return src; }

template<>
typename TypeConverter<uint8_t>::type read<uint8_t>(uint8_t src) { return src; }

//
// Declaration test cases for each data type
//

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, FP16))
{
ICV_TESTS_REGISTER_SUITE(PermuteTests<fp16>)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TESTS_SUITE_NAME, I32))
{
ICV_TESTS_REGISTER_SUITE(PermuteTests<int32_t>)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TESTS_SUITE_NAME, I8))
{
ICV_TESTS_REGISTER_SUITE(PermuteTests<uint8_t>)
}

} // namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
