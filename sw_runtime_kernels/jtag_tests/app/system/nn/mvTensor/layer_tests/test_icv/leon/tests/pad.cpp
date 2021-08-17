/*
* {% copyright %}
*/

#define ICV_TEST_SUITE_NAME PAD

#include "icv_test_suite.h"

#include "Pad.h"

#include <float.h> // FLT_EPSILON

using namespace icv_tests;

namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
{
const bool save_to_file = false;

struct Dimensions
{
    int width;
    int height;
    int channels;
};

//#define ALL_TESTS

const std::initializer_list<Dimensions> dimensions_list =
{
        { 23, 27, 87 },
        { 281, 193, 23 },

#ifdef ALL_TESTS
    { 1 *  25, 1 * 25, 1 * 28 },
    { 1 *  25, 1 * 25, 2 * 28 },
    { 1 *  25, 2 * 25, 1 * 28 },
    { 1 *  25, 2 * 25, 2 * 28 },
    { 2 *  25, 1 * 25, 1 * 28 },
    { 2 *  25, 1 * 25, 2 * 28 },
    { 2 *  25, 2 * 25, 1 * 28 },
    { 2 *  25, 2 * 25, 2 * 28 },

    { 1 *  25, 1 * 25, 1 * 28 },
    { 1 *  25, 1 * 25, 5 * 28 },
    { 1 *  25, 5 * 25, 1 * 28 },
    { 1 *  25, 5 * 25, 5 * 28 },
    { 5 *  25, 1 * 25, 1 * 28 },
    { 5 *  25, 1 * 25, 5 * 28 },
    { 5 *  25, 5 * 25, 1 * 28 },
    { 5 *  25, 5 * 25, 5 * 28 },

    { 1 *  25, 1 * 25, 1 * 28 },
    { 1 *  25, 1 * 25, 7 * 28 },
    { 1 *  25, 7 * 25, 1 * 28 },
    { 1 *  25, 7 * 25, 7 * 28 },
    { 7 *  25, 1 * 25, 1 * 28 },
    { 7 *  25, 1 * 25, 7 * 28 },
    { 7 *  25, 7 * 25, 1 * 28 },
    { 7 *  25, 7 * 25, 7 * 28 },

// big tensors
    { 1 *  100, 1 * 100, 1 * 28 },
    { 1 *  100, 1 * 100, 5 * 28 },
    { 1 *  100, 5 * 100, 1 * 28 },
    { 1 *  100, 5 * 100, 5 * 28 },
    { 5 *  100, 1 * 100, 1 * 28 },
    { 5 *  100, 1 * 100, 5 * 28 },
#endif
        };

struct PadParams
{
    uint32_t padc_begin;
    uint32_t padc_end;
    uint32_t padh_begin;
    uint32_t padh_end;
    uint32_t padw_begin;
    uint32_t padw_end;
    uint32_t padb_begin;
    uint32_t padb_end;

    ePadMode pad_mode;

    t_MvTensorStorageOrder storageOrder;

    bool withStride;
};

const float padValue = 42.0f;

const std::initializer_list<PadParams> params_list =
        {
                {1,   1,  0, 1, 0 , 1,  0, 0, ePadMode::Constant,  orderNZYX, true},
                {0,   0,  0, 1, 1 , 0,  0, 0, ePadMode::Constant,  orderNHWC, false},
                {0,   0,  3, 0, 10, 7,  0, 0, ePadMode::Constant,  orderNHWC, true},

                {1,   1,  0, 1, 0 , 1,  0, 0, ePadMode::ConstantMTL,  orderNZYX, true},
                {0,   0,  0, 1, 1 , 0,  0, 0, ePadMode::ConstantMTL,  orderNHWC, false},
                {0,   0,  3, 0, 10, 7,  0, 0, ePadMode::ConstantMTL,  orderNHWC, true},

                {1,   1,  0, 1, 0 , 1,  0, 0, ePadMode::Edge,      orderNZYX, true},
                {0,   0,  0, 1, 1 , 0,  0, 0, ePadMode::Edge,      orderNZYX, true},
                {0,   0,  3, 0, 10, 7,  0, 0, ePadMode::Edge,      0x4132, true}, //NWCH

                {1,   1,  0, 1, 0 , 1,  0, 0, ePadMode::Reflect,   orderNHWC, true},
                {0,   0,  0, 1, 1 , 0,  0, 0, ePadMode::Reflect,   orderNZYX, true},
                {0,   0,  3, 0, 10, 7,  0, 0, ePadMode::Reflect,   orderNHCW, true},

                {1,   1,  0, 1, 0 , 1,  0, 0, ePadMode::Symmetric, orderNHWC, true},
                {0,   0,  0, 1, 1 , 0,  0, 0, ePadMode::Symmetric, orderNHCW, true},
                {0,   0,  3, 0, 10, 7,  0, 0, ePadMode::Symmetric, orderNHWC, true},
        };

class Tests: public TestSuite
{
public:
    explicit Tests()
            : m_paramsLoop(params_list, "param")
            , m_dimensionsLoop(dimensions_list, "dim")
    {}
    virtual ~Tests()
    {}
protected:
    const char* suiteName() const override
    { return ICV_TESTS_STRINGIFY(ICV_TEST_SUITE_NAME); }
    void userLoops() override
    {
        addLoop(m_paramsLoop);
        addLoop(m_dimensionsLoop);
    }
    void initData() override
    {
        const auto& dim = m_dimensionsLoop.value();
        const auto& param = m_paramsLoop.value();

        TensorDims dims3(dim.width, dim.height, dim.channels, 1);
        TensorAlign align3((param.withStride ? 16 : 0), 0, 0, 0);
        TensorDims pad(param.padw_begin + param.padw_end, param.padh_begin + param.padh_end, param.padc_begin + param.padc_end, 0);

        m_inputTensor.init(param.storageOrder, dims3, align3);
        m_outputTensor.init(param.storageOrder, dims3 + pad, align3);
        m_referenceOutput.init(param.storageOrder, dims3 + pad, TensorAlign(0, 0, 0, 0));

        allocBuffer(m_inputTensor);
        allocBuffer(m_outputTensor);
        allocBuffer(m_referenceOutput);
    }
    void formatTestParams(char* str, int maxLength) const override
    {
        const PadParams& test = m_paramsLoop.value();
        const auto& param = m_paramsLoop.value();

        const auto& id = m_inputTensor.tensorDims();
        const auto& il = m_inputTensor.tensorLimits();

        const char* pad_mode_text = padModeString(param.pad_mode);
        const char* layout_text = layoutString(param.storageOrder);
        const char* stride_text = (test.withStride? "output stride" : "no strides");
        const char* tensor_memory_text = (useCMXTensors ? "CMX tensors allocation" : "DDR tensors allocation");

        snprintf_append(str, maxLength, "H W C = %u %u %u (%u %u %u)",
                        id.height, id.width, id.channels, il.height, il.width, il.channels);

        snprintf_append(str, maxLength, "; pad: begin=(%i,%i,%i,%i), end=(%i,%i,%i,%i), mode=%s",
                        param.padb_begin, param.padc_begin, param.padh_begin, param.padw_begin,
                        param.padb_end, param.padc_end, param.padh_end, param.padw_end,
                        pad_mode_text);

        snprintf_append(str, maxLength, "; %s, %s", layout_text, stride_text);
        snprintf_append(str, maxLength, "; %s", tensor_memory_text);
    }
    t_MvTensorOpType opType() const override
    { return kPad; }
    void initParserRunner() override
    {
        const auto& param = m_paramsLoop.value();

        initMyriadResources();
        initDebugInfo();

        Pad* PadOp = static_cast<Pad*>(m_op);

        m_inputTensor.exportToBuffer(PadOp->input);
        m_outputTensor.exportToBuffer(PadOp->output);

        PadOp->padValue() = padValue;
        PadOp->pad_mode() = param.pad_mode;

        PadOp->pad0_begin() = param.padb_begin;
        PadOp->pad0_end()   = param.padb_end;
        PadOp->pad1_begin() = param.padc_begin;
        PadOp->pad1_end()   = param.padc_end;
        PadOp->pad2_begin() = param.padh_begin;
        PadOp->pad2_end()   = param.padh_end;
        PadOp->pad3_begin() = param.padw_begin;
        PadOp->pad3_end()   = param.padw_end;
    }
    void generateData() override
    {
        const auto id = m_inputTensor.tensorDims();

        // input
        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
        {
            const TensorDims ti = m_inputTensor.toTensor(indices);

            float tmp = float((ti.width + id.width * (ti.height + id.height * ti.channels)) % 65000);
            m_inputTensor.at(indices) = f32Tof16(tmp);
        });
    }
    fp16 calcReferenceOutput(TensorDims ti)
    {
        const auto& param = m_paramsLoop.value();
        const auto& dims = m_inputTensor.tensorDims();

        ti.width  -= param.padw_begin;
        ti.height -= param.padh_begin;
        ti.channels -= param.padc_begin;

        // inside input tensor
        if ((ti.width >= 0 && ti.width < dims.width) &&
            (ti.height >= 0 && ti.height < dims.height) &&
            (ti.channels >= 0 && ti.channels < dims.channels))
        {
            return m_inputTensor.at(ti);
        }
        else
        {
            if (param.pad_mode == ePadMode::Constant || param.pad_mode == ePadMode::ConstantMTL)
            {
                return f32Tof16(padValue);
            }
            else if (param.pad_mode == ePadMode::Edge)
            {
                ti.width = std::min(std::max(ti.width, 0), dims.width - 1);
                ti.height = std::min(std::max(ti.height, 0), dims.height - 1);
                ti.channels = std::min(std::max(ti.channels, 0), dims.channels - 1);

                return m_inputTensor.at(ti);
            }
            else if (param.pad_mode == ePadMode::Reflect || param.pad_mode == ePadMode::Symmetric)
            {
                int mode_offset = (param.pad_mode == ePadMode::Symmetric) ? 1 : 0;

                if (ti.width > dims.width - 1) ti.width = dims.width-1 - (ti.width - (dims.width-1)) + mode_offset;
                if (ti.width < 0) ti.width = -ti.width - mode_offset;

                if (ti.height > dims.height - 1) ti.height = dims.height-1 - (ti.height - (dims.height-1)) + mode_offset;
                if (ti.height < 0) ti.height = -ti.height - mode_offset;

                if (ti.channels > dims.channels - 1) ti.channels = dims.channels-1 - (ti.channels - (dims.channels-1)) + mode_offset;
                if (ti.channels < 0) ti.channels = -ti.channels - mode_offset;

                return m_inputTensor.at(ti);
            }
        }
        return 0;
    }
    void resetOutputData() override
    { resetTensorBuffer(m_outputTensor); }
    bool checkResult() override
    {
        m_outputTensor.confirmBufferData();

        // save output data
        if (save_to_file)
        {
            saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(), "outMyriad_Pad.bin");
        }

        bool threshold_test_failed = false;

        m_outputTensor.forEach(false, [&](const MemoryDims& indices)
        {
            float value = f16Tof32(m_outputTensor.at(indices));

            const TensorDims ti = m_outputTensor.toTensor(indices);
            float ref_value = f16Tof32(calcReferenceOutput(ti));

            bool differ = !bool(value == ref_value);

            m_referenceOutput.at(indices) = calcReferenceOutput(ti);

            threshold_test_failed |= differ;

            if (differ && GlobalData::doPrintDiffs)
            {
                const TensorDims ti = m_outputTensor.toTensor(indices);
                printf("DIFF HWC [%d:%d:%d] %f %f\n", ti.height, ti.width, ti.channels, value, ref_value);
            }
        });

        return !threshold_test_failed;
    }
    static const char* padModeString(ePadMode pad_mode)
    {
        switch (pad_mode)
        {
            case ePadMode::ConstantMTL:  return "ConstantMTL";
            case ePadMode::Constant:  return "Constant";
            case ePadMode::Edge:      return "Edge";
            case ePadMode::Reflect:   return "Reflect";
            case ePadMode::Symmetric: return "Symmetric";
            default:                  return "<unknown>";
        }
    }
protected:
    ListIterator<PadParams> m_paramsLoop;
    ListIterator<Dimensions> m_dimensionsLoop;
    bool useCMXTensors;
    Tensor<fp16> m_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceOutput;
};

ICV_TESTS_REGISTER_SUITE(Tests)

} // namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
