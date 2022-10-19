// {% copyright %}

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"
#include "layers/param_custom_cpp.h"

#include "param_softmax.h"

__attribute__((aligned(1024)))
#include "sk.singleShaveSoftmax.3720xx.text.xdat"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Softmax))
{
static constexpr std::initializer_list<SingleTest> softmax_test_list
{
    {{2, 2, 2}, {2, 2, 2}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{0 /*axis*/, sw_params::Location::UPA_CMX /*mem type*/,}}},
    {{2, 2, 2}, {2, 2, 2}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{1 /*axis*/, sw_params::Location::UPA_CMX /*mem type*/,}}},
    {{4, 4, 4}, {4, 4, 4}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{0 /*axis*/, sw_params::Location::NN_CMX/*mem type*/,}}},
    {{1, 4, 4}, {1, 4, 4}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{0 /*axis*/, sw_params::Location::NN_CMX/*mem type*/,}}},
    {{4, 4, 4}, {4, 4, 4}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{1 /*axis*/, sw_params::Location::NN_CMX/*mem type*/,}}},
    {{4, 4, 4}, {4, 4, 4}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{2 /*axis*/, sw_params::Location::NN_CMX/*mem type*/,}}},
//    {{3, 4, 5}, {3, 4, 5}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{2 /*axis*/, sw_params::Location::DDR/*mem type*/,}}},
//    {{8, 8, 8}, {8, 8, 8}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */nn::shave_lib::SOFTMAX, {2 /*axis*/,}}},
//    {{224, 128, 24}, {224, 128, 24}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */nn::shave_lib::SOFTMAX, {0 /*axis*/,}}},
//    {{224, 128, 24}, {224, 128, 24}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */nn::shave_lib::SOFTMAX, {1 /*axis*/,}}},
};

class CustomCppSoftmaxTest: public CustomCppTests<fp16> {
public:
    explicit CustomCppSoftmaxTest(): m_testsLoop(softmax_test_list, "test") {}
    virtual ~CustomCppSoftmaxTest() {}
protected:
    const char* suiteName() const override
    {
        return "CustomCppSoftmaxTest";
    }
    void userLoops() override
    {
        addLoop(m_testsLoop);
    }

    void initData() override {
        sw_params::BaseKernelParams emptyParamData;
        m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

        CustomCppTests<fp16>::initData();
        const SingleTest* test = m_currentTest;
        int32_t ind[subspace::MAX_DIMS] = {0, };
        subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
        m_axis = ind[test->customLayerParams.layerParams[0]];
        m_temp = allocData<float>(m_inputTensor.memoryDims().dims[m_axis]);
        m_softmaxParams = reinterpret_cast<sw_params::SoftmaxParams *>(paramContainer);
        *m_softmaxParams = sw_params::SoftmaxParams();
        m_softmaxParams->axis = (int64_t)m_axis;
        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
        m_params.paramDataLen = sizeof(sw_params::SoftmaxParams);
        m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[1]);
        m_params.baseParamData = sw_params::softmaxParamsToBaseKernelParams(m_softmaxParams);
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.001f;

//        const StorageOrder& storageOrder = m_currentTest->storageOrder;
//        const auto& dimIn = m_currentTest->inDim;
//        const TensorDims dims3In(dimIn.width, dimIn.height, dimIn.channels, 1);
//        m_refInputTensor.init(storageOrder, dims3In);
//        allocBuffer(m_refInputTensor);
    }

    void generateInputData() override {
        const auto customData = false;//m_testLoop.value().customData;

        m_params.kernel  = reinterpret_cast<uint64_t>(sk_singleShaveSoftmax_3720xx_text);

        rand_seed();

        if (customData)
        {
            static const float data[] = { -6.1601562, -10.28125  , -3.9277344 , -13.375     , -13.8046875 , -9.2421875 ,
                                          -6.6015625,  -2.3320312, -1.34375   ,  -1.7880859 ,  -3.6308594 ,  9.5       ,
                                           4.3984375,   6.9257812,  0.85058594,  11.53125   ,   0.26367188,  7.1640625 ,
                                         -10.0078125,  -7.9140625, -0.6176758 ,  -6.6679688 ,   4.9453125 , -3.6660156 ,
                                          -8.734375 ,  23.1875   ,  3.359375  ,  15.5234375 ,  -4.3242188 ,  3.0800781 ,
                                          -1.5019531,   4.015625 , -0.84521484,   0.61279297,   5.3554688 , -0.85546875, 4.265625 };
            // input
            int i = 0;
            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                m_inputTensor.at(indices) = f32Tof16(data[i++]);
            });
        }
        else
        {
            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                float tmp = float(rand() % 1000) / 100 - 5.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }

        // reference output
        generateReferenceData();
    }
    void generateReferenceData() override {
//        const auto axis = m_axis;//decodeNegativeAxis(m_axis);

        MemoryDims iDims = m_inputTensor.memoryDims();
        MemoryDims ind = MemoryDims();
        int totalSets = m_inputTensor.totalLines(m_axis);

        for(int s = 0; s < totalSets; ++s)
        {
            ind.dims[m_axis] = 0;
            float largest = f16Tof32(m_inputTensor.at(ind));
            for (int i = 1; i < iDims.dims[m_axis]; ++i)
            {
                ind.dims[m_axis] = i;
                float val = f16Tof32(m_inputTensor.at(ind));
                largest = std::max(largest, val);
            }

            float sum = 0.f;
            for (int i = 0; i < iDims.dims[m_axis]; ++i)
            {
                ind.dims[m_axis] = i;
                float val = f16Tof32(m_inputTensor.at(ind));
                m_temp[i] = pow(exp(1), val - largest)/*exp(val - largest)*/;
                sum = sum + m_temp[i];
            }

            for (int i = 0; i < iDims.dims[m_axis]; ++i)
            {
                ind.dims[m_axis] = i;
                float val = m_temp[i] / sum;
                m_referenceOutputTensor.at(ind) = f32Tof16(val);
            }

            m_inputTensor.incrementLine(ind, m_axis);
        }
    }
    virtual bool checkResult() override
        {
            m_outputTensor.confirmBufferData();

//            // save output data
//            if (save_to_file)
//            {
//                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(), "outMyriad.bin");
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
    ListIterator<SingleTest> m_testsLoop;

    // Additional buffer to avoid convertion back and forth
    int m_axis;
    float* m_temp;
    sw_params::SoftmaxParams * m_softmaxParams;
//    Tensor<fp16> m_referenceTensor;
};

ICV_TESTS_REGISTER_SUITE(CustomCppSoftmaxTest)
}
