// {% copyright %}

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"
#include "layers/param_custom_cpp.h"
// #include <iostream>

#ifdef CONFIG_TARGET_SOC_3720
extern void*  (&shvNN0_sigmoid_fp16);
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_sigmoid.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Sigmoid))
{
static constexpr std::initializer_list<SingleTest> sigmoid_test_list
{
    {{2, 2, 2}, {2, 2, 2}, orderZYX, FPE("sigmoid_fp16.elf"), {sw_params::Location::NN_CMX}},
    // {{2, 2, 2}, {2, 2, 2}, orderZYX, FPE("sigmoid.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{0 /*axis*/, sw_params::Location::UPA_CMX /*mem type*/,}}},
    // {{2, 2, 2}, {2, 2, 2}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{1 /*axis*/, sw_params::Location::UPA_CMX /*mem type*/,}}},
    // {{4, 4, 4}, {4, 4, 4}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{0 /*axis*/, sw_params::Location::NN_CMX/*mem type*/,}}},
    // {{4, 4, 4}, {4, 4, 4}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{1 /*axis*/, sw_params::Location::NN_CMX/*mem type*/,}}},
    // {{4, 4, 4}, {4, 4, 4}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{2 /*axis*/, sw_params::Location::NN_CMX/*mem type*/,}}},
    // {{3, 4, 5}, {3, 4, 5}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */{2 /*axis*/, sw_params::Location::DDR/*mem type*/,}}},
//    {{8, 8, 8}, {8, 8, 8}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */nn::shave_lib::SOFTMAX, {2 /*axis*/,}}},
//    {{224, 128, 24}, {224, 128, 24}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */nn::shave_lib::SOFTMAX, {0 /*axis*/,}}},
//    {{224, 128, 24}, {224, 128, 24}, orderZYX, FPE("softmax.elf"), {/*{{224, 1, 1}, {1, 128, 1}, {0, 0, 0}, 3, 0}, */nn::shave_lib::SOFTMAX, {1 /*axis*/,}}},
};

class CustomCppSigmoidTest: public CustomCppTests<fp16> {
public:
    explicit CustomCppSigmoidTest(): m_testsLoop(sigmoid_test_list) {}
    virtual ~CustomCppSigmoidTest() {}
protected:
    const char* suiteName() const override
    {
        return "CustomCppSigmoidTest";
    }
    void userLoops() override
    {
        addLoop(m_testsLoop);
    }

    void initData() override {
        // std::cout << "init data" << std::endl;

        m_params = {
            0xFFFFFFFF,
            m_elfBuffer,
            0,
            nullptr,
            MAX_LOCAL_PARAMS,
            0,
            0
        };

        paramContainer.resize(((int)sizeof(sw_params::SigmoidParams) + 7) / 8);
        CustomCppTests<fp16>::initData();
        const SingleTest* test = m_currentTest;
        int32_t ind[subspace::MAX_DIMS] = {0, };
        subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
        m_sigmoidParams = reinterpret_cast<sw_params::SigmoidParams *>(paramContainer.data());
        *m_sigmoidParams = sw_params::SigmoidParams();
        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer.data());
        m_params.paramDataLen = paramContainer.size() * sizeof(uint64_t);
        m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_sigmoidParams);

        // std::cout << "init data end" << std::endl;
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.0005f;

    //    const StorageOrder& storageOrder = m_currentTest->storageOrder;
    //    const auto& dimIn = m_currentTest->inDim;
    //    const TensorDims dims3In(dimIn.width, dimIn.height, dimIn.channels, 1);
    //    m_inputTensor.init(storageOrder, dims3In);
    //    allocBuffer(m_inputTensor);
    }

    void generateInputData() override {
        // std::cout << "generate input data" << std::endl;
        const auto customData = false;//m_testLoop.value().customData;

#ifdef CONFIG_TARGET_SOC_3720
        m_params.kernel  = reinterpret_cast<uint64_t>(&shvNN0_sigmoid_fp16);
#else
        m_params.kernel  = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(sigmoid_fp16));
#endif

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
        // std::cout << "generate input data end" << std::endl;
    }
    void generateReferenceData() override {
        m_inputTensor.forEach(false, [&](const MemoryDims& indices){
            float val = f16Tof32(m_inputTensor.at(indices));
            float ref = val * -1.0f;
            ref = 1.0f + expf(ref);
            ref = 1.0f / ref;
            m_referenceOutputTensor.at(indices) = f32Tof16(ref);
        });
    }
    virtual bool checkResult() override
        {
            // std::cout << "checkResult" << std::endl;
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

                if ((differ && GlobalData::doPrintDiffs) || true)
                {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value, abs_diff);
                }
            });

            // std::cout << "checkResult end" << std::endl;
            return !threshold_test_failed;
        }
private:
    ListIterator<SingleTest> m_testsLoop;

    // Additional buffer to avoid convertion back and forth
    int m_axis;
    std::vector<uint64_t> paramContainer;
    sw_params::SigmoidParams * m_sigmoidParams;
//    Tensor<fp16> m_referenceTensor;
};

ICV_TESTS_REGISTER_SUITE(CustomCppSigmoidTest)
}
