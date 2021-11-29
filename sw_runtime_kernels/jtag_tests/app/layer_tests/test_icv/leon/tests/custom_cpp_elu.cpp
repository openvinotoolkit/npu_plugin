// {% copyright %}

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"
#include "layers/param_custom_cpp.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.elu_fp16.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_elu.h"

// To workaround integer array 'layerParams'.
union Hex {
    float f;
    int32_t i;
};

#define F_0_5 0x3f000000 // Hex representation of 0.5f

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Elu))
{
static constexpr std::initializer_list<SingleTest> elu_test_list
{
    {{2, 2, 2}, {2, 2, 2}, orderZYX, FPE("elu.elf"), {{F_0_5 /*alpha*/, sw_params::Location::NN_CMX /*mem type*/,}}},
};

class CustomCppEluTest: public CustomCppTests<fp16> {
public:
    explicit CustomCppEluTest(): m_testsLoop(elu_test_list, "test") {}
    virtual ~CustomCppEluTest() {}
protected:
    const char* suiteName() const override
    {
        return "CustomCppEluTest";
    }
    void userLoops() override
    {
        addLoop(m_testsLoop);
    }

    void initData() override {

        m_params = {
            0xFFFFFFFF,
            m_elfBuffer,
            0,
            nullptr,
            MAX_LOCAL_PARAMS,
            0,
            0
        };

        paramContainer.resize(((int)sizeof(sw_params::EluParams) + 7) / 8);
        CustomCppTests<fp16>::initData();
        const SingleTest* test = m_currentTest;
        const Hex alpha_hex = { .i = test->customLayerParams.layerParams[0] };
	m_alpha = alpha_hex.f;
        m_eluParams = reinterpret_cast<sw_params::EluParams *>(paramContainer.data());
        *m_eluParams = sw_params::EluParams();
        m_eluParams->alpha = m_alpha;
        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer.data());
        m_params.paramDataLen = paramContainer.size() * sizeof(uint64_t);
        m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[1]);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_eluParams);
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.0005f;

//        const StorageOrder& storageOrder = m_currentTest->storageOrder;
//        const auto& dimIn = m_currentTest->inDim;
//        const TensorDims dims3In(dimIn.width, dimIn.height, dimIn.channels, 1);
//        m_refInputTensor.init(storageOrder, dims3In);
//        allocBuffer(m_refInputTensor);
    }

    void generateInputData() override {
        const auto customData = false;//m_testLoop.value().customData;

#ifdef CONFIG_TARGET_SOC_3720
        m_params.kernel  = reinterpret_cast<uint64_t>(sk_elu_fp16_3010xx_text);
#else
        m_params.kernel  = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(elu_fp16));
#endif

        rand_seed();

        // set random seed
        u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
        srand(ticks_for_seed);

        // input
        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
        {
            float tmp = float(rand() % 1000) / 100 - 5.0f;
            m_inputTensor.at(indices) = f32Tof16(tmp);
        });

        // reference output
        generateReferenceData();
    }
    void generateReferenceData() override {
        m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            float val = f16Tof32(m_inputTensor.at(indices));
            float min = std::min(val, 0.0f);
            float max = std::max(val, 0.0f);
            float ref = max + 0.3f * (exp((double)min) - 1.0f);
            m_referenceOutputTensor.at(indices) = f32Tof16(ref);
        });

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
    float m_alpha;
    std::vector<uint64_t> paramContainer;
    sw_params::EluParams * m_eluParams;
//    Tensor<fp16> m_referenceTensor;
};

ICV_TESTS_REGISTER_SUITE(CustomCppEluTest)
}
