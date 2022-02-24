// {% copyright %}

#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.subtract_fp16.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_subtract.h"

#define USE_SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Subtract)) {

    const bool save_to_file = true;

    static constexpr std::initializer_list<SingleTest> subtract_test_list{
            {{2, 2, 2}, {2, 2, 2}, orderZXY, FPE("subtract_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{4, 4, 4}, {4, 4, 4}, orderZXY, FPE("subtract_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{16, 16, 16}, {16, 16, 16}, orderZXY, FPE("subtract_fp16.elf"), {sw_params::Location::NN_CMX}}};

    class CustomCppSubtractTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppSubtractTest(): m_testsLoop(subtract_test_list, "test") {
        }
        virtual ~CustomCppSubtractTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppSubtractTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            CustomCppTests<fp16>::initData();

            initElfBuffer();
            initTestCase();
            const Dimensions& dimIn = m_currentTest->inDim;
            const Dimensions& dimIn2 = m_currentTest->inDim;
            const Dimensions& dimOut = m_currentTest->outDim;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            const TensorDims dims3In(dimIn.width,   dimIn.height,  dimIn.channels,  1);
            const TensorDims dims3In2(dimIn2.width,   dimIn2.height,  dimIn2.channels,  1);
            const TensorDims dims3Out(dimOut.width, dimOut.height, dimOut.channels, 1);

            m_inputTensor.init(storageOrder, dims3In);
            m_inputTensor2.init(storageOrder, dims3In2);
            m_outputTensor.init(storageOrder, dims3Out);
            m_referenceOutputTensor.init(storageOrder, dims3Out);

            allocBuffer(m_inputTensor);
            allocBuffer(m_inputTensor2);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            const SingleTest* test = m_currentTest;
            m_subtractParams = reinterpret_cast<sw_params::SubtractParams*>(paramContainer);
            *m_subtractParams = sw_params::SubtractParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::SubtractParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_subtractParams);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_subtract_fp16_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(subtract_fp16));
#endif
        }
        void initParserRunner() override
        {
            initMyriadResources();
            initDebugInfo();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inBuff;
            OpTensor inBuff2;
            OpTensor outBuff;
            m_inputTensor.exportToBuffer(inBuff);
            m_inputTensor2.exportToBuffer(inBuff2);
            m_outputTensor.exportToBuffer(outBuff);

            customCppOp->addInputBuffer(inBuff, m_requiredTensorLocation);
            customCppOp->addInputBuffer(inBuff2, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }
        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.002f;
        }

        void generateInputData() override {
#if defined(USE_SEED_VALUE)
            auto seedValue = USE_SEED_VALUE;
#else
            u64 systemTicks;
            DrvTimerGetSystemTicks64(&systemTicks);
            auto seedValue = static_cast<unsigned int>(systemTicks);
#endif
            std::mt19937 generator(seedValue);
            m_inputTensor.forEach(false, [this, &generator](const MemoryDims& indices) {
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
            m_inputTensor2.forEach(false, [this, &generator](const MemoryDims& indices) {
                // We generate the random value between -8.f and 8f and the kernel do x * relu6(x+3) / 6
                // So the minimum resolution is 2^(-7) = 0.00781f and the kernel may calculate 0 output value
                // if input value is less than this resolution. In such cases, relative difference would be 1.
                const float precisionLimitations = 0.00781f;
                float fp32Value = 0.f;
                do {
                    fp32Value = float(generator()) / generator.max() * 16.f - 8.f;
                } while (fabs(fp32Value) < precisionLimitations && fp32Value != 0.f);

                m_inputTensor2.at(indices) = f32Tof16(fp32Value);
            });
        }
        void generateReferenceData() override {
            // no need to remap memory indices between tensors
            mvTensorAssert(m_inputTensor.storageOrder() == m_referenceOutputTensor.storageOrder());

            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float val2 = f16Tof32(m_inputTensor2.at(indices));
                float ref = val - val2;
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }
        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");
            }

            // no need to remap memory indices between tensors
            mvTensorAssert(m_outputTensor.storageOrder() == m_inputTensor.storageOrder());
            mvTensorAssert(m_outputTensor.storageOrder() == m_referenceOutputTensor.storageOrder());

            bool threshold_test_failed = false;
            float max_abs_diff = 0.0;
            float max_rel_diff = 0.0;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));

                float abs_diff = fabs(value - gt_value);
                float rel_diff = gt_value != 0.0 ? fabs(abs_diff / gt_value) : abs_diff;
                max_abs_diff = std::max(max_abs_diff, abs_diff);
                max_rel_diff = std::max(max_rel_diff, rel_diff);

                float abs_threshold = (fabs(gt_value) * m_test_threshold);
                bool differ = bool(!(abs_diff <= abs_threshold));

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    char indices_str[64];
                    printf("DIFF [%s] %f %f %f %f abs_diff: %f rel_diff: %f\n",
                           m_outputTensor.indicesToString(indices, indices_str), f16Tof32(m_inputTensor.at(indices)), f16Tof32(m_inputTensor2.at(indices)),
                           value, gt_value, abs_diff, rel_diff);
                }
            });

            if (GlobalData::doPrintDiffMax)
                printf("MAX DIFF ABS=%f REL=%f\n", max_abs_diff, max_rel_diff);

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        Tensor<fp16> m_inputTensor2;
        sw_params::SubtractParams* m_subtractParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppSubtractTest)
}  // namespace )
