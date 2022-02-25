//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <custom_cpp_tests.h>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.divide_fp16.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_divide.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Divide)) {

    const bool save_to_file = false;

    static constexpr std::initializer_list<SingleTest> divide_test_list{
            {{2, 2, 2}, {2, 2, 2}, orderZYX, FPE("divide_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1, 1, 20}, {1, 1, 20}, orderZYX, FPE("divide_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("divide_fp16.elf"), {sw_params::Location::NN_CMX}}};

    class CustomCppDivideTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppDivideTest(): m_testsLoop(divide_test_list, "test") {
        }
        virtual ~CustomCppDivideTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppDivideTest";
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
            const Dimensions& dimOut = m_currentTest->outDim;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            const TensorDims dims3In(dimIn.width, dimIn.height, dimIn.channels, 1);
            const TensorDims dims3Out(dimOut.width, dimOut.height, dimOut.channels, 1);

            m_inputTensor1.init(storageOrder, dims3In);
            m_inputTensor2.init(storageOrder, dims3In);

            allocBuffer(m_inputTensor1);
            allocBuffer(m_inputTensor2);

            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            const SingleTest* test = m_currentTest;

            m_divideParams = reinterpret_cast<sw_params::DivideParams*>(paramContainer);
            *m_divideParams = sw_params::DivideParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::DivideParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_divideParams);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_divide_fp16_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(divide_fp16));
#endif
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void initParserRunner() override {
            initMyriadResources();
            initDebugInfo();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inBuff1;
            OpTensor inBuff2;

            OpTensor outBuff;
            m_inputTensor1.exportToBuffer(inBuff1);
            m_inputTensor2.exportToBuffer(inBuff2);

            m_outputTensor.exportToBuffer(outBuff);

            customCppOp->addInputBuffer(inBuff1, m_requiredTensorLocation);
            customCppOp->addInputBuffer(inBuff2, m_requiredTensorLocation);

            customCppOp->addOutputBuffer(outBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        void generateInputData() override {
            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor1.forEach(false, [&](const MemoryDims& indices) {
                float tmp1 = float(rand() % 600) / 10.0f - 3.0f;
                m_inputTensor1.at(indices) = f32Tof16(tmp1);
            });

            m_inputTensor2.forEach(false, [&](const MemoryDims& indices) {
                float tmp2 = float(rand() % 600) - 0.5f;
                m_inputTensor2.at(indices) = f32Tof16(tmp2);
            });
        }

        void generateReferenceData() override {
            m_referenceOutputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val1 = f16Tof32(m_inputTensor1.at(indices));
                float val2 = f16Tof32(m_inputTensor2.at(indices));

                float ref = val1 / val2;
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor1.buffer()), m_inputTensor1.bufferSize(),
                                 "in1Myriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor2.buffer()), m_inputTensor2.bufferSize(),
                                 "in2Myriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                                 m_referenceOutputTensor.bufferSize(), "refOutMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float input1 = f16Tof32(m_inputTensor1.at(indices));
                float input2 = f16Tof32(m_inputTensor2.at(indices));
                float abs_diff = fabs(value - gt_value);

                bool differ = !bool(abs_diff <= m_test_threshold);
                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] in1 = %f in2 = %f out = %f ref =  %f abs_diff = %f\n", ti.height,
                           ti.width, ti.channels, input1, input2, value, gt_value, abs_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        Tensor<fp16> m_inputTensor1;
        Tensor<fp16> m_inputTensor2;

        sw_params::DivideParams* m_divideParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppDivideTest)
}  // namespace )
