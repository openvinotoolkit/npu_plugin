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
#include <cmath>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.tanh_fp16.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_tanh.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Tanh)) {
    static constexpr std::initializer_list<SingleTest> tanh_test_list {
        {{1, 1, 7}, {1, 1, 7}, orderZYX, FPE("tanh_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1, 1, 20}, {1, 1, 20}, orderZYX, FPE("tanh_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("tanh_fp16.elf"), {sw_params::Location::NN_CMX}}};

    class CustomCppTanhTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppTanhTest(): m_testsLoop(tanh_test_list, "test") {
        }
        virtual ~CustomCppTanhTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppTanhTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            m_tanhParams = reinterpret_cast<sw_params::TanhParams*>(paramContainer);
            *m_tanhParams = sw_params::TanhParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::TanhParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_tanhParams);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_tanh_fp16_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(tanh_fp16));
#endif
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void generateInputData() override {

            rand_seed();

            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }
        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = val * -2.0f;
                ref = 1.0f + exp((double)ref);
                ref = 2.0f / ref - 1.0f;
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

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value,
                           abs_diff);
                }
            });
            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        sw_params::TanhParams* m_tanhParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppTanhTest)
}  // namespace )
