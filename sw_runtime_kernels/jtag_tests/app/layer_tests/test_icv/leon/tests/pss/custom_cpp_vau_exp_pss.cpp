// {% copyright %}

#include <custom_cpp_tests.h>
#include <nn_cache.h>
#include <cmath>
#include "layers/param_custom_cpp.h"

__attribute__((aligned(1024)))
#include "sk.vau_exp_fp16.3720xx.text.xdat"

#include "pss/param_vau_exp.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Exp_pss)) {
    static constexpr std::initializer_list<SingleTest> exp_test_list{
            {{1, 1, 8}, {1, 1, 8}, orderZYX, FPE("vau_exp_fp16.3720xx.elf"), {sw_params::Location::NN_CMX}},
            {{1, 2, 8}, {1, 2, 8}, orderZYX, FPE("vau_exp_fp16.3720xx.elf"), {sw_params::Location::NN_CMX}},
            {{1, 1, 1000}, {1, 1, 1000}, orderZYX, FPE("vau_exp_fp16.3720xx.elf"), {sw_params::Location::NN_CMX}},
    };

    class CustomCppExpTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppExpTest(): m_testsLoop(exp_test_list, "test") {
        }
        virtual ~CustomCppExpTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppExpTestpss";
        }

        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_vauExpParams = reinterpret_cast<sw_params::VauExpParams*>(paramContainer);
            *m_vauExpParams = sw_params::VauExpParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::VauExpParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_vauExpParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_vau_exp_fp16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 1.0f;
        }

        void generateInputData() override {
            rand_seed();

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }

        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = exp((double)val);
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(),
                                 "inMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                                 m_referenceOutputTensor.bufferSize(), "refOutMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float ulp_diff = ulp::absdiff_fp32(value, gt_value);

                bool differ = !bool(ulp_diff <= m_test_threshold);
                threshold_test_failed |= differ;

                float input = f16Tof32(m_inputTensor.at(indices));

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] input: %f value: %f ref_value: %f ulp: %f\n", ti.height, ti.width,
                           ti.channels, input, value, gt_value, ulp_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        sw_params::VauExpParams* m_vauExpParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppExpTest)
}  // namespace )
