// {% copyright %}

#include <custom_cpp_tests.h>
#include <nn_cache.h>
#include <cmath>
#include "layers/param_custom_cpp.h"

__attribute__((aligned(1024)))
#include "sk.vau_log_fp16.3720xx.text.xdat"

#include "pss/param_vau_log.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Log_pss)) {
    static constexpr std::initializer_list<SingleTest> log_test_list{
            {{1, 1, 8}, {1, 1, 8}, orderZYX, FPE("vau_log_fp16.3720xx.elf"), {sw_params::Location::NN_CMX}},
            {{1, 2, 8}, {1, 2, 8}, orderZYX, FPE("vau_log_fp16.3720xx.elf"), {sw_params::Location::NN_CMX}},
            {{1, 1, 1000}, {1, 1, 1000}, orderZYX, FPE("vau_log_fp16.3720xx.elf"), {sw_params::Location::NN_CMX}},
    };

    class CustomCppLogTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppLogTest(): m_testsLoop(log_test_list, "test") {
        }
        virtual ~CustomCppLogTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppLogTestpss";
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
            m_vauLogParams = reinterpret_cast<sw_params::VauLogParams*>(paramContainer);
            *m_vauLogParams = sw_params::VauLogParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::VauLogParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_vauLogParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_vau_log_fp16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 1.0f;
        }

        void generateInputData() override {
            rand_seed();

            // input
            const int ndims = m_inputTensor.ndims();
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                int tmp = 1;
                for (int i = 0; i < ndims; ++i)
                    tmp *= (3 + (indices.dims[i] % 13));
                float tmp2 = 0.001f + (tmp % 33);
                fp16 val = f32Tof16(tmp2);

                m_inputTensor.at(indices) = val;
            });
        }

        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float val = f16Tof32(m_inputTensor.at(indices));
                float ref = (float)std::log((double)val);
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

        sw_params::VauLogParams* m_vauLogParams;
    };

    // `__builtin_shave_vau_log_v8f16_r` (output is always the same as input)
    // To Do: add ticket
    ICV_TESTS_REGISTER_SUITE(CustomCppLogTest)
}  // namespace )
