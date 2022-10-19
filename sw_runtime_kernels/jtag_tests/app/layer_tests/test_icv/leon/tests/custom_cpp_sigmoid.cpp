// {% copyright %}

#include <custom_cpp_tests.h>
#include <cmath>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"
#include <nn_cache.h>

__attribute__((aligned(1024)))
#include "sk.sigmoid_fp16.3720xx.text.xdat"

#include "param_sigmoid.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Sigmoid)) {
    static constexpr std::initializer_list<SingleTest> sigmoid_test_list{
            {{1, 1, 20}, {1, 1, 20}, orderZYX, FPE("sigmoid_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{100, 1, 1}, {100, 1, 1}, orderZYX, FPE("sigmoid_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1, 111, 1}, {1, 111, 1}, orderZYX, FPE("sigmoid_fp16.elf"), {sw_params::Location::NN_CMX}},
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("sigmoid_fp16.elf"), {sw_params::Location::NN_CMX}}
    };

    class CustomCppSigmoidTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppSigmoidTest(): m_testsLoop(sigmoid_test_list, "test") {
        }
        virtual ~CustomCppSigmoidTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppSigmoidTest";
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
            m_sigmoidParams = reinterpret_cast<sw_params::SigmoidParams*>(paramContainer);
            *m_sigmoidParams = sw_params::SigmoidParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::SigmoidParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_sigmoidParams);
            nn::cache::flush(*m_sigmoidParams);
            nn::cache::flush(m_params.baseParamData);
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
            m_params.kernel = reinterpret_cast<uint64_t>(sk_sigmoid_fp16_3720xx_text);

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
                float ref = val * -1.0f;
                ref = 1.0f + exp((double)ref);
                ref = 1.0f / ref;
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }
        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            //            // save output data
            //            if (save_to_file)
            //            {
            //                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()),
            //                m_outputTensor.bufferSize(), "outMyriad.bin");
            //            }

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

        sw_params::SigmoidParams* m_sigmoidParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppSigmoidTest)
}  // namespace )
