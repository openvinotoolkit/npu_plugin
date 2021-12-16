// {% copyright %}

#include <custom_cpp_tests.h>
#include <cmath>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.singleShaveMVN.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_mvn.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Mvn)) {

    const bool save_to_file = false;
    const bool customData = true;

    static constexpr std::initializer_list<SingleTest> mvn_test_list{
            {{20, 2, 10},
             {20, 2, 10},
             orderHWC,
             FPE("mvn.elf"),
             {/*across_channels*/ 0, /*normalize_variance*/ 0, /*eps = 10^(-5)*/ 925353388, sw_params::Location::NN_CMX}},
            {{20, 2, 10},
             {20, 2, 10},
             orderHWC,
             FPE("mvn.elf"),
             {/*across_channels*/ 0, /*normalize_variance*/ 1, /*eps = 10^(-5)*/ 925353388, sw_params::Location::NN_CMX}},
            {{20, 2, 10},
             {20, 2, 10},
             orderCHW,
             FPE("mvn.elf"),
             {/*across_channels*/ 0, /*normalize_variance*/ 0, /*eps = 10^(-5)*/ 925353388, sw_params::Location::NN_CMX}},
            {{20, 2, 10},
             {20, 2, 10},
             orderCHW,
             FPE("mvn.elf"),
             {/*across_channels*/ 0, /*normalize_variance*/ 1, /*eps = 10^(-5)*/ 925353388, sw_params::Location::NN_CMX}},
    };

    class CustomCppMvnTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppMvnTest(): m_testsLoop(mvn_test_list, "test") {
        }
        virtual ~CustomCppMvnTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppMvnTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            paramContainer.resize(((int)sizeof(sw_params::MvnParams) + 7) / 8);
            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_mvnParams = reinterpret_cast<sw_params::MvnParams*>(paramContainer.data());
            *m_mvnParams = sw_params::MvnParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer.data());
            m_params.paramDataLen = paramContainer.size() * sizeof(uint64_t);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[3]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_mvnParams);

            uint32_t acrossChannels = test->customLayerParams.layerParams[0];
            uint32_t normalize = test->customLayerParams.layerParams[1];
            float eps = *((float *)(&test->customLayerParams.layerParams[2]));

            m_mvnParams->acrossChannels = acrossChannels;
            m_mvnParams->normalize = normalize;
            m_mvnParams->eps = eps;

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_singleShaveMVN_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(singleShaveMVN));
#endif
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
            // set random seed
            rand_seed();
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 1000) / 100 - 5.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
        }
        void generateReferenceData() override {
            const auto &dims = m_inputTensor.tensorDims();
            const bool normalize_variance = m_mvnParams->normalize;
            const bool across_channels = m_mvnParams->acrossChannels;
            const float epsilon = m_mvnParams->eps;
            int b = 0;

            if(across_channels) {
                float sum = 0.f;
                float sqsum = 0.f;

                for(int c = 0; c < dims.channels; c++){
                    for(int h = 0; h < dims.height; h++) {
                        for(int w = 0; w < dims.width; w++) {
                            sum += f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b)));
                            sqsum += f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b))) *
                                        f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b)));
                        }
                    }
                }
                float mean = sum / (dims.channels * dims.height * dims.width);

                float variance = sqsum / (dims.channels * dims.height * dims.width);
                variance = variance - mean * mean;
                variance = variance < 0 ? 1.f : variance;
                variance = sqrtf(variance) + epsilon;

                for(int c = 0; c < dims.channels; c++){
                    for(int h = 0; h < dims.height; h++) {
                        for(int w = 0; w < dims.width; w++) {
                            m_referenceOutputTensor.at(TensorDims(w, h, c, b)) =
                                f32Tof16(f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b))) - mean);
                            if(normalize_variance){
                                m_referenceOutputTensor.at(TensorDims(w, h, c, b)) =
                                    f32Tof16(f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b))) / variance);
                            }
                        }
                    }
                }
            } else { // across_channels == false
                for(int c = 0; c < dims.channels; c++) {
                    float sum = 0.f;
                    float sqsum = 0.f;

                    for(int h = 0; h < dims.height; h++) {
                        for(int w = 0; w < dims.width; w++) {
                            sum += f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b)));
                            sqsum += f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b))) *
                                        f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b)));
                        }
                    }
                    float mean = sum / (dims.height * dims.width);

                    float variance = sqsum / (dims.height * dims.width);
                    variance = variance - mean * mean;
                    variance = variance < 0 ? 1.f : variance;
                    variance = sqrtf(variance) + epsilon;

                    for(int h = 0; h < dims.height; h++) {
                        for(int w = 0; w < dims.width; w++) {
                            m_referenceOutputTensor.at(TensorDims(w, h, c, b)) =
                                f32Tof16(f16Tof32(m_inputTensor.at(TensorDims(w, h, c, b))) - mean);
                            if(normalize_variance){
                                m_referenceOutputTensor.at(TensorDims(w, h, c, b)) =
                                    f32Tof16(f16Tof32(m_referenceOutputTensor.at(TensorDims(w, h, c, b))) / variance);
                            }
                        }
                    }
                }
            }
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(),
                                 "inMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()), m_referenceOutputTensor.bufferSize(),
                                 "refOutMyriad.bin");
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

        int m_axis;
        std::vector<uint64_t> paramContainer;
        sw_params::MvnParams* m_mvnParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppMvnTest)
}  // namespace )
