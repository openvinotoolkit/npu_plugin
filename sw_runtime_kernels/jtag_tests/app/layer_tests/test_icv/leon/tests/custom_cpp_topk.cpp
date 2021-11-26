// {% copyright %}

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"
#include "layers/param_custom_cpp.h"

#ifdef CONFIG_TARGET_SOC_3720
extern void*  (shvNN0_singleShaveTopK);
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_topk.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, TopK)) {
    static constexpr std::initializer_list <SingleTest> topk_test_list {
        {{2, 3, 4}, {2, 3, 4}, orderZYX, FPE("topk.elf"), {sw_params::Location::NN_CMX}}
    };

    class CustomCppTopKTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppTopKTest(): m_testsLoop(topk_test_list, "test") {}
        virtual ~CustomCppTopKTest() {}

    protected:
        const char* suiteName() const override {
            return "CustomCppTopKTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            printf("init data.\n");
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            paramContainer.resize(((int)sizeof(sw_params::TopKParams) + 7) / 8);
            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_TopKParams = reinterpret_cast<sw_params::TopKParams*>(paramContainer.data());
            *m_TopKParams = sw_params::TopKParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer.data());
            m_params.paramDataLen = paramContainer.size() * sizeof(uint64_t);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_TopKParams);
        }

        void initTestCase() override {
            printf("init test case.\n");
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.01f;
        }

        void generateInputData() override {
            printf("generate input data.\n");
            const auto customData = false;

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(&shvNN0_singleShaveTopK);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(singleShaveTopK));
#endif
            // input
            float val = 1.0;
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                m_inputTensor.at(indices) = f32Tof16(val);
                val++;
            });
        }
        void generateReferenceData() override {
            printf("generate reference data.\n");
        }

        virtual bool checkResult() override {
            printf("check result.\n");
            m_outputTensor.confirmBufferData();

            //bool threshold_test_failed = false;

            float value;
            MemoryDims ind1(0, 0, 0, 0, 0, 0, 0, 0);
            value = f16Tof32(m_outputTensor.at(ind1));
            printf("value: %f\n", value);
            MemoryDims ind2(1, 0, 0, 0, 0, 0, 0, 0);
            value = f16Tof32(m_outputTensor.at(ind2));
            printf("value: %f\n", value);

            return true;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        std::vector<uint64_t> paramContainer;
        sw_params::TopKParams* m_TopKParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppTopKTest)
};