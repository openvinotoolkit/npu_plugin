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
        {{4, 3, 2}, {4, 3, 2}, orderZYX, FPE("topk.elf"), {sw_params::Location::NN_CMX}}
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
            //CustomCppTests<fp16>::initData();

            initElfBuffer();
            initTestCase();
            const Dimensions& dimIn = m_currentTest->inDim;
            const Dimensions& dimOut = m_currentTest->outDim;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            const TensorDims dims3In(dimIn.width,   dimIn.height,  dimIn.channels,  1);
            const TensorDims dims3K(1, 1, 1, 1);
            const TensorDims dims3Out(dimOut.width, dimOut.height, dimOut.channels, 1);

            m_inputTensor.init(storageOrder, dims3In);
            m_kTensor.init(storageOrder, dims3K);
            m_valueTensor.init(storageOrder, dims3Out);
            m_indexTensor.init(storageOrder, dims3Out);
            m_referenceValueTensor.init(storageOrder, dims3Out);
            m_referenceIndexTensor.init(storageOrder, dims3Out);

            allocBuffer(m_inputTensor);
            allocBuffer(m_kTensor);
            allocBuffer(m_valueTensor);
            allocBuffer(m_indexTensor);
            allocBuffer(m_referenceValueTensor);
            allocBuffer(m_referenceIndexTensor);

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

        void initParserRunner() override {
            printf("init parser runner.\n");
            initMyriadResources();
            initDebugInfo();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);

            Buffer inBuff;
            m_inputTensor.exportToBuffer(inBuff);
            customCppOp->addInputBuffer(inBuff, m_requiredTensorLocation);

            Buffer kBuff;
            m_kTensor.exportToBuffer(kBuff);
            customCppOp->addInputBuffer(kBuff, m_requiredTensorLocation);

            Buffer valueBuff;
            m_valueTensor.exportToBuffer(valueBuff);
            customCppOp->addOutputBuffer(valueBuff, m_requiredTensorLocation);

            Buffer indexBuff;
            m_indexTensor.exportToBuffer(indexBuff);
            customCppOp->addOutputBuffer(indexBuff, m_requiredTensorLocation);

            customCppOp->ops = *getParams();
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

            // k value
            int32_t k = 13;
            m_kTensor.forEach(false, [&](const MemoryDims& indices) {
                m_kTensor.at(indices) = k;
            });
        }
        void generateReferenceData() override {
            printf("generate reference data.\n");
        }

        virtual bool checkResult() override {
            printf("check result.\n");
            m_valueTensor.confirmBufferData();

            //bool threshold_test_failed = false;

            // Test value tensor
            float value;
            MemoryDims ind1(0, 0, 0, 0, 0, 0, 0, 0);
            value = f16Tof32(m_valueTensor.at(ind1));
            printf("value: %f\n", value);
            MemoryDims ind2(3, 2, 1, 0, 0, 0, 0, 0);
            value = f16Tof32(m_valueTensor.at(ind2));
            printf("value: %f\n", value);

            // Test index tensor
            int32_t index;
            index = m_indexTensor.at(ind1);
            printf("index: %d\n", index);

            return true;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        Tensor<int32_t> m_kTensor;
        Tensor<fp16> m_valueTensor;
        Tensor<int32_t> m_indexTensor;

        Tensor<fp16> m_referenceValueTensor;
        Tensor<int32_t> m_referenceIndexTensor;

        std::vector<uint64_t> paramContainer;
        sw_params::TopKParams* m_TopKParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppTopKTest)
};