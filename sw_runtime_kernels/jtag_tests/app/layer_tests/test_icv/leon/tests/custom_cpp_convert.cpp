//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <cmath>
#include <random>

__attribute__((aligned(1024)))
#include "sk.single_shave_convert.3720xx.text.xdat"

#include "param_convert.h"

namespace
{

    static constexpr std::initializer_list<SingleTest> convert_test_list{
            {{1, 1, 7}, {1, 1, 7}, orderZYX, {sw_params::Location::NN_CMX}},
            {{1, 1, 20}, {1, 1, 20}, orderZYX, {sw_params::Location::NN_CMX}},
#ifdef CONFIG_RUN_LARGE_TESTS
            {{1000, 1, 1}, {1000, 1, 1}, orderZYX, {sw_params::Location::NN_CMX}}
#endif
            };

    template<class T_DST, class T_SRC>
    T_DST convert(T_SRC src) { return static_cast<T_DST>(src); };

    //
    // Specializations of convertation template
    //

    template<>
    float convert<float>(int32_t src) {
        if(src >= 0) {
            return static_cast<float>(src);
        } else {
            src = -src;
            float res = static_cast<float>(src);
            res = -res;
            return res;
        }
    };

    template<>
    float convert<float>(int8_t src) { return convert<float>(static_cast<int32_t>(src)); };

    template<>
    uint8_t convert<uint8_t>(fp16 src) { return static_cast<uint8_t>(f16Tof32(src)); };

    template<>
    int8_t convert<int8_t>(fp16 src) { return static_cast<int8_t>(f16Tof32(src)); };

    template<>
    float convert<float>(fp16 src) { return f16Tof32(src); };

    template<>
    int32_t convert<int32_t>(fp16 src) { return static_cast<int32_t>(f16Tof32(src)); };

    template <>
    int64_t convert<int64_t>(fp16 src) { return static_cast<int64_t>(f16Tof32(src)); };

    template<>
    fp16 convert<fp16>(float src) { return f32Tof16(src); };

    template<>
    fp16 convert<fp16>(int32_t src) { return f32Tof16(convert<float>(src)); };

    template<>
    fp16 convert<fp16>(uint8_t src) { return f32Tof16(static_cast<float>(src)); };

    template<>
    fp16 convert<fp16>(int8_t src) { return f32Tof16(convert<float>(static_cast<int32_t>(src))); };

    template <>
    fp16 convert<fp16>(int64_t src) { return f32Tof16(convert<float>(static_cast<int32_t>(src))); };

    template <typename T_SRC, typename T_DST>
    class CustomCppConvertTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppConvertTest(): m_testsLoop(convert_test_list, "test") {
        }
        virtual ~CustomCppConvertTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppConvertTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            initTestCase();
            const Dims& inputDims = m_currentTest->inputDims;
            const Dims& outputDims = m_currentTest->outputDims;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            const TensorDims dims3In(inputDims.begin()[0], inputDims.begin()[1], inputDims.begin()[2], 1);
            const TensorDims dims3Out(outputDims.begin()[0], outputDims.begin()[1], outputDims.begin()[2], 1);

            m_inputTensor.init(storageOrder, dims3In);
            m_outputTensor.init(storageOrder, dims3Out);
            m_referenceOutputTensor.init(storageOrder, dims3Out);

            allocBuffer(m_inputTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            const SingleTest* test = m_currentTest;
            m_convertParams = reinterpret_cast<sw_params::ConvertParams*>(paramContainer);
            *m_convertParams = sw_params::ConvertParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::ConvertParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_convertParams);

            m_params.kernel = reinterpret_cast<uint32_t>(sk_single_shave_convert_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void initParserRunner() override
        {
            initMyriadResources();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inBuff;
            OpTensor outBuff;
            m_inputTensor.exportToBuffer(inBuff);
            m_outputTensor.exportToBuffer(outBuff);

            customCppOp->addInputBuffer(inBuff, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        void resetOutputData() override
        {
            resetTensorBuffer(m_outputTensor);
        }

        void generateInputData() override {

            rand_seed();

            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                if(std::is_same<T_SRC, uint8_t>::value || std::is_same<T_DST, uint8_t>::value ||
                   std::is_same<T_SRC, uint16_t>::value || std::is_same<T_DST, uint16_t>::value ||
                   std::is_same<T_SRC, uint32_t>::value || std::is_same<T_DST, uint32_t>::value) {
                      tmp = fabs(tmp);
                }
                m_inputTensor.at(indices) = convert<T_SRC>(tmp);
            });
        }

        void generateReferenceData() override {
            m_referenceOutputTensor.forEach(false, [&](const MemoryDims& indices) {
                const auto val = m_inputTensor.at(indices);
                T_DST ref = convert<T_DST>(val);
                m_referenceOutputTensor.at(indices) = ref;
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
                const auto value = m_outputTensor.at(indices);
                const auto gt_value = m_referenceOutputTensor.at(indices);
                const auto abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f %f\n", ti.height, ti.width, ti.channels, convert<float>(m_inputTensor.at(indices)), convert<float>(value), convert<float>(gt_value), abs_diff);
                }
            });
            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        Tensor<T_SRC> m_inputTensor;
        Tensor<T_DST> m_outputTensor;
        Tensor<T_DST> m_referenceOutputTensor;

        sw_params::ConvertParams* m_convertParams;
    };

}  // namespace

//
// Test declarations
//

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP16FP32))
{
typedef CustomCppConvertTest<fp16, float> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP32FP16))
{
typedef CustomCppConvertTest<float, fp16> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP16INT32))
{
typedef CustomCppConvertTest<fp16, int32_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_INT32FP16))
{
typedef CustomCppConvertTest<int32_t, fp16> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_U8FP16))
{
typedef CustomCppConvertTest<uint8_t, fp16> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_U8FP32))
{
typedef CustomCppConvertTest<uint8_t, float> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP16U8))
{
typedef CustomCppConvertTest<fp16, uint8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP32U8))
{
typedef CustomCppConvertTest<float, uint8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S32U8))
{
typedef CustomCppConvertTest<int32_t, uint8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S32FP32))
{
typedef CustomCppConvertTest<int32_t, float> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP32S32))
{
typedef CustomCppConvertTest<float, int32_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_U8S32))
{
typedef CustomCppConvertTest<uint8_t, int32_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S8U8))
{
typedef CustomCppConvertTest<int8_t, uint8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S8S32))
{
typedef CustomCppConvertTest<int8_t, int32_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S8FP16))
{
typedef CustomCppConvertTest<int8_t, fp16> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S8FP32))
{
typedef CustomCppConvertTest<int8_t, float> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_U8S8))
{
typedef CustomCppConvertTest<uint8_t, int8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S32S8))
{
typedef CustomCppConvertTest<int32_t, int8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP16S8))
{
typedef CustomCppConvertTest<fp16, int8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP32S8))
{
typedef CustomCppConvertTest<float, int8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S32S64))
{
typedef CustomCppConvertTest<int32_t, int64_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S8S64))
{
typedef CustomCppConvertTest<int8_t, int64_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP16S64))
{
typedef CustomCppConvertTest<fp16, int64_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_FP32S64))
{
typedef CustomCppConvertTest<fp32, int64_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_U8S64))
{
typedef CustomCppConvertTest<uint8_t, int64_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S64S32))
{
typedef CustomCppConvertTest<int64_t, int32_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S64S8))
{
typedef CustomCppConvertTest<int64_t, int8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S64FP16))
{
typedef CustomCppConvertTest<int64_t, fp16> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S64FP32))
{
typedef CustomCppConvertTest<int64_t, fp32> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Convert_S64U8))
{
typedef CustomCppConvertTest<int64_t, uint8_t> convert_tests;
ICV_TESTS_REGISTER_SUITE(convert_tests)
}
