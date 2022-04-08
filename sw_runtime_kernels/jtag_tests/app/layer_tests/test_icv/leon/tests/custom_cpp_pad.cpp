//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

#include "param_pad.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_pad.3720xx.text.xdat"

#define USE_SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Pad))
{
    enum class padMode {
        Constant = 0, Edge = 1, Reflect = 2, Symmetric = 3
    };

    #define CONSTANT static_cast<int>(padMode::Constant)
    #define EDGE static_cast<int>(padMode::Edge)
    #define REFLECT static_cast<int>(padMode::Reflect)
    #define SYMMETRIC static_cast<int>(padMode::Symmetric)

    static constexpr std::initializer_list<SingleTest> pad_test_list
    {                //W H C                              //N1 N2 C1 C2 H1 H2 W1 W2
        {{1, 2, 3}, {1, 2, 3}, orderZYX, {{0, 0, 1, 1, 1, 0, 0, 0, CONSTANT, sw_params::Location::NN_CMX,}}},
        {{2, 32, 4}, {2, 32, 4}, orderZYX, {{0, 0, 11, 1, 1, 12, 1, 1, CONSTANT, sw_params::Location::NN_CMX,}}},

        {{2, 3, 4}, {2, 3, 4}, orderZYX, {{0, 0, 2, 2, 1, 1, 1, 1, EDGE, sw_params::Location::NN_CMX,}}},
        {{2, 3, 4}, {2, 3, 4}, orderZYX, {{0, 0, 0, 0, 0, 0, 0, 0, EDGE, sw_params::Location::NN_CMX,}}},
        {{2, 3, 14}, {2, 3, 14}, orderZYX, {{0, 0, 2, 23, 1, 1, 1, 1, EDGE, sw_params::Location::NN_CMX,}}},
        {{20, 3, 4}, {20, 3, 4}, orderZYX, {{0, 0, 0, 0, 6, 7, 1, 0, EDGE, sw_params::Location::NN_CMX,}}},
        {{2, 2, 2}, {2, 2, 2}, orderZYX, {{0, 0, 1, 1, 0, 6, 0, 0, EDGE, sw_params::Location::NN_CMX,}}},

        {{2, 3, 4}, {2, 3, 4}, orderZYX, {{0, 0, 2, 2, 1, 1, 1, 1, REFLECT, sw_params::Location::NN_CMX,}}},
        {{21, 3, 4}, {21, 3, 4}, orderZYX, {{0, 0, 2, 2, 1, 1, 1, 1, REFLECT, sw_params::Location::NN_CMX,}}},
        {{2, 3, 14}, {2, 3, 14}, orderZYX, {{0, 0, 2, 2, 1, 1, 0, 1, REFLECT, sw_params::Location::NN_CMX,}}},
        {{2, 13, 4}, {2, 13, 4}, orderZYX, {{0, 0, 0, 0, 1, 1, 0, 0, REFLECT, sw_params::Location::NN_CMX,}}},

        {{2, 3, 41}, {2, 3, 41}, orderZYX, {{0, 0, 2, 2, 1, 1, 1, 1, SYMMETRIC, sw_params::Location::NN_CMX,}}},
        {{2, 31, 4}, {2, 31, 4}, orderZYX, {{0, 0, 2, 2, 3, 1, 1, 1, SYMMETRIC, sw_params::Location::NN_CMX,}}},
        {{2, 3, 4}, {2, 3, 4}, orderZYX, {{0, 0, 2, 0, 1, 1, 0, 0, SYMMETRIC, sw_params::Location::NN_CMX,}}},
    };

    class CustomCppPadTest: public CustomCppTests<fp16> {
    public:
        explicit CustomCppPadTest(): m_testsLoop(pad_test_list, "test") {}
        virtual ~CustomCppPadTest() {}
    protected:
        const char* suiteName() const override
        {
            return "CustomCppPadTest";
        }
        void userLoops() override
        {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            CustomCppTests<fp16>::initData();

            const Dims& inputDims = m_currentTest->inputDims;
            const Dims& outputDims = m_currentTest->outputDims;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            const TensorDims dims3In(inputDims.begin()[0], inputDims.begin()[1], inputDims.begin()[2], 1);
            const TensorDims dims3Out(outputDims.begin()[0], outputDims.begin()[1], outputDims.begin()[2], 1);

            const SingleTest* test = m_currentTest;

            m_padw_begin = static_cast<int>(test->customLayerParams.layerParams[6]);
            m_padw_end = static_cast<int>(test->customLayerParams.layerParams[7]);
            m_padh_begin = static_cast<int>(test->customLayerParams.layerParams[4]);
            m_padh_end = static_cast<int>(test->customLayerParams.layerParams[5]);
            m_padc_begin = static_cast<int>(test->customLayerParams.layerParams[2]);
            m_padc_end = static_cast<int>(test->customLayerParams.layerParams[3]);
            TensorDims pad(m_padw_begin + m_padw_end, m_padh_begin + m_padh_end, m_padc_begin + m_padc_end, 0);
            
            m_inputTensor.init(storageOrder, dims3In);
            m_outputTensor.init(storageOrder, dims3Out + pad);
            m_referenceOutputTensor.init(storageOrder, dims3Out + pad);

            allocBuffer(m_inputTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            m_padParams = reinterpret_cast<sw_params::PadParams*>(paramContainer);
            *m_padParams = sw_params::PadParams();

            m_padParams->pad_begin[0] = m_padw_begin;
            m_padParams->pad_begin[1] = m_padh_begin;
            m_padParams->pad_begin[2] = m_padc_begin;
            
            m_padParams->pad_end[0] = m_padw_end;
            m_padParams->pad_end[1] = m_padh_end;
            m_padParams->pad_end[2] = m_padc_end;
            
            m_pad_mode = static_cast<int>(test->customLayerParams.layerParams[8]);
            m_pad_value = -3.0f;

            m_padParams->pad_value = m_pad_value;
            m_padParams->pad_mode = m_pad_mode;

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::PadParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[9]);
            m_params.baseParamData = sw_params::padParamsToBaseKernelParams(m_padParams);

            m_params.kernel = reinterpret_cast<uint32_t>(sk_single_shave_pad_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.0f;
        }

        void generateInputData() override {
            auto seedValue = USE_SEED_VALUE;

            std::mt19937 generator(seedValue);
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                m_inputTensor.at(indices) = f32Tof16(float(generator()) / generator.max() * 256);
            });
        }

        void generateReferenceData() override {}

        fp16 calcReferenceOutput(TensorDims ti)
        {
            const auto& dims = m_inputTensor.tensorDims();

            ti.width  -= m_padw_begin;
            ti.height -= m_padh_begin;
            ti.channels -= m_padc_begin;

            // inside input tensor
            if ((ti.width >= 0 && ti.width < dims.width) &&
                (ti.height >= 0 && ti.height < dims.height) &&
                (ti.channels >= 0 && ti.channels < dims.channels))
            {
                return m_inputTensor.at(ti);
            }
            else
            {
                if (m_pad_mode == CONSTANT)
                {
                    return f32Tof16(m_pad_value);
                }
                else if (m_pad_mode == EDGE)
                {
                    ti.width = std::min(std::max(ti.width, 0), dims.width - 1);
                    ti.height = std::min(std::max(ti.height, 0), dims.height - 1);
                    ti.channels = std::min(std::max(ti.channels, 0), dims.channels - 1);

                    return m_inputTensor.at(ti);
                }
                else if (m_pad_mode == REFLECT || m_pad_mode == SYMMETRIC)
                {
                    int mode_offset = (m_pad_mode == SYMMETRIC) ? 1 : 0;

                    if (ti.width > dims.width - 1) ti.width = dims.width-1 - (ti.width - (dims.width-1)) + mode_offset;
                    if (ti.width < 0) ti.width = -ti.width - mode_offset;

                    if (ti.height > dims.height - 1) ti.height = dims.height-1 - (ti.height - (dims.height-1)) + mode_offset;
                    if (ti.height < 0) ti.height = -ti.height - mode_offset;

                    if (ti.channels > dims.channels - 1) ti.channels = dims.channels-1 - (ti.channels - (dims.channels-1)) + mode_offset;
                    if (ti.channels < 0) ti.channels = -ti.channels - mode_offset;

                    return m_inputTensor.at(ti);
                }
            }
            return 0;
        }

        virtual bool checkResult() override
        {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                const TensorDims ti = m_outputTensor.toTensor(indices);
                float ref_value = f16Tof32(calcReferenceOutput(ti));

                float abs_diff = fabs(value - ref_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF WHC [%d:%d:%d] %f %f %f %f\n", ti.width, ti.height, ti.channels, f16Tof32(m_inputTensor.at(indices)), value, ref_value,
                           abs_diff);
                }
            });

            return !threshold_test_failed;
        }
    private:
        ListIterator<SingleTest> m_testsLoop;

        float m_pad_value;
        int m_pad_mode;
        int m_padw_begin;
        int m_padw_end;
        int m_padh_begin;
        int m_padh_end;
        int m_padc_begin;
        int m_padc_end;
        sw_params::PadParams * m_padParams;

    };

    ICV_TESTS_REGISTER_SUITE(CustomCppPadTest)
}
