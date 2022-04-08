//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_reverse_sequence.3720xx.text.xdat"

#include "param_reverse_sequence.h"

struct ReverseSequenceTest {
    Dims inputDims;
    Dims sequence_length;
    Dims outputDims;
    StorageOrder storageOrder;
    CustomParams customLayerParams;
};

static inline StorageOrder maskOrder(StorageOrder fullOrder, int nOrd) {
    return static_cast<StorageOrder>(fullOrder & (0xffffffffu >> ((MAX_DIMS - nOrd) * HEX_DIGIT_BITS)));
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, ReverseSequence))
{

    const std::initializer_list<ReverseSequenceTest> reverse_sequence_tests_list = {
        {{5, 4, 3}, {3}, {5, 4, 3}, orderCHW, {2/*batch_axis*/, 1/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},
        {{15, 41, 3}, {15}, {15, 41, 3}, orderCHW, {0/*batch_axis*/, 1/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},
        {{5, 4, 32}, {4}, {5, 4, 32}, orderCHW, {1/*batch_axis*/, 2/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},

        {{2, 2, 2}, {2}, {2, 2, 2}, orderCHW, {2/*batch_axis*/, 1/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},
        {{1, 1, 32}, {1}, {1, 1, 32}, orderCHW, {1/*batch_axis*/, 2/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},
        {{5, 1, 1}, {1}, {5, 1, 1}, orderCHW, {2/*batch_axis*/, 1/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},
        {{1, 41, 1}, {1}, {1, 41, 1}, orderCHW, {0/*batch_axis*/, 1/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},
        {{6, 23, 32}, {23}, {6, 23, 32}, orderCHW, {1/*batch_axis*/, 2/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},
        {{17, 65, 2}, {65}, {17, 65, 2}, orderCHW, {1/*batch_axis*/, 0/*seq_axis*/, sw_params::Location::NN_CMX/*mem type*/}},
    };

    class CustomCppReverseSequenceTest: public CustomCppTests<fp16, ReverseSequenceTest> {
    public:
        explicit CustomCppReverseSequenceTest(): m_testsLoop(reverse_sequence_tests_list, "test") {}
        virtual ~CustomCppReverseSequenceTest() {}
    protected:
        const char* suiteName() const override
        {
            return "CustomCppReverseSequenceTest";
        }
        void userLoops() override
        {
            addLoop(m_testsLoop);
        }

        void initData() override {

            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            initTestCase();

            const Dims& inputDims = m_currentTest->inputDims;
            const Dims& sequenceLengthDims = m_currentTest->sequence_length;
            const Dims& outputDims = m_currentTest->outputDims;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            int32_t numInputDims = inputDims.size();
            int32_t numSequenceLengthDims = sequenceLengthDims.size();
            int32_t numOutputDims = outputDims.size();

            const StorageOrder inputOrder = maskOrder(storageOrder, numInputDims);
            const StorageOrder sequenceLengthOrder = maskOrder(storageOrder, numSequenceLengthDims);
            const StorageOrder outputOrder = maskOrder(storageOrder, numOutputDims);

            const MemoryDims inputMemDims(inputDims.begin(), numInputDims);
            const MemoryDims sequenceLengthMemDims(sequenceLengthDims.begin(), numSequenceLengthDims);
            const MemoryDims outputMemDims(outputDims.begin(), numOutputDims);

            m_inputTensor.init(inputOrder, inputMemDims);
            m_sequenceLengthTensor.init(sequenceLengthOrder, sequenceLengthMemDims);
            m_outputTensor.init(outputOrder, outputMemDims);
            m_referenceOutputTensor.init(outputOrder, outputMemDims);

            allocBuffer(m_inputTensor);
            allocBuffer(m_sequenceLengthTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            m_batch_axis = static_cast<int>(m_currentTest->customLayerParams.layerParams[0]);
            m_seq_axis = static_cast<int>(m_currentTest->customLayerParams.layerParams[1]);
            m_ndims = numInputDims;
            m_batch_axis = m_batch_axis < 0 ? m_batch_axis + m_ndims : m_batch_axis;
            m_seq_axis = m_seq_axis < 0 ? m_seq_axis + m_ndims : m_seq_axis;

            m_reverseSequenceParams = reinterpret_cast<sw_params::ReverseSequenceParams*>(paramContainer);
            *m_reverseSequenceParams = sw_params::ReverseSequenceParams();

            m_reverseSequenceParams->batch_axis = m_batch_axis;
            m_reverseSequenceParams->seq_axis = m_seq_axis;

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::ReverseSequenceParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(m_currentTest->customLayerParams.layerParams[2]);
            m_params.baseParamData = sw_params::reverseSequenceParamsToBaseKernelParams(m_reverseSequenceParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_single_shave_reverse_sequence_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.0f;
        }

        void initParserRunner() override {
            initMyriadResources();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);

            OpTensor inputBuff;
            m_inputTensor.exportToBuffer(inputBuff);
            customCppOp->addInputBuffer(inputBuff, m_requiredTensorLocation);

            OpTensor sequenceLengthBuff;
            m_sequenceLengthTensor.exportToBuffer(sequenceLengthBuff);
            customCppOp->addInputBuffer(sequenceLengthBuff, m_requiredTensorLocation);

            OpTensor outputBuff;
            m_outputTensor.exportToBuffer(outputBuff);
            customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);

            customCppOp->ops = *getParams();
        }

        void generateInputData() override {
            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });

            MemoryDims md_dims(m_currentTest->inputDims.begin(), m_ndims);
            int32_t *psequence_length = m_sequenceLengthTensor.data();
            int psequence_length_size = m_sequenceLengthTensor.dataSize();
            for (int i = 0; i < psequence_length_size; i++) {
                psequence_length[i] = md_dims.dims[m_seq_axis];
            }
        }

        void generateReferenceData() override {
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                int32_t *psequence_length = m_sequenceLengthTensor.data();
                auto ti_mem = indices;
                int new_ind = psequence_length[ti_mem.dims[m_batch_axis]] - 1 - ti_mem.dims[m_seq_axis];
                if (new_ind >= 0)
                    ti_mem.dims[m_seq_axis] = new_ind;
                m_referenceOutputTensor.at(indices) = m_inputTensor.at(ti_mem);
            });
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
                float ref_value = f16Tof32(m_referenceOutputTensor.at(indices));
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
        ListIterator<ReverseSequenceTest> m_testsLoop;

        int m_batch_axis;
        int m_seq_axis;
        int m_ndims;

        Tensor<int32_t> m_sequenceLengthTensor;

        sw_params::ReverseSequenceParams * m_reverseSequenceParams;

    };

    ICV_TESTS_REGISTER_SUITE(CustomCppReverseSequenceTest)
}
