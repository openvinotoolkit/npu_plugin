//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.permute_quantize.3720xx.text.xdat"

#include "param_permute_quantize.h"

using namespace sw_params;

#define USE_SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed

using namespace sw_params;

uint8_t quantization(fp16 value, double scale, double zero) {
    const auto quantize = f16Tof32(value) / scale + zero + 0.5f;
    return static_cast<uint8_t>(quantize);
}

struct PermuteQuantizeTestParams {
    std::initializer_list<int32_t> inputDims;
    std::initializer_list<int32_t> outputDims;
    PermuteQuantizeOptMode mode;
    StorageOrder storageOrder;
    CustomParams customLayerParams;
};

#define PERMUTE_QUANTIZE_TEST_NAMESPACE ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, PermuteQuantize))

namespace PERMUTE_QUANTIZE_TEST_NAMESPACE {

static constexpr std::initializer_list<PermuteQuantizeTestParams> permute_quantize_test_list{
        // C1
        // W  H  C    C  W  H
        {{3, 3, 1}, {16, 3, 3}, PQ_NCHW_NHWC_C1, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},
        {{12, 5, 1}, {4, 12, 5}, PQ_NCHW_NHWC_C1, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},
        {{32, 3, 1}, {4, 32, 3}, PQ_NCHW_NHWC_C1EXP4, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},

        // C3
        // W  H  C    C  W  H
        {{4, 4, 3}, {16, 4, 4}, PQ_NCHW_NHWC_C3, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},
        {{12, 5, 3}, {16, 12, 5}, PQ_NCHW_NHWC_C3, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},
        {{32, 3, 3}, {4, 32, 3}, PQ_NCHW_NHWC_C3EXP4, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},

        // C4
        // W  H  C    C  W  H
        {{2, 4, 4}, {16, 2, 4}, PQ_NCHW_NHWC_C4, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},
        {{8, 8, 4}, {8, 8, 8}, PQ_NCHW_NHWC_C4, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},
        {{9, 9, 4}, {16, 9, 9}, PQ_NCHW_NHWC_C4, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},
        {{32, 4, 4}, {4, 32, 4}, PQ_NCHW_NHWC_C4EXP4, FULL_ORDER, {{2, 0, 1, sw_params::Location::NN_CMX}}},

        // Custom tests
        // W  H  C    C  H  W
        {{12, 5, 2}, {4, 5, 12}, PQ_NONE, FULL_ORDER, {{2, 1, 0, sw_params::Location::NN_CMX}}},
        {{12, 5, 2}, {6, 6, 6}, PQ_NONE, FULL_ORDER, {{2, 1, 0, sw_params::Location::NN_CMX}}},
};

class CustomCppPermuteQuantizeTest : public CustomCppTests<fp16, PermuteQuantizeTestParams> {
public:
    explicit CustomCppPermuteQuantizeTest(): m_testsLoop(permute_quantize_test_list, "test") {
    }
    virtual ~CustomCppPermuteQuantizeTest() {
    }

protected:
    const char* suiteName() const override {
        return "CustomCppPermuteQuantizeTest";
    }
    void userLoops() override {
        addLoop(m_testsLoop);
    }
    void initData() override {
        BaseKernelParams emptyParamData;
        m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

        initTestCase();
        const PermuteQuantizeTestParams* test = m_currentTest;

        // Build tensors
        std::vector<int32_t> inputDims = test->inputDims;
        std::vector<int32_t> outputDims = test->outputDims;
        const StorageOrder& storageOrder = test->storageOrder;
        const PermuteQuantizeOptMode mode = test->mode;

        MemoryDims md_input_dims(inputDims.data(), inputDims.size());
        MemoryDims md_output_dims(outputDims.data(), outputDims.size());

        m_inputTensor.init(maskOrder(storageOrder, inputDims.size()), md_input_dims, md_input_dims);
        m_outputTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);
        m_referenceTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);

        allocBuffer(m_inputTensor);
        allocBuffer(m_outputTensor);
        allocBuffer(m_referenceTensor);

        // Build parameters
        m_permute_quantizeParams = reinterpret_cast<PermuteQuantizeParams*>(paramContainer);
        *m_permute_quantizeParams = PermuteQuantizeParams();

        m_permute_quantizeParams->opt_mode = (int64_t)mode;

        // Set permutation parameters
        const int ndims = m_inputTensor.ndims();
        for (int i = 0; i < ndims; ++i) {
            m_permute_quantizeParams->perm[i] = (int64_t)test->customLayerParams.layerParams[i];
        }

        // Set quantize parameters

        const auto rng = [&] {
            return m_distribution(generator);
        };
        const auto rngz = [&] {
            return m_distributionz(generator);
        };

        m_scales = rng();
        m_zeros = rngz();

        m_permute_quantizeParams->scale = (float)m_scales;
        m_permute_quantizeParams->zero = (int64_t)m_zeros;

        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
        m_params.paramDataLen = sizeof(PermuteQuantizeParams);
        m_requiredTensorLocation = static_cast<Location>(test->customLayerParams.layerParams[ndims]);
        m_params.baseParamData = ToBaseKernelParams(m_permute_quantizeParams);
        m_params.kernel = reinterpret_cast<uint32_t>(sk_permute_quantize_3720xx_text);
    }
    void formatTestParams(char* str, int maxLength) const override {
        char inSizes_str[100];
        char outSizes_str[100];

        snprintf_append(str, maxLength, "input: %s, output: %s", m_inputTensor.dimsToStringNCHW(inSizes_str),
                        m_outputTensor.dimsToStringNCHW(outSizes_str));
    }
    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 1;
    }
    void initParserRunner() override {
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

    void resetOutputData() override {
        resetTensorBuffer(m_outputTensor, 0x0);
    }
    void generateInputData() override {
        // input
        std::uniform_real_distribution<float> distrubution(0.0f, 30.0f);
        m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            m_inputTensor.at(indices) = f32Tof16(distrubution(generator));
        });
    }
    void generateReferenceData() override {
        const PermuteQuantizeTestParams* test = m_currentTest;
        const int ndims = m_inputTensor.ndims();

        // reset output
        m_referenceTensor.forEach(false, [&](const MemoryDims& out) {
            m_referenceTensor.at(out) = (unsigned char)0;
        });

        // permute + quantize
        m_inputTensor.forEach(false, [&](const MemoryDims& in) {
            MemoryDims out;
            // Permutation
            permuteArray(in.dims, test->customLayerParams.layerParams, out.dims, ndims);

            // Quantization of permuted value
            const double scaleVal = m_scales;
            const double zeroVal = m_zeros;

            m_referenceTensor.at(out) = quantization(m_inputTensor.at(in), scaleVal, zeroVal);
        });
    }
    bool checkResult() override {
        m_outputTensor.confirmBufferData();
        m_referenceTensor.confirmBufferData();
        const int ndims = m_inputTensor.ndims();

        // save output data
        if (m_save_to_file) {
            saveMemoryToFile(reinterpret_cast<uint32_t>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                             "outMyriad.bin");
        }

        bool threshold_test_failed = false;

        const PermuteQuantizeTestParams* test = m_currentTest;
        int32_t inC = m_inputTensor.memoryDims().dims[2];  // W H C
        int32_t outC = 0;
        for (int i = 0; i < ndims; i++) {
            if (test->customLayerParams.layerParams[i] == 2) {
                outC = m_outputTensor.memoryDims().dims[i];
                break;
            }
        }

        int kC = 0;
        m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
            if (kC < inC) {
                const uint8_t value = m_outputTensor.at(indices);
                const uint8_t gt_value = m_referenceTensor.at(indices);
                const uint8_t abs_diff = abs(value - gt_value);
                const bool differ = !bool(abs_diff <= (uint8_t)m_test_threshold);

                threshold_test_failed |= differ;
                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF NCHW [%d:%d:%d:%d] %d %d %d | %d\n", ti.batch, ti.channels, ti.height, ti.width, value,
                           gt_value, abs_diff, kC);
                }
            }
            kC++;
            if (kC >= outC)
                kC = 0;
        });

        return !threshold_test_failed;
    }

private:
    ListIterator<PermuteQuantizeTestParams> m_testsLoop;
    // Tensors
    Tensor<fp16> m_inputTensor;
    Tensor<uint8_t> m_outputTensor;
    Tensor<uint8_t> m_referenceTensor;

    // Parameters
    float m_scales;
    int64_t m_zeros;

    // Additional test parameters
    PermuteQuantizeParams* m_permute_quantizeParams;
    std::mt19937 generator = std::mt19937(USE_SEED_VALUE);
    std::uniform_real_distribution<double> m_distribution = std::uniform_real_distribution<double>(0.2f, 0.5f);
    std::uniform_real_distribution<double> m_distributionz = std::uniform_real_distribution<double>(0.0f, 0.0f);
};

ICV_TESTS_REGISTER_SUITE(CustomCppPermuteQuantizeTest)

}  // namespace PERMUTE_QUANTIZE_TEST_NAMESPACE
