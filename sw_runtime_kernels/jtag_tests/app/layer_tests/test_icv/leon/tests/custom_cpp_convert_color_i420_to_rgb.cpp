
//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_convert_color_i420_to_rgb.3720xx.text.xdat"

#include "param_convert_color_i420_to_rgb.h"

#define NAMESPACE ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, I420ToRGB))

enum class RgbFormat : int32_t {
    RGB = 0,
    BGR = 1
};

struct I420ToRGBTestParams {
    RgbFormat color_format;
    int32_t layerParams[MAX_LOCAL_PARAMS];
};

struct I420ToRGBTest {
    Dims inputDims;
    Dims inputDims2;
    Dims inputDims3;
    Dims outputDims;
    StorageOrder storageOrder;
    I420ToRGBTestParams customLayerParams;
};

namespace NAMESPACE {
const bool save_to_file = false;
static const size_t N_DIM = 3;
static const size_t H_DIM = 2;
static const size_t W_DIM = 1;
static const size_t C_DIM = 0;

inline fp16 genData() {
    float val = (float)(rand() % 256);
    return static_cast<fp16>(f32Tof16(val));
}

static std::initializer_list<I420ToRGBTest> i420torgb_test_list{{{1, 4, 6, 1},
                                                                 {1, 2, 3, 1},
                                                                 {1, 2, 3, 1},
                                                                 {3, 4, 6, 1},
                                                                 FULL_ORDER,
                                                                 {RgbFormat::RGB, {sw_params::Location::NN_CMX}}},
                                                                {{1, 4, 6, 1},
                                                                 {1, 2, 3, 1},
                                                                 {1, 2, 3, 1},
                                                                 {3, 4, 6, 1},
                                                                 FULL_ORDER,
                                                                 {RgbFormat::BGR, {sw_params::Location::NN_CMX}}}};

class CustomCppI420ToRGBTest : public CustomCppTests<fp16, I420ToRGBTest> {
public:
    explicit CustomCppI420ToRGBTest(): m_testsLoop(i420torgb_test_list, "test") {
    }
    virtual ~CustomCppI420ToRGBTest() {
    }

protected:
    const char* suiteName() const override {
        return "CustomCppI420ToRGBTest";
    }

    void userLoops() override {
        addLoop(m_testsLoop);
    }

    void initData() override {
        sw_params::BaseKernelParams emptyParamData;
        m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};
        CustomCppTests<fp16, I420ToRGBTest>::initData();
        const I420ToRGBTest* test = m_currentTest;
        const auto& params = m_testsLoop.value();
        const StorageOrder& storageOrder = m_currentTest->storageOrder;

        y_inputDims = params.inputDims;
        m_color_format = params.customLayerParams.color_format;

        u_inputDims = {0};
        v_inputDims = {0};
        u_inputDims = params.inputDims2;
        v_inputDims = params.inputDims3;

        const auto y_inputDimsArr = y_inputDims.begin();
        const auto u_inputDimsArr = u_inputDims.begin();
        const auto v_inputDimsArr = v_inputDims.begin();

        // construct output dims
        auto outputDims = y_inputDims;
        outputDims[C_DIM] = 3;  // 3 is number of channels (R, G, B) or (B, G, R)

        mvTensorAssert(y_inputDimsArr[H_DIM] == u_inputDimsArr[H_DIM] * 2,
                       "Y input height shall be 2 times bigger that U input height.");
        mvTensorAssert(y_inputDimsArr[H_DIM] == v_inputDimsArr[H_DIM] * 2,
                       "Y input height shall be 2 times bigger that V input height.");
        mvTensorAssert(y_inputDimsArr[W_DIM] == u_inputDimsArr[W_DIM] * 2,
                       "Y input width shall be 2 times bigger that U input width.");
        mvTensorAssert(y_inputDimsArr[W_DIM] == v_inputDimsArr[W_DIM] * 2,
                       "Y input width shall be 2 times bigger that V input width.");
        mvTensorAssert(u_inputDimsArr[C_DIM] == 1, "U channels dimension shall be either dynamic or equal to 1.");
        mvTensorAssert(v_inputDimsArr[C_DIM] == 1, "U channels dimension shall be either dynamic or equal to 1.");
        mvTensorAssert(y_inputDimsArr[N_DIM] == u_inputDimsArr[N_DIM], "Y input batch shall be same as U input batch.");
        mvTensorAssert(y_inputDimsArr[N_DIM] == v_inputDimsArr[N_DIM], "Y input batch shall be same as V input batch.");
        mvTensorAssert(outputDims[H_DIM] % 2 == 0, "Image height must be even.");
        mvTensorAssert(outputDims[W_DIM] % 2 == 0, "Image width must be even.");

        MemoryDims md_y_input_dims(y_inputDims.data(), y_inputDims.size());
        MemoryDims md_u_input_dims(u_inputDims.data(), u_inputDims.size());
        MemoryDims md_v_input_dims(v_inputDims.data(), v_inputDims.size());
        MemoryDims md_output_dims(outputDims.data(), outputDims.size());

        m_y_inputTensor.init(maskOrder(storageOrder, y_inputDims.size()), md_y_input_dims, md_y_input_dims);
        m_u_inputTensor.init(maskOrder(storageOrder, u_inputDims.size()), md_u_input_dims, md_u_input_dims);
        m_v_inputTensor.init(maskOrder(storageOrder, v_inputDims.size()), md_v_input_dims, md_v_input_dims);
        m_outputTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);
        m_referenceTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);

        allocBuffer(m_y_inputTensor);
        allocBuffer(m_u_inputTensor);
        allocBuffer(m_v_inputTensor);

        allocBuffer(m_outputTensor);
        allocBuffer(m_referenceTensor);

        m_i420torgbParams = reinterpret_cast<sw_params::ConvertColorI420ToRGBParams*>(paramContainer);
        *m_i420torgbParams = sw_params::ConvertColorI420ToRGBParams();
        m_i420torgbParams->outFmt = (int64_t)m_color_format;

        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
        m_params.paramDataLen = sizeof(sw_params::ConvertColorI420ToRGBParams);
        m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_i420torgbParams);
        m_params.kernel = reinterpret_cast<uint64_t>(sk_single_shave_convert_color_i420_to_rgb_3720xx_text);
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.005f;
    }

    void initParserRunner() override {
        initMyriadResources();
        static_assert(std::is_base_of<Op, CustomCpp>());
        CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
        OpTensor inputBuff1;
        OpTensor inputBuff2;
        OpTensor inputBuff3;
        OpTensor outputBuff;
        m_y_inputTensor.exportToBuffer(inputBuff1);
        m_u_inputTensor.exportToBuffer(inputBuff2);
        m_v_inputTensor.exportToBuffer(inputBuff3);
        m_outputTensor.exportToBuffer(outputBuff);
        customCppOp->addInputBuffer(inputBuff1, m_requiredTensorLocation);
        customCppOp->addInputBuffer(inputBuff2, m_requiredTensorLocation);
        customCppOp->addInputBuffer(inputBuff3, m_requiredTensorLocation);
        customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);
        customCppOp->ops = *getParams();
    }

    void resetOutputData() override {
        resetTensorBuffer(m_outputTensor);
    }

    void generateReferenceData() override {
        const auto& test_value = m_testsLoop.value();

        const auto y_src = m_y_inputTensor.data();
        auto u_src = m_u_inputTensor.data();
        auto v_src = m_v_inputTensor.data();
        auto dst = m_referenceTensor.data();

        std::vector<int32_t> y_dims(test_value.inputDims.begin(), test_value.inputDims.end());

        std::vector<int32_t> u_dims(test_value.inputDims2.begin(), test_value.inputDims2.end());

        std::vector<int32_t> v_dims(test_value.inputDims3.begin(), test_value.inputDims3.end());

        size_t batch_size = y_dims[N_DIM];
        size_t image_h = y_dims[H_DIM];
        size_t image_w = y_dims[W_DIM];
        size_t stride_y = image_w * image_h;
        size_t stride_uv = image_w / 2 * image_h / 2;

        for (size_t batch = 0; batch < batch_size; batch++) {
            auto out = dst + batch * image_w * image_h * 3;
            auto y_ptr = y_src + batch * stride_y;
            auto u_ptr = u_src + batch * stride_uv;
            auto v_ptr = v_src + batch * stride_uv;
            for (size_t h = 0; h < image_h; h++) {
                for (size_t w = 0; w < image_w; w++) {
                    auto y_index = h * image_w + w;
                    auto uv_index = (h / 2) * (image_w / 2) + (w / 2);
                    float y_val, u_val, v_val;
                    y_val = f16Tof32(y_ptr[y_index]);
                    u_val = f16Tof32(u_ptr[uv_index]);
                    v_val = f16Tof32(v_ptr[uv_index]);
                    auto c = y_val - 16.f;
                    auto d = u_val - 128.f;
                    auto e = v_val - 128.f;

                    auto clip = [](float a) {
                        return f32Tof16(static_cast<float>(std::min(std::max(a, 0.f), 255.f)));
                    };

                    auto b = clip(1.164f * c + 2.018f * d);
                    auto g = clip(1.164f * c - 0.391f * d - 0.813f * e);
                    auto r = clip(1.164f * c + 1.596f * e);

                    if (m_color_format == RgbFormat::RGB) {
                        out[y_index * 3] = r;
                        out[y_index * 3 + 1] = g;
                        out[y_index * 3 + 2] = b;
                    } else if (m_color_format == RgbFormat::BGR) {
                        out[y_index * 3] = b;
                        out[y_index * 3 + 1] = g;
                        out[y_index * 3 + 2] = r;
                    }
                }
            }
        }
    }

    void generateInputData() override {
        m_y_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            m_y_inputTensor.at(indices) = genData();
        });
        m_u_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            m_v_inputTensor.at(indices) = genData();
        });
        m_v_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            m_u_inputTensor.at(indices) = genData();
        });
    }

    virtual bool checkResult() override {
        m_outputTensor.confirmBufferData();
        m_referenceTensor.confirmBufferData();

        // save output data
        if (save_to_file) {
            saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                             "outMyriad.bin");
        }

        bool threshold_test_failed = false;
        m_outputTensor.forEach(true, [&](const MemoryDims& indices) {
            const auto gt_value = f16Tof32(m_referenceTensor.at(indices, true));
            const auto value = f16Tof32(m_outputTensor.at(indices, true));

            float abs_diff = fabs(value - gt_value);
            bool differ = !bool(abs_diff <= m_test_threshold);
            threshold_test_failed |= differ;
            if (differ) {
                char indices_str[64];
                printf("DIFF [%s] val: %f ref: %f\n", m_outputTensor.indicesToString(indices, indices_str), value,
                       gt_value);
            }
        });

        resetTensorBuffer(m_referenceTensor);

        return !threshold_test_failed;
    }

private:
    ListIterator<I420ToRGBTest> m_testsLoop;
    sw_params::ConvertColorI420ToRGBParams* m_i420torgbParams;
    RgbFormat m_color_format;
    std::vector<int32_t> y_inputDims;
    std::vector<int32_t> u_inputDims;
    std::vector<int32_t> v_inputDims;
    Tensor<fp16> m_y_inputTensor;
    Tensor<fp16> m_u_inputTensor;
    Tensor<fp16> m_v_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceTensor;
};
ICV_TESTS_REGISTER_SUITE(CustomCppI420ToRGBTest)
}  // namespace NAMESPACE
