//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_convert_color_nv12_to_rgb_single_plane.3720xx.text.xdat"

#include "param_convert_color_nv12_to_rgb_single_plane.h"

#define NAMESPACE ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, ConvertColorNV12ToRGBSinglePlane))

enum class RgbFormat : int {
    RGB = 0,
    BGR = 1
};

struct ConvertColorNV12ToRGBSinglePlaneTestParams {
    RgbFormat color_format;
    int32_t layerParams[MAX_LOCAL_PARAMS];
};

struct ConvertColorNV12ToRGBSinglePlaneTest {
    Dims inputDims;
    Dims outputDims;
    StorageOrder storageOrder;
    ConvertColorNV12ToRGBSinglePlaneTestParams convertColorNV12ToRGBSinglePlaneTestParams;
};

namespace NAMESPACE {
const bool save_to_file = false;

static const size_t N_DIM = 3;
static const size_t H_DIM = 2;
static const size_t W_DIM = 1;
static const size_t C_DIM = 0;

static constexpr std::initializer_list<ConvertColorNV12ToRGBSinglePlaneTest>
        ConvertColorNV12ToRGBSinglePlane_test_list = {
                {{1, 2, 3, 6}, {}, FULL_ORDER, {RgbFormat::RGB, {sw_params::Location::NN_CMX}}},
                {{1, 8, 6, 1}, {}, FULL_ORDER, {RgbFormat::RGB, {sw_params::Location::NN_CMX}}},

                {{1, 2, 3, 6}, {}, FULL_ORDER, {RgbFormat::BGR, {sw_params::Location::NN_CMX}}},
                {{1, 8, 6, 1}, {}, FULL_ORDER, {RgbFormat::BGR, {sw_params::Location::NN_CMX}}},

};

class CustomCppConvertColorNV12ToRGBSinglePlaneTest :
        public CustomCppTests<fp16, ConvertColorNV12ToRGBSinglePlaneTest> {
public:
    explicit CustomCppConvertColorNV12ToRGBSinglePlaneTest()
            : m_testsLoop(ConvertColorNV12ToRGBSinglePlane_test_list, "test") {
    }
    virtual ~CustomCppConvertColorNV12ToRGBSinglePlaneTest() {
    }

protected:
    const char* suiteName() const override {
        return "CustomCppConvertColorNV12ToRGBSinglePlaneTest";
    }
    void userLoops() override {
        addLoop(m_testsLoop);
    }

    void initData() override {
        sw_params::BaseKernelParams emptyParamData;
        m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};
        CustomCppTests<fp16, ConvertColorNV12ToRGBSinglePlaneTest>::initData();

        const StorageOrder& storageOrder = m_currentTest->storageOrder;

        std::vector<int32_t> y_inputDims = m_testsLoop.value().inputDims;
        m_color_format = m_testsLoop.value().convertColorNV12ToRGBSinglePlaneTestParams.color_format;

        const auto y_inputDimsArr = y_inputDims.begin();

        // construct output dims
        auto outputDims = y_inputDims;
        outputDims[C_DIM] = 3;  // 3 is number of channels (R, G, B) or (B, G, R)

        mvTensorAssert(y_inputDimsArr[H_DIM] % 3 == 0, "NV12 image height shall be divisible by 3.");
        outputDims[H_DIM] = y_inputDimsArr[H_DIM] * 2 / 3;

        mvTensorAssert(outputDims[H_DIM] % 2 == 0, "Image height must be even.");
        mvTensorAssert(outputDims[W_DIM] % 2 == 0, "Image width must be even.");

        MemoryDims md_y_input_dims(y_inputDims.data(), y_inputDims.size());
        MemoryDims md_output_dims(outputDims.data(), outputDims.size());

        m_y_inputTensor.init(maskOrder(storageOrder, y_inputDims.size()), md_y_input_dims, md_y_input_dims);
        m_outputTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);
        m_referenceTensor.init(maskOrder(storageOrder, outputDims.size()), md_output_dims, md_output_dims);

        allocBuffer(m_y_inputTensor);
        allocBuffer(m_outputTensor);
        allocBuffer(m_referenceTensor);

        m_ConvertColorNV12ToRGBSinglePlaneParams =
                reinterpret_cast<sw_params::ConvertColorNV12ToRGBSinglePlaneParams*>(paramContainer);
        *m_ConvertColorNV12ToRGBSinglePlaneParams = sw_params::ConvertColorNV12ToRGBSinglePlaneParams();
        m_ConvertColorNV12ToRGBSinglePlaneParams->rgbFormat = (int64_t)m_color_format;

        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
        m_params.paramDataLen = sizeof(sw_params::ConvertColorNV12ToRGBSinglePlaneParams);
        m_requiredTensorLocation = static_cast<sw_params::Location>(
                m_currentTest->convertColorNV12ToRGBSinglePlaneTestParams.layerParams[0]);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_ConvertColorNV12ToRGBSinglePlaneParams);

        m_params.kernel =
                reinterpret_cast<uint32_t>(sk_single_shave_convert_color_nv12_to_rgb_single_plane_3720xx_text);
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.005f;
    }

    void initParserRunner() override {
        initMyriadResources();

        static_assert(std::is_base_of<Op, CustomCpp>());
        CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
        OpTensor yBuff;
        OpTensor outBuff;
        m_y_inputTensor.exportToBuffer(yBuff);
        m_outputTensor.exportToBuffer(outBuff);

        customCppOp->addInputBuffer(yBuff, m_requiredTensorLocation);
        customCppOp->addOutputBuffer(outBuff, m_requiredTensorLocation);
        customCppOp->ops = *getParams();
    }

    void generateReferenceData() override {
        const auto& test_value = m_testsLoop.value();

        const auto y_src = m_y_inputTensor.data();
        auto dst = m_referenceTensor.data();

        std::vector<int32_t> y_dims(test_value.inputDims.begin(), test_value.inputDims.end());

        std::vector<int32_t> output_dims;
        for (int i = 0; i < m_outputTensor.ndims(); i++)
            output_dims.push_back(m_outputTensor.memoryDims().dims[i]);

        size_t batch_size = y_dims[N_DIM];
        size_t image_h = y_dims[H_DIM] * 2 / 3;
        size_t image_w = y_dims[W_DIM];
        size_t stride_y = image_w * image_h * 3 / 2;
        size_t stride_uv = image_w * image_h * 3 / 2;

        auto uv_src = y_src + image_w * image_h;

        for (size_t batch = 0; batch < batch_size; batch++) {
            auto out = dst + batch * image_w * image_h * 3;
            auto y_ptr = y_src + batch * stride_y;
            auto uv_ptr = uv_src + batch * stride_uv;
            for (size_t h = 0; h < image_h; h++) {
                for (size_t w = 0; w < image_w; w++) {
                    auto y_index = h * image_w + w;
                    auto uv_index = (h / 2) * image_w + (w / 2) * 2;
                    float y_val, u_val, v_val;

                    y_val = f16Tof32(y_ptr[y_index]);
                    u_val = f16Tof32(uv_ptr[uv_index]);
                    v_val = f16Tof32(uv_ptr[uv_index + 1]);

                    auto c = y_val - 16.f;
                    auto d = u_val - 128.f;
                    auto e = v_val - 128.f;

                    auto clip = [](float a) -> fp16 {
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
            float val = (float)(rand() % 256);
            m_y_inputTensor.at(indices) = static_cast<fp16>(f32Tof16(val));
        });

        generateReferenceData();
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
            const auto gt_value = m_referenceTensor.at(indices, true);
            const auto value = m_outputTensor.at(indices, true);

            bool differ = !(fabsf(value - gt_value) <= m_test_threshold * fabsf(gt_value));

            threshold_test_failed |= differ;

            if (differ && GlobalData::doPrintDiffs) {
                const TensorDims ti = m_outputTensor.toTensor(indices);
                float print_value = f16Tof32(value);
                float print_gt_value = f16Tof32(gt_value);

                printf("DIFF NCHW [%d:%d:%d:%d] %f %f\n", ti.batch, ti.channels, ti.height, ti.width, print_value,
                       print_gt_value);
            }
        });

        return !threshold_test_failed;
    }

private:
    ListIterator<ConvertColorNV12ToRGBSinglePlaneTest> m_testsLoop;
    RgbFormat m_color_format;
    Tensor<fp16> m_y_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceTensor;

    sw_params::ConvertColorNV12ToRGBSinglePlaneParams* m_ConvertColorNV12ToRGBSinglePlaneParams;
};

ICV_TESTS_REGISTER_SUITE(CustomCppConvertColorNV12ToRGBSinglePlaneTest)
}  // namespace NAMESPACE
