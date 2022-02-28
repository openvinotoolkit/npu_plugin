//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <custom_cpp_tests.h>
#include <cmath>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.singleShaveInterpolate.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_interpolate.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Interpolate)) {

#define DNEAREST static_cast<int>(InterpolationMethod::NEAREST)
#define DBILINEAR static_cast<int>(InterpolationMethod::BILINEAR)
#define DLINEAR_ONNX static_cast<int>(InterpolationMethod::LINEAR_ONNX)

#define DASYMMETRIC static_cast<int>(InterpolationCoordTransMode::ASYMMETRIC)
#define DHALF_PIXEL static_cast<int>(InterpolationCoordTransMode::HALF_PIXEL)
#define DALIGN_CORNERS static_cast<int>(InterpolationCoordTransMode::ALIGN_CORNERS)
#define DPYTORCH_HALF_PIXEL static_cast<int>(InterpolationCoordTransMode::PYTORCH_HALF_PIXEL)
#define DTF_HALF_PIXEL_FOR_NN static_cast<int>(InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN)

#define DCEIL static_cast<int>(InterpolationNearestMode::CEIL)
#define DFLOOR static_cast<int>(InterpolationNearestMode::FLOOR)
#define DSIMPLE static_cast<int>(InterpolationNearestMode::SIMPLE)
#define DROUND_PREFER_CEIL static_cast<int>(InterpolationNearestMode::ROUND_PREFER_CEIL)
#define DROUND_PREFER_FLOOR static_cast<int>(InterpolationNearestMode::ROUND_PREFER_FLOOR)

    const bool save_to_file = false;
// #define ALL_PARAMS_SET

    static constexpr std::initializer_list<SingleTest> interpolate_test_list{
        // CustomParams order: InterpolationMethod, InterpolationCoordTransMode, InterpolationNearestMode, antialias, sw_params::Location
        {{64, 32, 4}, {16, 16, 4}, orderCHW, FPE("interpolate.elf"), {DNEAREST,     DASYMMETRIC,    DFLOOR,              false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderHWC, FPE("interpolate.elf"), {DNEAREST,     DALIGN_CORNERS, DROUND_PREFER_FLOOR, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderHWC, FPE("interpolate.elf"), {DBILINEAR,    DALIGN_CORNERS, 0,                   false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderCHW, FPE("interpolate.elf"), {DBILINEAR,    DALIGN_CORNERS, 0,                   false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderCHW, FPE("interpolate.elf"), {DLINEAR_ONNX, DALIGN_CORNERS, 0,                   false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderHWC, FPE("interpolate.elf"), {DLINEAR_ONNX, DALIGN_CORNERS, 0,                   false, sw_params::Location::NN_CMX}},
#ifdef ALL_PARAMS_SET
        {{16, 16, 4}, {64, 32, 4}, orderCHW, FPE("interpolate.elf"), {DNEAREST,     DASYMMETRIC,           DFLOOR,              false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderCHW, FPE("interpolate.elf"), {DNEAREST,     DALIGN_CORNERS,        DROUND_PREFER_FLOOR, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderHWC, FPE("interpolate.elf"), {DNEAREST,     DHALF_PIXEL,           DSIMPLE,             false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderCHW, FPE("interpolate.elf"), {DNEAREST,     DHALF_PIXEL,           DROUND_PREFER_CEIL,  false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderHWC, FPE("interpolate.elf"), {DNEAREST,     DPYTORCH_HALF_PIXEL,   DROUND_PREFER_CEIL,  false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderCHW, FPE("interpolate.elf"), {DNEAREST,     DPYTORCH_HALF_PIXEL,   DROUND_PREFER_CEIL,  false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderCHW, FPE("interpolate.elf"), {DNEAREST,     DTF_HALF_PIXEL_FOR_NN, DCEIL,               false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderHWC, FPE("interpolate.elf"), {DNEAREST,     DTF_HALF_PIXEL_FOR_NN, DROUND_PREFER_CEIL,  false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderHWC, FPE("interpolate.elf"), {DLINEAR_ONNX, DASYMMETRIC,           0, false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderCHW, FPE("interpolate.elf"), {DLINEAR_ONNX, DASYMMETRIC,           0, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderCHW, FPE("interpolate.elf"), {DLINEAR_ONNX, DHALF_PIXEL,           0, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderCHW, FPE("interpolate.elf"), {DLINEAR_ONNX, DPYTORCH_HALF_PIXEL,   0, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderCHW, FPE("interpolate.elf"), {DLINEAR_ONNX, DTF_HALF_PIXEL_FOR_NN, 0, false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderHWC, FPE("interpolate.elf"), {DLINEAR_ONNX, DTF_HALF_PIXEL_FOR_NN, 0, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderHWC, FPE("interpolate.elf"), {DBILINEAR,    DHALF_PIXEL,           0, false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderCHW, FPE("interpolate.elf"), {DBILINEAR,    DHALF_PIXEL,           0, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderHWC, FPE("interpolate.elf"), {DBILINEAR,    DASYMMETRIC,           0, false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderHWC, FPE("interpolate.elf"), {DBILINEAR,    DASYMMETRIC,           0, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderCHW, FPE("interpolate.elf"), {DBILINEAR,    DPYTORCH_HALF_PIXEL,   0, false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderHWC, FPE("interpolate.elf"), {DBILINEAR,    DPYTORCH_HALF_PIXEL,   0, false, sw_params::Location::NN_CMX}},
        {{64, 32, 4}, {16, 16, 4}, orderHWC, FPE("interpolate.elf"), {DBILINEAR,    DTF_HALF_PIXEL_FOR_NN, 0, false, sw_params::Location::NN_CMX}},
        {{16, 16, 4}, {64, 32, 4}, orderCHW, FPE("interpolate.elf"), {DBILINEAR,    DTF_HALF_PIXEL_FOR_NN, 0, false, sw_params::Location::NN_CMX}},
#endif
    };

    #define MIN(a, b) ((a)<(b)?(a):(b))
    #define MAX(a, b) ((a)>(b)?(a):(b))

    float alphaCoordinateTransform(InterpolationCoordTransMode coordTransMode, float rotateValue)
    {
        if (coordTransMode == InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN)
            return rotateValue / 2.0f;
        else if (coordTransMode == InterpolationCoordTransMode::PYTORCH_HALF_PIXEL)
            return rotateValue / 2.0f - 0.5f;
        else if (coordTransMode == InterpolationCoordTransMode::ASYMMETRIC)
            return 0.0f;
        else
            return rotateValue / 2.0f - .5f;
    }

    float coordinateTransform(InterpolationCoordTransMode coordTransMode, int x_resized, float x_scale, int length_resized, int length_original)
    {
        if (coordTransMode == InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN) {
            return (x_resized + 0.5f) * x_scale;
        } else if (coordTransMode == InterpolationCoordTransMode::PYTORCH_HALF_PIXEL) {
            float out = (length_resized > 1)? ((x_resized + 0.5f) * x_scale) - 0.5f : 0.0f;
            return out;
        } else if (coordTransMode == InterpolationCoordTransMode::ASYMMETRIC) {
            return x_resized * x_scale;
        } else if (coordTransMode == InterpolationCoordTransMode::HALF_PIXEL) {
            return ((x_resized + 0.5f) * x_scale) - 0.5f;
        } else if (coordTransMode == InterpolationCoordTransMode::ALIGN_CORNERS) {
            float out = (length_resized == 1)? 0.0f : x_resized * static_cast<float>(length_original - 1) / (length_resized - 1);
            return out;
        } else {
            return x_resized * x_scale;
        }
    }

    float round_prefer_floor(float x)
    {
        float temp; float frac = std::modf(x, &temp);
        return (frac == 0.5f) ? std::floor(x) : std::round(x);
    }

    static void calcReferenceOutputBilinear(const int OH, const int IH, const int OW,
                                            const int IW, const int C,
                                            const float rw, const float rh,
                                            const Tensor<fp16>& input, Tensor<fp16>& output,
                                            const InterpolationCoordTransMode coordTransMode) {
        int b = 0; //for (int b = 0; b < N; ++b)
        {
            for (int h = 0; h < OH; h++)
            {
                float fh = (coordTransMode == InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN && OH <= 1) ?
                            0.0f : coordinateTransform(coordTransMode, h, rh, OH, IH);
                // fh ~= h * rh = input h
                // rh = h ratio IH/OH
                int ih0 = static_cast<int>(fh);
                int ih1 = iif(ih0 < IH - 1, ih0 + 1, ih0);

                float h_lambda0 = fh - ih0; // analog - dicret
                float h_lambda1 = 1.0f - h_lambda0;

                for (int w = 0; w < OW; w++)
                {
                    float fw = (coordTransMode == InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN && OW <= 1) ?
                                0.0f : coordinateTransform(coordTransMode, w, rw, OW, IW);
                    int iw0 = static_cast<int>(fw);
                    int iw1 = iif(iw0 < IW - 1, iw0 + 1, iw0);

                    float w_lambda0 = fw - iw0;
                    float w_lambda1 = 1.0f - w_lambda0;

                    for (int c = 0; c < C; c++)
                    {
                        float src00 = f16Tof32(input.at(TensorDims(iw0, ih0, c, b)));
                        float src01 = f16Tof32(input.at(TensorDims(iw1, ih0, c, b)));
                        float src10 = f16Tof32(input.at(TensorDims(iw0, ih1, c, b)));
                        float src11 = f16Tof32(input.at(TensorDims(iw1, ih1, c, b)));

                        output.at(TensorDims(w, h, c, b)) =
                            f32Tof16(h_lambda1 * (w_lambda1 * src00 + w_lambda0 * src01) +
                                     h_lambda0 * (w_lambda1 * src10 + w_lambda0 * src11));
                    }
                }
            }
        }
    }

    static void calcReferenceOutputONNX(const int OH, const int IH, const int OW,
                                        const int IW, const int C,
                                        const float rw, const float rh,
                                        const Tensor<fp16>& input, Tensor<fp16>& output,
                                        const InterpolationCoordTransMode coordTransMode) {
        int in_y1 = 0;
        int in_y2 = 0;
        int in_x1[OW];
        int in_x2[OW];

        float dy1 = 0;
        float dy2 = 0;

        float dx1[OW];
        float dx2[OW];

        for (int y = 0; y < OH; y++)
        {
            int b = 0; //for (int b = 0; b < N; ++b)
            {
                float in_y = (coordTransMode == InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN && OH <= 1) ?
                                0.0f : coordinateTransform(coordTransMode, y, rh, OH, IH);
                in_y  = MAX(0, MIN(in_y, IH - 1));
                in_y1 = MAX(0, MIN(static_cast<int>(in_y), IH - 1));
                in_y2 = MIN(in_y1 + 1, IH - 1);

                dy1 = iif(in_y1 == in_y2, 0.5f, fabsf(in_y - in_y1));
                dy2 = iif(in_y1 == in_y2, 0.5f, fabsf(in_y - in_y2));

                for (int x = 0; x < OW; x++)
                {
                    float in_x = (coordTransMode == InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN && OW <= 1) ?
                                    0.0f : coordinateTransform(coordTransMode, x, rw, OW, IW);
                    in_x     = MIN(in_x, IW - 1);
                    in_x1[x] = MIN(static_cast<int>(in_x), IW - 1);
                    in_x2[x] = MIN(in_x + 1, IW - 1);

                    dx1[x] = iif(in_x1[x] == in_x2[x], 0.5f, fabsf(in_x - in_x1[x]));
                    dx2[x] = iif(in_x1[x] == in_x2[x], 0.5f, fabsf(in_x - in_x2[x]));

                    for (int c = 0; c < C; c++)
                    {
                        float x11 = f16Tof32(input.at(TensorDims(in_x1[x], in_y1, c, b)));
                        float x21 = f16Tof32(input.at(TensorDims(in_x2[x], in_y1, c, b)));
                        float x12 = f16Tof32(input.at(TensorDims(in_x1[x], in_y2, c, b)));
                        float x22 = f16Tof32(input.at(TensorDims(in_x2[x], in_y2, c, b)));

                        float temp = dx2[x] * dy2 * x11 + dx1[x] * dy2 * x21 + dx2[x] * dy1 * x12 + dx1[x] * dy1 * x22;

                        output.at(TensorDims(x, y, c, b)) = f32Tof16(temp);
                    }
                }
            }
        }
    }

    static void calcReferenceOutputNearest(const int OH, const int IH, const int OW,
                                        const int IW, const int C,
                                        const Tensor<fp16>& input, Tensor<fp16>& output,
                                        const InterpolationNearestMode nearestMode, const InterpolationCoordTransMode coordTransMode) {
        float rh = static_cast<float>(IH) / static_cast<float>(OH);
        float rw = static_cast<float>(IW) / static_cast<float>(OW);

        float (*roundingFunc)(float) = &std::round;
        if (nearestMode == InterpolationNearestMode::FLOOR
        || nearestMode == InterpolationNearestMode::SIMPLE) {
            roundingFunc = &std::floor;
        }
        if (nearestMode == InterpolationNearestMode::CEIL) {
            roundingFunc = &std::ceil;
        }
        if (nearestMode == InterpolationNearestMode::ROUND_PREFER_FLOOR) {
            roundingFunc = &round_prefer_floor;
        }

        int b = 0; //for (int b = 0; b < N; ++b)
        {
            for (int h = 0; h < OH; h++)
            {
                for (int w = 0; w < OW; w++)
                {
                    float alpha = alphaCoordinateTransform(coordTransMode, rh);
                    float fw = rw*w + alpha;
                    if (coordTransMode == InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN && OW <= 1)
                        fw = 0;
                    int iw = MIN(static_cast<int>(roundingFunc(fw)), IW-1);

                    alpha = alphaCoordinateTransform(coordTransMode, rw);
                    float fh = rh * h + alpha;
                    if (coordTransMode == InterpolationCoordTransMode::TF_HALF_PIXEL_FOR_NN && OH <= 1)
                        fh = 0;
                    int ih = MIN(static_cast<int>(roundingFunc(fh)), IH-1);

                    for (int c = 0; c < C; c++)
                    {
                        output.at(TensorDims(w, h, c, b)) = input.at(TensorDims(iw, ih, c, b));
                    }
                }
            }
        }
    }

    class CustomCppInterpolateTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppInterpolateTest(): m_testsLoop(interpolate_test_list, "test") {
        }
        virtual ~CustomCppInterpolateTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppInterpolateTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_interpolateParams = reinterpret_cast<sw_params::InterpolateParams*>(paramContainer);
            *m_interpolateParams = sw_params::InterpolateParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::InterpolateParams);
            m_requiredTensorLocation =
                    static_cast<sw_params::Location>(test->customLayerParams.layerParams[4]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_interpolateParams);

            m_interpolateParams->interpolation_mode = static_cast<InterpolationMethod>(test->customLayerParams.layerParams[0]);
            m_interpolateParams->coord_transform_mode = static_cast<InterpolationCoordTransMode>(test->customLayerParams.layerParams[1]);
            m_interpolateParams->nearest_mode = static_cast<InterpolationNearestMode>(test->customLayerParams.layerParams[2]);
            m_interpolateParams->antialias = static_cast<int>(test->customLayerParams.layerParams[3]);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_singleShaveInterpolate_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(singleShaveInterpolate));
#endif
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.07f;
        }

        void generateInputData() override {
            const auto il = m_inputTensor.tensorLimits();
            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                uint32_t index = m_inputTensor.index(indices);

                const TensorDims ti = m_inputTensor.toTensor(indices);

                uint32_t cval = ti.channels - il.channels/2;
                uint32_t wval = ti.width    - il.width/2;
                uint32_t hval = ti.height   - il.height/2;

                float val = ((float)((cval % 7) * (hval % 13) * (wval % 17)) / 25.f);
                if (index % 3) val = -val;

                m_inputTensor.at(indices) = f32Tof16(val);
            });
        }

        void generateReferenceData() override {
            const auto in_dims  = m_inputTensor.tensorDims();
            const auto out_dims = m_referenceOutputTensor.tensorDims();

            int OH = out_dims.height;
            int OW = out_dims.width;

            int C = in_dims.channels;
            int IH = in_dims.height;
            int IW = in_dims.width;

            if (IH == OH && IW == OW)
            {
                int b = 0; //for (int b = 0; b < N; b++)
                for (int c = 0; c < C; c++)
                    for (int h = 0; h < IH; h++)
                        for (int w = 0; w < IW; w++)
                            m_referenceOutputTensor.at(TensorDims(w, h, c, b)) = m_inputTensor.at(TensorDims(w, h, c, b));
                return;
            }
            bool align_corners = m_interpolateParams->coord_transform_mode == InterpolationCoordTransMode::ALIGN_CORNERS;
            float rh = (OH > 1 && align_corners) ? static_cast<float>(IH - 1) / (OH - 1) :
                        static_cast<float>(IH) / OH;
            float rw = (OW > 1 && align_corners) ? static_cast<float>(IW - 1) / (OW - 1) :
                        static_cast<float>(IW) / OW;

            if (m_interpolateParams->interpolation_mode == InterpolationMethod::BILINEAR) {
                calcReferenceOutputBilinear(OH, IH, OW, IW, C, rw, rh, m_inputTensor, m_referenceOutputTensor, m_interpolateParams->coord_transform_mode);
            } else if (m_interpolateParams->interpolation_mode == InterpolationMethod::LINEAR_ONNX) {
                calcReferenceOutputONNX(OH, IH, OW, IW, C, rw, rh, m_inputTensor, m_referenceOutputTensor, m_interpolateParams->coord_transform_mode);
            } else if (m_interpolateParams->interpolation_mode == InterpolationMethod::NEAREST) {
                calcReferenceOutputNearest(OH, IH, OW, IW, C, m_inputTensor, m_referenceOutputTensor, m_interpolateParams->nearest_mode, m_interpolateParams->coord_transform_mode);
            }

            printf("interpolation_mode = %d\n", m_interpolateParams->interpolation_mode);
            printf("coord_transform_mode = %d\n", m_interpolateParams->coord_transform_mode);
            printf("nearest_mode = %d\n", m_interpolateParams->nearest_mode);
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(),
                                 "inMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                                 m_referenceOutputTensor.bufferSize(), "refOutMyriad.bin");
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

        sw_params::InterpolateParams* m_interpolateParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppInterpolateTest)
}  // namespace )
