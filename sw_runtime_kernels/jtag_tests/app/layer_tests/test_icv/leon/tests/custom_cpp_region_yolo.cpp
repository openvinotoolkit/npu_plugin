//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_region_yolo.3720xx.text.xdat"

#include "param_region_yolo.h"

#define NAMESPACE ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, RegionYolo))

#define MAX_REGION_YOLO_INT_PARAMS 3

struct RegionYoloTest {
    Dims inputDims;
    Dims outputDims;
    StorageOrder storageOrder;
    StorageOrder out_storageOrder;
    int64_t int_params[MAX_REGION_YOLO_INT_PARAMS];
    uint64_t do_softmax;
    int64_t mask[MAX_MASK_SIZE];
    uint64_t mask_size;
    float anchors[MAX_ANCHOR_SIZE];
    uint64_t anchors_size;
    CustomParams customLayerParams;
};

namespace NAMESPACE {
static constexpr std::initializer_list<RegionYoloTest> region_yolo_test_list{
        {{3, 3, 25},
         {3, 3, 25},
         orderYXZ, /*in order*/
         orderYXZ, /*out_order*/
         {4 /*coords*/, 20 /*classes*/, 1 /*regions*/},
         0 /*softmax*/,
         {0}, /*mask*/
         1,   /*mask_size*/
         {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319},
         12, /*anchors_size*/
         {{
                 sw_params::Location::NN_CMX /*mem type*/
         }}},                                /*Yolo v3*/
        {{3, 3, 25},
         {3, 3, 25},
         orderZYX, /*in order*/
         orderZYX, /*out_order*/
         {4 /*coords*/, 20 /*classes*/, 1 /*regions*/},
         0 /*softmax*/,
         {0}, /*mask*/
         1,   /*mask_size*/
         {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319},
         12, /*anchors_size*/
         {{
                 sw_params::Location::NN_CMX /*mem type*/
         }}},                                /*YoloO v3*/
        {{3, 3, 25},
         {225},
         orderYXZ, /*in_order*/
         orderW,   /*out_order*/
         {4 /*coords*/, 20 /*classes*/, 1 /*regions*/},
         1 /*softmax*/,
         {0}, /*mask*/
         1,   /*mask_size*/
         {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52},
         10, /*anchors_size*/
         {{
                 sw_params::Location::NN_CMX /*mem type*/
         }}},                                /*Yolo v2*/
        {{3, 3, 25},
         {225},
         orderZYX, /*in_order*/
         orderW,   /*out_order*/
         {4 /*coords*/, 20 /*classes*/, 1 /*regions*/},
         1 /*softmax*/,
         {0}, /*mask*/
         1,   /*mask_size*/
         {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52},
         10, /*anchors_size*/
         {{
                 sw_params::Location::NN_CMX /*mem type*/
         }}}                                 /*Yolo v2*/
};

class CustomCppRegionYoloTest : public CustomCppTests<fp16, RegionYoloTest> {
public:
    explicit CustomCppRegionYoloTest(): m_testsLoop(region_yolo_test_list, "test") {
    }
    virtual ~CustomCppRegionYoloTest() {
    }

protected:
    const char* suiteName() const override {
        return "CustomCppRegionYoloTest";
    }
    void userLoops() override {
        addLoop(m_testsLoop);
    }

    void initData() override {
        sw_params::BaseKernelParams emptyParamData;
        m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};
        initTestCase();
        const RegionYoloTest* test = m_currentTest;
        int32_t ind[subspace::MAX_DIMS] = {0};
        subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
        subspace::orderToIndices((t_D8StorageOrder)(test->out_storageOrder), ind);
        m_coords = test->int_params[0];
        m_classes = test->int_params[1];
        m_regions = test->int_params[2];
        m_do_softmax = test->do_softmax;
        for (int32_t i = 0; i < test->mask_size; ++i) {
            m_mask[i] = test->mask[i];
        }
        m_mask_size = test->mask_size;
        for (int32_t i = 0; i < test->anchors_size; ++i) {
            m_anchors[i] = test->anchors[i];
        }
        m_anchors_size = test->anchors_size;
        m_regionYoloParams = reinterpret_cast<sw_params::RegionYoloParams*>(paramContainer);
        *m_regionYoloParams = sw_params::RegionYoloParams();
        m_regionYoloParams->coords = m_coords;
        m_regionYoloParams->classes = m_classes;
        m_regionYoloParams->regions = m_regions;
        m_regionYoloParams->mask_size = m_mask_size;
        m_regionYoloParams->do_softmax = m_do_softmax;
        for (int32_t i = 0; i < test->mask_size; ++i) {
            m_regionYoloParams->mask[i] = m_mask[i];
        }
        for (int32_t i = 0; i < m_anchors_size; ++i) {
            m_regionYoloParams->anchors[i] = m_anchors[i];
        }
        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
        m_params.paramDataLen = sizeof(sw_params::RegionYoloParams);
        m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_regionYoloParams);

        const Dims& inputDims = test->inputDims;
        const Dims& outputDims = test->outputDims;
        const StorageOrder& in_storageOrder = test->storageOrder;
        const StorageOrder& out_storageOrder = test->out_storageOrder;

        const TensorDims dimsIn(inputDims.begin()[0], inputDims.begin()[1], inputDims.begin()[2],
                                inputDims.size() > 3 ? inputDims.begin()[3] : 1);
        TensorDims dimsOut(outputDims.begin()[0], outputDims.begin()[1], outputDims.begin()[2],
                           outputDims.size() > 3 ? outputDims.begin()[3] : 1);
        if (m_do_softmax) {
            dimsOut.height = 1;
            dimsOut.channels = 1;
            dimsOut.batch = 1;
        }
        m_inputTensor.init(in_storageOrder, dimsIn);
        m_outputTensor.init(out_storageOrder, dimsOut);
        m_referenceOutputTensor.init(out_storageOrder, dimsOut);

        allocBuffer(m_inputTensor);
        allocBuffer(m_outputTensor);
        allocBuffer(m_referenceOutputTensor);

        m_params.kernel = reinterpret_cast<uint64_t>(sk_single_shave_region_yolo_3720xx_text);
    }

    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.008f;
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

    void generateInputData() override {
        rand_seed();

        // set random seed
        u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
        srand(ticks_for_seed);

        // input
        m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            float tmp = float(rand() % 600) / 100 - 3.0f;
            m_inputTensor.at(indices) = f32Tof16(tmp);
        });
    }

    void generateReferenceData() override {
        const auto& in_dims = m_inputTensor.tensorDims();
        const auto& out_dims = m_outputTensor.tensorDims();

        const int batch = in_dims.batch == 0 ? 1 : in_dims.batch;
        const int channels = in_dims.channels;
        const int height = in_dims.height;
        const int width = in_dims.width;
        const auto order = m_inputTensor.storageOrder();

        for (int n = 0; n < batch; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int32_t index;
                        if (order == 0x4321 || order == 0x321) {  // NCHW or CHW
                            index = n * channels * height * width + c * height * width + h * width + w;
                        } else {
                            index = n * height * width * channels + h * width * channels + w * channels + c;
                        }
                        m_referenceOutputTensor.data()[index] = m_inputTensor.at(TensorDims(w, h, c, n));
                    }
                }
            }
        }

        int32_t end_index;
        if (m_do_softmax == 1) {  // Yolo V2
            end_index = 1;
        } else {  // Yolo V3
            end_index = 1 + m_classes;
            m_regions = m_mask_size;
        }

        for (int n = 0; n < batch; ++n) {
            for (int reg = 0; reg < m_regions; ++reg) {
                for (int c = 0; c < 2; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            int32_t index;
                            if (order == 0x4321 || order == 0x321) {  // NCHW or CHW
                                index = n * channels * height * width +
                                        reg * (m_coords + 1 + m_classes) * height * width + c * height * width +
                                        h * width + w;
                            } else {  // NHWC or HWC
                                index = n * height * width * channels +
                                        m_regions * (m_coords + 1 + m_classes) * (h * width + w) +
                                        reg * (m_coords + 1 + m_classes) + c;
                            }
                            m_referenceOutputTensor.data()[index] = f32Tof16(sigmoid<float>(f16Tof32(
                                    m_inputTensor.at(TensorDims(w, h, reg * (m_coords + 1 + m_classes) + c, n)))));
                        }
                    }
                }
                for (int c = m_coords; c < m_coords + end_index; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            int32_t index;
                            if (order == 0x4321 || order == 0x321) {  // NCHW or CHW
                                index = n * channels * height * width +
                                        reg * (m_coords + 1 + m_classes) * height * width + c * height * width +
                                        h * width + w;
                            } else {  // NHWC or HWC
                                index = n * height * width * channels +
                                        m_regions * (m_coords + 1 + m_classes) * (h * width + w) +
                                        reg * (m_coords + 1 + m_classes) + c;
                            }
                            m_referenceOutputTensor.data()[index] = f32Tof16(sigmoid<float>(f16Tof32(
                                    m_inputTensor.at(TensorDims(w, h, reg * (m_coords + 1 + m_classes) + c, n)))));
                        }
                    }
                }
            }
        }
        if (m_do_softmax) {
            for (int n = 0; n < batch; ++n) {
                for (int reg = 0; reg < m_regions; ++reg) {
                    softmax(n, reg, channels, height, width, order);
                }
            }
        }
    }

    void softmax(int n, int reg, int channels, int height, int width, t_MvTensorStorageOrder order) {
        int chan_offset = reg * (m_coords + 1 + m_classes);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float max = f16Tof32(m_inputTensor.at(TensorDims(w, h, (m_coords + 1) + chan_offset, n)));
                for (int cla = 0; cla < m_classes; ++cla) {
                    float val = f16Tof32(m_inputTensor.at(TensorDims(w, h, (m_coords + 1 + cla) + chan_offset, n)));
                    max = std::max(max, val);
                }
                float sum = 0;
                for (int cla = 0; cla < m_classes; ++cla) {
                    int32_t index;
                    if (order == 0x4321 || order == 0x321) {  // NCHW or CHW
                        index = n * channels * height * width + reg * (m_coords + 1 + m_classes) * height * width +
                                (m_coords + 1 + cla) * height * width + h * width + w;
                    } else {  // NHWC or HWC
                        index = n * height * width * channels +
                                m_regions * (m_coords + 1 + m_classes) * (h * width + w) +
                                reg * (m_coords + 1 + m_classes) + m_coords + 1 + cla;
                    }
                    m_referenceOutputTensor.data()[index] = f32Tof16(exp_cal<float>(
                            f16Tof32(m_inputTensor.at(TensorDims(w, h, (m_coords + 1 + cla) + chan_offset, n))) - max));
                    sum += f16Tof32(m_referenceOutputTensor.data()[index]);
                }
                for (int cla = 0; cla < m_classes; ++cla) {
                    int32_t index;
                    if (order == 0x4321 || order == 0x321) {  // NCHW or CHW
                        index = n * channels * height * width + reg * (m_coords + 1 + m_classes) * height * width +
                                (m_coords + 1 + cla) * height * width + h * width + w;
                    } else {  // NHWC or HWC
                        index = n * height * width * channels +
                                m_regions * (m_coords + 1 + m_classes) * (h * width + w) +
                                reg * (m_coords + 1 + m_classes) + m_coords + 1 + cla;
                    }
                    float val = f16Tof32(m_referenceOutputTensor.data()[index]) / sum;
                    m_referenceOutputTensor.data()[index] = f32Tof16(val);
                }
            }
        }
    }

    template <typename T>
    static inline T sigmoid(float x) {
        T val;
        val = static_cast<T>(1.f / (1.f + exp_cal<float>(-x)));
        return val;
    }

    template <typename T>
    static inline T exp_cal(float x) {
        T val;
        if (x < 0) {
            val = static_cast<T>(1.f / std::exp(-x));
        } else {
            val = static_cast<T>(std::exp(x));
        }
        return val;
    }

    virtual bool checkResult() override {
        m_outputTensor.confirmBufferData();

        // save output data
        if (m_save_to_file) {
            saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(), "output.bin");
        }

        bool threshold_test_failed = false;
        m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
            float input = f16Tof32(m_inputTensor.at(indices));
            float value = f16Tof32(m_outputTensor.at(indices));
            float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
            float abs_diff = fabs(value - gt_value);
            bool differ = !bool(abs_diff <= m_test_threshold);

            threshold_test_failed |= differ;

            if (differ && GlobalData::doPrintDiffs) {
                const TensorDims ti = m_outputTensor.toTensor(indices);
                printf("DIFF HWCN [%d:%d:%d:%d] %f %f %f %f\n", ti.height, ti.width, ti.channels, ti.batch, input,
                       value, gt_value, abs_diff);
            }
        });
        return !threshold_test_failed;
    }

private:
    ListIterator<RegionYoloTest> m_testsLoop;

    int64_t m_coords;
    int64_t m_classes;
    int64_t m_regions;
    uint64_t m_do_softmax;
    int64_t m_mask[MAX_MASK_SIZE];
    uint64_t m_mask_size;
    float m_anchors[MAX_ANCHOR_SIZE];
    uint64_t m_anchors_size;

    Tensor<fp16> m_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceOutputTensor;

    sw_params::RegionYoloParams* m_regionYoloParams;
};

ICV_TESTS_REGISTER_SUITE(CustomCppRegionYoloTest)
}  // namespace NAMESPACE
