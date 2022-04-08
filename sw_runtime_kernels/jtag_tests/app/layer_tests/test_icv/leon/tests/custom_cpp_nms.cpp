//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.nms_fp16.3720xx.text.xdat"

#include "param_nms.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, NMS)) {
    typedef std::initializer_list<int32_t> Dims;
    typedef std::initializer_list<int32_t> InitIntScalar;
    typedef std::initializer_list<float> InitFloatScalar;
    typedef std::initializer_list<std::initializer_list<std::initializer_list<float>>> Init3DFloat;
    typedef std::initializer_list<std::initializer_list<int32_t>> Init2DInt;
    struct NMS_testParams {
        Dims dims;
        int32_t center_point_box;
        int32_t MaxOutBoxesPerClass;
        float IOUThreshold;
        float ScoreThreshold;
        float SoftNmsSigma;
        Init3DFloat boxes;
        Init3DFloat scores;
        Init2DInt reference_indices;
        StorageOrder storageOrder;
        Dims inputDims;
        Dims outputDims;
    };

    struct boxAndScore {
        float score;
        int32_t box_id;
    };

    struct filteredBoxes {
        float score;
        int32_t batch_index;
        int32_t class_index;
        int32_t box_index;
    };

    static constexpr std::initializer_list<NMS_testParams> nms_test_list{
            {{6, 1, 1},  // {spatial_dimension, num_classes, num_batches}
             1,
             3,
             0.5f,
             0.f,
             0.f,
             {
                     // batches
                     {
                             // spatial_dimension
                             // center_point_box=0 {y1, x1, y2, x2}  center_point_box=1 {y0, x0, w, h}
                             {0.5f, 0.5f, 1.0f, 1.0f},
                             {0.5f, 0.6f, 1.0f, 1.0f},
                             {0.5f, 0.4f, 1.0f, 1.0f},
                             {0.5f, 10.5f, 1.0f, 1.0f},
                             {0.5f, 10.6f, 1.0f, 1.0f},
                             {0.5f, 100.5f, 1.0f, 1.0f},
                     },
             },
             {
                     // batches
                     {
                             // classes
                             {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f},  // spatial_dimension
                     },
             },
             {
                     // num_selected_indices
                     {0, 0, 3},  // {batch_index, class_index, box_index}
                     {0, 0, 0},
                     {0, 0, 5},
             }}};

    float IoU(fp16 * boxesI, fp16 * boxesJ, bool center_point_box) {
        float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
        if (center_point_box) {
            //  box format: x_center, y_center, width, height
            yminI = f16Tof32(boxesI[1]) - f16Tof32(boxesI[3]) / 2.f;
            xminI = f16Tof32(boxesI[0]) - f16Tof32(boxesI[2]) / 2.f;
            ymaxI = f16Tof32(boxesI[1]) + f16Tof32(boxesI[3]) / 2.f;
            xmaxI = f16Tof32(boxesI[0]) + f16Tof32(boxesI[2]) / 2.f;
            yminJ = f16Tof32(boxesJ[1]) - f16Tof32(boxesJ[3]) / 2.f;
            xminJ = f16Tof32(boxesJ[0]) - f16Tof32(boxesJ[2]) / 2.f;
            ymaxJ = f16Tof32(boxesJ[1]) + f16Tof32(boxesJ[3]) / 2.f;
            xmaxJ = f16Tof32(boxesJ[0]) + f16Tof32(boxesJ[2]) / 2.f;
        } else {
            //  box format: y1, x1, y2, x2
            yminI = std::min(f16Tof32(boxesI[0]), f16Tof32(boxesI[2]));
            xminI = std::min(f16Tof32(boxesI[1]), f16Tof32(boxesI[3]));
            ymaxI = std::max(f16Tof32(boxesI[0]), f16Tof32(boxesI[2]));
            xmaxI = std::max(f16Tof32(boxesI[1]), f16Tof32(boxesI[3]));
            yminJ = std::min(f16Tof32(boxesJ[0]), f16Tof32(boxesJ[2]));
            xminJ = std::min(f16Tof32(boxesJ[1]), f16Tof32(boxesJ[3]));
            ymaxJ = std::max(f16Tof32(boxesJ[0]), f16Tof32(boxesJ[2]));
            xmaxJ = std::max(f16Tof32(boxesJ[1]), f16Tof32(boxesJ[3]));
        }

        float areaI = (ymaxI - yminI) * (xmaxI - xminI);
        float areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
        if (areaI <= 0.f || areaJ <= 0.f)
            return 0.f;

        float intersection_area = std::max(std::min(ymaxI, ymaxJ) - std::max(yminI, yminJ), 0.f) *
                                  std::max(std::min(xmaxI, xmaxJ) - std::max(xminI, xminJ), 0.f);
        return intersection_area / (areaI + areaJ - intersection_area);
    }

    int calcTempBufSize(int spatDim) {
        return (sizeof(half) * 4 * spatDim + sizeof(half) * spatDim + sizeof(s32) * spatDim) * 2 + 128;
    }

    class CustomCppNMSTest : public CustomCppTests<fp16, NMS_testParams> {
    public:
        explicit CustomCppNMSTest(): m_testsLoop(nms_test_list, "test") {
        }
        virtual ~CustomCppNMSTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppNMSTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            printf("Entering initData\n");
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            initTestCase();

            const Dims& dims = m_currentTest->dims;
            const std::initializer_list<int32_t> boxesDims = {4, dims.begin()[0], dims.begin()[2]};
            const std::initializer_list<int32_t> scoresDims = {dims.begin()[0], dims.begin()[1], dims.begin()[2]};
            const auto maxNumOutBoxes = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int32_t>());
            const std::initializer_list<int32_t> outUpperBoundDims = {3, maxNumOutBoxes};
            const std::initializer_list<int32_t> outRealDims = {1};

            printf("maxNumOutBoxes = %d\n", maxNumOutBoxes);

            const MemoryDims boxesTDims(boxesDims.begin(), static_cast<int>(boxesDims.size()));
            const MemoryDims scoresTDims(scoresDims.begin(), static_cast<int>(scoresDims.size()));
            const MemoryDims outUbTDims(outUpperBoundDims.begin(), static_cast<int>(outUpperBoundDims.size()));
            const MemoryDims outRealTDims(outRealDims.begin(), static_cast<int>(outRealDims.size()));

            const StorageOrder& inputsOrder = orderZYX;
            const StorageOrder& outputIndicesOrder = orderNC;
            const StorageOrder& outputDimsOrder = orderC;

            m_boxesTensor.init(inputsOrder, boxesTDims, boxesTDims);
            m_scoresTensor.init(inputsOrder, scoresTDims, scoresTDims);
            m_outSelectedIndices.init(outputIndicesOrder, outUbTDims, outUbTDims);
            m_outSelectedScores.init(outputIndicesOrder, outUbTDims, outUbTDims);
            m_validOutputs.init(outputDimsOrder, outRealTDims, outRealTDims);
            m_refSelectedIndices.init(outputIndicesOrder, outUbTDims, outUbTDims);
            m_refSelectedScores.init(outputIndicesOrder, outUbTDims, outUbTDims);
            m_refValidOutputs.init(outputDimsOrder, outRealTDims, outRealTDims);

            allocBuffer(m_boxesTensor);
            allocBuffer(m_scoresTensor);
            allocBuffer(m_outSelectedIndices);
            allocBuffer(m_outSelectedScores);
            allocBuffer(m_validOutputs);
            allocBuffer(m_refSelectedIndices);
            allocBuffer(m_refSelectedScores);
            allocBuffer(m_refValidOutputs);

            const NMS_testParams* test = m_currentTest;
            m_maxOutBoxesPerClassTensor = static_cast<int64_t>(test->MaxOutBoxesPerClass);
            m_IOU_thresholdTensor = test->IOUThreshold;
            m_scoreThreshold = test->ScoreThreshold;
            m_softNmsSigmaTensor = test->SoftNmsSigma;
            m_boxEncoding = static_cast<int64_t>(test->center_point_box);
            m_nmsParams = reinterpret_cast<sw_params::NMSParams*>(paramContainer);
            *m_nmsParams = sw_params::NMSParams();
            m_nmsParams->maxOutputBoxesPerClass = m_maxOutBoxesPerClassTensor;
            m_nmsParams->iouThreshold = m_IOU_thresholdTensor;
            m_nmsParams->scoreThreshold = m_scoreThreshold;
            m_nmsParams->softNmsSigma = m_softNmsSigmaTensor;
            m_nmsParams->boxEncoding = m_boxEncoding;
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::NMSParams);
            m_requiredTensorLocation = sw_params::Location::NN_CMX;
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_nmsParams);
            m_params.kernel = reinterpret_cast<uint32_t>(sk_nms_fp16_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void initParserRunner() override {
            printf("Entering initParserRunner\n");
            initMyriadResources();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);

            OpTensor boxesBuff;
            m_boxesTensor.exportToBuffer(boxesBuff);
            customCppOp->addInputBuffer(boxesBuff, m_requiredTensorLocation);

            OpTensor scoresBuff;
            m_scoresTensor.exportToBuffer(scoresBuff);
            customCppOp->addInputBuffer(scoresBuff, m_requiredTensorLocation);

            OpTensor selectedIndicesBuff;
            m_outSelectedIndices.exportToBuffer(selectedIndicesBuff);
            customCppOp->addOutputBuffer(selectedIndicesBuff, m_requiredTensorLocation);

            OpTensor selectedScoresBuff;
            m_outSelectedScores.exportToBuffer(selectedScoresBuff);
            customCppOp->addOutputBuffer(selectedScoresBuff, m_requiredTensorLocation);

            OpTensor validOutputsBuff;
            m_validOutputs.exportToBuffer(validOutputsBuff);
            customCppOp->addOutputBuffer(validOutputsBuff, m_requiredTensorLocation);

            customCppOp->ops = *getParams();
        }

        void resetOutputData() override {
            resetTensorBuffer(m_outSelectedIndices);
            resetTensorBuffer(m_outSelectedScores);
            resetTensorBuffer(m_validOutputs);
        }

        void generateInputData() override {
            printf("Entering generateInputData\n");
            const Dims& dims = m_currentTest->dims;

            if (m_testsLoop.value().boxes.size()) {
                m_boxesTensor.forEach(false, [&](const MemoryDims& indices) {
                    m_boxesTensor.at(indices) = f32Tof16(m_testsLoop.value()
                                                                 .boxes.begin()[indices.dims[2]]
                                                                 .begin()[indices.dims[1]]
                                                                 .begin()[indices.dims[0]]);
                });
            } else {
                for (int batch = 0; batch < m_testsLoop.value().dims.begin()[2]; batch++) {
                    for (int spat_dim = 0; spat_dim < m_testsLoop.value().dims.begin()[0]; spat_dim++) {
                        float width = (float)rand() / RAND_MAX * 10.f;
                        float height = (float)rand() / RAND_MAX * 10.f;
                        float x_center = (float)rand() / RAND_MAX * 200.f - 100.f;
                        float y_center = (float)rand() / RAND_MAX * 200.f - 100.f;
                        if (m_boxEncoding == 0) {
                            m_boxesTensor.at(MemoryDims(0, spat_dim, batch)) = f32Tof16(y_center - height / 2);
                            m_boxesTensor.at(MemoryDims(1, spat_dim, batch)) = f32Tof16(x_center - width / 2);
                            m_boxesTensor.at(MemoryDims(2, spat_dim, batch)) = f32Tof16(y_center + height / 2);
                            m_boxesTensor.at(MemoryDims(3, spat_dim, batch)) = f32Tof16(x_center + width / 2);
                        } else {
                            m_boxesTensor.at(MemoryDims(0, spat_dim, batch)) = f32Tof16(y_center);
                            m_boxesTensor.at(MemoryDims(1, spat_dim, batch)) = f32Tof16(x_center);
                            m_boxesTensor.at(MemoryDims(2, spat_dim, batch)) = f32Tof16(width);
                            m_boxesTensor.at(MemoryDims(3, spat_dim, batch)) = f32Tof16(height);
                        }
                    }
                }
            }

            if (m_testsLoop.value().scores.size()) {
                m_scoresTensor.forEach(false, [&](const MemoryDims& indices) {
                    m_scoresTensor.at(indices) = f32Tof16(m_testsLoop.value()
                                                                  .scores.begin()[indices.dims[2]]
                                                                  .begin()[indices.dims[1]]
                                                                  .begin()[indices.dims[0]]);
                    printf("input score if = %.2f\n", m_testsLoop.value()
                                                              .scores.begin()[indices.dims[2]]
                                                              .begin()[indices.dims[1]]
                                                              .begin()[indices.dims[0]]);
                });
            } else {
                m_scoresTensor.forEach(false, [&](const MemoryDims& indices) {
                    float score = (float)(rand() % 101) / 100;
                    printf("input score else = %.2f\n", score);
                    m_scoresTensor.at(indices) = f32Tof16(score);
                });
            }

            // reference output
            generateReferenceData();
        }
        void generateReferenceData() override {
            printf("Entering generateReferenceData\n");
            int64_t boxEncoding = m_boxEncoding;
            int64_t spat_dim = m_testsLoop.value().dims.begin()[0];
            int64_t num_classes = m_testsLoop.value().dims.begin()[1];
            int64_t num_batches = m_testsLoop.value().dims.begin()[2];

            int64_t maxOutBoxes = m_maxOutBoxesPerClassTensor;
            fp32 IOU_threshold = f16Tof32(m_IOU_thresholdTensor);
            fp32 scoreThreshold = f16Tof32(m_scoreThreshold);

            std::vector<filteredBoxes> fb;
            for (int64_t batch = 0; batch < num_batches; batch++) {
                for (int64_t classNum = 0; classNum < num_classes; classNum++) {
                    std::vector<boxAndScore> filtered_boxes;
                    for (int box = 0; box < spat_dim; box++) {
                        if (f16Tof32(m_scoresTensor.at(MemoryDims(box, classNum, batch))) > scoreThreshold)
                            filtered_boxes.push_back(
                                    {f16Tof32(m_scoresTensor.at(MemoryDims(box, classNum, batch))), box});
                    }
                    if (filtered_boxes.size()) {
                        std::stable_sort(filtered_boxes.begin(), filtered_boxes.end(),
                                         [](const boxAndScore& l, const boxAndScore& r) {
                                             return l.score > r.score;
                                         });
                        int io_selection_size = 1;
                        fb.push_back({filtered_boxes[0].score, batch, classNum, filtered_boxes[0].box_id});
                        for (int box_idx = 1;
                             (box_idx < static_cast<int>(filtered_boxes.size()) && io_selection_size < maxOutBoxes);
                             box_idx++) {
                            bool box_is_selected = true;
                            for (int idx = io_selection_size - 1; idx >= 0; idx--) {
                                float iou =
                                        IoU(&(m_boxesTensor.at(MemoryDims(0, filtered_boxes[box_idx].box_id, batch))),
                                            &(m_boxesTensor.at(MemoryDims(0, filtered_boxes[idx].box_id, batch))),
                                            static_cast<bool>(boxEncoding));
                                if (iou > IOU_threshold) {
                                    box_is_selected = false;
                                    break;
                                }
                            }
                            if (box_is_selected) {
                                filtered_boxes[io_selection_size] = filtered_boxes[box_idx];
                                io_selection_size++;
                                fb.push_back({filtered_boxes[box_idx].score, batch, classNum,
                                              filtered_boxes[box_idx].box_id});
                            }
                        }
                    }
                }
            }
            size_t idx;
            for (idx = 0; idx < fb.size() && idx < static_cast<size_t>(spat_dim); idx++) {
                m_refSelectedIndices.at(MemoryDims(0, idx)) = fb[idx].batch_index;
                m_refSelectedIndices.at(MemoryDims(1, idx)) = fb[idx].class_index;
                m_refSelectedIndices.at(MemoryDims(2, idx)) = fb[idx].box_index;
                m_refSelectedScores.at(MemoryDims(0, idx)) = f32Tof16(fb[idx].batch_index);
                m_refSelectedScores.at(MemoryDims(1, idx)) = f32Tof16(fb[idx].class_index);
                m_refSelectedScores.at(MemoryDims(2, idx)) = f32Tof16(fb[idx].score);
            }

            auto validOutputsPtr = m_refValidOutputs.data();
            validOutputsPtr[0] = fb.size();
        }

        virtual bool checkResult() override {
            printf("Entering checkResult\n");
            m_outSelectedIndices.confirmBufferData();
            m_outSelectedScores.confirmBufferData();
            m_validOutputs.confirmBufferData();
            m_refSelectedIndices.confirmBufferData();
            m_refSelectedScores.confirmBufferData();
            m_refValidOutputs.confirmBufferData();

            const int outNumBoxes = m_validOutputs.data()[0];
            const int refNumBoxes = m_refValidOutputs.data()[0];
            if ((outNumBoxes != refNumBoxes) || GlobalData::doPrintDiffs)
                printf("TOTAL STATIC SHAPE NMS(s) OUT/REF = %" PRId32 "/%" PRId32 "\n", (long int)outNumBoxes,
                       (long int)refNumBoxes);

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outSelectedIndices.buffer()),
                                 m_outSelectedIndices.bufferSize(), "outSelectedIndices.bin");
                saveMemoryToFile(reinterpret_cast<u32>(m_outSelectedScores.buffer()), m_outSelectedScores.bufferSize(),
                                 "outSelectedScores.bin");
            }

            bool test_failed = false;

            // check outDims
            mvTensorAssert(m_validOutputs.storageOrder() == m_refValidOutputs.storageOrder());
            m_validOutputs.forEach(true, [&](const MemoryDims& indices) {
                int32_t out_value = m_validOutputs.at(indices);
                int32_t gt_value = m_refValidOutputs.at(indices);
                printf("act_valid = %d, ref_valid = %d\n", (int)out_value, (int)gt_value);

                bool differ = bool(gt_value != out_value);

                test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    char indices_str[64];

                    printf("DIFF: OUT DIMS [%s] %" PRId32 " %" PRId32 "\n",
                           m_validOutputs.indicesToString(indices, indices_str), (long int)out_value,
                           (long int)gt_value);
                }
            });

            // check selectedIndices
            mvTensorAssert(m_outSelectedIndices.storageOrder() == m_refSelectedIndices.storageOrder());
            m_outSelectedIndices.forEach(true, [&](const MemoryDims& indices) {
                const bool unused = bool(indices.dims[1] >= refNumBoxes);
                if (!unused) {
                    const int out_value = m_outSelectedIndices.at(indices);
                    const int gt_value = m_refSelectedIndices.at(indices);
                    printf("act_idx = %d, ref_idx = %d\n", out_value, gt_value);

                    const bool differ = bool(!(out_value == gt_value));
                    test_failed |= differ;

                    if (differ && GlobalData::doPrintDiffs) {
                        char indices_str[64];
                        printf("DIFF: SELECTED INDICES [%s] %" PRId32 " %" PRId32 "\n",
                               m_outSelectedIndices.indicesToString(indices, indices_str), out_value, gt_value);
                    }
                }
            });

            // check selectedScores
            mvTensorAssert(m_outSelectedScores.storageOrder() == m_refSelectedScores.storageOrder());
            m_outSelectedScores.forEach(true, [&](const MemoryDims& indices) {
                const bool unused = bool(indices.dims[1] >= refNumBoxes);
                if (!unused) {
                    const float out_value = f16Tof32(m_outSelectedScores.at(indices));
                    const float gt_value = f16Tof32(m_refSelectedScores.at(indices));
                    printf("act_score = %.2f, ref_score = %.2f\n", out_value, gt_value);

                    const bool differ = bool(!(out_value == gt_value));
                    test_failed |= differ;

                    if (differ && GlobalData::doPrintDiffs) {
                        char indices_str[64];
                        printf("DIFF: SELECTED SCORES [%s] %f %f\n",
                               m_outSelectedScores.indicesToString(indices, indices_str), out_value, gt_value);
                    }
                }
            });

            return !test_failed;
        }

    private:
        ListIterator<NMS_testParams> m_testsLoop;

        // Additional buffer to avoid convertion back and forth
        sw_params::NMSParams* m_nmsParams;
        Tensor<fp16> m_boxesTensor;
        Tensor<fp16> m_scoresTensor;
        Tensor<int32_t> m_outSelectedIndices;
        Tensor<fp16> m_outSelectedScores;
        Tensor<int32_t> m_validOutputs;
        int64_t m_maxOutBoxesPerClassTensor;
        float m_IOU_thresholdTensor;
        float m_scoreThreshold;
        float m_softNmsSigmaTensor;
        int64_t m_boxEncoding;
        Tensor<int32_t> m_refSelectedIndices;
        Tensor<fp16> m_refSelectedScores;
        Tensor<int32_t> m_refValidOutputs;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppNMSTest)
}  // namespace ICV_TESTS_NAMESPACE
