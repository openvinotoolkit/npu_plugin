//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <moviVectorConvert.h>
#include <mvSubspaces.h>
#include <param_nms.h>

using namespace sw_params;
using namespace subspace;

#define MIN_HALF(a, b) __builtin_shave_cmu_min_f16_rr_half((a), (b))
#define MAX_HALF(a, b) __builtin_shave_cmu_max_f16_rr_half((a), (b))
#define MIN(a, b) __builtin_shave_cmu_min_f32_rr_float((a), (b))
#define MAX(a, b) __builtin_shave_cmu_max_f32_rr_float((a), (b))
#define MIN_VEC(a, b) __builtin_shave_cmu_min_f32_rr_float4((a), (b))
#define MAX_VEC(a, b) __builtin_shave_cmu_max_f32_rr_float4((a), (b))
#define MUL_VEC(a, b) __builtin_shave_vau_mul_f32_rr((a), (b))
#define ADD_VEC(a, b) __builtin_shave_vau_add_f32_rr((a), (b))
#define SUB_VEC(a, b) __builtin_shave_vau_sub_f32_rr((a), (b))
#define DIV_VEC(a, b, c)        \
    ((a)[0] = (b)[0] / (c)[0]); \
    ((a)[1] = (b)[1] / (c)[1]); \
    ((a)[2] = (b)[2] / (c)[2]); \
    ((a)[3] = (b)[3] / (c)[3]);

int32_t selectedIndicesData[3];
int32_t selectedScoresData[3];
struct boxAndScore {
    half* ymin;
    half* xmin;
    half* ymax;
    half* xmax;
    half* score;
    int32_t* boxIdx;
    bool operator>(const boxAndScore& r) {
        return *score > *(r.score);
    }
    boxAndScore operator=(const boxAndScore& r) {
        *(ymin) = *(r.ymin);
        *(xmin) = *(r.xmin);
        *(ymax) = *(r.ymax);
        *(xmax) = *(r.xmax);
        *(score) = *(r.score);
        *(boxIdx) = *(r.boxIdx);
        return {ymin, xmin, ymax, xmax, score, boxIdx};
    }
    void copyPointers(const boxAndScore& r) {
        ymin = r.ymin;
        xmin = r.xmin;
        ymax = r.ymax;
        xmax = r.xmax;
        score = r.score;
        boxIdx = r.boxIdx;
    }
    void swapBNS(const boxAndScore& r2) {
        half yminTmp = *ymin;
        half xminTmp = *xmin;
        half ymaxTmp = *ymax;
        half xmaxTmp = *xmax;
        half scoreTmp = *score;
        half boxIdxTmp = *boxIdx;

        *ymin = *(r2.ymin);
        *xmin = *(r2.xmin);
        *ymax = *(r2.ymax);
        *xmax = *(r2.xmax);
        *score = *(r2.score);
        *boxIdx = *(r2.boxIdx);

        *(r2.ymin) = yminTmp;
        *(r2.xmin) = xminTmp;
        *(r2.ymax) = ymaxTmp;
        *(r2.xmax) = xmaxTmp;
        *(r2.score) = scoreTmp;
        *(r2.boxIdx) = boxIdxTmp;
    }
};

class BoxAndScoreContainer {
public:
    half* boxPtr;
    half* scorePtr;
    int32_t* boxIdxPtr;

    bool centerPointBox;
    int32_t maxBoxesNum;
    int32_t actualBoxesNum;

public:
    BoxAndScoreContainer(half* boxPtr, half* scorePtr, int32_t* boxIdxPtr, int32_t maxBoxesNum, bool centerPointBox)
            : boxPtr(boxPtr),
              scorePtr(scorePtr),
              boxIdxPtr(boxIdxPtr),
              centerPointBox(centerPointBox),
              maxBoxesNum(maxBoxesNum),
              actualBoxesNum(0) {
    }
    boxAndScore operator[](int32_t n) const {
        return {&boxPtr[n],
                &boxPtr[n + maxBoxesNum],
                &boxPtr[n + 2 * maxBoxesNum],
                &boxPtr[n + 3 * maxBoxesNum],
                &scorePtr[n],
                &boxIdxPtr[n]};
    }
    void append(half val1, half val2, half val3, half val4, half score, int32_t boxIdx) {
        if (centerPointBox) {
            boxPtr[actualBoxesNum] = val2 - val4 / 2;
            boxPtr[actualBoxesNum + maxBoxesNum] = val1 - val3 / 2;
            boxPtr[actualBoxesNum + 2 * maxBoxesNum] = val2 + val4 / 2;
            boxPtr[actualBoxesNum + 3 * maxBoxesNum] = val1 + val3 / 2;
        } else {
            boxPtr[actualBoxesNum] = MIN_HALF(val1, val3);
            boxPtr[actualBoxesNum + maxBoxesNum] = MIN_HALF(val2, val4);
            boxPtr[actualBoxesNum + 2 * maxBoxesNum] = MAX_HALF(val1, val3);
            boxPtr[actualBoxesNum + 3 * maxBoxesNum] = MAX_HALF(val2, val4);
        }
        scorePtr[actualBoxesNum] = score;
        boxIdxPtr[actualBoxesNum] = boxIdx;
        actualBoxesNum++;
    }
    bool isEmpty() {
        return actualBoxesNum == 0;
    }
    int32_t size() {
        return actualBoxesNum;
    }
    void copyActualSize(const BoxAndScoreContainer& src) {
        actualBoxesNum = src.actualBoxesNum;
    }
    void clear() {
        actualBoxesNum = 0;
    }
};

class SelectedIndices {
    int32_t* selectedIndices = nullptr;
    half* selectedScores = nullptr;
    const int32_t* stride = nullptr;
    NDOrder order;
    int32_t maxSize;
    int32_t actualSize;

public:
    SelectedIndices(int32_t* selectedIndices, half* selectedScores, int32_t maxSize, const int32_t* stride,
                    NDOrder order)
            : selectedIndices(selectedIndices),
              selectedScores(selectedScores),
              stride(stride),
              order(order),
              maxSize(maxSize),
              actualSize(0) {
    }
    void push_back(int32_t batchIdx, int32_t classIdx, int32_t boxIdx, half score) {
        // selectedIndices and selectedScores have the same shape, order of axes
        // and their strides are default without paddings so strides of int32_t *selectedIndices
        // are just twice of the strides of half *selectedScores
        *((int32_t*)((uint8_t*)selectedIndices + actualSize * stride[1])) = batchIdx;
        *((int32_t*)((uint8_t*)selectedIndices + actualSize * stride[1] + stride[0])) = classIdx;
        *((int32_t*)((uint8_t*)selectedIndices + actualSize * stride[1] + 2 * stride[0])) = boxIdx;
        *((half*)((uint8_t*)selectedScores + (actualSize * stride[1]) / 2)) = batchIdx;
        *((half*)((uint8_t*)selectedScores + (actualSize * stride[1] + stride[0]) / 2)) = classIdx;
        *((half*)((uint8_t*)selectedScores + (actualSize * stride[1] + 2 * stride[0]) / 2)) = score;
        actualSize++;
    }
    bool isFull() {
        return actualSize == maxSize;
    }
    int32_t getActualSize() {
        return actualSize;
    }
};

static bool SuppressByIoU(const BoxAndScoreContainer* bns, const int32_t boxIdx, const int32_t selection_size,
                          const half IoUThreshold) {
    boxAndScore tmp_box;
    tmp_box.copyPointers((*bns)[boxIdx]);

    float yminI = (float)*(tmp_box.ymin);
    float xminI = (float)*(tmp_box.xmin);
    float ymaxI = (float)*(tmp_box.ymax);
    float xmaxI = (float)*(tmp_box.xmax);

    float areaI = (ymaxI - yminI) * (xmaxI - xminI);
    if (areaI <= 0.f)
        return false;

    int32_t iter = 0;

    for (; iter < selection_size - 3; iter += 4) {
        tmp_box.copyPointers((*bns)[iter]);
        float4 yminJ = {(float)*((half*)(tmp_box.ymin)), (float)*((half*)(tmp_box.ymin) + 1),
                        (float)*((half*)(tmp_box.ymin) + 2), (float)*((half*)(tmp_box.ymin) + 3)};
        float4 xminJ = {(float)*((half*)(tmp_box.xmin)), (float)*((half*)(tmp_box.xmin) + 1),
                        (float)*((half*)(tmp_box.xmin) + 2), (float)*((half*)(tmp_box.xmin) + 3)};
        float4 ymaxJ = {(float)*((half*)(tmp_box.ymax)), (float)*((half*)(tmp_box.ymax) + 1),
                        (float)*((half*)(tmp_box.ymax) + 2), (float)*((half*)(tmp_box.ymax) + 3)};
        float4 xmaxJ = {(float)*((half*)(tmp_box.xmax)), (float)*((half*)(tmp_box.xmax) + 1),
                        (float)*((half*)(tmp_box.xmax) + 2), (float)*((half*)(tmp_box.xmax) + 3)};

        float4 areaJ = MUL_VEC(SUB_VEC(ymaxJ, yminJ), SUB_VEC(xmaxJ, xminJ));
        int4 areaJ_neg = (areaJ <= 0.0f);

        float4 result =
                MUL_VEC(MAX_VEC(SUB_VEC(MIN_VEC((float4)ymaxI, ymaxJ), MAX_VEC((float4)yminI, yminJ)), (float4)0.f),
                        MAX_VEC(SUB_VEC(MIN_VEC((float4)xmaxI, xmaxJ), MAX_VEC((float4)xminI, xminJ)), (float4)0.f));
        float4 union_area = SUB_VEC(ADD_VEC((float4)areaI, areaJ), result);

        int4 upper_threshold = (result >= (float)IoUThreshold * union_area);
        int4 both_criteria = ~areaJ_neg & upper_threshold;
        if (__builtin_shave_sau_orx_x32_r(both_criteria))
            return true;
    }

    for (; iter < selection_size; iter++) {
        tmp_box.copyPointers((*bns)[iter]);
        float yminJ = (float)*(tmp_box.ymin);
        float xminJ = (float)*(tmp_box.xmin);
        float ymaxJ = (float)*(tmp_box.ymax);
        float xmaxJ = (float)*(tmp_box.xmax);

        float areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
        if (areaJ <= 0.f)
            continue;

        float result =
                MAX((MIN(ymaxI, ymaxJ) - MAX(yminI, yminJ)), 0.f) * MAX((MIN(xmaxI, xmaxJ) - MAX(xminI, xminJ)), 0.f);
        float union_area = areaI + areaJ - result;
        if (result >= (float)IoUThreshold * union_area)
            return true;
    }

    return false;
}

static void bubble_sort(BoxAndScoreContainer& bsMgr) {
    for (int i = 0; i < bsMgr.size() - 1; i++) {
        for (int j = 0; j < bsMgr.size() - i - 1; j++) {
            if (bsMgr[j + 1] > bsMgr[j]) {
                boxAndScore tmp_box;
                tmp_box.copyPointers(bsMgr[j]);
                half ymin = *(tmp_box.ymin);
                half xmin = *(tmp_box.xmin);
                half ymax = *(tmp_box.ymax);
                half xmax = *(tmp_box.xmax);
                half score = *(tmp_box.score);
                half boxIdx = *(tmp_box.boxIdx);
                bsMgr[j] = bsMgr[j + 1];
                *bsMgr[j + 1].ymin = ymin;
                *bsMgr[j + 1].xmin = xmin;
                *bsMgr[j + 1].ymax = ymax;
                *bsMgr[j + 1].xmax = xmax;
                *bsMgr[j + 1].score = score;
                *bsMgr[j + 1].boxIdx = boxIdx;
            }
        }
    }
}

namespace nn {
namespace shave_lib {

extern "C" {

void nms_fp16(uint32_t lParams) {
    auto layerParams = reinterpret_cast<const NMSParams*>(lParams);

    half* p_act_boxes = (half*)(layerParams->boxes.dataAddr);
    half* p_act_scores = (half*)(layerParams->scores.dataAddr);
    int32_t* p_act_selected_indices = (int32_t*)(layerParams->selectedIndices.dataAddr);
    half* p_act_selected_scores = (half*)(layerParams->selectedScores.dataAddr);
    int32_t* p_act_valid_outputs = (int32_t*)(layerParams->validOutputs.dataAddr);

    int32_t* boxesDims = (int32_t*)(layerParams->boxes.dimsAddr);
    int32_t* scoresDims = (int32_t*)(layerParams->scores.dimsAddr);
    int32_t* selectedIndicesDims = (int32_t*)(layerParams->selectedIndices.dimsAddr);
    int32_t* selectedScoresDims = (int32_t*)(layerParams->selectedScores.dimsAddr);

    const int64_t* boxesStridesBits = (int64_t*)(layerParams->boxes.stridesAddr);
    const int64_t* scoresStridesBits = (int64_t*)(layerParams->scores.stridesAddr);
    const int64_t* outputStridesBits = (int64_t*)(layerParams->selectedIndices.stridesAddr);

    const int32_t numInDims = layerParams->boxes.numDims;
    const int32_t numOutDims = layerParams->selectedIndices.numDims;

    int64_t boxesStrides[numInDims], scoresStrides[numInDims];
    int32_t outputStrides[numOutDims];

    for (int32_t i = 0; i < numInDims; i++) {
        boxesStrides[i] = boxesStridesBits[i] / CHAR_BIT;
        scoresStrides[i] = scoresStridesBits[i] / CHAR_BIT;
    }

    for (int32_t i = 0; i < numOutDims; i++) {
        outputStrides[i] = outputStridesBits[i] / CHAR_BIT;
    }

    NDOrder boxesOrder = layerParams->boxes.dimsOrder;
    NDOrder scoresOrder = layerParams->scores.dimsOrder;
    NDOrder selectedIndicesOrder = layerParams->selectedIndices.dimsOrder;
    NDOrder selectedScoresOrder = layerParams->selectedScores.dimsOrder;

    const int32_t spat_dim = boxesDims[1];
    const int32_t num_classes = scoresDims[1];
    const int32_t num_batches = boxesDims[2];
    const int32_t max_selected_indices = selectedIndicesDims[1];

    int64_t maxOutBoxes = layerParams->maxOutputBoxesPerClass;
    float IOU_threshold = layerParams->iouThreshold;
    float scoreThreshold = layerParams->scoreThreshold;
    float softNmsSigma = layerParams->softNmsSigma;
    bool centerPointBox = (int64_t)(layerParams->boxEncoding) == 1 ? true : false;

    half boxesPtrCMX[4 * spat_dim];
    half scoresPtrCMX[spat_dim];
    int32_t boxIdxPtrCMX[spat_dim];

    BoxAndScoreContainer* bns = nullptr;
    BoxAndScoreContainer bsMgr(boxesPtrCMX, scoresPtrCMX, boxIdxPtrCMX, spat_dim, centerPointBox);
    SelectedIndices fb(p_act_selected_indices, p_act_selected_scores, max_selected_indices, outputStrides,
                       selectedIndicesOrder);
    for (int32_t batch = 0; batch < num_batches; batch++) {
        for (int32_t classNum = 0; classNum < num_classes; classNum++) {
            bsMgr.clear();
            uint8_t* boxes = (uint8_t*)p_act_boxes + batch * boxesStrides[2];
            uint8_t* scores = (uint8_t*)p_act_scores + (batch * scoresStrides[2] + classNum * scoresStrides[1]);
            int32_t perElBoxStrideDim = 0;
            for (int32_t boxIdx = 0; boxIdx < spat_dim; boxIdx++) {
                if (*((half*)scores) > scoreThreshold) {
                    bsMgr.append(*((half*)boxes), *((half*)(boxes + boxesStrides[perElBoxStrideDim])),
                                 *((half*)(boxes + 2 * boxesStrides[perElBoxStrideDim])),
                                 *((half*)(boxes + 3 * boxesStrides[perElBoxStrideDim])), *((half*)scores), boxIdx);
                }
                boxes += boxesStrides[1];
                scores += scoresStrides[0];
            }

            if (!bsMgr.isEmpty()) {
                bubble_sort(bsMgr);
                bns = &bsMgr;
                fb.push_back(batch, classNum, *((*bns)[0].boxIdx), *((*bns)[0].score));
                int32_t selection_size = 1;
                for (int32_t boxIdx = 1; boxIdx < bns->size() && selection_size < maxOutBoxes; boxIdx++) {
                    if (!SuppressByIoU(bns, boxIdx, selection_size, IOU_threshold)) {
                        (*bns)[selection_size] = (*bns)[boxIdx];
                        selection_size++;
                        fb.push_back(batch, classNum, *((*bns)[boxIdx].boxIdx), *((*bns)[boxIdx].score));
                    }
                }
            }
        }
    }

    if (p_act_valid_outputs)  // StaticShapeNMS
    {
        p_act_valid_outputs[0] = fb.getActualSize();
    }
    while (!fb.isFull())
        fb.push_back(-1, -1, -1, -1);
}
}
}  // namespace shave_lib
}  // namespace nn
