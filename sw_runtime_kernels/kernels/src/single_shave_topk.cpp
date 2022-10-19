//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <math.h>
#include <moviVectorTypes.h>
#include <mvSubspaces.h>
#include <param_topk.h>
#include <stdio.h>
struct Pack {
    half value;
    int32_t index;
};
typedef bool (*ComparePackedFunc)(const Pack&, const Pack&);
struct PartialHeapSortPacked {
    Pack* m_items;
    int m_cnt;
    int m_fill;
    PartialHeapSortPacked(Pack* items, int cnt): m_items(items), m_cnt(cnt), m_fill(0) {
    }
    void clear() {
        m_fill = 0;
    }
    int INDEX_PARENT(int i) const {
        return (i - 1) / 2;
    }
    void pushAll(int n, ComparePackedFunc comparePacked) {
        m_fill = m_cnt;
        int start = INDEX_PARENT(m_cnt - 1);
        while (start >= 0) {
            siftDown(start, -1, comparePacked);
            start--;
        }
        for (int i = m_fill; i < n; ++i) {
            if (comparePacked(m_items[i], m_items[0])) {
                m_items[0] = m_items[i];
                siftDown(0, -1, comparePacked);
            }
        }
    }
    // full sort the array by putting smallest item to the end one by one.
    // after full sort, the heap is empty and array is in descending order
    int fullSort(ComparePackedFunc comparePacked) {
        if (m_cnt > 1) {
            int N = m_cnt;
            while (N > 1) {
                std::swap(m_items[0], m_items[N - 1]);
                N--;
                siftDown(0, N, comparePacked);
            }
        }
        return m_cnt;
    }
    int simpleSort(ComparePackedFunc comparePacked) {
        int tmpValue = -INFINITY;
        int tmpIndex = 0;
        for (int i = 0; i < m_cnt; i++) {
            if (m_items[i].value > tmpValue) {
                tmpValue = m_items[i].value;
                tmpIndex = i;
            }
        }
        std::swap(m_items[0], m_items[tmpIndex]);
        return m_cnt;
    }
    // after top is changed, siftDown can recover the heap order,
    // ensure top is the smallest
    void siftDown(int start, int end, ComparePackedFunc comparePacked) {
        if (end < start)
            end = m_cnt;
        // make a copy of the new item on top
        Pack itemx = m_items[start];
        int root = start;
        int child0 = (2 * root + 1);
        // the root element is going down along the heap-tree structure
        // and smaller child along its path will bubble-up
        while (child0 < end) {
            int child1 = child0 + 1;
            int swapx = root;
            Pack* pItemx = &itemx;
            if (comparePacked(*pItemx, m_items[child0])) {
                swapx = child0;
                pItemx = &m_items[child0];
            }
            if (child1 < end) {
                if (comparePacked(*pItemx, m_items[child1])) {
                    swapx = child1;
                    pItemx = &m_items[child1];
                }
            }
            if (swapx == root)
                break;
            // bubble-up smallest child to root
            m_items[root] = m_items[swapx];
            // sift following sub-tree
            root = swapx;
            child0 = (2 * root + 1);
        }
        // final location of the new element put into the heap
        if (start != root)
            m_items[root] = itemx;
    }
};
bool isSmallValue(const Pack& a, const Pack& b) {
    return !(a.value <= b.value) | (!(a.value != b.value) & (a.index < b.index));
}
bool isLargeValue(const Pack& a, const Pack& b) {
    return !(a.value >= b.value) | (!(a.value != b.value) & (a.index < b.index));
}
bool isSmallIndex(const Pack& a, const Pack& b) {
    return bool(a.index < b.index);
}
using namespace sw_params;
namespace nn {
namespace shave_lib {
extern "C" {
void single_shave_topk(uint32_t lParamsAddr) {
    half* p_act_input = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->inputValues.dataAddr);
    int32_t* kaddr = (int32_t*)(reinterpret_cast<TopKParams*>(lParamsAddr)->k.dataAddr);
    half* p_act_value = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->outputValues.dataAddr);
    int32_t* p_act_index = (int32_t*)(reinterpret_cast<TopKParams*>(lParamsAddr)->outputIndex.dataAddr);
    const TopKParams* lParams = reinterpret_cast<const TopKParams*>(lParamsAddr);
    int32_t numInputDims = (int32_t)lParams->inputValues.numDims;
    int32_t* pInputDims = (int32_t*)(lParams->inputValues.dimsAddr);
    int64_t* pInputStrides = (int64_t*)(lParams->inputValues.stridesAddr);
    int32_t* pValueDims = (int32_t*)(lParams->outputValues.dimsAddr);
    int64_t* pValueStrides = (int64_t*)(lParams->outputValues.stridesAddr);
    int32_t numIndexDims = (int32_t)lParams->outputIndex.numDims;
    int32_t* pIndexDims = (int32_t*)(lParams->outputIndex.dimsAddr);
    int64_t* pIndexStrides = (int64_t*)(lParams->outputIndex.stridesAddr);
    int axis = lParams->axis;
    int32_t k = *kaddr;
    int64_t mode = (int64_t)lParams->mode;  // max: 0, min: 1
    int64_t sort = (int64_t)lParams->sort;

    int numLines = 1;
    for (int i = 0; i < numInputDims; i++) {
        numLines *= pInputDims[i];
    }
    numLines = numLines / pInputDims[axis];

    int inputStride, valueStride, indexStride;
    if (axis == 0) {
        inputStride = pInputStrides[1] / CHAR_BIT;
        valueStride = pValueStrides[1] / CHAR_BIT;
        indexStride = pIndexStrides[1] / CHAR_BIT;
    } else {
        inputStride = pInputStrides[0] / CHAR_BIT;
        valueStride = pValueStrides[0] / CHAR_BIT;
        indexStride = pIndexStrides[0] / CHAR_BIT;
    }

    int inputOffset = 1;
    int valueOffset = 1;
    int indexOffset = 1;
    for (size_t i = 0; i < axis; i++) {
        inputOffset *= pInputDims[i];
        valueOffset *= pValueDims[i];
        indexOffset *= pIndexDims[i];
    }

    half* inputLinePtr;
    half* valueLinePtr;
    int32_t* indexLinePtr;
    for (int line = 0; line < numLines; line++) {
        if (axis == 1) {
            inputLinePtr = p_act_input + line / pInputDims[0] * pInputDims[0] * pInputDims[1] +
                           line * inputStride / sizeof(half) % pInputDims[0];
            valueLinePtr = p_act_value + line / pInputDims[0] * pValueDims[0] * k +
                           line * valueStride / sizeof(half) % pInputDims[0];
            indexLinePtr = p_act_index + line / pInputDims[0] * pValueDims[0] * k +
                           line * indexStride / sizeof(int32_t) % pInputDims[0];
        } else {
            inputLinePtr = p_act_input + line * inputStride / sizeof(half);
            valueLinePtr = p_act_value + line * valueStride / sizeof(half);
            indexLinePtr = p_act_index + line * indexStride / sizeof(int32_t);
        }
        Pack lineBuffer[pInputDims[axis]];
        for (int i = 0; i < pInputDims[axis]; i++) {
            lineBuffer[i].value = *inputLinePtr;
            lineBuffer[i].index = i + 1;
            inputLinePtr += inputOffset;
        }
        PartialHeapSortPacked hsort(lineBuffer, k);
        if (mode == 0) {
            if (k == 1) {
                hsort.pushAll(pInputDims[axis], isSmallValue);
                hsort.simpleSort(isSmallValue);
            } else {
                hsort.pushAll(pInputDims[axis], isSmallValue);
                hsort.fullSort(isSmallValue);
            }
        } else {
            if (k == 1) {
                hsort.pushAll(pInputDims[axis], isLargeValue);
                hsort.simpleSort(isLargeValue);
            } else {
                hsort.pushAll(pInputDims[axis], isLargeValue);
                hsort.fullSort(isLargeValue);
            }
        }
        if (sort == 2) {
            hsort.clear();
            hsort.pushAll(k, isSmallIndex);
            hsort.fullSort(isSmallIndex);
        }
        for (int i = 0; i < k; i++) {
            *valueLinePtr = lineBuffer[i].value;
            *indexLinePtr = lineBuffer[i].index - 1;
            valueLinePtr += valueOffset;
            indexLinePtr += indexOffset;
        }
    }
}
}
}  // namespace shave_lib
}  // namespace nn
