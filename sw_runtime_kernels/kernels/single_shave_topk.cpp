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
    half* p_act_value = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->outputValues.dataAddr);
    int32_t* p_act_index = (int32_t*)(reinterpret_cast<TopKParams*>(lParamsAddr)->outputIndex.dataAddr);

    const TopKParams* lParams = reinterpret_cast<const TopKParams*>(lParamsAddr);

    int32_t numInputDims = (int32_t)lParams->inputValues.numDims;
    int32_t* pInputDims = (int32_t*)(lParams->inputValues.dimsAddr);
    int64_t* pInputStrides = (int64_t*)(lParams->inputValues.stridesAddr);

    int32_t numValueDims = (int32_t)lParams->outputValues.numDims;
    int32_t* pValueDims = (int32_t*)(lParams->outputValues.dimsAddr);
    int64_t* pValueStrides = (int64_t*)(lParams->outputValues.stridesAddr);

    int32_t numIndexDims = (int32_t)lParams->outputIndex.numDims;
    int32_t* pIndexDims = (int32_t*)(lParams->outputIndex.dimsAddr);
    int64_t* pIndexStrides = (int64_t*)(lParams->outputIndex.stridesAddr);

    int32_t k = (int32_t)lParams->k;
    int32_t axis = (int32_t)lParams->axis;
    int32_t mode = (int32_t)lParams->mode;  // max: 0, min: 1
    int32_t sort = (int32_t)lParams->sort;

    // calculate top K inner (axis = 0)
    if (axis == 0) {
        int numLines = 1;
        for (int i = 1; i < numInputDims; i++) {
            numLines *= pInputDims[i];
        }

        int inputStride = pInputStrides[1] / CHAR_BIT;
        int valueStride = k * 2;  // k * 2;//pValueStrides[1] / CHAR_BIT;
        int indexStride = k * 4;  // k * 4;//pIndexStrides[1] / CHAR_BIT;

        for (int line = 0; line < numLines; line++) {
            half* inputLinePtr = p_act_input + line * inputStride / sizeof(half);
            half* valueLinePtr = p_act_value + line * valueStride / sizeof(half);
            int32_t* indexLinePtr = p_act_index + line * indexStride / sizeof(int32_t);

            Pack lineBuffer[pInputDims[0]];
            for (int i = 0; i < pInputDims[0]; i++) {
                lineBuffer[i].value = *inputLinePtr;
                lineBuffer[i].index = i + 1;
                inputLinePtr++;
            }

            PartialHeapSortPacked hsort(lineBuffer, k);
            if (mode == 0) {
                hsort.pushAll(pInputDims[0], isSmallValue);
                hsort.fullSort(isSmallValue);
            } else {
                hsort.pushAll(pInputDims[0], isLargeValue);
                hsort.fullSort(isLargeValue);
            }
            if (sort) {
                hsort.clear();
                hsort.pushAll(k, isSmallIndex);
                hsort.fullSort(isSmallIndex);
            }

            for (int i = 0; i < k; i++) {
                *valueLinePtr = lineBuffer[i].value;
                *indexLinePtr = lineBuffer[i].index - 1;
                valueLinePtr++;
                indexLinePtr++;
            }
        }
    }

    // calculate top K outer (axis = 1 and axis = 2)
    if (axis == 1) {
        int numLines = pInputDims[0] * pInputDims[2];

        int inputStride = pInputStrides[0] / CHAR_BIT;
        int valueStride = pValueStrides[0] / CHAR_BIT;
        int indexStride = pIndexStrides[0] / CHAR_BIT;

        for (int line = 0; line < numLines; line++) {
            int blockNum = line / pInputDims[0];
            int blockStep = line % pInputDims[0];
            int inputBlockOffset = blockNum * pInputDims[0] * pInputDims[1];
            int outputBlockOffset = blockNum * pValueDims[0] * k;
            half* inputLinePtr = p_act_input + inputBlockOffset + blockStep * inputStride / sizeof(half);
            half* valueLinePtr = p_act_value + outputBlockOffset + blockStep * valueStride / sizeof(half);
            int32_t* indexLinePtr = p_act_index + outputBlockOffset + blockStep * indexStride / sizeof(int32_t);

            Pack lineBuffer[pInputDims[1]];
            for (int i = 0; i < pInputDims[1]; i++) {
                lineBuffer[i].value = *inputLinePtr;
                lineBuffer[i].index = i + 1;
                inputLinePtr += pInputDims[0];
            }

            PartialHeapSortPacked hsort(lineBuffer, k);
            if (mode == 0) {
                hsort.pushAll(pInputDims[1], isSmallValue);
                hsort.fullSort(isSmallValue);
            } else {
                hsort.pushAll(pInputDims[1], isLargeValue);
                hsort.fullSort(isLargeValue);
            }
            if (sort) {
                hsort.clear();
                hsort.pushAll(k, isSmallIndex);
                hsort.fullSort(isSmallIndex);
            }

            for (int i = 0; i < k; i++) {
                *valueLinePtr = lineBuffer[i].value;
                *indexLinePtr = lineBuffer[i].index - 1;
                valueLinePtr += pValueDims[0];
                indexLinePtr += pIndexDims[0];
            }
        }
    }

    if (axis == 2) {
        int numLines = 1;
        for (int i = 0; i < numInputDims - 1; i++) {
            numLines *= pInputDims[i];
        }

        int inputStride = pInputStrides[0] / CHAR_BIT;
        int valueStride = pValueStrides[0] / CHAR_BIT;
        int indexStride = pIndexStrides[0] / CHAR_BIT;

        for (int line = 0; line < numLines; line++) {
            half* inputLinePtr = p_act_input + line * inputStride / sizeof(half);
            half* valueLinePtr = p_act_value + line * valueStride / sizeof(half);
            int32_t* indexLinePtr = p_act_index + line * indexStride / sizeof(int32_t);

            Pack lineBuffer[pInputDims[2]];
            for (int i = 0; i < pInputDims[2]; i++) {
                lineBuffer[i].value = *inputLinePtr;
                lineBuffer[i].index = i + 1;
                inputLinePtr += numLines;
            }

            PartialHeapSortPacked hsort(lineBuffer, k);
            if (mode == 0) {
                hsort.pushAll(pInputDims[2], isSmallValue);
                hsort.fullSort(isSmallValue);
            } else {
                hsort.pushAll(pInputDims[2], isLargeValue);
                hsort.fullSort(isLargeValue);
            }
            if (sort) {
                hsort.clear();
                hsort.pushAll(k, isSmallIndex);
                hsort.fullSort(isSmallIndex);
            }

            for (int i = 0; i < k; i++) {
                *valueLinePtr = lineBuffer[i].value;
                *indexLinePtr = lineBuffer[i].index - 1;
                valueLinePtr += numLines;
                indexLinePtr += numLines;
            }
        }
    }
}
}
}  // namespace shave_lib
}  // namespace nn
