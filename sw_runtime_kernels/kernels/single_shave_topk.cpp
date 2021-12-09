// {% copyright %}

#include <moviVectorTypes.h>
#include <mvSubspaces.h>
#include <math.h>
#include <stdio.h>
#include <param_topk.h>

struct Pack {
    half value;
    int32_t index;
};

using ComparePackedFunc = bool (*)(const Pack& a, const Pack& b);

struct PartialHeapSortPacked
{
    Pack* m_items;
    int m_cnt;
    int m_fill;

    PartialHeapSortPacked(Pack *items, int cnt)
            : m_items(items)
            , m_cnt(cnt)
            , m_fill(0)
    {}

    int INDEX_PARENT(int i) const {
        return (i - 1) / 2;
    }

    void pushAll(int n, ComparePackedFunc comparePacked) {
        m_fill = m_cnt;

        int start = INDEX_PARENT(m_cnt - 1);
        while (start >= 0)
        {
            siftDown(start, -1, comparePacked);
            start--;
        }
        for (int i = m_fill; i < n; ++i)
        {
            if (comparePacked(m_items[i], m_items[0]))
            {
                m_items[0] = m_items[i];
                siftDown(0, -1, comparePacked);
            }
        }
    }

    //full sort the array by putting smallest item to the end one by one.
    //after full sort, the heap is empty and array is in descending order
    int fullSort(ComparePackedFunc comparePacked) {
        if (m_cnt > 1)
        {
            int N = m_cnt;
            while (N > 1)
            {
                std::swap(m_items[0], m_items[N-1]);
                N --;
                siftDown(0, N, comparePacked);
            }
        }
        return m_cnt;
    }

    //after top is changed, siftDown can recover the heap order,
    //ensure top is the smallest
    void siftDown(int start, int end, ComparePackedFunc comparePacked) {
        if (end < start)
            end = m_cnt;

        // make a copy of the new item on top
        Pack itemx = m_items[start];
        int root = start;
        int child0 = (2*root + 1);
        //the root element is going down along the heap-tree structure
        //and smaller child along its path will bubble-up
        while (child0 < end)
        {
            int child1 = child0 + 1;
            int swapx = root;
            Pack* pItemx = &itemx;

            if (comparePacked(*pItemx, m_items[child0]))
            {
                swapx = child0;
                pItemx = &m_items[child0];
            }

            if (child1 < end)
            {
                if (comparePacked(*pItemx, m_items[child1]))
                {
                    swapx = child1;
                    pItemx = &m_items[child1];
                }
            }

            if (swapx == root)
                break;

            //bubble-up smallest child to root
            m_items[root] = m_items[swapx];

            //sift following sub-tree
            root = swapx;
            child0 = (2*root + 1);
        }

        //final location of the new element put into the heap
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

extern "C" {

void singleShaveTopK(uint32_t lParamsAddr) {

    half* p_act_input = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->input.dataAddr);
    int32_t k = *(int32_t*)(reinterpret_cast<TopKParams*>(lParamsAddr)->k.dataAddr);
    half* p_act_value = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->value.dataAddr);
    int32_t* p_act_index = (int32_t*)(reinterpret_cast<TopKParams*>(lParamsAddr)->index.dataAddr);

    const TopKParams* lParams = reinterpret_cast<const TopKParams *>(lParamsAddr);

    half* p_act_input_tmp = p_act_input;
    half* p_act_value_tmp = p_act_value;
    for (int i = 0; i < 2 * 3 * 4; i++) {
        //*p_act_value_tmp = *p_act_input_tmp;
        *p_act_value_tmp = 0;
        if (i == 0) {
            *p_act_value_tmp = 1234;
        }
        p_act_input_tmp++;
        p_act_value_tmp++;
    }

    // Case dimension NCHW 4, 3, 2 (0,0,0)=1, (3,2,1)=24
    int32_t axis = 0;
    int32_t mode = 0; // max: 0, min: 1
    int32_t sort = 0; // value: 0, index: 1

    int32_t numInputDims = (int32_t)lParams->input.numDims;
    int32_t* pInputDims = (int32_t*)(lParams->input.dimsAddr);
    int64_t* pInputStrides = (int64_t*)(lParams->input.stridesAddr);

    int32_t numValueDims = (int32_t)lParams->value.numDims;
    int32_t* pValueDims = (int32_t*)(lParams->value.dimsAddr);
    int64_t* pValueStrides = (int64_t*)(lParams->value.stridesAddr);

    int32_t numIndexDims = (int32_t)lParams->index.numDims;
    int32_t* pIndexDims = (int32_t*)(lParams->index.dimsAddr);
    int64_t* pIndexStrides = (int64_t*)(lParams->index.stridesAddr);

    p_act_index[0] = k;

    // input dim number
    p_act_index[1] = numInputDims; // 3
    // input dims
    p_act_index[2] = pInputDims[0]; // 4
    p_act_index[3] = pInputDims[1]; // 3
    p_act_index[4] = pInputDims[2]; // 2
    p_act_index[5] = pInputDims[3]; // -1
    // input strides
    p_act_index[6] = pInputStrides[0] / CHAR_BIT; // 2
    p_act_index[7] = pInputStrides[1] / CHAR_BIT; // 8
    p_act_index[8] = pInputStrides[2] / CHAR_BIT;; // 24
    p_act_index[9] = pInputStrides[3] / CHAR_BIT;; // -842150451 or 0

//    // value dim number
//    p_act_index[10] = numValueDims; // 3
//    // value dims
//    p_act_index[11] = pValueDims[0]; // 4
//    p_act_index[12] = pValueDims[1]; // 3
//    p_act_index[13] = pValueDims[2]; // 2
//    p_act_index[14] = pValueDims[3]; // -1
//    // value strides
//    p_act_index[15] = pValueStrides[0] / CHAR_BIT; // 2
//    p_act_index[16] = pValueStrides[1] / CHAR_BIT; // 8
//    p_act_index[17] = pValueStrides[2] / CHAR_BIT; // 24
//    p_act_index[18] = pValueStrides[3] / CHAR_BIT; // -842150451 or 0

    // index dim number
    p_act_index[10] = numIndexDims; // 3
    // index dims
    p_act_index[11] = pIndexDims[0]; // 4
    p_act_index[12] = pIndexDims[1]; // 3
    p_act_index[13] = pIndexDims[2]; // 2
    p_act_index[14] = pIndexDims[3]; // -1
    // index strides
    p_act_index[15] = pIndexStrides[0] / CHAR_BIT; // 4
    p_act_index[16] = pIndexStrides[1] / CHAR_BIT; // 16
    p_act_index[17] = pIndexStrides[2] / CHAR_BIT; // 48
    p_act_index[18] = pIndexStrides[3] / CHAR_BIT; // -842150451 or 0

    ComparePackedFunc comparePacked = nullptr;
    switch (mode) {
        case 0: comparePacked = isSmallValue; break;
        case 1: comparePacked = isLargeValue; break;
        default: return;
    }

    // cases for k = 1
    // calculate top K inner (axis = 0)
    k = 1;
    if (axis == 0) {

        int numLines = 1;
        for (int i = 1; i < numInputDims; i++) {
            numLines *= pInputDims[i];
        }

        int inputStride = pInputStrides[1] / CHAR_BIT;
        int valueStride = k * 2; // pValueStrides[1] / CHAR_BIT
        int indexStride = k * 4; // pIndexStrides[1] / CHAR_BIT

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
            hsort.pushAll(pInputDims[0], comparePacked);
            hsort.fullSort(comparePacked);
            // TODO: sort by index

            for (int i = 0; i < k; i++) {
                *valueLinePtr = lineBuffer[i].value;
                *indexLinePtr = lineBuffer[i].index;
                valueLinePtr++;
                indexLinePtr++;
            }
        }
    }

}
}
