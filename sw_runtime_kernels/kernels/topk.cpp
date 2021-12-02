#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <param_topk.h>
#include <algorithm>
#include <mvSubspaces.h>
#include <sw_shave_lib_common.h>
#include <moviVectorConvert.h>
#ifdef CONFIG_TARGET_SOC_3720
#include <dma_shave_nn.h>
#else
#include <dma_shave.h>
#endif

typedef sw_params::TopKParams MTLTopKParams;
#define MAX_DIMS MAX_ND_DIMS

#define INPUT_BPP  sizeof(fp16)

#define MIN(_a, _b) (__builtin_shave_cmu_min_i32_rr_int((_a), (_b)))

#define DIVR(_val, _size) (((_val) + ((_size) - 1)) / (_size))
#define ALIGN_TO_MULTIPLE(_size, _val) (DIVR((_val), (_size)) * (_size))
#define UNROLL_SIZE 8  // Changes to this should be reflected in the code as well.

using namespace sw_params;
using namespace subspace;

namespace  {
using Pack = MTLTopKParams;
using FindValueOuterFunc = void (*)(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines);
using FindValueMultipleFunc = void (*)(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines);
using FindValueSingleFunc = int32_t (*)(fp16* data, int32_t prevElems, int32_t numElems, int32_t startIndex, int32_t maxIndex);
using SortPackedFunc = void (*)(Pack* data, int32_t k, int32_t n);

constexpr int vectorSizeOuter = 16; // vector size for findValueOuter() implementation

template<class CompareVectors>
inline void findValueOuter(int32_t* _indices, fp16* _values, int32_t lineSize, int32_t numLines,
                           CompareVectors&& compareVectors){
    const int numVectors = DIVR(numLines, vectorSizeOuter);
    const bool hasIndices = (_indices != nullptr);
    
    int16* indices = reinterpret_cast<int16*>(_indices);
    half16* values = reinterpret_cast<half16*>(_values);
    
    for (int v = 0; v < numVectors; ++v){
        short16 index = (short16)0;
        half16 value = values[0];
        
        if (lineSize > 1){
            half16 val0 = values[1 * numVectors];
            for (int i = 2; i < lineSize; ++i){
                const half16 val1 = values[i * numVectors];
                
                const short16 mask = compareVectors(val0, value);
                index = ((short16)(i - 1) & mask) | (index & ~mask);
                value = (half16)(((short16)val0 & mask) | ((short16)value & ~mask));

                val0 = val1;
            }
            
            const short16 mask = compareVectors(val0, value);
            index = ((short16)(lineSize - 1) & mask) | (index & ~mask);
            value = (half16)(((short16)val0 & mask) | ((short16)value & ~mask));
        }
        if (hasIndices)
            indices[0] = mvuConvert_int16(index);
        values[0] = value;
        
        ++indices;
        ++values;
    }
}

__attribute__((noinline))
void findValueOuterMax(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines)
{
    return findValueOuter(indices, values, lineSize, numLines,
                          [](half16 a, half16 b) -> short16 { return !(a <= b); });
}

__attribute__((noinline))
void findValueOuterMin(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines)
{
    return findValueOuter(indices, values, lineSize, numLines,
                          [](half16 a, half16 b) -> short16 { return !(a >= b); });
}


// For short-axis case (a whole axis fit in CMX), max length <= 100K/sizeof(half) == 50K < 2^16,
// so we can maintain index as 16-bit integers for optimization.
// try: provide an assertion statement in run() method
template<class CompareVectors, class CompareScalars>
inline void findValueMultiple(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines,
                              CompareVectors&& compareVectors, CompareScalars&& compareScalars){
    int32_t line = 0;
    
    for (; line < numLines - 7; line += 8){
        short8 iindex = (short8)0;
        
        half8 hvalue = (half8)0;
        for (int v = 0; v < 8; ++v)
            hvalue[v] = values[(lineSize * (line + v)) + 0];
        
        if (lineSize > 1){
            half8 hval1 = (half8)0;
            for (int v = 0; v < 8; ++v)
                hval1[v] = values[(lineSize * (line + v)) + 1];
            
            for (int16_t i = 2; i < (int16_t)lineSize; ++i)
            {
                half8 hval2 = (half8)0;
                for (int v = 0; v < 8; ++v)
                    hval2[v] = values[(lineSize * (line + v)) + i];
                
                const short8 imask = compareVectors(hval1, hvalue);
                iindex = ((short8)(i - 1) & imask) | (iindex & ~imask);
                hvalue = (half8)(((short8)hval1 & imask) | ((short8)hvalue & ~imask));

                hval1 = hval2;
            }
            
            const short8 imask = compareVectors(hval1, hvalue);
            iindex = ((short8)(lineSize - 1) & imask) | (iindex & ~imask);
            hvalue = (half8)(((short8)hval1 & imask) | ((short8)hvalue & ~imask));
        }
        
        
        *reinterpret_cast<int8*>(&indices[line]) = mvuConvert_int8(iindex);
        
        for (int v = 0; v < 8; ++v)
            values[(lineSize * (line + v)) + 0] = hvalue[v];
    }
    
    for (; line < numLines; ++line){
        int16_t index = 0;
        fp16 value = values[(lineSize * line) + 0];
        for (int16_t i = 1; i < (int16_t)lineSize; ++i)
        {
            if (compareScalars(values[(lineSize * line) + i], value))
            {
                index = i;
                value = values[(lineSize * line) + i];
            }
        }
        indices[line] = index;
        values[(lineSize * line) + 0] = value;
    }
}

__attribute__((noinline))
void findValueMultipleMax(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines){
    return findValueMultiple(indices, values, lineSize, numLines,
            [](half8 a, half8 b) -> short8 { return !(a <= b); },
            [](fp16 a, fp16 b) -> bool { return !(a <= b); });
}

__attribute__((noinline))
void findValueMultipleMin(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines){
    return findValueMultiple(indices, values, lineSize, numLines,
            [](half8 a, half8 b) -> short8 { return !(a >= b); },
            [](fp16 a, fp16 b) -> bool { return !(a >= b); });
}

template<class CompareScalars>
inline int32_t findValueSingle(fp16* data, int32_t prevElems, int32_t numElems, int32_t startIndex, int32_t maxIndex,
                               CompareScalars&& compareScalars){
    int32_t index = maxIndex;
    fp16 value = data[0];
    for (int32_t i = 1; i < numElems; ++i)
    {
        if (compareScalars(data[prevElems + i], value))
        {
            index = startIndex + i;
            value = data[prevElems + i];
        }
    }
    if (prevElems > 0)
    {
        if (compareScalars(data[0], value))
        {
            index = maxIndex;
            value = data[0];
        }
    }
    data[0] = value;
    return index;
}

__attribute__((noinline))
int32_t findValueSingleMax(fp16* data, int32_t prevElems, int32_t numElems, int32_t startIndex, int32_t maxIndex){
    return findValueSingle(data, prevElems, numElems, startIndex, maxIndex,
                           [](fp16 a, fp16 b) -> bool { return !(a <= b); });
}

__attribute__((noinline))
int32_t findValueSingleMin(fp16* data, int32_t prevElems, int32_t numElems, int32_t startIndex, int32_t maxIndex)
{
    return findValueSingle(data, prevElems, numElems, startIndex, maxIndex,
                           [](fp16 a, fp16 b) -> bool { return !(a >= b); });
}

// note: try to implement parallel (vector) sort
template<class ComparePacked>
class PartialHeapSortPacked{
public:
    PartialHeapSortPacked(Pack *items, int cnt)
            : m_items(items)
              , m_cnt(cnt)
              , m_fill(0)
    {}
    
    int INDEX_PARENT(int i) const
    {
        return (i - 1) / 2;
    }

    void pushall(int n, ComparePacked comparePacked)
    {
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
    int fullsort(ComparePacked comparePacked)
    {
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
    void siftDown(int start, int end, ComparePacked comparePacked)
    {
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
private:
    Pack* m_items;
    int m_cnt;
    int m_fill;
};

template<class ComparePacked>
inline void partialHeapsortPacked(Pack* data, int32_t k, int32_t n,
                                  ComparePacked&& comparePacked){
    PartialHeapSortPacked<ComparePacked> hsort(data, k);
    hsort.pushall(n, std::forward<ComparePacked>(comparePacked));
    hsort.fullsort(std::forward<ComparePacked>(comparePacked));
}

void sortPackedByValuesMax(Pack* data, int32_t k, int32_t n){
    partialHeapsortPacked(data, k, n,
                          [](const Pack& a, const Pack& b) -> bool
                          { return !(a.value <= b.value) | (!(a.value != b.value) & (a.index < b.index)); });
}

void sortPackedByValuesMin(Pack* data, int32_t k, int32_t n){
    partialHeapsortPacked(data, k, n,
                          [](const Pack& a, const Pack& b) -> bool
                          { return !(a.value >= b.value) | (!(a.value != b.value) & (a.index < b.index)); });
}

void sortPackedByIndices(Pack* data, int32_t k, int32_t n){
    partialHeapsortPacked(data, k, n,
                          [](const Pack& a, const Pack& b) -> bool
                          { return bool(a.index < b.index); });
}

void fillPackedIndices(Pack* lines, int32_t numLines, int32_t lineSize, int32_t offset){
    int32_t line_i = 0;
    for (; line_i < numLines; ++line_i)
    {
        Pack* line = &lines[lineSize * line_i];
        int8 seq = (int8)offset + (int8) { 0, 1, 2, 3, 4, 5, 6, 7 };
        
        int32_t i = 0;
        for (; i < lineSize - 7; i += 8)
        {
            line[0].index = seq[0];
            line[1].index = seq[1];
            line[2].index = seq[2];
            line[3].index = seq[3];
            line[4].index = seq[4];
            line[5].index = seq[5];
            line[6].index = seq[6];
            line[7].index = seq[7];

            line += 8;
            seq += (int8)8;
        }

        for (; i < lineSize; ++i)
        {
            lines[lineSize * line_i + i].index = static_cast<int32_t>(offset + i);
        }
    }
}

static void dma_start_3d(DmaAlShave& dma, const void* src, void* dst, uint32_t byteLength,
                         uint32_t srcWidth, uint32_t dstWidth, uint32_t srcStride, uint32_t dstStride,
                         uint32_t numPlanes, uint32_t srcPlaneStride, uint32_t dstPlaneStride){
    if (((byteLength % srcWidth) == 0) && (srcStride * byteLength == srcPlaneStride * srcWidth) &&
        ((byteLength % dstWidth) == 0) && (dstStride * byteLength == dstPlaneStride * dstWidth))
    {
        byteLength *= numPlanes;
        numPlanes = 1;
    }
    if (srcWidth == srcStride)
        srcWidth = srcStride = byteLength;
    if (dstWidth == dstStride)
        dstWidth = dstStride = byteLength;
    dma.start(src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride, numPlanes, srcPlaneStride, dstPlaneStride);
}

static void dma_start_2d(DmaAlShave& dma, const void* src, void* dst, uint32_t byteLength,
                         uint32_t srcWidth, uint32_t dstWidth, uint32_t srcStride, uint32_t dstStride){
    if (srcWidth == srcStride)
        srcWidth = srcStride = byteLength;
    if (dstWidth == dstStride)
        dstWidth = dstStride = byteLength;
    dma.start(src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride);
}

bool topKSimpleFindOuter(MTLTopKParams * p){
    nnLog(MVLOG_DEBUG, "topKSimpleFindOuter()");
    
    DmaAlShave dma1;

    const int32_t* dims = p->inputValueDims;
    const int32_t ndims = p->inNdims;

    const bool hasIndices = (p->hasIndices != 0);
    const bool hasValues  = (p->hasValues != 0);

    const uint8_t* inputValues = reinterpret_cast<const uint8_t*>(p->inputValues.get());
    uint8_t* outputValues = reinterpret_cast<uint8_t*>(p->outputValues.get());
    uint8_t* outputIndices = reinterpret_cast<uint8_t*>(p->outputIndices.get());

    const TopKMode mode = static_cast<TopKMode>(p->mode);

    FindValueOuterFunc findValue = nullptr;
    switch (mode)
    {
    case TopKMode::max: findValue = findValueOuterMax; break;
    case TopKMode::min: findValue = findValueOuterMin; break;
    default: return false;
    }

    const int indexBPP = hasIndices ? sizeof(int32_t) : 0;

    const int lineBytes = (p->inputDim * INPUT_BPP) + indexBPP;
    const int cmxLines = vectorSizeOuter * (p->availableCmxBytes / (lineBytes * vectorSizeOuter));

    if (!(cmxLines > 0))
        return false;

    const int32_t linesLimit = p->start + p->toProcess;
    int32_t currentLine = p->start;

    uint8_t* cmx = reinterpret_cast<uint8_t*>(p->cmxData.get());
    int32_t* indexBuffer = hasIndices ? reinterpret_cast<int32_t*>(cmx) : nullptr; cmx += indexBPP * cmxLines;
    fp16* valueBuffer = reinterpret_cast<fp16*>(cmx); // uncomment for next allocations: cmx += (cmxLines * p->inputDim * INPUT_BPP) * cmxLines;

    int32_t setCoords[MAX_DIMS];
    subspace::getCoord(currentLine, dims, ndims, setCoords);

    const bool needToSortData = !bool(p->k == p->inputDim);

    while (currentLine < linesLimit)
    {
        const int32_t numLines = MIN(cmxLines, MIN(linesLimit - currentLine, dims[0] - setCoords[0]));
        const int32_t numLinesA = ALIGN_TO_MULTIPLE(vectorSizeOuter, numLines);

        unsigned inputValuesOffset, outputValuesOffset, outputIndicesOffset;
        subspace::getOffsetsU8(setCoords, p->inputValueStrides, p->outputValueStrides, p->outputIndicesStrides,
                               ndims, inputValuesOffset, outputValuesOffset, outputIndicesOffset);

        const uint8_t* inputValuesPtr = inputValues + inputValuesOffset;
        uint8_t* outputValuesPtr = outputValues + outputValuesOffset;
        uint8_t* outputIndicesPtr = outputIndices + outputIndicesOffset;

        dma_start_2d(dma1, inputValuesPtr, reinterpret_cast<uint8_t*>(&valueBuffer[0]),
                     numLines * p->inputDim * INPUT_BPP,
                     numLines * INPUT_BPP,
                     numLines * INPUT_BPP,
                     p->inputValueStride,
                     numLinesA * INPUT_BPP
        );
        dma1.wait();

        if (hasIndices)
        {
            std::fill_n(&indexBuffer[0], numLines, 0);
        }

        if (needToSortData)
        {
            findValue(&indexBuffer[0], &valueBuffer[0], p->inputDim, numLines);
        }

        if (hasValues)
        {
            dma1.start(reinterpret_cast<uint8_t*>(&valueBuffer[0]), outputValuesPtr, numLines * p->outputDim * INPUT_BPP);
            dma1.wait();
        }

        if (hasIndices)
        {
            dma1.start(reinterpret_cast<uint8_t*>(&indexBuffer[0]), outputIndicesPtr, numLines * p->outputDim * sizeof(int32_t));
            dma1.wait();
        }

        subspace::incrementNCoord(setCoords, dims, ndims, numLines);
        currentLine += numLines;
    }

    return true;
}

void topKMergedFindSingle(MTLTopKParams * p){
    nnLog(MVLOG_DEBUG, "topKMergedFindSingle()");
    
    DmaAlShave dma1;

    const int32_t* dims = p->inputValueDims;
    const int32_t ndims = p->inNdims;

    const bool hasIndices = (p->hasIndices != 0);
    const bool hasValues  = (p->hasValues != 0);

    const uint8_t* inputValues = reinterpret_cast<const uint8_t*>(p->inputValues.get());
    uint8_t* outputValues = reinterpret_cast<uint8_t*>(p->outputValues.get());
    uint8_t* outputIndices = reinterpret_cast<uint8_t*>(p->outputIndices.get());

    const TopKMode mode = static_cast<TopKMode>(p->mode);

    FindValueSingleFunc findValue = nullptr;
    switch (mode)
    {
    case TopKMode::max: findValue = findValueSingleMax; break;
    case TopKMode::min: findValue = findValueSingleMin; break;
    default: return;
    }

    const auto cmxBytes = p->availableCmxBytes;
    const auto cmxSlice = p->cmxData;

    const int32_t cmxElems = cmxBytes / sizeof(fp16);
    const int32_t cmxLines = 1;

    const int32_t linesLimit = p->start + p->toProcess;
    int32_t currentLine = p->start;

    fp16* lineBuffer = reinterpret_cast<fp16*>(cmxSlice.get());

    int32_t setCoords[MAX_DIMS];
    subspace::getCoord(currentLine, dims, ndims, setCoords);

    while (currentLine < linesLimit)
    {
        const int32_t numLines = MIN(cmxLines, MIN(linesLimit - currentLine, dims[0] - setCoords[0]));

        unsigned inputValuesOffset, outputValuesOffset, outputIndicesOffset;
        subspace::getOffsetsU8(setCoords, p->inputValueStrides, p->outputValueStrides, p->outputIndicesStrides,
                               ndims, inputValuesOffset, outputValuesOffset, outputIndicesOffset);

        const uint8_t* inputValuesPtr = inputValues + inputValuesOffset;
        uint8_t* outputValuesPtr = outputValues + outputValuesOffset;
        uint8_t* outputIndicesPtr = outputIndices + outputIndicesOffset;

        const int32_t totalElems = p->inputDim;

        int32_t currentElem = 0;
        int32_t maxElems = cmxElems;
        int32_t prevElems = 0;

        int32_t maxIndex = 0;
        while (currentElem < totalElems)
        {
            int32_t newElems = MIN(totalElems - currentElem, maxElems);

            dma_start_2d(dma1, inputValuesPtr, reinterpret_cast<uint8_t*>(&lineBuffer[prevElems]),
                         newElems * INPUT_BPP,
                         INPUT_BPP,
                         INPUT_BPP,
                         p->inputValueStride,
                         sizeof(fp16)
            );
            dma1.wait();

            maxIndex = findValue(&lineBuffer[0], prevElems, newElems, currentElem, maxIndex);

            inputValuesPtr += newElems * p->inputValueStride;
            currentElem += newElems;

            maxElems = cmxElems - 1;
            prevElems = 1;
        }

        if (hasValues)
        {
            dma_start_2d(dma1, reinterpret_cast<uint8_t*>(&lineBuffer[0]), outputValuesPtr,
                         p->outputDim * INPUT_BPP,
                         INPUT_BPP,
                         INPUT_BPP,
                         sizeof(fp16),
                         p->outputValueStride);
            dma1.wait();
        }

        if (hasIndices)
        {
            dma_start_2d(dma1, reinterpret_cast<uint8_t*>(&maxIndex), outputIndicesPtr,
                         p->outputDim * sizeof(int32_t),
                         sizeof(int32_t),
                         sizeof(int32_t),
                         sizeof(Pack),
                         p->outputIndicesStride);
            dma1.wait();
        }

        subspace::incrementNCoord(setCoords, dims, ndims, numLines);
        currentLine += numLines;
    }
}

void topKSimpleFindMultiple(MTLTopKParams * p){
    nnLog(MVLOG_DEBUG, "topKSimpleFindMultiple()");
    
    DmaAlShave dma1;
    const int32_t* dims = p->inputValueDims;
    const int32_t ndims = p->inNdims;
    
    const bool hasIndices = (p->hasIndices != 0);
    const bool hasValues  = (p->hasValues != 0);
    
    const uint8_t* inputValues = reinterpret_cast<const uint8_t*>(p->inputValues.get());
    uint8_t* outputValues = reinterpret_cast<uint8_t*>(p->outputValues.get());
    uint8_t* outputIndices = reinterpret_cast<uint8_t*>(p->outputIndices.get());
    
    const TopKMode mode = static_cast<TopKMode>(p->mode);
    
    FindValueMultipleFunc findValue = nullptr;
    
    switch (mode)
    {
    case 0: findValue = findValueMultipleMax; break;
    case 1: findValue = findValueMultipleMin; break;
    default: return;
    }
    
    const auto cmxBytes = p->availableCmxBytes;
    const auto cmxSlice = p->cmxData;
    
    const int32_t lineBytes = sizeof(int32_t) + (sizeof(fp16) * p->inputDim);
    const int32_t maxLines = MIN(dims[0], DmaAlShave::max_3D_planes);
    const int32_t cmxLines = MIN(maxLines, cmxBytes / lineBytes);
    
    const int32_t linesLimit = p->start + p->toProcess;
    int32_t currentLine = p->start;
    
    uint8_t* cmx = reinterpret_cast<uint8_t*>(cmxSlice.get());
    int32_t* indexBuffer = reinterpret_cast<int32_t*>(cmx); cmx += sizeof(int32_t) * cmxLines;
    fp16* valueBuffer = reinterpret_cast<fp16*>(cmx); // uncomment for next allocations: cmx += (sizeof(fp16) * p->inputDim) * cmxLines;
    
    int32_t setCoords[MAX_DIMS];
    subspace::getCoord(currentLine, dims, ndims, setCoords);
    
    const bool needToSortData = !bool(p->k == p->inputDim);
    
    while (currentLine < linesLimit){
        const int32_t numLines = MIN(cmxLines, MIN(linesLimit - currentLine, dims[0] - setCoords[0]));
        
        unsigned inputValuesOffset, outputValuesOffset, outputIndicesOffset;
        subspace::getOffsetsU8(setCoords, p->inputValueStrides, p->outputValueStrides, p->outputIndicesStrides,
                               ndims, inputValuesOffset, outputValuesOffset, outputIndicesOffset);
        
        const uint8_t* inputValuesPtr = inputValues + inputValuesOffset;
        uint8_t* outputValuesPtr = outputValues + outputValuesOffset;
        uint8_t* outputIndicesPtr = outputIndices + outputIndicesOffset;
        
        dma_start_3d(dma1, inputValuesPtr, reinterpret_cast<uint8_t*>(&valueBuffer[0]),
                     p->inputDim * INPUT_BPP,
                     INPUT_BPP,
                     INPUT_BPP,
                     p->inputValueStride,
                     sizeof(fp16),
                     numLines,
                     p->inputValueStrides[0],
                     p->inputDim * sizeof(fp16)
        );
        std::fill_n(&indexBuffer[0], numLines, 0);
        dma1.wait();
        
        if (needToSortData){
            findValue(&indexBuffer[0], &valueBuffer[0], p->inputDim, numLines);
        }
        
        if (hasValues){
            dma_start_3d(dma1, reinterpret_cast<uint8_t*>(&valueBuffer[0]), outputValuesPtr,
                         p->outputDim * INPUT_BPP,
                         INPUT_BPP,
                         INPUT_BPP,
                         sizeof(fp16),
                         p->outputValueStride,
                         numLines,
                         p->inputDim * sizeof(fp16),
                         p->outputValueStrides[0]
            );
            dma1.wait();
        }
        
        if (hasIndices){
            dma_start_3d(dma1, reinterpret_cast<uint8_t*>(&indexBuffer[0]), outputIndicesPtr,
                         p->outputDim * sizeof(int32_t),
                         sizeof(int32_t),
                         sizeof(int32_t),
                         sizeof(int32_t),
                         p->outputIndicesStride,
                         numLines,
                         sizeof(int32_t),
                         p->outputIndicesStrides[0]
            );
            dma1.wait();
        }
        
        subspace::incrementNCoord(setCoords, dims, ndims, numLines);
        currentLine += numLines;
    }
}

void topKMergedSort(MTLTopKParams* p){
    nnLog(MVLOG_DEBUG, "topKMergedSort()");
    
    DmaAlShave dma1;

    const int32_t* dims = p->inputValueDims;
    const int32_t ndims = p->inNdims;
    
    const bool hasIndices = (p->hasIndices != 0);
    const bool hasValues  = (p->hasValues != 0);
    
    const uint8_t* inputValues = reinterpret_cast<const uint8_t*>(p->inputValues.get());
    uint8_t* outputValues = reinterpret_cast<uint8_t*>(p->outputValues.get());
    uint8_t* outputIndices = reinterpret_cast<uint8_t*>(p->outputIndices.get());
    
    const TopKMode mode = static_cast<TopKMode>(p->mode);
    const TopKSort sort = static_cast<TopKSort>(p->sort);
    
    SortPackedFunc sortPackedByValues = nullptr;
    switch (mode){
    case TopKMode::max: sortPackedByValues = sortPackedByValuesMax; break;
    case TopKMode::min: sortPackedByValues = sortPackedByValuesMin; break;
    default: return;
    }
    
    const auto cmxBytes = p->availableCmxBytes;
    const auto cmxSlice = p->cmxData;
    
    const int32_t cmxElems = cmxBytes / sizeof(Pack);
    const int32_t cmxLines = 1;
    
    const int32_t linesLimit = p->start + p->toProcess;
    int32_t currentLine = p->start;
    
    Pack* lineBuffer = reinterpret_cast<Pack*>(cmxSlice.get());

    int32_t setCoords[MAX_DIMS];
    subspace::getCoord(currentLine, dims, ndims, setCoords);

    while (currentLine < linesLimit){
        const int32_t numLines = MIN(cmxLines, MIN(linesLimit - currentLine, dims[0] - setCoords[0]));
        
        unsigned inputValuesOffset, outputValuesOffset, outputIndicesOffset;
        subspace::getOffsetsU8(setCoords, p->inputValueStrides, p->outputValueStrides, p->outputIndicesStrides,
                               ndims, inputValuesOffset, outputValuesOffset, outputIndicesOffset);

        const uint8_t* inputValuesPtr = inputValues + inputValuesOffset;
        uint8_t* outputValuesPtr = outputValues + outputValuesOffset;
        uint8_t* outputIndicesPtr = outputIndices + outputIndicesOffset;

        const int32_t totalElems = p->inputDim;
        
        int32_t currentElem = 0;
        int32_t maxElems = cmxElems;
        int32_t prevElems = 0;
        
        while (currentElem < totalElems){
            int32_t newElems = MIN(totalElems - currentElem, maxElems);
            
            dma_start_2d(dma1, inputValuesPtr, reinterpret_cast<uint8_t*>(&lineBuffer[prevElems].value),
                         newElems * INPUT_BPP,
                         INPUT_BPP,
                         INPUT_BPP,
                         p->inputValueStride,
                         sizeof(Pack)
            );
            fillPackedIndices(&lineBuffer[prevElems], numLines, newElems, currentElem);
            dma1.wait();

            sortPackedByValues(&lineBuffer[0], p->k, prevElems + newElems);

            inputValuesPtr += newElems * p->inputValueStride;
            currentElem += newElems;

            maxElems = cmxElems - p->k;
            prevElems = p->k;
        }
        
        if (sort == TopKSort::index){// also, p->k != 1
            sortPackedByIndices(&lineBuffer[0], p->k, p->k);
        }
        
        if (hasValues){
            dma_start_2d(dma1, reinterpret_cast<uint8_t*>(&lineBuffer[0].value), outputValuesPtr,
                         p->outputDim * INPUT_BPP,
                         INPUT_BPP,
                         INPUT_BPP,
                         sizeof(Pack),
                         p->outputValueStride);
            dma1.wait();
        }
        
        if (hasIndices){
            dma_start_2d(dma1, reinterpret_cast<uint8_t*>(&lineBuffer[0].index), outputIndicesPtr,
                         p->outputDim * sizeof(int32_t),
                         sizeof(int32_t),
                         sizeof(int32_t),
                         sizeof(Pack),
                         p->outputIndicesStride);
            dma1.wait();
        }
        
        subspace::incrementNCoord(setCoords, dims, ndims, numLines);
        currentLine += numLines;
    }
}

void topKSimpleSort(MTLTopKParams * p){
    nnLog(MVLOG_DEBUG, "topKSimpleSort()");
    
    DmaAlShave dma1;

    const int32_t* dims = p->inputValueDims;
    const int32_t ndims = p->inNdims;

    const bool hasIndices = (p->hasIndices != 0);
    const bool hasValues  = (p->hasValues != 0);

    const uint8_t* inputValues = reinterpret_cast<const uint8_t*>(p->inputValues.get());
    uint8_t* outputValues = reinterpret_cast<uint8_t*>(p->outputValues.get());
    uint8_t* outputIndices = reinterpret_cast<uint8_t*>(p->outputIndices.get());

    const TopKMode mode = static_cast<TopKMode>(p->mode);
    const TopKSort sort = static_cast<TopKSort>(p->sort);

    SortPackedFunc sortPackedByValues = nullptr;
    switch (mode){
    case TopKMode::max: sortPackedByValues = sortPackedByValuesMax; break;
    case TopKMode::min: sortPackedByValues = sortPackedByValuesMin; break;
    default: return;
    }
    
    const auto cmxBytes = p->availableCmxBytes;
    const auto cmxSlice = p->cmxData;
    
    const int32_t lineBytes = sizeof(Pack) * p->inputDim;
    const int32_t maxLines = MIN(dims[0], DmaAlShave::max_3D_planes);
    const int32_t cmxLines = MIN(maxLines, cmxBytes / lineBytes);

    const int32_t linesLimit = p->start + p->toProcess;
    int32_t currentLine = p->start;

    Pack* lineBuffer = reinterpret_cast<Pack*>(cmxSlice.get());

    int32_t setCoords[MAX_DIMS];
    subspace::getCoord(currentLine, dims, ndims, setCoords);

    const bool needToSortData = !bool( (p->k == p->inputDim) && (sort == TopKSort::index) );
    while (currentLine < linesLimit){
        int32_t numLines = MIN(cmxLines, MIN(linesLimit - currentLine, dims[0] - setCoords[0]));
        
        unsigned inputValuesOffset, outputValuesOffset, outputIndicesOffset;
        subspace::getOffsetsU8(setCoords, p->inputValueStrides, p->outputValueStrides, p->outputIndicesStrides,
                               ndims, inputValuesOffset, outputValuesOffset, outputIndicesOffset);

        const uint8_t* inputValuesPtr = inputValues + inputValuesOffset;
        uint8_t* outputValuesPtr = outputValues + outputValuesOffset;
        uint8_t* outputIndicesPtr = outputIndices + outputIndicesOffset;
        
        dma_start_3d(dma1, inputValuesPtr, reinterpret_cast<uint8_t*>(&lineBuffer[0].value),
                     p->inputDim * INPUT_BPP,
                     INPUT_BPP,
                     INPUT_BPP,
                     p->inputValueStride,
                     sizeof(Pack),
                     numLines,
                     p->inputValueStrides[0],
                     p->inputDim * sizeof(Pack)
        );
        fillPackedIndices(lineBuffer, numLines, p->inputDim, 0);
        dma1.wait();
        
        if (needToSortData){
            for (int32_t line = 0; line < numLines; ++line){
                sortPackedByValues(&lineBuffer[p->inputDim * line], p->k, p->inputDim);
                if (sort == TopKSort::index){ // also, p->k != 1
                    sortPackedByIndices(&lineBuffer[p->inputDim * line], p->k, p->k);
                }
            }
        }
        
        if (hasValues){
            dma_start_3d(dma1, reinterpret_cast<uint8_t*>(&lineBuffer[0].value), outputValuesPtr,
                         p->outputDim * INPUT_BPP,
                         INPUT_BPP,
                         INPUT_BPP,
                         sizeof(Pack),
                         p->outputValueStride,
                         numLines,
                         p->inputDim * sizeof(Pack),
                         p->outputValueStrides[0]
            );
            dma1.wait();
        }
        
        if (hasIndices){
            dma_start_3d(dma1, reinterpret_cast<uint8_t*>(&lineBuffer[0].index), outputIndicesPtr,
                         p->outputDim * sizeof(int32_t),
                         sizeof(int32_t),
                         sizeof(int32_t),
                         sizeof(Pack),
                         p->outputIndicesStride,
                         numLines,
                         p->inputDim * sizeof(Pack),
                         p->outputIndicesStrides[0]
            );
            dma1.wait();
        }
        
        subspace::incrementNCoord(setCoords, dims, ndims, numLines);
        currentLine += numLines;
    }
}
};

extern "C" void mvTopK(t_MvTopKParams* p) {
    const int32_t cmxBytes = p->availableCmxBytes;

    if (p->k == 1) {
        bool status = false;
        if ((p->k == 1) && (p->axis > 0))
            status = topKSimpleFindOuter(p);
        if (!status) {
            const int32_t lineBytes = sizeof(int32_t) + (sizeof(fp16) * p->inputDim);
            if (lineBytes <= cmxBytes)
                topKSimpleFindMultiple(p);
            else
                topKMergedFindSingle(p);
        }
    } else {
        const int32_t lineBytes = sizeof(Pack) * p->inputDim;
        if (lineBytes <= cmxBytes)
            topKSimpleSort(p);
        else
            topKMergedSort(p);
    }
}