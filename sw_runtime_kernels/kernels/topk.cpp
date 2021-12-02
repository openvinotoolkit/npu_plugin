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

namespace {
using Pack = MTLTopKParams;
using FindValueOuterFunc = void (*)(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines);
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

    const uint8_t* inputValues = reinterpret_cast<const uint8_t*>(p->inputValues.dataAddr);
    uint8_t* outputValues = reinterpret_cast<uint8_t*>(p->outputValues.dataAddr);
    uint8_t* outputIndices = reinterpret_cast<uint8_t*>(p->outputIndices.dataAddr);

    int mode = p->mode;

    FindValueOuterFunc findValue = nullptr;
    switch (mode)
    {
    case 0: findValue = findValueOuterMax; break;
    case 1: findValue = findValueOuterMin; break;
    default: return false;
    }

    const int indexBPP = hasIndices ? sizeof(int32_t) : 0;

    const int lineBytes = (p->inputDim * INPUT_BPP) + indexBPP;
    const int cmxLines = vectorSizeOuter * (p->availableCmxBytes / (lineBytes * vectorSizeOuter));

    if (!(cmxLines > 0))
        return false;

    const int32_t linesLimit = p->start + p->toProcess;
    int32_t currentLine = p->start;

    uint8_t* cmx = reinterpret_cast<uint8_t*>(p->cmxData);
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
};

extern "C" void topk(MTLTopKParams * p) {
    bool status = topKSimpleFindOuter(p);
}