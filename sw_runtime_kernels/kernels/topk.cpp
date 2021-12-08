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

namespace{

typedef half* phalf;

#define WORK_BUFFER_SIZE (((p->availableCmxBytes)/4))

constexpr int vectorSizeOuter = 16; // vector size for findValueOuter() implementation
using FindValueOuterFunc = void (*)(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines);


template<class CompareVectors>
inline void findValueOuter(int32_t* _indices, fp16* _values, int32_t lineSize, int32_t numLines,
                           CompareVectors&& compareVectors)
{
    const int numVectors = DIVR(numLines, vectorSizeOuter);
    
    const bool hasIndices = (_indices != nullptr);

    int16* indices = reinterpret_cast<int16*>(_indices);
    half16* values = reinterpret_cast<half16*>(_values);

    for (int v = 0; v < numVectors; ++v)
    {
        short16 index = (short16)0;
        half16 value = values[0];

        if (lineSize > 1)
        {
            half16 val0 = values[1 * numVectors];

            for (int i = 2; i < lineSize; ++i)
            {
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

        if (hasIndices==1)
            indices[0] = mvuConvert_int16(index);
        values[0] = value;

        ++indices;
        ++values;
    }
}

template<class CompareScalars>
inline int32_t findValueSingle(fp16* data, int32_t prevElems, int32_t numElems, int32_t startIndex, int32_t maxIndex,
                               CompareScalars&& compareScalars)
{
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

struct t_MvTopKParamNClasses
{
    half* input;
    Location inLocation;
    half* output;
    Location outLocation;
    half* outputInd;
    Location outIndLocation;
    u8* cmxslice;
    int32_t availableCmxBytes;
    
    s32 ndims;
    int32_t k;
    int32_t inputValueDims[MAX_ND_DIMS];
    int32_t outputValueDims[MAX_ND_DIMS];
    int32_t outputIndicesDims[MAX_ND_DIMS];
    int32_t in_strides[MAX_ND_DIMS];
    int32_t out_strides[MAX_ND_DIMS];
    int32_t ind_strides[MAX_ND_DIMS];

    s32 axis;
    s32 axisDim;
    s32 axisIStride;
    s32 axisOStride;
    s32 axisOIStride;
    
    s32 start;
    s32 toProcess;
    
    s32 mode;
    int32_t inputDim;
    int32_t outputDim;
    int32_t indicesDim;
    bool hasIndices;
    bool inputInCmx;
    bool outputInCmx;
};

void mvTopKSingle(t_MvTopKParamNClasses *p)
{
    half* in  = p->input;
    half* out = p->output;
    //half* ind = p->outputInd;
    
    s32* dims = p->inputValueDims;
    s32* istrides = p->in_strides;
    s32* ostrides = p->out_strides;
    //s32* oIstrides = p->ind_strides;
    s32 ndims = p->ndims;
    
    half* p_input0  = (p->inputInCmx) ? in : reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    half* p_output0 = (p->outputInCmx) ? out : reinterpret_cast<half*>(p->cmxslice + 2 * WORK_BUFFER_SIZE);
    //half* p_outputInd0 = (p->hasIndices) ? ind : reinterpret_cast<half*>(p->cmxslice + 4 * WORK_BUFFER_SIZE);
    
    int sets_per_step = (WORK_BUFFER_SIZE) / (sizeof(half) * p->axisDim);
    int32_t setCoords[MAX_ND_DIMS];
    
    void (*findValue)(int32_t* indices, fp16* values, int32_t lineSize, int32_t numLines);
    
    const auto cmxBytes = p->availableCmxBytes;
    const auto cmxSlice = p->cmxslice;
    
    findValue = (p->mode = 0) ? findValueOuterMax : findValueOuterMin;
    {
        sets_per_step = (p->inputInCmx && p->outputInCmx) ? p->toProcess : (2 * WORK_BUFFER_SIZE) / (sizeof(half) * p->axisDim);
        int32_t i = p->start;
        subspace::getCoord(i, dims, ndims, setCoords);
        s32 r_step;
        s32 axisDim = p->axisDim;
        
        //s32 iStride = p->axisIStride / sizeof(fp16);
        //s32 oStride = p->axisOStride / sizeof(fp16);
        //s32 oInStride = p->axisOIStride / sizeof(fp16);
        
        bool hasIndices = p->hasIndices;
        
        const int indexBPP = hasIndices ? sizeof(int32_t) : 0;
        
        const int lineBytes = (p->inputDim * INPUT_BPP) + indexBPP;
        const int cmxLines = vectorSizeOuter * (p->availableCmxBytes / (lineBytes * vectorSizeOuter));
        
        uint8_t* cmx = reinterpret_cast<uint8_t*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
        int32_t* indexBuffer = reinterpret_cast<int32_t*>(cmx); 
        cmx += sizeof(int32_t) * cmxLines;
        fp16* valueBuffer = reinterpret_cast<fp16*>(cmx);
        
        while(i < p->start + p->toProcess)
        {
            r_step = __builtin_shave_cmu_min_i32_rr_int(sets_per_step, dims[0] - setCoords[0]);

            unsigned inOffset, outOffset;// outIndOffset;
            subspace::getOffsetsU8(setCoords, istrides, ostrides, ndims, inOffset, outOffset);

            p_input0 = (half*)((u8*)in + inOffset);
            p_output0 = (half*)((u8*)out + outOffset);
            //p_outputInd0 = (half*)((u8*)ind + outIndOffset);
            
            valueBuffer = p_input0;
            std::fill_n(&indexBuffer[0], r_step, 0);
            findValue(&indexBuffer[0], &valueBuffer[0], p->inputDim, r_step);
            
            p_output0 = valueBuffer;
            //p_outputInd0 = indexBuffer;
            
            i += r_step;
            subspace::incrementNCoord(setCoords, dims, ndims, r_step);
        }
    }
}
};

using namespace subspace;

namespace nn {
namespace shave_lib {
extern "C" void topk(uint32_t lParamsAddr) {
    uint8_t* cmxData = nullptr;
    int32_t availableCmxBytes = 0;
    
    half* p_act_data = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->input.dataAddr);  // 0x1F000000
    half* p_act_out = (half*)(reinterpret_cast<TopKParams*>(lParamsAddr)->value.dataAddr);   // 0x1F004000
    
//    for(int i=0;i<2*2*2;i++)
//    {
//        *p_act_out = *p_act_data;
//        p_act_out++;
//        p_act_data++;
//    }
    
    t_MvTopKParamNClasses topkParamsCMX;
    t_MvTopKParamNClasses* tp = &topkParamsCMX;
    
    const TopKParams* layerParams = reinterpret_cast<const TopKParams*>(lParamsAddr);
    
    tp->axis = layerParams->axis;
    tp->inputInCmx = true;
    tp->outputInCmx = true;
    tp->inputDim = layerParams->input.numDims;
    tp->outputDim = layerParams->value.numDims;
    //tp->indicesDim = layerParams->outputIndices.numDims;
    tp->k = *(int32_t*)(reinterpret_cast<TopKParams*>(lParamsAddr)->k.dataAddr);
    //tp->hasIndices = (layerParams->hasIndices == 1) ? true:false;
    
    int32_t* iPDims = (int32_t*)(layerParams->input.dimsAddr);
    int32_t* oPDims = (int32_t*)(layerParams->value.dimsAddr);
    //int32_t* oPIDims = (int32_t*)(layerParams->outputIndices.dimsAddr);
    
    int32_t* iPStrides = (int32_t*)(layerParams->input.stridesAddr);
    int32_t* oPStrides = (int32_t*)(layerParams->value.stridesAddr);
    //int32_t* oPIStrides = (int32_t*)(layerParams->outputIndices.stridesAddr);

    const int32_t mode = layerParams->mode;
    //const int32_t hasIndices = layerParams->hasIndices;
    
    memcpy_s(tp->inputValueDims, MAX_ND_DIMS * sizeof(int32_t), iPDims, MAX_ND_DIMS * sizeof(int32_t));
    memcpy_s(tp->outputValueDims, MAX_ND_DIMS * sizeof(int32_t), oPDims, MAX_ND_DIMS * sizeof(int32_t));
    //memcpy_s(tp->outputIndicesDims, MAX_ND_DIMS * sizeof(int32_t), oPIDims, MAX_ND_DIMS * sizeof(int32_t));
    
    memcpy_s(tp->in_strides, MAX_ND_DIMS * sizeof(int32_t), iPStrides, MAX_ND_DIMS * sizeof(int32_t));
    memcpy_s(tp->out_strides, MAX_ND_DIMS * sizeof(int32_t), oPStrides, MAX_ND_DIMS * sizeof(int32_t));
    //memcpy_s(tp->ind_strides, MAX_ND_DIMS * sizeof(int32_t), oPIStrides, MAX_ND_DIMS * sizeof(int32_t));
    
    const auto *lp = &topkParamsCMX;
    
    int to_process = getTotal(lp->inputValueDims, lp->ndims);
    unsigned int shaves_no = 1;
    int32_t firstShave = 0;
    int32_t lastShave = firstShave + static_cast<int>(shaves_no) - 1;
    nnLog(MVLOG_DEBUG, "singleShaveSoftmax: run on %d SHAVEs\n", shaves_no);
    {
        nnLog(MVLOG_DEBUG, "softMaxParamNClasses %d\n", __LINE__);
        // one or many softmax sets on one shave
        int step_size = to_process / shaves_no;
        int step_size_rem = to_process % shaves_no;
        
        nnLog(MVLOG_DEBUG, "axis %d, step_size %d, to_process %d, shaves_no %d\n", lp->axis,
              step_size, to_process, shaves_no);

        int i = firstShave;
        int processed = 0;

        for (; i <= lastShave/* && processed < to_process*/; i++) {
            t_MvTopKParamNClasses *topkParamNClasses = &topkParamsCMX;;
            int to_process_on_shave = step_size + ((step_size_rem-- > 0) ? 1 : 0);
            nnLog(MVLOG_DEBUG, "i %d, to_process_on_shave %d lines, started from %d\n", i, to_process_on_shave, processed);

            topkParamNClasses->input = reinterpret_cast<half *>(p_act_data);
            topkParamNClasses->inLocation = sw_params::Location::NN_CMX;//layerParams->input.location;
            topkParamNClasses->inputInCmx = true;//layerParams->input.location;
            topkParamNClasses->output = reinterpret_cast<half *>(p_act_out);
            topkParamNClasses->outLocation = sw_params::Location::NN_CMX;//layerParams->output.location;
            topkParamNClasses->outputInCmx = true;//layerParams->input.location;

            topkParamNClasses->cmxslice = cmxData;
            topkParamNClasses->availableCmxBytes = availableCmxBytes;
            topkParamNClasses->ndims = lp->ndims;
            topkParamNClasses->inputDim = lp->inputDim;
            //topkParamNClasses->hasIndices = lp->hasIndices;
            
            for (int32_t i = 0; i < MAX_ND_DIMS; i++) {
                topkParamNClasses->inputValueDims[i] = lp->inputValueDims[i];
                topkParamNClasses->outputValueDims[i] = lp->outputValueDims[i];
                //topkParamNClasses->outputIndicesDims[i] = lp->outputIndicesDims[i];
                topkParamNClasses->in_strides[i] = lp->in_strides[i];
                topkParamNClasses->out_strides[i] = lp->out_strides[i];
                //topkParamNClasses->ind_strides[i] = lp->ind_strides[i];
            }
            topkParamNClasses->mode = mode;
            topkParamNClasses->axis = lp->axis;
            topkParamNClasses->axisDim = lp->axisDim;
            topkParamNClasses->axisIStride = lp->axisIStride;
            topkParamNClasses->axisOStride = lp->axisOStride;
            topkParamNClasses->start = processed;
            topkParamNClasses->toProcess = to_process_on_shave;

            mvTopKSingle(topkParamNClasses);
            processed += to_process_on_shave;
        }
    }
    }
}
}
