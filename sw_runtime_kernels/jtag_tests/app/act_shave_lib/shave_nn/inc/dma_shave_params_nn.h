/*
 * {% copyright %}
 */

#pragma once

#include "dma_shave_nn.h"

#include <cstddef>

// Layer parameters copy subroutines
//
// Stricter code safety would distinguish physical vs virtual DDR,
// which would require processing LayerParams (and its extensions)
// in especial manner - as layer parameters would be allocated in
// physical memory, which SHAVE would be able to read only via DMA.
//
// Thus, a layer's preamble would have to allocate and initialize
// a copy (in CMX) of the layer parameters structure. Presumably,
// this may look like the following example, provided MyParamsType
// does not contain non-trivial constructor or destructor:
//
//     #include "dma_shave_params.h"
//
//     void preLayerName(const LayerParams *layerParams, ...)
//     {
//         MyParamsType myParamsCMX;
//         dmaShaveParams(myParamsCMX, layerParams); // DDR to CMX
//         const MyParamsType *params = myParamsCMX;
//         .   .   .
//     }
//
// CAUTION: Please remember about the following limitation!
//   * Parameters class destructor must NOT free/delete
//     (as pointers inside may be copies of original)
//
// If your layer params points to additional data chuncks,
// for example points to inputs/outputs arrays of TensorRef,
// you need to copy them as well like following, e.g.:
//
//         TensorRef inputs[3], outputs[1];
//         dmaShaveParams(inputs,  params->inputs,  3);
//         dmaShaveParams(outputs, params->outputs, 1);
//
// Then, please do not forget to update pointers, e.g.:
//
//         params->inputs  = inputs;
//         params->outputs = outputs;
//
// Or simply use the copies, e.g.: inputs, not params->inputs

inline static
bool dmaShaveParams(void *dst, const void* src, const size_t size, const size_t count = 1)
{
    if (count > 0) {
        // assume src in physical DDR, so use start_pa
        // note: dma destructor waits for started task
        DmaAlShave dma;
        return dma.start_pa(src, dst, size * count);
    } else {
        return true;
    }
}

template<typename T> inline static
bool dmaShaveParams(T* dst, const void* src, const size_t count = 1)
{
    return dmaShaveParams(dst, src, sizeof(T), count);
}

template<typename T> inline static
bool dmaShaveParams(T& dst, const void* src)
{
    return dmaShaveParams<T>(&dst, src);
}
