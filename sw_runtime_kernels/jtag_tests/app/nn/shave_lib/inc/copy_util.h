/*
* {% copyright %}
*/
#pragma once
#include <mv_types.h>
#include <mvMacros.h>
#include <sw_shave_lib_common.h>
#include <sw_layer_params.h>

const uint32_t MAX_DIMS_DMA = 3;

struct CopyInitializer {
    /* Set by constructor */
    const uint8_t* in;
    uint8_t* out;
    uint32_t ndims;
    int32_t dims[MAX_ND_DIMS];
    int32_t strides[MAX_ND_DIMS];
    int32_t out_strides[MAX_ND_DIMS];

    #ifndef __shave__
    void initDims(nn::TensorRef in, nn::TensorRef out, bool merge_dims);
    void initDims(s32* in_dims, s32* in_strides, s32* out_strides, uint32_t ndims, bool merge_dims);
    void print();
    #endif
};

#ifdef __shave__

#include <dma_shave.h>

class CopyManager {
private:
    /* Tensor shape and I/O parameters */
    CopyInitializer cp;

    /* Calculated copy characteristics */
    uint32_t copy_num;
    uint64_t byte_per_copy;
    uint32_t top_dim;
    uint32_t top_dim_count;
    uint32_t top_dim_remainder;
    uint64_t remainder_byte_per_copy;
    int32_t comp_strides[MAX_DIMS_DMA];

    uint32_t increment[MAX_ND_DIMS + 1];
    uint32_t out_increment[MAX_ND_DIMS + 1];

    void init(const uint8_t * in_pointer, uint8_t * out_pointer, uint32_t available_bytes, s32* in_dims, s32* in_strides, s32* out_strides, uint32_t ndims, bool merge_dims);

    /* Copy state handling */
    DmaAlShave dmaTask;
    uint64_t byteLength = 0;
    uint32_t in_copy_counters[MAX_ND_DIMS + 1] = {0};
    uint32_t out_copy_counters[MAX_ND_DIMS + 1] = {0};

    /* Internal methods */
    bool startCopy(const void * src, void * dst, uint64_t length, int32_t* dims, int32_t* strides_src, int32_t* strides_dst, uint32_t ndims);
    void incrementInCounters();
    void incrementOutCounters();
public:
    CopyManager(CopyInitializer cp, uint32_t available_bytes);

    void copyNextIn(void * dst, bool async);
    void copyNextOut(const void * src, bool async);
    void waitLastJob();
    uint64_t getLastCopyLength();
    int32_t getNumberOfCopies();

};

#endif
