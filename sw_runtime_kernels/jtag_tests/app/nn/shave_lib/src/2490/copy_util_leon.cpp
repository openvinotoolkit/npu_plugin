/*
* {% copyright %}
*/
#include "copy_util.h"

#include <nn_log.h>
#include <sw_layer.h>

void CopyInitializer::print()
{
    nnLog(MVLOG_DEBUG, "IN: %p\nOUT: %p\nndims: %u\n", this->in, this->out, this->ndims);

    nnLog(MVLOG_DEBUG, "dims: [%d", this->dims[0]);
    for(int i = 1; i < MAX_ND_DIMS; i++)
    {
        nnLog(MVLOG_DEBUG, ", %d", this->dims[i]);
    }
    nnLog(MVLOG_DEBUG, "]\n");

    nnLog(MVLOG_DEBUG, "Strides: [%d", this->strides[0]);
    for(uint32_t i = 1; i < MAX_DIMS_DMA; i++)
    {
        nnLog(MVLOG_DEBUG, ", %d", this->strides[i]);
    }
    nnLog(MVLOG_DEBUG, "]\n");

    nnLog(MVLOG_DEBUG, "Out Strides: [%d", this->out_strides[0]);
    for(uint32_t i = 1; i < MAX_DIMS_DMA; i++)
    {
        nnLog(MVLOG_DEBUG, ", %d", this->out_strides[i]);
    }
    nnLog(MVLOG_DEBUG, "]\n");
}

void CopyInitializer::initDims(nn::TensorRef in, nn::TensorRef out, bool merge_dims)
{
    this->initDims(in.dims, in.strides, out.strides, in.ndims, merge_dims);
}

void CopyInitializer::initDims(s32* in_dims, s32* in_strides, s32* out_strides, uint32_t ndims, bool merge_dims)
{
    if(merge_dims)
    {
        this->dims[0] = in_dims[0];
        this->strides[0] = in_strides[0]/sizeof(half);
        this->out_strides[0] = out_strides[0]/sizeof(half);
        uint32_t cnt = 0;
        for(uint32_t i = 1; i < ndims; i++)
        {
            if((in_strides[i - 1] * in_dims[i - 1] == in_strides[i]) && (out_strides[i - 1] * in_dims[i - 1] == out_strides[i]))
            {
                this->dims[cnt] *= in_dims[i];
            }
            else
            {
                cnt++;
                this->dims[cnt] = in_dims[i];
                this->strides[cnt] = in_strides[i];
                this->out_strides[cnt] = out_strides[i];
            }

        }
        cnt++;

        this->dims[0] *= sizeof(half);
        this->ndims = cnt;
    }
    else
    {
        this->ndims = ndims;
        for(uint32_t i = 0; i < ndims; i++)
        {
            this->dims[i] = in_dims[i];
            this->strides[i] = in_strides[i];
            this->out_strides[i] = out_strides[i];
        }
        this->dims[0] *= sizeof(half);
    }
}
