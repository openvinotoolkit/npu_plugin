/*
* {% copyright %}
*/
#include "copy_util.h"

#include <mvMacros.h>

CopyManager::CopyManager(CopyInitializer cp, uint32_t available_bytes):cp(cp)
{
    int dimByteSize = 1;

    /* Initialize top dimension count in order to avoid uninit warning */
    this->top_dim_count = 1;

    /* At most 3 dimensions are supported by DMA */
    uint32_t max_dims = MIN(MAX_DIMS_DMA, this->cp.ndims);

    uint32_t dim;

    /* Iterate tensor dimensions to find maximum elements that can be copied */
    for(dim = 0; dim < max_dims; dim++)
    {
        /* Check if entire dimension can be copied locally (to CMX) */
        if(static_cast<uint32_t>(dimByteSize * this->cp.dims[dim]) > available_bytes)
        {
            /* Calculate how many elements of the highest dim can be copied */
            this->top_dim_count = available_bytes/dimByteSize;

            /* Highest dimension that can be copied found. Exit loop */
            break;
        }
        else
        {
            /* Calculate next dimension size in bytes */
            dimByteSize = dimByteSize * this->cp.dims[dim];
        }
    }

    /* In case loop exits because dim reached max range, set parameters accordingly */
    if(dim == max_dims)
    {
        /* Set top dimension to highest dim */
        dim = max_dims - 1;

        /* Set top dim count to the highest dimension's count, since entire dimension can be copied can be copied at once */
        this->top_dim_count = this->cp.dims[dim];

        /* Reset dimension bytesize with one dimension */
        dimByteSize /= this->cp.dims[dim];
    }

    /* In case 3D DMA transfers, number of planes is limited */
    if(dim == 2)
    {
        this->top_dim_count = MIN(this->top_dim_count, DmaAlShave::max_3D_planes);
    }

    this->top_dim_remainder = this->cp.dims[dim] % this->top_dim_count;

    /* Calculate number of copy iterations needed */
    uint32_t mul = static_cast<uint32_t>(this->cp.dims[dim]) / this->top_dim_count + (this->top_dim_remainder > 0);
    for(uint32_t i = dim + 1; i < this->cp.ndims; i++)
    {
        mul *= static_cast<uint32_t>(this->cp.dims[i]);
    }

    /* Calculate input address increments */
    int sub = 0;

    this->increment[dim] = this->cp.strides[dim] * this->top_dim_count;

    if(dim < this->cp.ndims - 1)
    {
        sub += (this->cp.dims[dim] - this->top_dim_remainder) * this->cp.strides[dim];
        this->increment[dim + 1] = this->cp.strides[dim + 1] - sub;
    }

    for(uint32_t i = dim + 2; i < this->cp.ndims; i++)
    {
        sub += (this->cp.dims[i - 1] - 1) * this->cp.strides[i - 1];
        this->increment[i] = this->cp.strides[i] - sub;
    }

    /* Set for unlikely scenarios */
    this->increment[this->cp.ndims] = 0;

    /* Calculate output address increments */
    sub = 0;

    this->out_increment[dim] = this->cp.out_strides[dim] * this->top_dim_count;

    if(dim < this->cp.ndims - 1)
    {
        sub += (this->cp.dims[dim] - this->top_dim_remainder) * this->cp.out_strides[dim];
        this->out_increment[dim + 1] = this->cp.out_strides[dim + 1] - sub;
    }

    for(uint32_t i = dim + 2; i < this->cp.ndims; i++)
    {
        sub += (this->cp.dims[i - 1] - 1) * this->cp.out_strides[i - 1];
        this->out_increment[i] = this->cp.out_strides[i] - sub;
    }

    /* Set for unlikely scenarios */
    this->out_increment[this->cp.ndims] = 0;

    this->copy_num = mul;

    /* Calculate number of bytes per copy */
    this->byte_per_copy = dimByteSize*this->top_dim_count;

    /* Calculate number of bytes for last copy in top dim */
    this->remainder_byte_per_copy = dimByteSize*this->top_dim_remainder;

    /* Set top dim */
    this->top_dim = dim;

    this->comp_strides[0] = 1;
    for(uint32_t i = 1; i <= top_dim; i++)
    {
        this->comp_strides[i] = this->cp.dims[i - 1] * this->comp_strides[i - 1];
    }
}

bool CopyManager::startCopy(const void * src, void * dst, uint64_t length, int32_t* dims, int32_t* strides_src, int32_t* strides_dst, uint32_t ndims)
{
    bool result = false;

    switch (ndims)
    {
    case 1:
        result = this->dmaTask.start(src, dst, length);
        break;
    case 2:
        result = this->dmaTask.start(src, dst, length, dims[0], dims[0], strides_src[1], strides_dst[1]);
        break;
    case 3:
        result = this->dmaTask.start(src, dst, length, dims[0], dims[0], strides_src[1], strides_dst[1], dims[1], strides_src[2], strides_dst[2]);
        break;
    default:
        break;
    }

    return result;
}

void CopyManager::copyNextIn(void * dst, bool async)
{
    if(in_copy_counters[top_dim] + top_dim_count <= static_cast<uint32_t>(cp.dims[top_dim]))
    {
        byteLength = byte_per_copy;
    }
    else
    {
        byteLength = remainder_byte_per_copy;
    }

    /* Start input copy */
    startCopy(reinterpret_cast<const void*>(cp.in), dst, byteLength, cp.dims, cp.strides, &comp_strides[0], top_dim + 1);
    if(!async)
    {
        dmaTask.wait();
    }
    incrementInCounters();
}


void CopyManager::copyNextOut(const void * src, bool async)
{
    if(out_copy_counters[top_dim] + top_dim_count <= static_cast<uint32_t>(cp.dims[top_dim]))
    {
        byteLength = byte_per_copy;
    }
    else
    {
        byteLength = remainder_byte_per_copy;
    }

    /* Start output copy */
    startCopy(reinterpret_cast<const void*>(src), cp.out, byteLength, cp.dims, &comp_strides[0], cp.out_strides, top_dim + 1);
    if(!async)
    {
        dmaTask.wait();
    }
    incrementOutCounters();
}


void CopyManager::waitLastJob()
{
    this->dmaTask.wait();
}

uint64_t CopyManager::getLastCopyLength()
{
    return byteLength;
}

int32_t CopyManager::getNumberOfCopies()
{
    return copy_num;
}

void CopyManager::incrementInCounters()
{
    uint32_t max = MAX(static_cast<uint32_t>(MAX_ND_DIMS), cp.ndims);

    in_copy_counters[top_dim] += top_dim_count;
    for(uint32_t i = top_dim; i < max; i++)
    {
        if(in_copy_counters[i] >= static_cast<uint32_t>(cp.dims[i]))
        {
            /* Reset counter for current dim and increment the next one */
            in_copy_counters[i] = 0;
            ++in_copy_counters[i + 1];
        }
        else
        {
            /* Increment copy pointer */
            cp.in += increment[i];
            break;
        }

    }
}

void CopyManager::incrementOutCounters()
{
    uint32_t max = MAX(static_cast<uint32_t>(MAX_ND_DIMS), cp.ndims);

    out_copy_counters[top_dim] += top_dim_count;

    for(uint32_t i = top_dim; i < max; i++)
    {
        if(out_copy_counters[i] >= static_cast<uint32_t>(cp.dims[i]))
        {
            /* Reset counter for current dim and increment the next one */
            out_copy_counters[i] = 0;
            ++out_copy_counters[i + 1];
        }
        else
        {
            /* Increment copy pointer */
            cp.out += out_increment[i];
            break;
        }

    }
}
