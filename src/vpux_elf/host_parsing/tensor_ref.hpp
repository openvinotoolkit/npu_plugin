/*
* {% copyright %}
*/

#pragma once

constexpr uint64_t MAX_ND_DIMS = 15;

struct TensorRef {
    uint64_t dims[MAX_ND_DIMS];
    uint32_t ndims;
    uint32_t elemSize;
};
