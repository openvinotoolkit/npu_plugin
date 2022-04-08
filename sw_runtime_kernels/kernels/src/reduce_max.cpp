//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include <mvSubspaces.h>
#include <numeric>

namespace {

#include <reduce_base.hpp>

template <>
class Max<fp16> {
    typedef fp16 DataType;

public:
    void init() { m_val = INT16_MIN; }
    void accumulate(const DataType &val) {
        auto fval = static_cast<float>(val);
        m_val = (m_val > fval ? m_val : fval);
    }
    DataType result() const { return static_cast<fp16>(m_val); }
    int getBpp() const { return sizeof(DataType); }

private:
    float m_val;
};

template <>
class Max<int32_t> {
    typedef int32_t DataType;

public:
    void init() { m_val = INT32_MIN; }
    void accumulate(const DataType &val) {
        auto fval = val;
        m_val = (m_val > fval ? m_val : fval);
    }
    DataType result() const { return m_val; }
    int getBpp() const { return sizeof(DataType); }

private:
    DataType m_val;
};

} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void reduce_max(struct ReduceParams *lParams) {
    auto in_type = lParams->input.dataType;
    switch (in_type)
    {
    case NN_FP16:
        reduce<fp16, Max<fp16>>(lParams, Max<fp16>());
        break;
    case NN_INT32:
        reduce<int32_t, Max<int32_t>>(lParams, Max<int32_t>());
        break;
    default:
        break;
    }
}

}
}  // namespace shave_lib
}  // namespace nn
