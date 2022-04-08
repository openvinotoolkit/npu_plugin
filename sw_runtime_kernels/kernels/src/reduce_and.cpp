//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include <mvSubspaces.h>
#include <numeric>

namespace {

#include <reduce_base.hpp>

template <>
class LogicalAnd<fp16> {
    typedef fp16 DataType;
    const fp16 one = static_cast<fp16>(1.0f);
    const fp16 zero = static_cast<fp16>(0.0f);

public:
    void init() { m_val = true; }
    void accumulate(const DataType &val) { m_val &= bool(static_cast<float>(val) != 0.0); }
    DataType result() const { return (m_val ? one : zero); }
    int getBpp() const { return sizeof(DataType); }

private:
    bool m_val;
};

template <>
class LogicalAnd<int32_t> {
    typedef int32_t DataType;
    const int32_t one = 1;
    const int32_t zero = 0;

public:
    void init() { m_val = true; }
    void accumulate(const DataType &val) { m_val &= bool(val != zero); }
    DataType result() const { return (m_val ? one : zero); }
    int getBpp() const { return sizeof(DataType); }

private:
    bool m_val;
};

} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void reduce_and(struct ReduceParams *lParams) {
    auto in_type = lParams->input.dataType;
    switch (in_type)
    {
    case NN_FP16:
        reduce<fp16, LogicalAnd<fp16>>(lParams, LogicalAnd<fp16>());
        break;
    case NN_INT32:
        reduce<int32_t, LogicalAnd<int32_t>>(lParams, LogicalAnd<int32_t>());
        break;
    default:
        break;
    }
}

}
}  // namespace shave_lib
}  // namespace nn
