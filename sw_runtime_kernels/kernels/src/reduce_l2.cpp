//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include <mvSubspaces.h>
#include <numeric>

namespace {

#include <reduce_base.hpp>

template <>
class L2<fp16> {
    typedef fp16 DataType;

public:
    void init() { m_val = 0.0f; }
    void accumulate(const DataType &val) { m_val += std::pow((float)val, 2); }
    DataType result() const {
        return m_val > static_cast<float>(F16_MAX) ? static_cast<fp16>(F16_MAX) : static_cast<fp16>(std::sqrt(m_val)); }
    int getBpp() const { return sizeof(DataType); }

private:
    float m_val;
};

template <>
class L2<int32_t> {
    typedef int32_t DataType;

public:
    void init() { m_val = 0; }
    void accumulate(const DataType &val) { m_val += std::pow(static_cast<int64_t>(val), 2); }
    DataType result() const {
        return m_val > INT32_MAX ? static_cast<DataType>(INT32_MAX) : static_cast<DataType>(std::sqrt(m_val)); }
    int getBpp() const { return sizeof(DataType); }

private:
    int64_t m_val;
};

} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void reduce_l2(struct ReduceParams *lParams) {
    auto in_type = lParams->input.dataType;
    switch (in_type)
    {
    case NN_FP16:
        reduce<fp16, L2<fp16>>(lParams, L2<fp16>());
        break;
    case NN_INT32:
        reduce<int32_t, L2<int32_t>>(lParams, L2<int32_t>());
        break;
    default:
        break;
    }
}

}
}  // namespace shave_lib
}  // namespace nn
