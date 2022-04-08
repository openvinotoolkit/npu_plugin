//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include <mvSubspaces.h>
#include <numeric>

namespace {

#include <reduce_base.hpp>

template <>
class Mean<fp16> {
    typedef fp16 DataType;

public:
    void init() {
        m_val = 0;
        m_count = 0;
    }
    void accumulate(const DataType &val) {
        m_val += (float)val;
        m_count++;
    }
    DataType result() const {
        if (m_count == 0)
            return m_val > static_cast<float>(F16_MAX) ? static_cast<fp16>(F16_MAX) : static_cast<fp16>(m_val);
        else{
            float result = m_val / m_count;
            return result > static_cast<float>(F16_MAX) ? static_cast<fp16>(F16_MAX) : static_cast<fp16>(result);
        }
    }
    int getBpp() const { return sizeof(DataType); }

private:
    float m_val;
    int m_count;
};

template <>
class Mean<int32_t> {
    typedef int32_t DataType;

public:
    void init() {
        m_val = 0;
        m_count = 0;
    }
    void accumulate(const DataType &val) {
        m_val += val;
        m_count++;
    }
    DataType result() const {
        if (m_count == 0)
            return m_val > INT32_MAX ? static_cast<DataType>(INT32_MAX) : static_cast<DataType>(m_val);
        else{
            int64_t result = m_val / m_count;
            return result > INT32_MAX ? static_cast<DataType>(INT32_MAX) : static_cast<DataType>(result);
        }
    }
    int getBpp() const { return sizeof(DataType); }

private:
    int64_t m_val;
    int m_count;
};
} // namespace

namespace nn {
namespace shave_lib {

extern "C" {

void reduce_mean(struct ReduceParams *lParams) {
    auto in_type = lParams->input.dataType;
    switch (in_type)
    {
    case NN_FP16:
        reduce<fp16, Mean<fp16>>(lParams, Mean<fp16>());
        break;
    case NN_INT32:
        reduce<int32_t, Mean<int32_t>>(lParams, Mean<int32_t>());
        break;
    default:
        break;
    }
}

}
}  // namespace shave_lib
}  // namespace nn
