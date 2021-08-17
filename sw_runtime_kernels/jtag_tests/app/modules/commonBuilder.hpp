// {% copyright %}

#pragma once

#include <memory>
#include <Op.h>
#include "mvTensor.h"

/**
 * @brief axis index in runtime layout
 */
enum class AxisIndex : uint32_t{
    N = 0, C = 1, H = 2, W = 3
};

class CommonFBFuilder {
 public:
    static std::unique_ptr <MVCNN::TensorReferenceT> buildTensorReferenceT(const Buffer &b);
    static MVCNN::DType buildDtype(t_MvTensorDataType type);
    /**
     * @brief build Runtime layout index from axis and given mvTensor order
     */
    static uint32_t buildAxisIndex(int memAxisInd, t_MvTensorStorageOrder order);
};
