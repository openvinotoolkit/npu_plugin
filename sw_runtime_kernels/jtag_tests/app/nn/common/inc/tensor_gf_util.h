/*
* {% copyright %}
*/
#pragma once

#include "sw_tensor_ref.h"
#include <graphfile_generated.h>

namespace nn {

/**
 * Tensor utility functions used to convert a graphfile to shave_lib's
 * internal tensor representation class TensorRef.
 * Sources intended for Inference Runtime may not depend on this file.
 */

void printTensorRef(const TensorRef *ref, const char *name = "");

DataType convertDataTypes(MVCNN::DType type);

bool compareTRs(const MVCNN::TensorReference *a, const MVCNN::TensorReference *b);

bool parseTensorRef(MVCNN::TensorReference const *tr, TensorRef *ref, NDOrder baseLineOrd = FULL_ND_NHWC);

} // namespace nn
