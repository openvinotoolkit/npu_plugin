/*
* {% copyright %}
*/
#include "tensor_gf_util.h"
#include <cassert>
#include <nn_cache.h>
#include <nn_log.h>
#include <nn_nce_lib_conversion_fbs.h>
#include <stdint.h>

namespace nn {

namespace {
enum : int { gfN = 0, gfC = 1, gfH = 2, gfW = 3};
}

const char *storageOrderToString(NDOrder order) {
    switch (order) {
    case ND_HWC:
    case ND_NHWC: return "HWC (ZM)";
    case ND_CHW:
    case ND_NCHW: return "CHW (CM)";
    default: return "Unknown";
    }
}

const char *dataTypeToString(DataType dType) {
    switch (dType) {
    case NN_FP16: return "FP16";
    case NN_U8: return "INT8 (Quantized)";
    case NN_I8: return "INT8 (Unquantized)";
    case NN_INT16: return "INT16";
    case NN_INT32: return "INT32";
    default: return "Unknown";
    }
}

void printTensorRef(TensorRef *ref, const char *name) {
    nnLog(MVLOG_INFO, "Tensor: %s\n", name);
    nnLog(MVLOG_INFO, "Dims (NCHW): %d %d %d %d\n", ref->dimN(), ref->dimC(), ref->dimH(), ref->dimW());
    nnLog(MVLOG_INFO, "Strides (NCHW): %d %d %d %d\n", ref->strideN(), ref->strideC(), ref->strideH(), ref->strideW());
    nnLog(MVLOG_INFO, "Storage Order: %s  DataType: %s\n", storageOrderToString(ref->ndOrder),
          dataTypeToString(ref->dType));
}

DataType convertDataTypes(MVCNN::DType type) {
    switch (type) {
    case MVCNN::DType_FP64: return NN_FP64;
    case MVCNN::DType_FP32: return NN_FP32;
    case MVCNN::DType_FP16: return NN_FP16;
    case MVCNN::DType_FP8: return NN_FP8;
    case MVCNN::DType_U64: return NN_U64;
    case MVCNN::DType_U32: return NN_U32;
    case MVCNN::DType_U16: return NN_U16;
    case MVCNN::DType_U8: return NN_U8;
    case MVCNN::DType_I64: return NN_I64;
    case MVCNN::DType_I32: return NN_I32;
    case MVCNN::DType_I16: return NN_I16;
    case MVCNN::DType_I8: return NN_I8;
    case MVCNN::DType_I4: return NN_I4;
    case MVCNN::DType_I2: return NN_I2;
    case MVCNN::DType_BIN: return NN_BIN;
    default : {
        nnLog(MVLOG_FATAL, "Unhandled datatype conversion!");
        return NN_UNDEFINED;
    }
    }
    assert(false && "Unsupported data type");
}

bool compareTRs(const MVCNN::TensorReference *a, const MVCNN::TensorReference *b) {
    (void)a; // bypass -Werror=unused-parameter
    (void)b; // bypass -Werror=unused-parameter

    // FIXME: implement comparitor
    return true;
}

bool parseTensorRef(MVCNN::TensorReference const *tr, TensorRef *ref, NDOrder baseLineOrder) {
    if (tr->dimensions() == nullptr) {
        nnLog(MVLOG_ERROR, "Tensor parsing fails. Missing tensor dimensions");
        return false;
    }

    if (tr->strides() == nullptr) {
        nnLog(MVLOG_ERROR, "Tensor parsing fails. Missing tensor strides");
        return false;
    }

    if (tr->dimensions()->size() > MAX_ND_DIMS) {
        nnLog(MVLOG_ERROR, "Tensor parsing fails. Number of dimensions exceeds maximal supported value");
        return false;
    }

    if (tr->strides()->size() != tr->dimensions()->size() + 1) {
        nnLog(MVLOG_ERROR, "Tensor parsing fails. Unexpected number of strides");
        return false;
    }

    uint32_t strides[MAX_ND_DIMS + 1] = {0};
    uint64_t stridesBits[MAX_ND_DIMS + 1] = {0};

    for (size_t i = 0; i < tr->strides()->size(); ++i) {
        strides[i] = tr->strides()->Get(i);
        stridesBits[i] = tr->strides()->Get(i) * CHAR_BIT;
    }

    if (tr->order() != 0) {
        baseLineOrder = tr->order();
    }

    if (!ref->setByStrides(
            convertDataTypes(tr->data_dtype()),
            tr->dimensions()->data(),
            &strides[0],
            &stridesBits[0],
            tr->dimensions()->size(),
            baseLineOrder)) {
        return false;
    }

    if (tr->order() != 0) {
        if (ref->ndOrder != tr->order()) {
            nnLog(MVLOG_ERROR, "Tensor parsing fails. order and shape(strides) mismatch");
            return false;
        }
    }

    nce_lib::transform(*tr, ref->dataAddr);

    return true;
}

} // namespace nn
