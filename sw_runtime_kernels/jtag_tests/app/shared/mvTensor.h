//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#ifndef _MV_TENSOR_H_
#define _MV_TENSOR_H_

#include <mv_types.h>
#include <mvTensorConfig.h>
#include "nn_tensor_ref.h"
#include <mvSubspaces.h>
#include <map>
#include <mvSubspaces8d.h>
#include "common_types.h"

/// version of mvtensor library
#define MVTENSOR_VERSION_MAJOR  5
#define MVTENSOR_VERSION_MINOR  0

using namespace subspace;

// the following 'flatbuffers' and 'MVCNN' definitions are copied 
// just to use codes of (sub)operations in testing system as previously

namespace flatbuffers {

// Check 'v' is out of closed range [low; high].
// Workaround for GCC warning [-Werror=type-limits]:
// comparison is always true due to limited range of data type.
template<typename T>
inline bool IsOutRange(const T &v, const T &low, const T &high) {
  return (v < low) || (high < v);
}

}  // namespace flatbuffers

namespace MVCNN {

enum DataType {
  DataType_UNKNOWN = 0,
  DataType_INT1 = 1,
  DataType_INT8 = 2,
  DataType_INT16 = 3,
  DataType_INT32 = 4,
  DataType_INT64 = 5,
  DataType_UINT8 = 6,
  DataType_UINT16 = 7,
  DataType_UINT32 = 8,
  DataType_UINT64 = 9,
  DataType_FLOAT16 = 10,
  DataType_FLOAT32 = 11,
  DataType_FLOAT64 = 12,
  DataType_MIN = DataType_UNKNOWN,
  DataType_MAX = DataType_FLOAT64
};

inline const DataType (&EnumValuesDataType())[13] {
  static const DataType values[] = {
    DataType_UNKNOWN,
    DataType_INT1,
    DataType_INT8,
    DataType_INT16,
    DataType_INT32,
    DataType_INT64,
    DataType_UINT8,
    DataType_UINT16,
    DataType_UINT32,
    DataType_UINT64,
    DataType_FLOAT16,
    DataType_FLOAT32,
    DataType_FLOAT64
  };
  return values;
}

inline const char * const *EnumNamesDataType() {
  static const char * const names[14] = {
    "UNKNOWN",
    "INT1",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "FLOAT16",
    "FLOAT32",
    "FLOAT64",
    nullptr
  };
  return names;
}

inline const char *EnumNameDataType(DataType e) {
  if (flatbuffers::IsOutRange(e, DataType_UNKNOWN, DataType_FLOAT64)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesDataType()[index];
}

enum SoftwareLayerParams {
  SoftwareLayerParams_NONE = 0,
  SoftwareLayerParams_CustomLayerCppParams = 1,
  SoftwareLayerParams_MIN = SoftwareLayerParams_NONE,
  SoftwareLayerParams_MAX = SoftwareLayerParams_CustomLayerCppParams
};

inline const SoftwareLayerParams (&EnumValuesSoftwareLayerParams())[2] {
  static const SoftwareLayerParams values[] = {
    SoftwareLayerParams_NONE,
    SoftwareLayerParams_CustomLayerCppParams,
  };
  return values;
}

inline const char * const *EnumNamesSoftwareLayerParams() {
  static const char * const names[3] = {
    "NONE",
    "CustomLayerCppParams",
    nullptr
  };
  return names;
}

inline const char *EnumNameSoftwareLayerParams(SoftwareLayerParams e) {
  if (flatbuffers::IsOutRange(e, SoftwareLayerParams_NONE, SoftwareLayerParams_CustomLayerCppParams)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesSoftwareLayerParams()[index];
}

}  // namespace MVCNN

typedef enum {
    t_fp16 = sw_params::NN_FP16, ///< Half precision floating point
    t_u8f = sw_params::NN_U8,    ///< Unsigned byte
    t_int = sw_params::NN_I32, ///< Signed integer (4 byte)
    t_i64 = sw_params::NN_I64,   ///< Signed integer (8 byte)
    t_uint = sw_params::NN_U32,  ///< Unsigned integer (4 byte)
    t_fp32 = sw_params::NN_FP32, ///< Single precision floating point
    t_i8 = sw_params::NN_I8,     ///< Signed byte
    t_i16 = sw_params::NN_INT16, ///< Signed integer (2 byte)
} t_MvTensorDataType;

typedef t_D8StorageOrder t_MvTensorStorageOrder;

/// MvTensor data storage order options

/// Filtering types

typedef enum : int32_t
{
    kEmpty = -1,
    kNone0 = MVCNN::SoftwareLayerParams_NONE,
    kCustomCpp = MVCNN::SoftwareLayerParams_CustomLayerCppParams,
}t_MvTensorOpType;

/// TODO[AP]: Remove!!!

struct t_MvTensorMyriadResources {
    int dataPartitionNo;          ///< Number of shave L2 cache data partition use by MvTensor
    int instrPartitionNo;         ///< Number of shave L2 cache instruction partition use by MvTensor
    int bypassPartitionNo;        ///< Number of shave L2 cache bypass partition use by MvTensor
    // MvTensorDmaDescriptor* dmaTransactions;
    int32_t firstShave;
    int32_t lastShave;
    unsigned int shaveNum;
    uint32_t dmaLinkAgent = 0;
};

struct t_MvTensorDebugInfo
{
    double ms;                               ///< Duration of the mvTensor call (in ms)
    char * debugMsg;   ///< Debug messages of size MV_TENSOR_DBG_MSG_SIZE
};

const char* getOpName(t_MvTensorOpType op);

#endif // _MV_TENSOR_H_
