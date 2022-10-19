//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux_compiler.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <transformations/utils/utils.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace ELF {

#define MAX_TENSOR_REF_DIMS 8
#define MAX_TENSOR_REF_STRIDES MAX_TENSOR_REF_DIMS + 1
#define MAX_METADATA_IO 5
#define MAX_OV_NODES 10
#define MAX_STRING_LEN 32

using TensorName = char[MAX_STRING_LEN];

enum class DType {
    DType_NOT_SET = 0,
    DType_FP64 = 1,
    DType_FP32 = 2,
    DType_FP16 = 3,
    DType_FP8 = 4,
    DType_U64 = 5,
    DType_U32 = 6,
    DType_U16 = 7,
    DType_U8 = 8,
    DType_I64 = 9,
    DType_I32 = 10,
    DType_I16 = 11,
    DType_I8 = 12,
    DType_I4 = 13,
    DType_I2 = 14,
    DType_I4X = 15,
    DType_BIN = 16,
    DType_LOG = 17,
    DType_I2X = 18,
    DType_BFP16 = 19,
    DType_U4 = 20,
    DType_MIN = DType_NOT_SET,
    DType_MAX = DType_U4
};

enum PreProcessColorSpace {
    PreProcessColorSpace_DEFAULT = 0,
    PreProcessColorSpace_BGR = 1,
    PreProcessColorSpace_RGB = 2,
    PreProcessColorSpace_NV12 = 3,
    PreProcessColorSpace_I420 = 4,
    PreProcessColorSpace_MIN = PreProcessColorSpace_DEFAULT,
    PreProcessColorSpace_MAX = PreProcessColorSpace_I420
};

enum PreProcessResizeAlgorithm {
    PreProcessResizeAlgorithm_NO_RESIZE = 0,
    PreProcessResizeAlgorithm_RESIZE_BILINEAR = 1,
    PreProcessResizeAlgorithm_RESIZE_AREA = 2,
    PreProcessResizeAlgorithm_MIN = PreProcessResizeAlgorithm_NO_RESIZE,
    PreProcessResizeAlgorithm_MAX = PreProcessResizeAlgorithm_RESIZE_AREA
};

enum OVNodeType {
    OVNodeType_UNDEFINED = 0,
    OVNodeType_DYNAMIC = 1,
    OVNodeType_BOOLEAN = 2,
    OVNodeType_BF16 = 3,
    OVNodeType_F16 = 4,
    OVNodeType_F32 = 5,
    OVNodeType_F64 = 6,
    OVNodeType_I4 = 7,
    OVNodeType_I8 = 8,
    OVNodeType_I16 = 9,
    OVNodeType_I32 = 10,
    OVNodeType_I64 = 11,
    OVNodeType_U1 = 12,
    OVNodeType_U4 = 13,
    OVNodeType_U8 = 14,
    OVNodeType_U16 = 15,
    OVNodeType_U32 = 16,
    OVNodeType_U64 = 17,
    OVNodeType_MIN = OVNodeType_UNDEFINED,
    OVNodeType_MAX = OVNodeType_U64
};

const EnumMap<ov::element::Type_t, ELF::OVNodeType> mapElementType = {
        {ov::element::Type_t::undefined, ELF::OVNodeType::OVNodeType_UNDEFINED},
        {ov::element::Type_t::dynamic, ELF::OVNodeType::OVNodeType_DYNAMIC},
        {ov::element::Type_t::boolean, ELF::OVNodeType::OVNodeType_BOOLEAN},
        {ov::element::Type_t::bf16, ELF::OVNodeType::OVNodeType_BF16},
        {ov::element::Type_t::f16, ELF::OVNodeType::OVNodeType_F16},
        {ov::element::Type_t::f32, ELF::OVNodeType::OVNodeType_F32},
        {ov::element::Type_t::f64, ELF::OVNodeType::OVNodeType_F64},
        {ov::element::Type_t::i4, ELF::OVNodeType::OVNodeType_I4},
        {ov::element::Type_t::i8, ELF::OVNodeType::OVNodeType_I8},
        {ov::element::Type_t::i16, ELF::OVNodeType::OVNodeType_I16},
        {ov::element::Type_t::i32, ELF::OVNodeType::OVNodeType_I32},
        {ov::element::Type_t::i64, ELF::OVNodeType::OVNodeType_I64},
        {ov::element::Type_t::u1, ELF::OVNodeType::OVNodeType_U1},
        {ov::element::Type_t::u4, ELF::OVNodeType::OVNodeType_U4},
        {ov::element::Type_t::u8, ELF::OVNodeType::OVNodeType_U8},
        {ov::element::Type_t::u16, ELF::OVNodeType::OVNodeType_U16},
        {ov::element::Type_t::u32, ELF::OVNodeType::OVNodeType_U32},
        {ov::element::Type_t::u64, ELF::OVNodeType::OVNodeType_U64},
};

const EnumMap<vpux::PreProcessColorSpace, ELF::PreProcessColorSpace> mapPreProcessColorFormat = {
        {vpux::PreProcessColorSpace::BGR, ELF::PreProcessColorSpace::PreProcessColorSpace_BGR},
        {vpux::PreProcessColorSpace::RGB, ELF::PreProcessColorSpace::PreProcessColorSpace_RGB},
        {vpux::PreProcessColorSpace::NV12, ELF::PreProcessColorSpace::PreProcessColorSpace_NV12},
        {vpux::PreProcessColorSpace::I420, ELF::PreProcessColorSpace::PreProcessColorSpace_I420},
        {vpux::PreProcessColorSpace::NONE, ELF::PreProcessColorSpace::PreProcessColorSpace_DEFAULT},
};

const EnumMap<vpux::PreProcessResizeAlgorithm, ELF::PreProcessResizeAlgorithm> mapPreProcessResizeAlgorithm = {
        {vpux::PreProcessResizeAlgorithm::RESIZE_BILINEAR,
         ELF::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_BILINEAR},
        {vpux::PreProcessResizeAlgorithm::RESIZE_AREA,
         ELF::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_AREA},
        {vpux::PreProcessResizeAlgorithm::NO_RESIZE,
         ELF::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_NO_RESIZE},
};

struct TensorRef {
    float strides[MAX_TENSOR_REF_STRIDES];
    uint32_t dimensions[MAX_TENSOR_REF_DIMS];
    TensorName name;
    uint64_t order;
    DType data_type;
    uint32_t dimensions_size, strides_size;
};

struct PreprocessingInfo {
    TensorName input_name;
    PreProcessColorSpace input_format;
    PreProcessColorSpace output_format;
    PreProcessResizeAlgorithm algorithm;
};

struct OVNode {
    TensorName tensor_names[MAX_METADATA_IO];
    uint64_t shape[MAX_TENSOR_REF_DIMS];
    TensorName friendly_name;
    TensorName input_name;
    OVNodeType type;
    uint32_t shape_size;
    uint32_t tensor_names_count = 0;
};

struct NetworkMetadata {
    TensorRef net_input[MAX_METADATA_IO];
    TensorRef net_output[MAX_METADATA_IO];

    TensorRef in_tensor_desc[MAX_METADATA_IO];
    TensorRef out_tensor_desc[MAX_METADATA_IO];

    TensorRef profiling_output[MAX_METADATA_IO];

    OVNode ov_parameters[MAX_OV_NODES];
    OVNode ov_results[MAX_OV_NODES];

    PreprocessingInfo pre_process_info[MAX_METADATA_IO];
    TensorName blob_name;

    uint32_t net_input_count = 0, net_output_count = 0;
    uint32_t in_tenosr_count = 0, out_tensor_count = 0;
    uint32_t profiling_output_count = 0;
    uint32_t ov_parameters_count = 0, ov_results_count = 0;
    uint32_t pre_process_info_count = 0;
};

NetworkMetadata constructMetadata(mlir::ModuleOp module, IE::CNNNetworkOp netOp, mlir::FuncOp netFunc,
                                  const std::vector<vpux::PreProcessInfo>& preprocessInfo,
                                  const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                  const std::vector<std::shared_ptr<const ov::Node>>& results);

TensorRef createTensorRef(mlir::Value val, StringRef name);
TensorRef createTensorRef(vpux::NDTypeInterface type, StringRef name);

ELF::DType createDType(mlir::Type type);

}  // namespace ELF
}  // namespace vpux
