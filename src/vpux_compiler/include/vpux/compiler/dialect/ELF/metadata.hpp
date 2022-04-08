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

#include <vpux_headers/metadata.hpp>

namespace vpux {
namespace ELF {

const EnumMap<ov::element::Type_t, elf::OVNodeType> mapElementType = {
        {ov::element::Type_t::undefined, elf::OVNodeType::OVNodeType_UNDEFINED},
        {ov::element::Type_t::dynamic, elf::OVNodeType::OVNodeType_DYNAMIC},
        {ov::element::Type_t::boolean, elf::OVNodeType::OVNodeType_BOOLEAN},
        {ov::element::Type_t::bf16, elf::OVNodeType::OVNodeType_BF16},
        {ov::element::Type_t::f16, elf::OVNodeType::OVNodeType_F16},
        {ov::element::Type_t::f32, elf::OVNodeType::OVNodeType_F32},
        {ov::element::Type_t::f64, elf::OVNodeType::OVNodeType_F64},
        {ov::element::Type_t::i4, elf::OVNodeType::OVNodeType_I4},
        {ov::element::Type_t::i8, elf::OVNodeType::OVNodeType_I8},
        {ov::element::Type_t::i16, elf::OVNodeType::OVNodeType_I16},
        {ov::element::Type_t::i32, elf::OVNodeType::OVNodeType_I32},
        {ov::element::Type_t::i64, elf::OVNodeType::OVNodeType_I64},
        {ov::element::Type_t::u1, elf::OVNodeType::OVNodeType_U1},
        {ov::element::Type_t::u4, elf::OVNodeType::OVNodeType_U4},
        {ov::element::Type_t::u8, elf::OVNodeType::OVNodeType_U8},
        {ov::element::Type_t::u16, elf::OVNodeType::OVNodeType_U16},
        {ov::element::Type_t::u32, elf::OVNodeType::OVNodeType_U32},
        {ov::element::Type_t::u64, elf::OVNodeType::OVNodeType_U64},
};

const EnumMap<vpux::PreProcessColorSpace, elf::PreProcessColorSpace> mapPreProcessColorFormat = {
        {vpux::PreProcessColorSpace::BGR, elf::PreProcessColorSpace::PreProcessColorSpace_BGR},
        {vpux::PreProcessColorSpace::RGB, elf::PreProcessColorSpace::PreProcessColorSpace_RGB},
        {vpux::PreProcessColorSpace::NV12, elf::PreProcessColorSpace::PreProcessColorSpace_NV12},
        {vpux::PreProcessColorSpace::I420, elf::PreProcessColorSpace::PreProcessColorSpace_I420},
        {vpux::PreProcessColorSpace::NONE, elf::PreProcessColorSpace::PreProcessColorSpace_DEFAULT},
};

const EnumMap<vpux::PreProcessResizeAlgorithm, elf::PreProcessResizeAlgorithm> mapPreProcessResizeAlgorithm = {
        {vpux::PreProcessResizeAlgorithm::RESIZE_BILINEAR,
         elf::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_BILINEAR},
        {vpux::PreProcessResizeAlgorithm::RESIZE_AREA,
         elf::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_RESIZE_AREA},
        {vpux::PreProcessResizeAlgorithm::NO_RESIZE,
         elf::PreProcessResizeAlgorithm::PreProcessResizeAlgorithm_NO_RESIZE},
};

elf::NetworkMetadata constructMetadata(mlir::ModuleOp module, IE::CNNNetworkOp netOp, mlir::FuncOp netFunc,
                                       const std::vector<vpux::PreProcessInfo>& preprocessInfo,
                                       const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                       const std::vector<std::shared_ptr<const ov::Node>>& results);

elf::TensorRef createTensorRef(mlir::Value val, StringRef name);
elf::TensorRef createTensorRef(vpux::NDTypeInterface type, StringRef name);

elf::DType createDType(mlir::Type type);

}  // namespace ELF
}  // namespace vpux
