//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s 
// REQUIRES: arch-VPUX37XX

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegister() {
    VPURegMapped.RegisterFiledWrapper regFieldAttr(<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(UINT)>>)
    VPURegMapped.RegisterFiledWrapper regFieldAttr(<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(SINT)>>)
    VPURegMapped.RegisterFiledWrapper regFieldAttr(<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(FP)>>)
    return
  }

// CHECK: VPURegMapped.RegisterFiledWrapper regFieldAttr(<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(UINT)>>)
// CHECK: VPURegMapped.RegisterFiledWrapper regFieldAttr(<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(SINT)>>)
// CHECK: VPURegMapped.RegisterFiledWrapper regFieldAttr(<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(FP)>>)
