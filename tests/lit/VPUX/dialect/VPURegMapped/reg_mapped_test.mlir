//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s 
// REQUIRES: arch-VPUX37XX

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegister() {
    VPURegMapped.RegisterMappedWrapper regMapped(<!VPURegMapped.RegMapped<name(regMappedForTest) regs([#VPURegMapped.Register<!VPURegMapped.Register<size(32) name(regForTest) address(12) regFields([#VPURegMapped.RegisterField<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(UINT)>>]) allowOverlap(false)>>])>>)
    return
  }

// CHECK: VPURegMapped.RegisterMappedWrapper regMapped(<!VPURegMapped.RegMapped<name(regMappedForTest) regs([#VPURegMapped.Register<!VPURegMapped.Register<size(32) name(regForTest) address(12) regFields([#VPURegMapped.RegisterField<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(UINT)>>]) allowOverlap(false)>>])>>)
