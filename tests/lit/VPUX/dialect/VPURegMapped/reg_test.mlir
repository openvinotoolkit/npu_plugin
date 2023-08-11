//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegisterAllowOverlapTrue() {
    VPURegMapped.RegisterWrapper regAttr(<!VPURegMapped.Register<size(32) name(regForTest) address(12) regFields([#VPURegMapped.RegisterField<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(UINT)>>]) allowOverlap(true)>>)
    return
  }

// CHECK: VPURegMapped.RegisterWrapper regAttr(<!VPURegMapped.Register<size(32) name(regForTest) address(12) regFields([#VPURegMapped.RegisterField<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(UINT)>>]) allowOverlap(true)>>)

// -----

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegisterAllowOverlapFalse() {
    VPURegMapped.RegisterWrapper regAttr(<!VPURegMapped.Register<size(32) name(regForTest) address(12) regFields([#VPURegMapped.RegisterField<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(UINT)>>]) allowOverlap(false)>>)
    return
  }

// CHECK: VPURegMapped.RegisterWrapper regAttr(<!VPURegMapped.Register<size(32) name(regForTest) address(12) regFields([#VPURegMapped.RegisterField<!VPURegMapped.RegField<width(8) pos(0) value(255) name(test) dataType(UINT)>>]) allowOverlap(false)>>)
