//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @FoldMemPermute
func @FoldMemPermute(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x16x2x3xf32> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x16x2x3xf32> -> tensor<1x16x2x3xf32>
    return %0 : tensor<1x16x2x3xf32>

    // CHECK-NOT: IE.MemPermute
    // CHECK:     return %arg0 : tensor<1x16x2x3xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @FuseMemPermutes
func @FuseMemPermutes(%arg0: tensor<1x16x2x3xf32>, %arg1: tensor<1x16x2x3xf32, {order = #NHWC}>) ->
        (tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NCWH} :
        tensor<1x3x16x2xf32> -> tensor<1x3x2x16xf32>

    %2 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NWCH} :
        tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x3x16x2xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32>

    return %1, %3 : tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>

    // CHECK-NOT: IE.MemPermute
    // CHECK-NOT: IE.MemPermute
    // CHECK:     %[[VAL_0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x16x2x3xf32> -> tensor<1x3x2x16xf32>
    // CHECK:     %[[VAL_1:.*]] = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32>
    // CHECK:     return %[[VAL_0]], %[[VAL_1]] : tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @ConvertToPermuteCast
func @ConvertToPermuteCast(
        %arg0: tensor<1x100x1x1xf32>,
        %arg1: tensor<1x100x1x1xf32, {order = #NHWC}>,
        %arg2: tensor<1x1x256x32xf32, {order = #NHWC}>,
        %arg3: tensor<1x512x2x1xf32, {order = #NHWC}>) ->
            (tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>, tensor<1x1x256x32xf32>, tensor<1x2x1x512xf32>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x100x1x1xf32> -> tensor<1x1x100x1xf32>

    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x100x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x100xf32>

    %2 = IE.MemPermute(%arg2) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x1x256x32xf32, {order = #NHWC}> -> tensor<1x1x256x32xf32>

    %3 = IE.MemPermute(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x512x2x1xf32, {order = #NHWC}> -> tensor<1x2x1x512xf32>

    return %0, %1, %2, %3 : tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>, tensor<1x1x256x32xf32>, tensor<1x2x1x512xf32>

    //CHECK: [[VAR0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK: [[VAR1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW}
    //CHECK: [[VAR2:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK: [[VAR3:%.+]] = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW}
    //CHECK: return [[VAR0]], [[VAR1]], [[VAR2]], [[VAR3]] : tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>, tensor<1x1x256x32xf32>, tensor<1x2x1x512xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @FusePermCastAndMemPerm
func @FusePermCastAndMemPerm(%arg0: tensor<1x1000x1x1xf32, {order = #NHWC}>) ->
            tensor<1x1x1000x1xf32> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NWCH} :
            tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x1x1000x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>

    return %1 : tensor<1x1x1000x1xf32>

    // CHECK:     %[[VAL_0:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>
    // CHECK:     return %[[VAL_0]] : tensor<1x1x1000x1xf32>
}
