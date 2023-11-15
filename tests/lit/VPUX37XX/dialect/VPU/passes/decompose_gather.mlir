//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --decompose-gather %s | FileCheck %s

// CHECK-LABEL: func.func @DecomposeGather
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1548288x1xf16>
func.func @DecomposeGather(%arg0: tensor<1548288x1xf16>) -> (tensor<1x100x1xf16>) {
      %cst = const.Declare tensor<1x100xsi32> = dense<1> : tensor<1x100xsi64>, [#const.ConvertElemType<si32>]
      %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<1548288x1xf16>, tensor<1x100xsi32> -> tensor<1x100x1xf16>
      return %0 : tensor<1x100x1xf16>

      // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x100xsi32> = dense<1> : tensor<1x100xsi64>, [#const.ConvertElemType<si32>]

      // GatherSlice 0

      // CHECK:     [[CST_TILE_0:%.+]] = VPU.Slice [[CST]] [0, 0] [1, 50] : tensor<1x100xsi32> to tensor<1x50xsi32>
      // CHECK:     [[INPUT_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0] [774144, 1] : tensor<1548288x1xf16> to tensor<774144x1xf16>
      // CHECK:     [[OUTPUT_0:%.+]], [[FLAG_0:%.+]] = VPU.GatherSlice([[INPUT_TILE_0]], [[CST_TILE_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, max_axis_dimension = 1548288 : i64, slice_head = 0 : i64, slice_tail = 774144 : i64} : tensor<774144x1xf16>, tensor<1x50xsi32> -> tensor<1x50x1xf16>, tensor<1x50xi8>
      // CHECK:     [[UNSQUEEZE_OUTPUT_0:%.+]] = VPU.Unsqueeze([[OUTPUT_0]]) {axes_value = [0]} : tensor<1x50x1xf16> -> tensor<1x1x50x1xf16>
      // CHECK:     [[UNSQUEEZE_FLAG_0:%.+]] = VPU.Unsqueeze([[FLAG_0]]) {axes_value = [0]} : tensor<1x50xi8> -> tensor<1x1x50xi8>

      // GatherSlice 1

      // CHECK:     [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [774144, 0] [774144, 1] : tensor<1548288x1xf16> to tensor<774144x1xf16>
      // CHECK:     [[OUTPUT_1:%.+]], [[FLAG_1:%.+]] = VPU.GatherSlice([[INPUT_TILE_1]], [[CST_TILE_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64, max_axis_dimension = 1548288 : i64, slice_head = 774144 : i64, slice_tail = 1548288 : i64} : tensor<774144x1xf16>, tensor<1x50xsi32> -> tensor<1x50x1xf16>, tensor<1x50xi8>
      // CHECK:     [[UNSQUEEZE_OUTPUT_1:%.+]] = VPU.Unsqueeze([[OUTPUT_1]]) {axes_value = [0]} : tensor<1x50x1xf16> -> tensor<1x1x50x1xf16>
      // CHECK:     [[UNSQUEEZE_FLAG_1:%.+]] = VPU.Unsqueeze([[FLAG_1]]) {axes_value = [0]} : tensor<1x50xi8> -> tensor<1x1x50xi8>

      // ExtractValue 0

      // CHECK:     [[CONCAT_OUTPUT_0:%.+]] = VPU.Concat([[UNSQUEEZE_OUTPUT_0]], [[UNSQUEEZE_OUTPUT_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x50x1xf16>, tensor<1x1x50x1xf16> -> tensor<2x1x50x1xf16>
      // CHECK:     [[CONCAT_FLAG_0:%.+]] = VPU.Concat([[UNSQUEEZE_FLAG_0]], [[UNSQUEEZE_FLAG_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x50xi8>, tensor<1x1x50xi8> -> tensor<2x1x50xi8>
      // CHECK:     [[EXTRACT_OUTPUT_0:%.+]] = VPU.ExtractValue([[CONCAT_OUTPUT_0]], [[CONCAT_FLAG_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<2x1x50x1xf16>, tensor<2x1x50xi8> -> tensor<1x50x1xf16>

      // GatherSlice 2

      // CHECK:     [[CST_TILE_1:%.+]] = VPU.Slice [[CST]] [0, 50] [1, 50] : tensor<1x100xsi32> to tensor<1x50xsi32>
      // CHECK:     [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0] [774144, 1] : tensor<1548288x1xf16> to tensor<774144x1xf16>
      // CHECK:     [[OUTPUT_2:%.+]], [[FLAG_2:%.+]]  = VPU.GatherSlice([[INPUT_TILE_2]], [[CST_TILE_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, max_axis_dimension = 1548288 : i64, slice_head = 0 : i64, slice_tail = 774144 : i64} : tensor<774144x1xf16>, tensor<1x50xsi32> -> tensor<1x50x1xf16>, tensor<1x50xi8>
      // CHECK:     [[UNSQUEEZE_OUTPUT_2:%.+]] = VPU.Unsqueeze([[OUTPUT_2]]) {axes_value = [0]} : tensor<1x50x1xf16> -> tensor<1x1x50x1xf16>
      // CHECK:     [[UNSQUEEZE_FLAG_2:%.+]] = VPU.Unsqueeze([[FLAG_2]]) {axes_value = [0]} : tensor<1x50xi8> -> tensor<1x1x50xi8>

      // GatherSlice 3

      // CHECK:     [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [774144, 0] [774144, 1] : tensor<1548288x1xf16> to tensor<774144x1xf16>
      // CHECK:     [[OUTPUT_3:%.+]], [[FLAG_3:%.+]] = VPU.GatherSlice([[INPUT_TILE_3]], [[CST_TILE_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, max_axis_dimension = 1548288 : i64, slice_head = 774144 : i64, slice_tail = 1548288 : i64} : tensor<774144x1xf16>, tensor<1x50xsi32> -> tensor<1x50x1xf16>, tensor<1x50xi8>
      // CHECK:     [[UNSQUEEZE_OUTPUT_3:%.+]] = VPU.Unsqueeze([[OUTPUT_3]]) {axes_value = [0]} : tensor<1x50x1xf16> -> tensor<1x1x50x1xf16>
      // CHECK:     [[UNSQUEEZE_FLAG_3:%.+]] = VPU.Unsqueeze([[FLAG_3]]) {axes_value = [0]} : tensor<1x50xi8> -> tensor<1x1x50xi8>

      // ExtractValue 1

      // CHECK:     [[CONCAT_OUTPUT_1:%.+]] = VPU.Concat([[UNSQUEEZE_OUTPUT_2]], [[UNSQUEEZE_OUTPUT_3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x50x1xf16>, tensor<1x1x50x1xf16> -> tensor<2x1x50x1xf16>
      // CHECK:     [[CONCAT_FLAG_1:%.+]] = VPU.Concat([[UNSQUEEZE_FLAG_2]], [[UNSQUEEZE_FLAG_3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x50xi8>, tensor<1x1x50xi8> -> tensor<2x1x50xi8>
      // CHECK:     [[EXTRACT_OUTPUT_1:%.+]] = VPU.ExtractValue([[CONCAT_OUTPUT_1]], [[CONCAT_FLAG_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<2x1x50x1xf16>, tensor<2x1x50xi8> -> tensor<1x50x1xf16>

      // CHECK:     [[RESULT:%.+]] = VPU.Concat([[EXTRACT_OUTPUT_0]], [[EXTRACT_OUTPUT_1]])
      // CHECK-SAME {static_offsets = [[0, 0, 0], [0, 50, 0]]} : tensor<1x50x1xf16>, tensor<1x50x1xf16> -> tensor<1x100x1xf16>
      // CHECK:     return [[RESULT]] : tensor<1x100x1xf16>
}

// -----

// CHECK-LABEL: func.func @NotDecomposeGather
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1500x1xf16>
func.func @NotDecomposeGather(%arg0: tensor<1500x1xf16>) -> (tensor<1x100x1xf16>) {
      %cst = const.Declare tensor<1x100xsi32> = dense<1> : tensor<1x100xsi64>, [#const.ConvertElemType<si32>]
      %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<1500x1xf16>, tensor<1x100xsi32> -> tensor<1x100x1xf16>
      return %0 : tensor<1x100x1xf16>

      // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x100xsi32> = dense<1> : tensor<1x100xsi64>, [#const.ConvertElemType<si32>]

      // CHECK:     [[RESULT:%.+]] = VPU.Gather([[INPUT]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<1500x1xf16>, tensor<1x100xsi32> -> tensor<1x100x1xf16>

      // CHECK:     return [[RESULT]] : tensor<1x100x1xf16>
}
