// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ParsePrintGroupSparseTensorPartial
func @ParsePrintGroupSparseTensorPartial(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %0 = const.Declare tensor<1x32x16x16xi1> = #const.Content<dense<1> : tensor<1x32x16x16xi1>>
    %1 = VPU.GroupSparseTensor(%arg0 : tensor<1x32x16x16xf16>, %0 : tensor<1x32x16x16xi1>)
            -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    %2:2 = builtin.unrealized_conversion_cast %1 : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
            to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>
    return %2#0 : tensor<1x32x16x16xf16>

    // CHECK:       [[SM:%.*]] = const.Declare tensor<1x32x16x16xi1> = #const.Content<dense<true> : tensor<1x32x16x16xi1>>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(
    // CHECK-SAME:                      %arg0 : tensor<1x32x16x16xf16>,
    // CHECK-SAME:                      [[SM]] : tensor<1x32x16x16xi1>)
    // CHECK-SAME:                 -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    // CHECK:       [[VAL1:%.*]]:2 = builtin.unrealized_conversion_cast
    // CHECK-SAME:                      [[VAL0]] : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    // CHECK-SAME:                      to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>
    // CHECK:       return [[VAL1]]#0 : tensor<1x32x16x16xf16>
}

// -----

// CHECK-LABEL: @ParsePrintGroupSparseTensor
func @ParsePrintGroupSparseTensor(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %0 = const.Declare tensor<1x32x16x16xi1> = #const.Content<dense<1> : tensor<1x32x16x16xi1>>
    %1 = const.Declare tensor<1x32x1x1xi32> = #const.Content<dense<1> : tensor<1x32x1x1xi32>>
    %2 = VPU.GroupSparseTensor(%arg0 : tensor<1x32x16x16xf16>, %0 : tensor<1x32x16x16xi1>, %1 : tensor<1x32x1x1xi32>)
            -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    %3:3 = builtin.unrealized_conversion_cast %2
            : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
            to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>, tensor<1x32x1x1xi32>
    return %3#0 : tensor<1x32x16x16xf16>

    // CHECK-DAG:   [[SM:%.*]] = const.Declare tensor<1x32x16x16xi1> = #const.Content<dense<true> : tensor<1x32x16x16xi1>>
    // CHECK-DAG:   [[SE:%.*]] = const.Declare tensor<1x32x1x1xi32> = #const.Content<dense<1> : tensor<1x32x1x1xi32>>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(
    // CHECK-SAME:                      %arg0 : tensor<1x32x16x16xf16>,
    // CHECK-SAME:                      [[SM]] : tensor<1x32x16x16xi1>,
    // CHECK-SAME:                      [[SE]] : tensor<1x32x1x1xi32>)
    // CHECK-SAME:                 -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    // CHECK:       [[VAL1:%.*]]:3 = builtin.unrealized_conversion_cast
    // CHECK-SAME:                      [[VAL0]] : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    // CHECK-SAME:                      to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>, tensor<1x32x1x1xi32>
    // CHECK:       return [[VAL1]]#0 : tensor<1x32x16x16xf16>
}
