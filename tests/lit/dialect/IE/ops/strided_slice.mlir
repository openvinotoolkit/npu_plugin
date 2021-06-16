// RUN: vpux-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x10x10x30xf16> {
    %begins = const.Declare tensor<4xsi64> = #const.Content<dense<[0, 0, 0, 0]> : tensor<4xsi64>>
    %ends = const.Declare tensor<4xsi64> = #const.Content<dense<[1, 5, 10, 20]> : tensor<4xsi64>>
    %strides = const.Declare tensor<4xsi64> = #const.Content<dense<[1, 1, 1, 1]> : tensor<4xsi64>>

    %0 = IE.StridedSlice(%arg0, %begins, %ends, %strides) {
        begin_mask = [0, 1, 1, 0],
        end_mask = [0, 1, 0, 1],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operand_segment_sizes = dense<1> : vector<4xi32>
    } : tensor<1x10x20x30xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<4xsi64> -> tensor<1x10x10x30xf16>

    return %0 : tensor<1x10x10x30xf16>
    // CHECK:       %[[VAL0:.*]] = IE.StridedSlice(%arg0)
    // CHECK-SAME:  begins_attr = [0, 0, 0, 0]
    // CHECK-SAME:  ends_attr = [1, 5, 10, 20]
    // CHECK-SAME:  strides_attr = [1, 1, 1, 1]
}
