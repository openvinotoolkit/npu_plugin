// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-paddings-to-floor-mode %s | FileCheck %s

// CHECK-LABEL: @MaxPool
func @MaxPool(%arg0: tensor<1x512x38x38xf32>) -> tensor<1x512x19x19xf32> {
    %0 = IE.MaxPool(%arg0)
        {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2],
            rounding_type = "FLOOR"
        } :
        tensor<1x512x38x38xf32> -> tensor<1x512x19x19xf32>

    return %0 : tensor<1x512x19x19xf32>

    // CHECK:        [[SLICE:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 37, 37] : tensor<1x512x38x38xf32> to tensor<1x512x37x37xf32>
    // CHECK:        [[MAX_POOL:%.+]] = IE.MaxPool([[SLICE]]) {
    // CHECK-SAME:       kernel_size = [1, 1],
    // CHECK-SAME:       pads_begin = [0, 0],
    // CHECK-SAME:       pads_end = [0, 0],
    // CHECK-SAME:       rounding_type = "FLOOR",
    // CHECK-SAME:       strides = [2, 2]
    // CHECK-SAME:   } :
    // CHECK:        tensor<1x512x37x37xf32> -> tensor<1x512x19x19xf32>
    // CHECK:        return [[MAX_POOL]]  : tensor<1x512x19x19xf32>
}
