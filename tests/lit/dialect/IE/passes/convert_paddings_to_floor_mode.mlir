// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --convert-paddings-to-floor-mode %s | FileCheck %s

// CHECK-LABEL: @MaxPool
func @MaxPool(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x30x30xf32> {
    %0 = IE.MaxPool(%arg0)
        {
            kernel_size = [3, 3],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2],
            rounding_type = "CEIL"
        } :
        tensor<1x48x60x60xf32> -> tensor<1x48x30x30xf32>

    return %0 : tensor<1x48x30x30xf32>

    // CHECK:       %[[VAL0:.*]] = IE.MaxPool(%arg0)
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      rounding_type = "FLOOR"
    // CHECK-SAME:      strides = [2, 2]
    // CHECK:       return %[[VAL0]] : tensor<1x48x30x30xf32>
}
