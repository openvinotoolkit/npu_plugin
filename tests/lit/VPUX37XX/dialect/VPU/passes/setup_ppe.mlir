// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --setup-ppe %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @NoopCase
func @NoopCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
               %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 2147483647 : i64,
            clamp_low = -2147483648 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = "ADD",
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        }
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          mode = "NOOP",
    // CHECK-SAME:          quant_mult = [16822],
    // CHECK-SAME:          quant_post_shift = 0 : i64,
    // CHECK-SAME:          quant_shift = [13]
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @ReLUCase
func @ReLUCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
               %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 2147483647 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = "ADD",
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        }
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          mode = "LRELU",
    // CHECK-SAME:          quant_mult = [16822],
    // CHECK-SAME:          quant_post_shift = 0 : i64,
    // CHECK-SAME:          quant_shift = [13]
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @ReLUXCase
func @ReLUXCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 6 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            mode = "ADD",
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        }
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 6 : i64,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 0 : i64,
    // CHECK-SAME:          mode = "LRELUX",
    // CHECK-SAME:          quant_mult = [16822],
    // CHECK-SAME:          quant_post_shift = 0 : i64,
    // CHECK-SAME:          quant_shift = [13]
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = type !quant.uniform<u8:f16, 0.054779411764705882>

// CHECK-LABEL: @PReLUCase
func @PReLUCase(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {
            clamp_high = 2147483647 : i64,
            clamp_low = 0 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 10 : i64,
            mode = "ADD",
            quant_mult = [16822],
            quant_post_shift = 0 : i64,
            quant_shift = [13]
        }
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    // CHECK-SAME:      op_type = "ADD",
    // CHECK-SAME:      ppe = {
    // CHECK-SAME:          clamp_high = 255 : i64,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          lrelu_mult = 1 : i64,
    // CHECK-SAME:          lrelu_shift = 10 : i64,
    // CHECK-SAME:          mode = "LPRELU",
    // CHECK-SAME:          quant_mult = [16822],
    // CHECK-SAME:          quant_post_shift = 0 : i64,
    // CHECK-SAME:          quant_shift = [13]
    // CHECK-SAME:      }
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}
