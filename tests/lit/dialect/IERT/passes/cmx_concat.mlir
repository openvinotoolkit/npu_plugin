// RUN: vpux-opt --split-input-file --cmx-concat %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CaseWithoutChildTiling
module @CaseWithoutChildTiling attributes {VPU.arch = "KMB"} {

IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
IE.MemoryResource 3311264 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x112x112xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x64x112x112xf16>
    }

func @main(
        %input: memref<1x32x112x112xf16, #NHWC, @DDR>,
        %output : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
         -> memref<1x64x112x112xf16, #NHWC, @CMX_NN>{
    // tile 1
    %0 = memref.alloc() : memref<1x32x112x112xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%input : memref<1x32x112x112xf16, #NHWC, @DDR>)
        outputs(%0 : memref<1x32x112x112xf16, #NHWC, @DDR>)
        -> memref<1x32x112x112xf16, #NHWC, @DDR>
    %cst_0 = const.Declare memref<64x32x3x3xf16> =
        #const.Content<dense<2.0> : tensor<64x32x3x3xf16>>
    %105 = IERT.SubView %1 [0, 0, 0, 0] [1, 32, 57, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    %106 = memref.alloc() : memref<1x32x57x112xf16, @DDR>
    %107 = memref.alloc() : memref<1x32x57x112xf16, #NHWC, @CMX_NN>
    %108 = IERT.Copy
        inputs(%105 : memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>)
        outputs(%107 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
        -> memref<1x32x57x112xf16, #NHWC, @CMX_NN>

    %109 = memref.alloc() : memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    %110 = IERT.Copy
        inputs(%cst_0 : memref<64x32x3x3xf16>)
        outputs(%109 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
        -> memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    %111 = memref.alloc() : memref<1x64x56x112xf16, #NHWC, @CMX_NN>
    %112 = const.Declare memref<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    %113 = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    %114 = IERT.Copy inputs(%112 : memref<64x1x1x4xsi32>) outputs(%113 : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>
    %115 = VPUIP.NCEClusterTask
        {
        kernel_padding = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "CONV"}
        input(%108 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
        weights(%110 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%114 : memref<64x1x1x4xsi32, @CMX_NN>)
        parent_input(%108 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
        parent_output(%111 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>)
        outputs(%111 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>)
        -> memref<1x64x56x112xf16, #NHWC, @CMX_NN> variants :  {
      DPUTask {end = [111, 10, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, start = [0, 0, 0]}
      DPUTask {end = [111, 21, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 11, 0]}
      DPUTask {end = [111, 32, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 22, 0]}
      DPUTask {end = [111, 43, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 33, 0]}
      DPUTask {end = [111, 55, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 44, 0]}
    } PPE :  {
    }
    %116 = memref.alloc() : memref<1x64x112x112xf16, #NHWC, @DDR>
    %117 = IERT.SubView %116 [0, 0, 0, 0] [1, 64, 56, 112] : memref<1x64x112x112xf16, #NHWC, @DDR> to memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %118 = IERT.Copy inputs(%115 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>) outputs(%117 : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) -> memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>

    // tile 2
    %119 = IERT.SubView %1 [0, 0, 55, 0] [1, 32, 57, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    %120 = memref.alloc() : memref<1x32x57x112xf16, #NHWC, @DDR>
    %121 = memref.alloc() : memref<1x32x57x112xf16, #NHWC, @CMX_NN>
    %122 = IERT.Copy inputs(%119 : memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>) outputs(%121 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>) -> memref<1x32x57x112xf16, #NHWC, @CMX_NN>
    %123 = memref.alloc() : memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    %124 = IERT.Copy inputs(%cst_0 : memref<64x32x3x3xf16>) outputs(%123 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>) -> memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    %125 = memref.alloc() : memref<1x64x56x112xf16, #NHWC, @CMX_NN>
    %126 = const.Declare memref<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    %127 = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    %128 = IERT.Copy inputs(%126 : memref<64x1x1x4xsi32>) outputs(%127 : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>
    %129 = VPUIP.NCEClusterTask {
    kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    kernel_size = [3, 3],
    kernel_strides = [1, 1],
    task_type = "CONV"}
    input(%122 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
    weights(%124 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    weight_table(%128 : memref<64x1x1x4xsi32, @CMX_NN>)
    parent_input(%122 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
    parent_output(%125 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>)
    outputs(%125 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>)
    -> memref<1x64x56x112xf16, #NHWC, @CMX_NN> variants :  {
      DPUTask {end = [111, 10, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 0, 0]}
      DPUTask {end = [111, 21, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 11, 0]}
      DPUTask {end = [111, 32, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 22, 0]}
      DPUTask {end = [111, 43, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 33, 0]}
      DPUTask {end = [111, 55, 63], mpe_mode = "MATRIX", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 44, 0]}
    } PPE :  {
    }
    %130 = IERT.SubView %116 [0, 0, 56, 0] [1, 64, 56, 112] : memref<1x64x112x112xf16, #NHWC, @DDR> to memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %131 = IERT.Copy inputs(%129 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>) outputs(%130 : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) -> memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %132 = IERT.ConcatView {CMXConcat = false} inputs(%118, %131 : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>, memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) outputs(%116 : memref<1x64x112x112xf16, #NHWC, @DDR>) -> memref<1x64x112x112xf16, #NHWC, @DDR>
    %135 = memref.alloc() : memref<1x64x112x112xf16, #NHWC, @CMX_NN>
    %136 = IERT.Copy inputs(%132 : memref<1x64x112x112xf16, #NHWC, @DDR>) outputs(%135 : memref<1x64x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x64x112x112xf16, #NHWC, @CMX_NN>
    %138 = IERT.ReLU inputs(%136 : memref<1x64x112x112xf16, #NHWC, @CMX_NN>) outputs(%output : memref<1x64x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x64x112x112xf16, #NHWC, @CMX_NN>
    return %138 : memref<1x64x112x112xf16, #NHWC, @CMX_NN>

    // input copy in
    // CHECK:       IERT.SubView
    // CHECK-SAME:      [0, 0, 0, 0] [1, 32, 57, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    // CHECK:       IERT.Copy

    // cmx concat buffers and sub views
    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x64x112x112xf16, #NHWC, @CMX_NN>
    // CHECK-NEXT:  [[VAR1:%.+]] = IERT.SubView
    // CHECK-SAME:      [0, 0, 0, 0] [1, 64, 56, 112] : memref<1x64x112x112xf16, #NHWC, @CMX_NN> to memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>
    // CHECK-NEXT:  [[VAR2:%.+]] = IERT.SubView
    // CHECK-SAME:      [0, 0, 56, 0] [1, 64, 56, 112] : memref<1x64x112x112xf16, #NHWC, @CMX_NN> to memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>

    // first tile
    // CHECK:       [[VAR3:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>)

    // second input copy in
    // CHECK:       IERT.SubView
    // CHECK-SAME:      [0, 0, 55, 0] [1, 32, 57, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    // CHECK:       IERT.Copy

    // second tile
    // CHECK:       [[VAR4:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>)

    // cmx concat
    // CHECK:       [[VAR5:%.+]] = IERT.ConcatView
    // CHECK-SAME       inputs([[VAR3, VAR4]] : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>, memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>)

    // CHECK:       IERT.ReLU
    // CHECK-SAME       inputs([[VAR5]] : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CaseWithChildTiling
module @CaseWithChildTiling attributes {VPU.arch = "KMB"} {

IE.MemoryResource 31457280 bytes of @DDR {VPU.bandwidth = 8, VPU.derateFactor = 6.000000e-01}
IE.MemoryResource 4194304 bytes of @CMX_UPA {VPU.bandwidth = 16, VPU.derateFactor = 8.500000e-01}
IE.MemoryResource 3311264 bytes of @CMX_NN {VPU.bandwidth = 32, VPU.derateFactor = 1.000000e+00}

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x32x112x112xf16>
    }
    outputsInfo : {
        DataInfo "prob1" : tensor<1x64x57x112xf16>
        DataInfo "prob2" : tensor<1x64x57x112xf16>
    }

func @main(
        %input: memref<1x32x112x112xf16, #NHWC, @DDR>,
        %output1: memref<1x64x57x112xf16, #NHWC, @CMX_NN>, %output2: memref<1x64x57x112xf16, #NHWC, @CMX_NN>)
            -> (memref<1x64x57x112xf16, #NHWC, @CMX_NN>, memref<1x64x57x112xf16, #NHWC, @CMX_NN>){
    // tile 1
    %0 = memref.alloc() : memref<1x32x112x112xf16, #NHWC, @DDR>
    %1 = IERT.Convert
        inputs(%input : memref<1x32x112x112xf16, #NHWC, @DDR>)
        outputs(%0 : memref<1x32x112x112xf16, #NHWC, @DDR>)
        -> memref<1x32x112x112xf16, #NHWC, @DDR>
    %cst_0 = const.Declare memref<64x32x3x3xf16> =
        #const.Content<dense<2.0> : tensor<64x32x3x3xf16>>
    %105 = IERT.SubView %1 [0, 0, 0, 0] [1, 32, 57, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    %106 = memref.alloc() : memref<1x32x57x112xf16, @DDR>
    %107 = memref.alloc() : memref<1x32x57x112xf16, #NHWC, @CMX_NN>
    %108 = IERT.Copy
        inputs(%105 : memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>)
        outputs(%107 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
        -> memref<1x32x57x112xf16, #NHWC, @CMX_NN>

    %109 = memref.alloc() : memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    %110 = IERT.Copy
        inputs(%cst_0 : memref<64x32x3x3xf16>)
        outputs(%109 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
        -> memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    %111 = memref.alloc() : memref<1x64x56x112xf16, #NHWC, @CMX_NN>
    %112 = const.Declare memref<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    %113 = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    %114 = IERT.Copy inputs(%112 : memref<64x1x1x4xsi32>) outputs(%113 : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>
    %115 = VPUIP.NCEClusterTask
        {
        kernel_padding = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "CONV"}
        input(%108 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
        weights(%110 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%114 : memref<64x1x1x4xsi32, @CMX_NN>)
        parent_input(%108 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
        parent_output(%111 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>)
        outputs(%111 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>)
        -> memref<1x64x56x112xf16, #NHWC, @CMX_NN> variants :  {
      DPUTask {end = [111, 10, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, start = [0, 0, 0]}
      DPUTask {end = [111, 21, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 11, 0]}
      DPUTask {end = [111, 32, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 22, 0]}
      DPUTask {end = [111, 43, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 33, 0]}
      DPUTask {end = [111, 55, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 44, 0]}
    } PPE :  {
    }
    %116 = memref.alloc() : memref<1x64x112x112xf16, #NHWC, @DDR>
    %117 = IERT.SubView %116 [0, 0, 0, 0] [1, 64, 56, 112] : memref<1x64x112x112xf16, #NHWC, @DDR> to memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %118 = IERT.Copy inputs(%115 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>) outputs(%117 : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) -> memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>

    // tile 2
    %119 = IERT.SubView %1 [0, 0, 55, 0] [1, 32, 57, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    %120 = memref.alloc() : memref<1x32x57x112xf16, #NHWC, @DDR>
    %121 = memref.alloc() : memref<1x32x57x112xf16, #NHWC, @CMX_NN>
    %122 = IERT.Copy inputs(%119 : memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>) outputs(%121 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>) -> memref<1x32x57x112xf16, #NHWC, @CMX_NN>
    %123 = memref.alloc() : memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    %124 = IERT.Copy inputs(%cst_0 : memref<64x32x3x3xf16>) outputs(%123 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>) -> memref<64x32x3x3xf16, #NHWC, @CMX_NN>
    %125 = memref.alloc() : memref<1x64x56x112xf16, #NHWC, @CMX_NN>
    %126 = const.Declare memref<64x1x1x4xsi32> = #const.Content<dense<10> : tensor<64x1x1x4xsi32>>
    %127 = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    %128 = IERT.Copy inputs(%126 : memref<64x1x1x4xsi32>) outputs(%127 : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>
    %129 = VPUIP.NCEClusterTask {
    kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    kernel_size = [3, 3],
    kernel_strides = [1, 1],
    task_type = "CONV"}
    input(%122 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
    weights(%124 : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    weight_table(%128 : memref<64x1x1x4xsi32, @CMX_NN>)
    parent_input(%122 : memref<1x32x57x112xf16, #NHWC, @CMX_NN>)
    parent_output(%125 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>)
    outputs(%125 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>)
    -> memref<1x64x56x112xf16, #NHWC, @CMX_NN> variants :  {
      DPUTask {end = [111, 10, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 0, 0]}
      DPUTask {end = [111, 21, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 11, 0]}
      DPUTask {end = [111, 32, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 22, 0]}
      DPUTask {end = [111, 43, 63], mpe_mode = "MATRIX", pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 33, 0]}
      DPUTask {end = [111, 55, 63], mpe_mode = "MATRIX", pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, start = [0, 44, 0]}
    } PPE :  {
    }
    %130 = IERT.SubView %116 [0, 0, 56, 0] [1, 64, 56, 112] : memref<1x64x112x112xf16, #NHWC, @DDR> to memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %131 = IERT.Copy inputs(%129 : memref<1x64x56x112xf16, #NHWC, @CMX_NN>) outputs(%130 : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) -> memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %132 = IERT.ConcatView {CMXConcat = false} inputs(%118, %131 : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>, memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>) outputs(%116 : memref<1x64x112x112xf16, #NHWC, @DDR>) -> memref<1x64x112x112xf16, #NHWC, @DDR>

    %133 = IERT.SubView %132 [0, 0, 0, 0] [1, 64, 57, 112] : memref<1x64x112x112xf16, #NHWC, @DDR> to memref<1x64x57x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>
    %134 = IERT.SubView %132 [0, 0, 55, 0] [1, 64, 57, 112] : memref<1x64x112x112xf16, #NHWC, @DDR> to memref<1x64x57x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>

    %135 = memref.alloc() : memref<1x64x57x112xf16, #NHWC, @CMX_NN>
    %136 = IERT.Copy
        inputs(%133 : memref<1x64x57x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>)
        outputs(%135 : memref<1x64x57x112xf16, #NHWC, @CMX_NN>)
        -> memref<1x64x57x112xf16, #NHWC, @CMX_NN>

    %137 = memref.alloc() : memref<1x64x57x112xf16, #NHWC, @CMX_NN>
    %138 = IERT.Copy
        inputs(%134 : memref<1x64x57x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @DDR>)
        outputs(%137 : memref<1x64x57x112xf16, #NHWC, @CMX_NN>)
        -> memref<1x64x57x112xf16, #NHWC, @CMX_NN>

    %139 = IERT.ReLU inputs(%136 : memref<1x64x57x112xf16, #NHWC, @CMX_NN>) outputs(%output1 : memref<1x64x57x112xf16, #NHWC, @CMX_NN>) -> memref<1x64x57x112xf16, #NHWC, @CMX_NN>
    %140 = IERT.ReLU inputs(%138 : memref<1x64x57x112xf16, #NHWC, @CMX_NN>) outputs(%output2 : memref<1x64x57x112xf16, #NHWC, @CMX_NN>) -> memref<1x64x57x112xf16, #NHWC, @CMX_NN>

    return %139, %140 : memref<1x64x57x112xf16, #NHWC, @CMX_NN>, memref<1x64x57x112xf16, #NHWC, @CMX_NN>

    // input copy in
    // CHECK:       IERT.SubView
    // CHECK-SAME:      [0, 0, 0, 0] [1, 32, 57, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    // CHECK:       IERT.Copy

    // cmx concat buffers and sub views
    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x64x112x112xf16, #NHWC, @CMX_NN>
    // CHECK-NEXT:  [[VAR1:%.+]] = IERT.SubView
    // CHECK-SAME:      [0, 0, 0, 0] [1, 64, 56, 112] : memref<1x64x112x112xf16, #NHWC, @CMX_NN> to memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>
    // CHECK-NEXT:  [[VAR2:%.+]] = IERT.SubView
    // CHECK-SAME:      [0, 0, 56, 0] [1, 64, 56, 112] : memref<1x64x112x112xf16, #NHWC, @CMX_NN> to memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>

    // first tile
    // CHECK:       [[VAR3:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>)

    // second input copy in
    // CHECK:       IERT.SubView
    // CHECK-SAME:      [0, 0, 55, 0] [1, 32, 57, 112] : memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x32x57x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    // CHECK:       IERT.Copy

    // second tile
    // CHECK:       [[VAR4:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>)

    // cmx concat
    // CHECK:       [[VAR5:%.+]] = IERT.ConcatView
    // CHECK-SAME       inputs([[VAR3, VAR4]] : memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>, memref<1x64x56x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>)

    // slices of input for next op
    // CHECK:       [[VAR6:%.+]] = IERT.SubView
    // CHECK-SAME:      [0, 0, 55, 0] [1, 64, 57, 112] : memref<1x64x112x112xf16, #NHWC, @CMX_NN> to memref<1x64x57x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>
    // CHECK:       [[VAR7:%.+]] = IERT.SubView
    // CHECK-SAME:      [0, 0, 0, 0] [1, 64, 57, 112] : memref<1x64x112x112xf16, #NHWC, @CMX_NN> to memref<1x64x57x112xf16, {order = #NHWC, strides = [802816, 1, 7168, 64]}, @CMX_NN>

    // CHECK:       IERT.ReLU
    // CHECK-SAME       inputs([[VAR7]] : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK:       IERT.ReLU
    // CHECK-SAME       inputs([[VAR6]] : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
}

}
