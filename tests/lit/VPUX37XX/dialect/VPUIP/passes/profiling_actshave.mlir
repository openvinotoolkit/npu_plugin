// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --act-shave-profiling %s | FileCheck %s

// CHECK-LABEL: @ActShaveProfiling
module @ActShaveProfiling {
    IE.CNNNetwork entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x3x224x224xf32>
    } outputsInfo :  {
        DataInfo "output" : tensor<1x150528xf32>
    } profilingOutputsInfo :  {
    }
    func @main(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x150528xf32>) -> memref<1x150528xf32> {
        %0 = memref.alloc() : memref<1x3x224x224xf16, @DDR>
        %1 = memref.alloc() : memref<1x3x224x224xf32, [@CMX_NN, 0]>
        %2 = VPUIP.Copy inputs(%arg0 : memref<1x3x224x224xf32>) outputs(%1 : memref<1x3x224x224xf32, [@CMX_NN, 0]>) -> memref<1x3x224x224xf32, [@CMX_NN, 0]>
        %3 = memref.alloc() : memref<1x3x224x224xf16, [@CMX_NN, 0]>
        %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%2 : memref<1x3x224x224xf32, [@CMX_NN, 0]>) outputs(%3 : memref<1x3x224x224xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x224x224xf16, [@CMX_NN, 0]>  {
        ^bb0(%arg2: memref<1x3x224x224xf32, [@CMX_NN, 0]>, %arg3: memref<1x3x224x224xf16, [@CMX_NN, 0]>):  // no predecessors
        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3) : memref<1x3x224x224xf32, [@CMX_NN, 0]>, memref<1x3x224x224xf16, [@CMX_NN, 0]>
        }
        %4 = VPUIP.Copy inputs(%results : memref<1x3x224x224xf16, [@CMX_NN, 0]>) outputs(%0 : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        %5 = VPUIP.GenericReshape inputs(%4 : memref<1x3x224x224xf16, @DDR>) -> memref<1x150528xf16, @DDR>
        %6 = memref.alloc() : memref<1x150528xf16, [@CMX_NN, 0]>
        %7 = VPUIP.Copy inputs(%5 : memref<1x150528xf16, @DDR>) outputs(%6 : memref<1x150528xf16, [@CMX_NN, 0]>) -> memref<1x150528xf16, [@CMX_NN, 0]>
        %8 = memref.alloc() : memref<1x150528xf16, [@CMX_NN, 0]>
        %results_0 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SoftMax inputs(%7 : memref<1x150528xf16, [@CMX_NN, 0]>) outputs(%8 : memref<1x150528xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x150528xf16, [@CMX_NN, 0]>  {
        ^bb0(%arg2: memref<1x150528xf16, [@CMX_NN, 0]>, %arg3: memref<1x150528xf16, [@CMX_NN, 0]>):  // no predecessors
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x150528xf16, [@CMX_NN, 0]>, memref<1x150528xf16, [@CMX_NN, 0]>
        }
        %9 = memref.alloc() : memref<1x150528xf32, [@CMX_NN, 0]>
        %results_1 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%results_0 : memref<1x150528xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x150528xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x150528xf32, [@CMX_NN, 0]>  {
        ^bb0(%arg2: memref<1x150528xf16, [@CMX_NN, 0]>, %arg3: memref<1x150528xf32, [@CMX_NN, 0]>):  // no predecessors
        VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3) : memref<1x150528xf16, [@CMX_NN, 0]>, memref<1x150528xf32, [@CMX_NN, 0]>
        }
        %10 = VPUIP.Copy inputs(%results_1 : memref<1x150528xf32, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x150528xf32>) -> memref<1x150528xf32>
        return %10 : memref<1x150528xf32>
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "actshave" : tensor<12xui32>
    //CHECK:        func @main(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x150528xf32>, %arg2: memref<12xui32>) -> (memref<1x150528xf32>, memref<12xui32>)
    //CHECK:        [[VAR0:%.+]] = memref.alloc() : memref<12xui32, [@CMX_NN, 0]>
    //CHECK:        [[VAR1:%.+]] = VPUIP.SubView [[VAR0]] [0] [4] : memref<12xui32, [@CMX_NN, 0]> to memref<4xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_Convert
    //CHECK-SAME:   profiling_data([[VAR1]] : memref<4xui32, [@CMX_NN, 0]>)
    //CHECK:        [[VAR2:%.+]] = VPUIP.SubView [[VAR0]] [4] [4] : memref<12xui32, [@CMX_NN, 0]> to memref<4xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_SoftMax
    //CHECK-SAME:   profiling_data([[VAR2]] : memref<4xui32, [@CMX_NN, 0]>)
    //CHECK:        [[VAR3:%.+]] = VPUIP.SubView [[VAR0]] [8] [4] : memref<12xui32, [@CMX_NN, 0]> to memref<4xui32, [@CMX_NN, 0]>
    //CHECK-NEXT:   VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_Convert
    //CHECK-SAME:   profiling_data([[VAR3]] : memref<4xui32, [@CMX_NN, 0]>)
    //CHECK:        return [[R1:%.+]], [[R2:%.+]] : memref<1x150528xf32>, memref<12xui32>
}
