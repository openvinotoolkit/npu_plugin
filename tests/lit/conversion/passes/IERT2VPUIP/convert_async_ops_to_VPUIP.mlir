// RUN: vpux-opt --split-input-file --convert-async-ops-to-VPUIP --canonicalize %s | FileCheck %s

// CHECK-LABEL: @LinearGraph
func @LinearGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<10xf16, "DDR">
    %t0, %f0 = async.execute -> !async.value<memref<10xf16, "DDR">>
            attributes { IERT.executor = "DMA_NN", IERT.num_units = 1 } {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, "DDR">) -> memref<10xf16, "DDR">
        async.yield %0 : memref<10xf16, "DDR">
    }

    %t1, %f1 = async.execute[%t0] (%f0 as %0: !async.value<memref<10xf16, "DDR">>) -> !async.value<memref<10xf16>>
            attributes { IERT.executor = "DMA_NN", IERT.num_units = 1 } {
        %1 = VPUIP.NNDMA inputs(%0 : memref<10xf16, "DDR">) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }

    %1 = async.await %f1 : !async.value<memref<10xf16>>
    return %1 : memref<10xf16>

    // CHECK-DAG:   [[BUF0:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<10xf16, "DDR">
    // CHECK-DAG:   [[B0:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAL0:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, "DDR">
    // CHECK-SAME:      updates([[B0]] : !VPUIP.Barrier)

    // CHECK-NOT:   VPUIP.DeclareVirtualBarrier

    // CHECK-NEXT:  [[VAL1:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAL0]] : memref<10xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<10xf16>)
    // CHECK-SAME:      waits([[B0]] : !VPUIP.Barrier)

    // CHECK-NEXT:  return [[VAL1]] : memref<10xf16>
}

// -----

// CHECK-LABEL: @IndependentBranchesLinearSched
func @IndependentBranchesLinearSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf0 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<10xf16, "DDR">
    %t0, %f0 = async.execute -> !async.value<memref<10xf16, "DDR">> {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, "DDR">) -> memref<10xf16, "DDR">
        async.yield %0 : memref<10xf16, "DDR">
    }

    %buf1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <20> -> memref<10xf16, "DDR">
    %t1, %f1 = async.execute[%t0] -> !async.value<memref<10xf16, "DDR">> {
        %1 = VPUIP.NNDMA inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16, "DDR">) -> memref<10xf16, "DDR">
        async.yield %1 : memref<10xf16, "DDR">
    }

    %buf2 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<20xf16, "DDR">
    %t2, %f2 = async.execute[%t1] (
                %f0 as %0 : !async.value<memref<10xf16, "DDR">>,
                %f1 as %1 : !async.value<memref<10xf16, "DDR">>
            ) -> !async.value<memref<20xf16>> {
        %2 = VPUIP.NNDMA inputs(%buf2 : memref<20xf16, "DDR">) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %2 : memref<20xf16>
    }

    %2 = async.await %f2 : !async.value<memref<20xf16>>
    return %2 : memref<20xf16>

    // CHECK-DAG:   [[BUF0:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<10xf16, "DDR">
    // CHECK-DAG:   [[B0:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAL0:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, "DDR">
    // CHECK-SAME:      updates([[B0]] : !VPUIP.Barrier)

    // CHECK-DAG:   [[BUF1:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <20> -> memref<10xf16, "DDR">
    // CHECK-DAG:   [[B1:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAL1:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg1 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF1]] : memref<10xf16, "DDR">)
    // CHECK-SAME:      waits([[B0]] : !VPUIP.Barrier)
    // CHECK-SAME:      updates([[B1]] : !VPUIP.Barrier)

    // CHECK-NOT:   VPUIP.DeclareVirtualBarrier
    // CHECK-NEXT:  [[BUF2:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<20xf16, "DDR">
    // CHECK-NOT:   VPUIP.DeclareVirtualBarrier

    // CHECK-NEXT:  [[VAL2:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF2]] : memref<20xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg2 : memref<20xf16>)
    // CHECK-SAME:      waits([[B1]] : !VPUIP.Barrier)

    // CHECK-NEXT:  return [[VAL2]] : memref<20xf16>
}

// -----

#map = affine_map<(d0) -> (d0 + 10)>

// CHECK-LABEL: @IndependentBranchesParallelSched
func @IndependentBranchesParallelSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<20xf16, "DDR">

    %t0, %f0 = async.execute -> !async.value<memref<10xf16, "DDR">> {
        %buf0 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<10xf16, "DDR">
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, "DDR">) -> memref<10xf16, "DDR">
        async.yield %0 : memref<10xf16, "DDR">
    }

    %t1, %f1 = async.execute -> !async.value<memref<10xf16, #map, "DDR">> {
        %buf1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <20> -> memref<10xf16, #map, "DDR">
        %1 = VPUIP.NNDMA inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16, #map, "DDR">) -> memref<10xf16, #map, "DDR">
        async.yield %1 : memref<10xf16, #map, "DDR">
    }

    %t3, %f3 = async.execute [%t0, %t1](
                %f0 as %0 : !async.value<memref<10xf16, "DDR">>,
                %f1 as %1 : !async.value<memref<10xf16, #map, "DDR">>
            ) -> !async.value<memref<20xf16>> {
        %3 = VPUIP.NNDMA inputs(%buf : memref<20xf16, "DDR">) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %3 : memref<20xf16>
    }

    %3 = async.await %f3 : !async.value<memref<20xf16>>
    return %3 : memref<20xf16>

    // CHECK:       [[BUF:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<20xf16, "DDR">

    // CHECK-DAG:   [[BUF0:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<10xf16, "DDR">
    // CHECK-DAG:   [[B0:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAL0:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, "DDR">
    // CHECK-SAME:      updates([[B0]] : !VPUIP.Barrier)

    // CHECK-DAG:   [[BUF1:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <20> -> memref<10xf16, #map, "DDR">
    // CHECK-DAG:   [[B1:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAL1:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg1 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF1]] : memref<10xf16, #map, "DDR">)
    // CHECK-SAME:      updates([[B1]] : !VPUIP.Barrier)

    // CHECK-NOT:   VPUIP.DeclareVirtualBarrier

    // CHECK-NEXT:  [[VAL2:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF]] : memref<20xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg2 : memref<20xf16>)
    // CHECK-SAME:      waits([[B0]], [[B1]] : !VPUIP.Barrier, !VPUIP.Barrier)

    // CHECK-NEXT:  return [[VAL2]] : memref<20xf16>
}

// -----

// CHECK-LABEL: @TwoOutputs
func @TwoOutputs(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %cst = const.Declare memref<2xf16, "DDR"> = #const.Content<dense<1.0> : tensor<2xf16>>

    %buf0 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<2xf16, "DDR">
    %t0, %f0 = async.execute -> !async.value<memref<2xf16, "DDR">> {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<2xf16>) outputs(%buf0 : memref<2xf16, "DDR">) -> memref<2xf16, "DDR">
        async.yield %0 : memref<2xf16, "DDR">
    }

    %buf1 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <4> -> memref<2xf16, "DDR">
    %t1, %f1 = async.execute[%t0] -> !async.value<memref<2xf16, "DDR">> {
        %1 = VPUIP.NNDMA inputs(%cst : memref<2xf16, "DDR">) outputs(%buf1 : memref<2xf16, "DDR">) -> memref<2xf16, "DDR">
        async.yield %1 : memref<2xf16, "DDR">
    }

    %t2, %f3 = async.execute[%t1] (%f0 as %0: !async.value<memref<2xf16, "DDR">>) -> !async.value<memref<2xf16>> {
        %3 = VPUIP.NNDMA inputs(%0 : memref<2xf16, "DDR">) outputs(%arg1 : memref<2xf16>) -> memref<2xf16>
        async.yield %3 : memref<2xf16>
    }

    %t4, %f4 = async.execute[%t2] (%f1 as %1: !async.value<memref<2xf16, "DDR">>) -> !async.value<memref<2xf16>> {
        %3 = VPUIP.NNDMA inputs(%1 : memref<2xf16, "DDR">) outputs(%arg2 : memref<2xf16>) -> memref<2xf16>
        async.yield %3 : memref<2xf16>
    }

    %1 = async.await %f3 : !async.value<memref<2xf16>>
    %2 = async.await %f4 : !async.value<memref<2xf16>>
    return %1, %2 : memref<2xf16>, memref<2xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<2xf16, "DDR"> =

    // CHECK-DAG:   [[BUF0:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<2xf16, "DDR">
    // CHECK-DAG:   [[B0:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAL0:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<2xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<2xf16, "DDR">)
    // CHECK-SAME:      updates([[B0]] : !VPUIP.Barrier)

    // CHECK-DAG:   [[BUF1:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <4> -> memref<2xf16, "DDR">
    // CHECK-DAG:   [[B1:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAL1:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST]] : memref<2xf16, "DDR">)
    // CHECK-SAME:      outputs([[BUF1]] : memref<2xf16, "DDR">)
    // CHECK-SAME:      waits([[B0]] : !VPUIP.Barrier)
    // CHECK-SAME:      updates([[B1]] : !VPUIP.Barrier)

    // CHECK-NEXT:  [[B2:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAL2:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAL0]] : memref<2xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<2xf16>)
    // CHECK-SAME:      waits([[B1]] : !VPUIP.Barrier)
    // CHECK-SAME:      updates([[B2]] : !VPUIP.Barrier)

    // CHECK-NOT:   VPUIP.DeclareVirtualBarrier

    // CHECK-NEXT:  [[VAL3:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAL1]] : memref<2xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg2 : memref<2xf16>)
    // CHECK-SAME:      waits([[B2]] : !VPUIP.Barrier)

    // CHECK-NEXT:  return [[VAL2]], [[VAL3]]
}

// -----

// CHECK-LABEL: @WithReshape
func @WithReshape(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %0 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512x1x1xf16, "DDR">

    %t2, %f2 = async.execute -> !async.value<memref<1x512x1x1xf16, "DDR">> {
        %1 = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<1x512x1x1xf16>
        %2 = VPUIP.SoftMaxUPA {axisInd = 1}
            inputs(%1 : memref<1x512x1x1xf16>)
            outputs(%0 : memref<1x512x1x1xf16, "DDR">)
            -> memref<1x512x1x1xf16, "DDR">
        async.yield %2 : memref<1x512x1x1xf16, "DDR">
    }

    %t4, %f4 = async.execute [%t2] (%f2 as %2: !async.value<memref<1x512x1x1xf16, "DDR">>) -> !async.value<memref<1x512xf16>> {
        %3 = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512xf16, "DDR">
        %4 = VPUIP.NNDMA inputs(%3 : memref<1x512xf16, "DDR">) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>
        async.yield %4 : memref<1x512xf16>
    }

    %4 = async.await %f4 : !async.value<memref<1x512xf16>>
    return %4 : memref<1x512xf16>

    // CHECK-DAG:   [[BUF0:%.+]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512x1x1xf16, "DDR">
    // CHECK-DAG:   [[ARG0:%.+]] = VPUIP.DeclareTensor "ProgrammableInput" [0] <0> -> memref<1x512x1x1xf16>
    // CHECK-DAG:   [[B0:%.+]] = VPUIP.DeclareVirtualBarrier -> !VPUIP.Barrier

    // CHECK-NEXT:  [[VAR1:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x512x1x1xf16, "DDR">)
    // CHECK-SAME:      updates([[B0]] : !VPUIP.Barrier)

    // CHECK-NEXT:  [[VAR2:%.*]] = VPUIP.DeclareTensor "VPU_DDR_Heap" [0] <0> -> memref<1x512xf16, "DDR">

    // CHECK-NEXT:  [[VAR3:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR2]] : memref<1x512xf16, "DDR">)
    // CHECK-SAME:      outputs(%arg1 : memref<1x512xf16>)
    // CHECK-SAME:      waits([[B0]] : !VPUIP.Barrier)

    // CHECK-NEXT:  return [[VAR3]] : memref<1x512xf16>
}
