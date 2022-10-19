// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-async-ops-to-tasks --canonicalize --move-declarations-to-top %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @LinearGraph
func @LinearGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<10xf16, @DDR>
    %t0, %f0 = async.execute -> !async.value<memref<10xf16, @DDR>>
            attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 0 : i64, cycleCost = 2 : i64, cycleEnd = 2 : i64} {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %buf0 : memref<10xf16, @DDR>
    }

    %t1, %f1 = async.execute[%t0] (%f0 as %0: !async.value<memref<10xf16, @DDR>>) -> !async.value<memref<10xf16>>
            attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 2 : i64, cycleCost = 2 : i64, cycleEnd = 4 : i64 } {
        %1 = VPUIP.NNDMA inputs(%buf0 : memref<10xf16, @DDR>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %arg1: memref<10xf16>
    }

    %1 = async.await %f1 : !async.value<memref<10xf16>>
    return %1 : memref<10xf16>

    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<10xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK-NEXT:  VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, @DDR>

    // CHECK:  VPURT.Task
    // CHECK-SAME:      waits([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF0]] : memref<10xf16, @DDR>)
    // CHECK-SAME:      outputs(%arg1 : memref<10xf16>)

    // CHECK:  return %arg1 : memref<10xf16>
}

// -----

// CHECK-LABEL: @IndependentBranchesLinearSched
func @IndependentBranchesLinearSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<10xf16, @DDR>
    %t0, %f0 = async.execute -> !async.value<memref<10xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 0 : i64, cycleCost = 2 : i64, cycleEnd = 2 : i64} {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %buf0 : memref<10xf16, @DDR>
    }

    %buf1 = VPURT.DeclareBuffer "DDR" <20> -> memref<10xf16, @DDR>
    %t1, %f1 = async.execute[%t0] -> !async.value<memref<10xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 2 : i64, cycleCost = 2 : i64, cycleEnd = 4 : i64} {
        %1 = VPUIP.NNDMA inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %buf1 : memref<10xf16, @DDR>
    }

    %buf2 = VPURT.DeclareBuffer "DDR" <0> -> memref<20xf16, @DDR>
    %t2, %f2 = async.execute[%t1] (
                %f0 as %0 : !async.value<memref<10xf16, @DDR>>,
                %f1 as %1 : !async.value<memref<10xf16, @DDR>>
            ) -> !async.value<memref<20xf16>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 4 : i64, cycleCost = 2 : i64, cycleEnd = 6 : i64} {
        %2 = VPUIP.NNDMA inputs(%buf2 : memref<20xf16, @DDR>) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %arg2 : memref<20xf16>
    }

    %2 = async.await %f2 : !async.value<memref<20xf16>>
    return %2 : memref<20xf16>

    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<10xf16, @DDR>
    // CHECK-DAG:   [[BUF1:%.+]] = VPURT.DeclareBuffer "DDR" <20> -> memref<10xf16, @DDR>
    // CHECK-DAG:   [[BUF2:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<20xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:   [[B1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, @DDR>

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B0]] : !VPURT.Barrier)
    // CHECK-SAME:      updates([[B1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg1 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF1]] : memref<10xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF2]] : memref<20xf16, @DDR>)
    // CHECK-SAME:      outputs(%arg2 : memref<20xf16>)

    // CHECK:       return %arg2 : memref<20xf16>
}

// -----

// CHECK-LABEL: @IndependentBranchesParallelSched
func @IndependentBranchesParallelSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = VPURT.DeclareBuffer "DDR" <0> -> memref<20xf16, @DDR>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 0 : i64, cycleCost = 2 : i64, cycleEnd = 2 : i64} {
        %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<10xf16, @DDR>
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %0 : memref<10xf16, @DDR>
    }

    %t1, %f1 = async.execute -> !async.value<memref<10xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 2 : i64, cycleCost = 2 : i64, cycleEnd = 4 : i64} {
        %buf1 = VPURT.DeclareBuffer "DDR" <20> -> memref<10xf16, @DDR>
        %1 = VPUIP.NNDMA inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %buf1 : memref<10xf16, @DDR>
    }

    %t3, %f3 = async.execute [%t0, %t1](
                %f0 as %0 : !async.value<memref<10xf16, @DDR>>,
                %f1 as %1 : !async.value<memref<10xf16, @DDR>>
            ) -> !async.value<memref<20xf16>>
            attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 4 : i64, cycleCost = 2 : i64, cycleEnd = 6 : i64} {
        %3 = VPUIP.NNDMA inputs(%buf : memref<20xf16, @DDR>) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %arg2 : memref<20xf16>
    }

    %3 = async.await %f3 : !async.value<memref<20xf16>>
    return %3 : memref<20xf16>

    // CHECK-DAG:   [[BUF:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<20xf16, @DDR>
    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<10xf16, @DDR>
    // CHECK-DAG:   [[BUF1:%.+]] = VPURT.DeclareBuffer "DDR" <20> -> memref<10xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:   [[B1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, @DDR>

    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg1 : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF1]] : memref<10xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B0]], [[B1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF]] : memref<20xf16, @DDR>)
    // CHECK-SAME:      outputs(%arg2 : memref<20xf16>)

    // CHECK:  return %arg2 : memref<20xf16>
}

// -----

// CHECK-LABEL: @TwoOutputs
func @TwoOutputs(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %cst = const.Declare memref<2xf16, @DDR> = #const.Content<dense<1.0> : tensor<2xf16>>

    %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<2xf16, @DDR>
    %t0, %f0 = async.execute -> !async.value<memref<2xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 0 : i64, cycleCost = 2 : i64, cycleEnd = 2 : i64} {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<2xf16>) outputs(%buf0 : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
        async.yield %buf0 : memref<2xf16, @DDR>
    }

    %buf1 = VPURT.DeclareBuffer "DDR" <4> -> memref<2xf16, @DDR>
    %t1, %f1 = async.execute[%t0] -> !async.value<memref<2xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 2 : i64, cycleCost = 2 : i64, cycleEnd = 4 : i64} {
        %1 = VPUIP.NNDMA inputs(%cst : memref<2xf16, @DDR>) outputs(%buf1 : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
        async.yield %buf1 : memref<2xf16, @DDR>
    }

    %t2, %f3 = async.execute[%t1] (%f0 as %0: !async.value<memref<2xf16, @DDR>>) -> !async.value<memref<2xf16>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 4 : i64, cycleCost = 2 : i64, cycleEnd = 6 : i64} {
        %3 = VPUIP.NNDMA inputs(%0 : memref<2xf16, @DDR>) outputs(%arg1 : memref<2xf16>) -> memref<2xf16>
        async.yield %arg1 : memref<2xf16>
    }

    %t4, %f4 = async.execute[%t2] (%f1 as %1: !async.value<memref<2xf16, @DDR>>) -> !async.value<memref<2xf16>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 6 : i64, cycleCost = 2 : i64, cycleEnd = 8 : i64} {
        %3 = VPUIP.NNDMA inputs(%1 : memref<2xf16, @DDR>) outputs(%arg2 : memref<2xf16>) -> memref<2xf16>
        async.yield %arg2 : memref<2xf16>
    }

    %1 = async.await %f3 : !async.value<memref<2xf16>>
    %2 = async.await %f4 : !async.value<memref<2xf16>>
    return %1, %2 : memref<2xf16>, memref<2xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<2xf16, @DDR> =

    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<2xf16, @DDR>
    // CHECK-DAG:   [[BUF1:%.+]] = VPURT.DeclareBuffer "DDR" <4> -> memref<2xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:   [[B1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:  [[B2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs(%arg0 : memref<2xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<2xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B0]] : !VPURT.Barrier)
    // CHECK-SAME:      updates([[B1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST]] : memref<2xf16, @DDR>)
    // CHECK-SAME:      outputs([[BUF1]] : memref<2xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B1]] : !VPURT.Barrier)
    // CHECK-SAME:      updates([[B2]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF0]] : memref<2xf16, @DDR>)
    // CHECK-SAME:      outputs(%arg1 : memref<2xf16>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B2]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF1]] : memref<2xf16, @DDR>)
    // CHECK-SAME:      outputs(%arg2 : memref<2xf16>)

    // CHECK:  return %arg1, %arg2
}

// -----

// CHECK-LABEL: @WithReshape
func @WithReshape(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512x1x1xf16, @DDR>

    %t2, %f2 = async.execute -> !async.value<memref<1x512x1x1xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 0 : i64, cycleCost = 2 : i64, cycleEnd = 2 : i64} {
        %1 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x512x1x1xf16>
        %2 = VPUIP.SoftMaxUPA {axisInd = 1}
            inputs(%1 : memref<1x512x1x1xf16>)
            outputs(%0 : memref<1x512x1x1xf16, @DDR>)
            -> memref<1x512x1x1xf16, @DDR>
        async.yield %0 : memref<1x512x1x1xf16, @DDR>
    }

    %t4, %f4 = async.execute [%t2] (%f2 as %2: !async.value<memref<1x512x1x1xf16, @DDR>>) -> !async.value<memref<1x512xf16>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 2 : i64, cycleCost = 2 : i64, cycleEnd = 4 : i64} {
        %3 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512xf16, @DDR>
        %4 = VPUIP.NNDMA inputs(%3 : memref<1x512xf16, @DDR>) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>
        async.yield %arg1 : memref<1x512xf16>
    }

    %4 = async.await %f4 : !async.value<memref<1x512xf16>>
    return %4 : memref<1x512xf16>

    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512x1x1xf16, @DDR>
    // CHECK-DAG:   [[ARG0:%.+]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x512x1x1xf16>
    // CHECK-DAG:   [[VAR2:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  [[VAR1:%.*]] = VPUIP.SoftMaxUPA
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<1x512x1x1xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  [[VAR3:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR2]] : memref<1x512xf16, @DDR>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x512xf16>)

    // CHECK:  return %arg1 : memref<1x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AwaitWithoutUsers
func @AwaitWithoutUsers(%arg0: memref<1x16x112x112xf16, #NHWC>, %arg1: memref<1x16x112x112xf16, #NHWC>, %arg2: memref<1x32x112x112xf16, #NHWC>) -> memref<1x32x112x112xf16, #NHWC> {
    %t1, %f1 = async.execute -> !async.value<memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 0 : i64, cycleCost = 2 : i64, cycleEnd = 2 : i64} {
        %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        %2 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true} inputs(%arg0 : memref<1x16x112x112xf16, #NHWC>) outputs(%1 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>) -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        async.yield %2 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    }

    %t2, %f2 = async.execute [%t1] -> !async.value<memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 2 : i64, cycleCost = 2 : i64, cycleEnd = 4 : i64} {
        %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <437248> -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        %2 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true} inputs(%arg1 : memref<1x16x112x112xf16, #NHWC>) outputs(%1 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>) -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        async.yield %2 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    }

    %1 = async.await %f1 : !async.value<memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>>
    %2 = async.await %f2 : !async.value<memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>>
    return %arg2 : memref<1x32x112x112xf16, #NHWC>
}
