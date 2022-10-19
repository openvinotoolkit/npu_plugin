// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-async-ops-to-tasks --canonicalize --move-declarations-to-top %s | FileCheck %s

// CHECK-LABEL: @WithProfiling
func @WithProfiling(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %t1, %f1 = async.execute -> !async.value<memref<1x512xf16>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1, cycleBegin = 0 : i64, cycleCost = 38 : i64, cycleEnd = 38 : i64 } {
        %3 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512xf16, @DDR>

        %4 = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui32, [@CMX_NN, 0]>
        %5 = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
        %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%5 : memref<1xui32, @Register>) outputs(%4 : memref<1xui32, [@CMX_NN, 0]>) -> memref<1xui32, [@CMX_NN, 0]>

        %7 = VPUIP.NNDMA inputs(%3 : memref<1x512xf16, @DDR>) outputs(%arg1 : memref<1x512xf16>) -> memref<1x512xf16>

        %8 = VPURT.DeclareBuffer "CMX_NN" [0] <12> -> memref<1xui32, [@CMX_NN, 0]>
        %9 = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
        %10 = VPUIP.NNDMA {port = 0 : i64} inputs(%9 : memref<1xui32, @Register>) outputs(%8 : memref<1xui32, [@CMX_NN, 0]>) -> memref<1xui32, [@CMX_NN, 0]>

        async.yield %arg1 : memref<1x512xf16>
    }

    %4 = async.await %f1 : !async.value<memref<1x512xf16>>
    return %4 : memref<1x512xf16>

    // CHECK-DAG:   [[VAR0:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x512xf16, @DDR>
    // CHECK-DAG:   [[VAR1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui32, [@CMX_NN, 0]>
    // CHECK-DAG:   [[VAR2:%.*]] = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>
    // CHECK-DAG:   [[VAR3:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <12> -> memref<1xui32, [@CMX_NN, 0]>
    // CHECK-DAG:   [[VAR4:%.*]] = VPURT.DeclareBuffer "Register" <545390780> -> memref<1xui32, @Register>

    // CHECK:       VPURT.Task
    // CHECK-SAME:      cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64
    // CHECK-NEXT:  [[NNDMA1:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR2]] : memref<1xui32, @Register>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1xui32, [@CMX_NN, 0]>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      cycleBegin = 1 : i64, cycleCost = 36 : i64, cycleEnd = 37 : i64
    // CHECK-NEXT:  [[NNDMA2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x512xf16, @DDR>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x512xf16>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      cycleBegin = 37 : i64, cycleCost = 1 : i64, cycleEnd = 38 : i64
    // CHECK-NEXT:  [[NNDMA3:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR4]] : memref<1xui32, @Register>)
    // CHECK-SAME:      outputs([[VAR3]] : memref<1xui32, [@CMX_NN, 0]>)

    // CHECK:  return %arg1 : memref<1x512xf16>
}
