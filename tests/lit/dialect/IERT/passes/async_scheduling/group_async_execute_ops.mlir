// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --group-async-execute-ops %s | FileCheck %s

// CHECK-LABEL: @MergeUPAAndDMA
func @MergeUPAAndDMA(%arg0: memref<1x3x1x1xui8>, %arg1: memref<1x3x1x1xf16>, %arg2: memref<1x3x1x1xf16>)
        -> (memref<1x3x1x1xf16>, memref<1x3x1x1xf16>) {
    %token, %results = async.execute
            -> !async.value<memref<1x3x1x1xf16, "DDR">>
            attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16 : i64} {
        %2 = IERT.StaticAlloc<0> -> memref<1x3x1x1xf16, "DDR">
        %3 = IERT.Convert inputs(%arg0 : memref<1x3x1x1xui8>) outputs(%2 : memref<1x3x1x1xf16, "DDR">) -> memref<1x3x1x1xf16, "DDR">
        async.yield %3 : memref<1x3x1x1xf16, "DDR">
    }
    async.await %token : !async.token

    %token_0, %results_1 = async.execute (%results as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>)
            -> !async.value<memref<1x3x1x1xf16, "DDR">>
            attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16 : i64} {
        %2 = IERT.StaticAlloc<64> -> memref<1x3x1x1xf16, "DDR">
        %3 = IERT.ReLU inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%2 : memref<1x3x1x1xf16, "DDR">) -> memref<1x3x1x1xf16, "DDR">
        async.yield %3 : memref<1x3x1x1xf16, "DDR">
    }
    async.await %token_0 : !async.token

    %token_2, %results_3 = async.execute (%results as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>)
            -> !async.value<memref<1x3x1x1xf16, "DDR">>
            attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16 : i64} {
        %2 = IERT.StaticAlloc<64> -> memref<1x3x1x1xf16, "DDR">
        %3 = IERT.ReLU inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%2 : memref<1x3x1x1xf16, "DDR">) -> memref<1x3x1x1xf16, "DDR">
        async.yield %3 : memref<1x3x1x1xf16, "DDR">
    }
    async.await %token_2 : !async.token

    %token_4, %results_5 = async.execute (%results_3 as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>)
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
        %2 = IERT.Copy inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%arg1 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %2 : memref<1x3x1x1xf16>
    }
    %0 = async.await %results_5 : !async.value<memref<1x3x1x1xf16>>

    %token_6, %results_7 = async.execute (%results_1 as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>)
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
      %2 = IERT.Copy inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%arg2 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
      async.yield %2 : memref<1x3x1x1xf16>
    }
    %1 = async.await %results_7 : !async.value<memref<1x3x1x1xf16>>

    return %0, %1 : memref<1x3x1x1xf16>, memref<1x3x1x1xf16>

    // CHECK:       [[T:%.+]], [[F:%.+]] = async.execute
    // CHECK:           IERT.StaticAlloc<0>
    // CHECK:           IERT.Convert
    // CHECK:       async.await [[T]]

    // CHECK:       [[T0:%.+]], [[F1:%.+]] = async.execute
    // CHECK:           [[VAR0:%.*]] = IERT.StaticAlloc<64> -> memref<1x3x1x1xf16, "DDR">
    // CHECK:           [[VAR1:%.*]] = IERT.ReLU
    // CHECK-SAME:          outputs([[VAR0]] : memref<1x3x1x1xf16, "DDR">)
    // CHECK:           [[VAR2:%.*]] = IERT.StaticAlloc<64> -> memref<1x3x1x1xf16, "DDR">
    // CHECK:           [[VAR3:%.*]] = IERT.ReLU
    // CHECK-SAME:          outputs([[VAR2]] : memref<1x3x1x1xf16, "DDR">)
    // CHECK:           async.yield [[VAR1]], [[VAR3]]
    // CHECK:       async.await [[T0]]

    // CHECK:       [[T1:%.+]], [[F2:%.+]]:2  = async.execute
    // CHECK:           [[VAR5:%.*]] = IERT.Copy
    // CHECK-SAME:          outputs(%arg1 : memref<1x3x1x1xf16>)
    // CHECK:           [[VAR7:%.*]] = IERT.Copy
    // CHECK-SAME:          outputs(%arg2 : memref<1x3x1x1xf16>)
    // CHECK:           async.yield [[VAR5]], [[VAR7]]
    // CHECK:       [[VAR8:%.*]] = async.await [[F2]]#0
    // CHECK:       [[VAR9:%.*]] = async.await [[F2]]#1

    // CHECK:       return [[VAR8]], [[VAR9]]
}

// -----

// CHECK-LABEL: @MergeDMAs
func @MergeDMAs(%arg0: memref<1x3x1x1xui8>, %arg1: memref<1x3x1x1xf16>, %arg2: memref<1x3x1x1xf16>)
        -> (memref<1x3x1x1xf16>, memref<1x3x1x1xf16>) {
    %token, %results = async.execute
            -> !async.value<memref<1x3x1x1xf16, "DDR">>
            attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16 : i64} {
        %2 = IERT.StaticAlloc<0> -> memref<1x3x1x1xf16, "DDR">
        %3 = IERT.Convert inputs(%arg0 : memref<1x3x1x1xui8>) outputs(%2 : memref<1x3x1x1xf16, "DDR">) -> memref<1x3x1x1xf16, "DDR">
        async.yield %3 : memref<1x3x1x1xf16, "DDR">
    }
    async.await %token : !async.token

    %token_4, %results_5 = async.execute (%results as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>)
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
        %2 = IERT.Copy inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%arg1 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %2 : memref<1x3x1x1xf16>
    }
    %0 = async.await %results_5 : !async.value<memref<1x3x1x1xf16>>

    %token_6, %results_7 = async.execute (%results as %arg3: !async.value<memref<1x3x1x1xf16, "DDR">>)
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
        %2 = IERT.Copy inputs(%arg3 : memref<1x3x1x1xf16, "DDR">) outputs(%arg2 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %2 : memref<1x3x1x1xf16>
    }
    %1 = async.await %results_7 : !async.value<memref<1x3x1x1xf16>>

    return %0, %1 : memref<1x3x1x1xf16>, memref<1x3x1x1xf16>

    // CHECK:       [[T:%.+]], [[F:%.+]] = async.execute
    // CHECK:           IERT.StaticAlloc<0>
    // CHECK:           IERT.Convert
    // CHECK:       async.await [[T]]

    // CHECK:       [[T1:%.+]], [[F2:%.+]]:2 = async.execute
    // CHECK:           [[VAR1:%.*]] = IERT.Copy
    // CHECK-SAME:          outputs(%arg1 : memref<1x3x1x1xf16>
    // CHECK:           [[VAR2:%.*]] = IERT.Copy
    // CHECK-SAME:          outputs(%arg2 : memref<1x3x1x1xf16>)
    // CHECK:           async.yield [[VAR1]], [[VAR2]]

    // CHECK:       [[VAR4:%.*]] = async.await [[F2]]#0
    // CHECK:       [[VAR5:%.*]] = async.await [[F2]]#1

    // CHECK:       return [[VAR4]], [[VAR5]]
}

// -----

// CHECK-LABEL: @TaskWithExclusiveUsers
func @TaskWithExclusiveUsers(%arg0: memref<16xf16>, %arg1: memref<16xf16>, %arg2: memref<16xf16>)
        -> (memref<16xf16>, memref<16xf16>) {
    %buf = IERT.StaticAlloc <0> -> memref<16xf16>

    %t0, %f0 = async.execute
            -> !async.value<memref<16xf16>>
            attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16} {
        %0 = IERT.ReLU inputs(%arg0 : memref<16xf16>) outputs(%buf : memref<16xf16>) -> memref<16xf16>
        async.yield %0 : memref<16xf16>
    }
    async.await %t0 : !async.token

    %t2, %f2 = async.execute
            -> !async.value<memref<16xf16>>
            attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16} {
        %2 = IERT.ReLU inputs(%arg0 : memref<16xf16>) outputs(%arg1 : memref<16xf16>) -> memref<16xf16>
        async.yield %2 : memref<16xf16>
    }
    %3 = async.await %f2 : !async.value<memref<16xf16>>

    %t4, %f4 = async.execute (%f0 as %0: !async.value<memref<16xf16>>)
            -> !async.value<memref<16xf16>>
            attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16} {
        %4 = IERT.ReLU inputs(%0 : memref<16xf16>) outputs(%arg2 : memref<16xf16>) -> memref<16xf16>
        async.yield %4 : memref<16xf16>
    }
    %5 = async.await %f4 : !async.value<memref<16xf16>>

    return %3, %5 : memref<16xf16>, memref<16xf16>

    // CHECK:       [[BUF:%.*]] = IERT.StaticAlloc

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK:           [[VAR0:%.*]] = IERT.ReLU
    // CHECK-SAME:          inputs(%arg0 : memref<16xf16>)
    // CHECK-SAME:          outputs([[BUF]] : memref<16xf16>)
    // CHECK:           async.yield [[VAR0]]
    // CHECK:       async.await [[T0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]]:2 = async.execute
    // CHECK-SAME:          ([[F0]] as [[VAR1:%.*]]: !async.value<memref<16xf16>>)
    // CHECK:           [[VAR2:%.*]] = IERT.ReLU
    // CHECK-SAME:          inputs(%arg0 : memref<16xf16>)
    // CHECK-SAME:          outputs(%arg1 : memref<16xf16>)
    // CHECK:           [[VAR3:%.*]] = IERT.ReLU
    // CHECK-SAME:          inputs([[VAR1]] : memref<16xf16>)
    // CHECK-SAME:          outputs(%arg2 : memref<16xf16>)
    // CHECK:           async.yield [[VAR2]], [[VAR3]]
    // CHECK:       [[VAR4:%.*]] = async.await [[F1]]#0
    // CHECK:       [[VAR5:%.*]] = async.await [[F1]]#1

    // CHECK:       return [[VAR4]], [[VAR5]]
}

// -----

// CHECK-LABEL: @MergeInputDMAs
func @MergeInputDMAs(%arg0: memref<1x3x1x1xf16>, %arg1: memref<1x3x1x1xf16>, %arg2: memref<1x3x1x1xf16>, %arg3: memref<1x3x1x1xf16>)
        -> (memref<1x3x1x1xf16>) {
    %token_0, %results_0 = async.execute
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
        %0 = IERT.Copy inputs(%arg0 : memref<1x3x1x1xf16>) outputs(%arg1 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %0 : memref<1x3x1x1xf16>
    }
    async.await %token_0 : !async.token

    %token_1, %results_1 = async.execute
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {IERT.executor = "DMA_NN", IERT.num_units = 1 : i64} {
        %1 = IERT.Copy inputs(%arg0 : memref<1x3x1x1xf16>) outputs(%arg2 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %1 : memref<1x3x1x1xf16>
    }
    async.await %token_1 : !async.token

    %token_2, %results_2 = async.execute (%results_0 as %0: !async.value<memref<1x3x1x1xf16>>, %results_1 as %1: !async.value<memref<1x3x1x1xf16>>)
            -> !async.value<memref<1x3x1x1xf16>>
            attributes {IERT.executor = "SHAVE_UPA", IERT.num_units = 16} {
        %2 = IERT.Add inputs(%0 : memref<1x3x1x1xf16>, %1 : memref<1x3x1x1xf16>) outputs(%arg3 : memref<1x3x1x1xf16>) -> memref<1x3x1x1xf16>
        async.yield %2 : memref<1x3x1x1xf16>
    }
    %res = async.await %results_2 : !async.value<memref<1x3x1x1xf16>>

    return %res : memref<1x3x1x1xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]]:2 = async.execute
    // CHECK:           [[VAR0:%.*]] = IERT.Copy
    // CHECK-SAME:          outputs(%arg1 : memref<1x3x1x1xf16>)
    // CHECK:           [[VAR1:%.*]] = IERT.Copy
    // CHECK-SAME:          outputs(%arg2 : memref<1x3x1x1xf16>)
    // CHECK:           async.yield [[VAR0]], [[VAR1]]
    // CHECK:       async.await [[T0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:               ([[F0]]#0 as [[VAR2:%.*]]: !async.value<memref<1x3x1x1xf16>>, [[F0]]#1 as [[VAR3:%.*]]: !async.value<memref<1x3x1x1xf16>>)
    // CHECK:           [[VAR4:%.*]] = IERT.Add
    // CHECK:               inputs([[VAR2]] : memref<1x3x1x1xf16>, [[VAR3]] : memref<1x3x1x1xf16>)
    // CHECK:       [[VAR5:%.*]] = async.await [[F1]]

    // CHECK:       return [[VAR5]]
}
