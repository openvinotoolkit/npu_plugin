// RUN: vpux-opt --move-view-ops-into-async-regions %s | FileCheck %s

#in_tile_1 = affine_map<(d0) -> (d0 + 50)>

#out_buf_tile_0 = affine_map<(d0, d1) -> (d0 * 10 + d1)>
#out_buf_tile_1 = affine_map<(d0, d1) -> (d0 * 10 + d1 + 50)>

func @main(%in : memref<10x10xf16>, %out_buf : memref<10x10xf16>) -> memref<10x10xf16> {
    %in_flat = IERT.GenericReshape inputs(%in : memref<10x10xf16>) -> memref<100xf16>

    %in_tile_0 = memref.subview %in_flat [ 0][50][1] : memref<100xf16> to memref<50xf16>
    %in_tile_1 = memref.subview %in_flat [50][50][1] : memref<100xf16> to memref<50xf16, #in_tile_1>

    %out_buf_tile_0 = memref.subview %out_buf [0, 0][5, 10][1, 1] : memref<10x10xf16> to memref<5x10xf16, #out_buf_tile_0>
    %out_buf_tile_1 = memref.subview %out_buf [5, 0][5, 10][1, 1] : memref<10x10xf16> to memref<5x10xf16, #out_buf_tile_1>

    // Tile 0

    %temp_buf_0 = memref.alloc() : memref<50xf16>

    %temp_token_0, %temp_future_0 = async.execute -> !async.value<memref<50xf16>> {
        %temp_0 = IERT.ReLU
            inputs(
                %in_tile_0 : memref<50xf16>
            ) outputs(
                %temp_buf_0 : memref<50xf16>
            ) -> memref<50xf16>
        async.yield %temp_0 : memref<50xf16>
    }
    %temp_0 = async.await %temp_future_0 : !async.value<memref<50xf16>>

    %temp_0_unflat = IERT.GenericReshape inputs(%temp_0 : memref<50xf16>) -> memref<5x10xf16>

    %out_tile_token_0, %out_tile_future_0 = async.execute -> !async.value<memref<5x10xf16, #out_buf_tile_0>> {
        %out_tile_0 = IERT.Copy
            inputs(
                %temp_0_unflat : memref<5x10xf16>
            ) outputs(
                %out_buf_tile_0 : memref<5x10xf16, #out_buf_tile_0>
            ) -> memref<5x10xf16, #out_buf_tile_0>
        async.yield %out_tile_0 : memref<5x10xf16, #out_buf_tile_0>
    }
    %out_tile_0 = async.await %out_tile_future_0 : !async.value<memref<5x10xf16, #out_buf_tile_0>>

    memref.dealloc %temp_buf_0 : memref<50xf16>

    // Tile 1

    %temp_buf_1 = memref.alloc() : memref<50xf16>

    %temp_token_1, %temp_future_1 = async.execute -> !async.value<memref<50xf16>> {
        %temp_1 = IERT.ReLU
            inputs(
                %in_tile_1 : memref<50xf16, #in_tile_1>
            ) outputs(
                %temp_buf_1 : memref<50xf16>
            ) -> memref<50xf16>
        async.yield %temp_1 : memref<50xf16>
    }
    %temp_1 = async.await %temp_future_1 : !async.value<memref<50xf16>>

    %temp_1_unflat = IERT.GenericReshape inputs(%temp_1 : memref<50xf16>) -> memref<5x10xf16>

    %out_tile_token_1, %out_tile_future_1 = async.execute -> !async.value<memref<5x10xf16, #out_buf_tile_1>> {
        %out_tile_1 = IERT.Copy
            inputs(
                %temp_1_unflat : memref<5x10xf16>
            ) outputs(
                %out_buf_tile_1 : memref<5x10xf16, #out_buf_tile_1>
            ) -> memref<5x10xf16, #out_buf_tile_1>
        async.yield %out_tile_1 : memref<5x10xf16, #out_buf_tile_1>
    }
    %out_tile_1 = async.await %out_tile_future_1 : !async.value<memref<5x10xf16, #out_buf_tile_1>>

    memref.dealloc %temp_buf_1 : memref<50xf16>

    // Concat

    %out = IERT.ConcatView
        inputs(
            %out_tile_0, %out_tile_1 : memref<5x10xf16, #out_buf_tile_0>, memref<5x10xf16, #out_buf_tile_1>
        ) outputs(
            %out_buf : memref<10x10xf16>
        ) -> memref<10x10xf16>

    return %out : memref<10x10xf16>
}

// CHECK:   func @main([[in:%.*]]: memref<10x10xf16>, [[out_buf:%.*]]: memref<10x10xf16>)

// CHECK:       [[temp_buf_0:%.*]] = memref.alloc()
// CHECK:       [[temp_token_0:%.*]], [[temp_future_0:%.*]] = async.execute
// CHECK:           [[in_flat_0:%.*]] = IERT.GenericReshape inputs([[in]] : memref<10x10xf16>)
// CHECK:           [[in_tile_0:%.*]] = memref.subview [[in_flat_0]][0] [50] [1]
// CHECK:           [[inner_temp_0:%.*]] = IERT.ReLU
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[in_tile_0]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[temp_buf_0]]
// CHECK-SAME:          )
// CHECK:           async.yield [[inner_temp_0]]
// CHECK:       [[temp_0:%.*]] = async.await [[temp_future_0]]
// CHECK:       [[out_tile_token_0:%.*]], [[out_tile_future_0:%.*]] = async.execute
// CHECK:           [[out_buf_tile_0:%.*]] = memref.subview [[out_buf]][0, 0] [5, 10] [1, 1]
// CHECK:           [[temp_0_unflat:%.*]] = IERT.GenericReshape inputs([[temp_0]] : memref<50xf16>)
// CHECK:           [[out_tile_0:%.*]] = IERT.Copy
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[temp_0_unflat]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[out_buf_tile_0]]
// CHECK-SAME:          )
// CHECK:           async.yield [[out_tile_0]]
// CHECK:       [[out_tile_0:%.*]] = async.await [[out_tile_future_0]]
// CHECK:       memref.dealloc [[temp_buf_0]]

// CHECK:       [[temp_buf_1:%.*]] = memref.alloc()
// CHECK:       [[temp_token_1:%.*]], [[temp_future_1:%.*]] = async.execute
// CHECK:           [[in_flat_1:%.*]] = IERT.GenericReshape inputs([[in]] : memref<10x10xf16>)
// CHECK:           [[in_tile_1:%.*]] = memref.subview [[in_flat_1]][50] [50] [1]
// CHECK:           [[inner_temp_1:%.*]] = IERT.ReLU
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[in_tile_1]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[temp_buf_1]]
// CHECK-SAME:          )
// CHECK:           async.yield [[inner_temp_1]]
// CHECK:       [[temp_1:%.*]] = async.await [[temp_future_1]]
// CHECK:       [[out_tile_token_1:%.*]], [[out_tile_future_1:%.*]] = async.execute
// CHECK:           [[out_buf_tile_1:%.*]] = memref.subview [[out_buf]][5, 0] [5, 10] [1, 1]
// CHECK:           [[temp_1_unflat:%.*]] = IERT.GenericReshape inputs([[temp_1]] : memref<50xf16>)
// CHECK:           [[out_tile_1:%.*]] = IERT.Copy
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[temp_1_unflat]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[out_buf_tile_1]]
// CHECK-SAME:          )
// CHECK:           async.yield [[out_tile_1]]
// CHECK:       [[out_tile_1:%.*]] = async.await [[out_tile_future_1]]
// CHECK:       memref.dealloc [[temp_buf_1]]

// CHECK:       [[out:%.*]] = IERT.ConcatView
// CHECK-SAME:      inputs(
// CHECK-SAME:          [[out_tile_0]], [[out_tile_1]]
// CHECK-SAME:      ) outputs(
// CHECK-SAME:          [[out_buf]]
// CHECK-SAME:      )
// CHECK:       return [[out]]
