// RUN: vpux-opt --split-input-file --optimize-parallel-copies %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @OptimizeParallelNonConstCopies(
        %input: memref<1x16x112x112xf32, #NHWC>,
        %output: memref<1x32x112x112xf16, #NHWC>)
         -> memref<1x32x112x112xf16, #NHWC>{
    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    %2 = memref.alloc() : memref<1x32x112x112xf16, #NHWC, @DDR>

    %3 = IERT.Convert
        inputs(%input : memref<1x16x112x112xf32, #NHWC>)
        outputs(%0 : memref<1x16x112x112xf16, #NHWC, @DDR>)
        -> memref<1x16x112x112xf16, #NHWC, @DDR>
    %4 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %5 = IERT.Copy
            inputs(%3 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%4 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %6 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %7 = IERT.ReLU
            inputs(%5: memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%6 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %8 = IERT.SubView %2 [0, 0, 0, 0] [1, 16, 112, 112] :
        memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    %9 = IERT.Copy
            inputs(%7 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%8 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>

    %10 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %11 = IERT.Copy
            inputs(%3 : memref<1x16x112x112xf16, #NHWC, @DDR>)
            outputs(%10 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %12 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %13 = IERT.ReLU
            inputs(%11: memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%12 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %14 = IERT.SubView %2 [0, 16, 0, 0] [1, 16, 112, 112] :
        memref<1x32x112x112xf16, #NHWC, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
    %15 = IERT.Copy
            inputs(%13 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
            outputs(%14 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>

    %16 = IERT.ConcatView
        inputs(%9, %15 :
            memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>,
            memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}, @DDR>
        )
        outputs(%2 : memref<1x32x112x112xf16, #NHWC, @DDR>)
        -> memref<1x32x112x112xf16, #NHWC, @DDR>

    %17 = IERT.Copy inputs(%16 : memref<1x32x112x112xf16, #NHWC, @DDR>)
        outputs(%output : memref<1x32x112x112xf16, #NHWC>)
        -> memref<1x32x112x112xf16, #NHWC>

    return %17 : memref<1x32x112x112xf16, #NHWC>

}

// CHECK-LABEL: func @OptimizeParallelNonConstCopies

// CHECK: [[VAR0:%.*]] =  IERT.Convert inputs(%arg0 : memref<1x16x112x112xf32, #NHWC>)
// CHECK: [[VAR1:%.*]] =  IERT.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, #NHWC, @DDR>)
// CHECK: [[VAR2:%.*]] =  IERT.ReLU inputs([[VAR1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK: [[VAR3:%.*]] =  IERT.SubView
// CHECK-SAME: [0, 0, 0, 0] [1, 16, 112, 112]
// CHECK: [[VAR4:%.*]] =  IERT.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// CHECK-NOT: IERT.COPY
// CHECK: [[VAR5:%.*]] =  IERT.ReLU inputs([[VAR1]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
// CHECK: [[VAR6:%.*]] =  IERT.SubView
// CHECK-SAME: [0, 16, 0, 0] [1, 16, 112, 112]
// CHECK: [[VAR7:%.*]] =  IERT.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)

// CHECK: [[VAR8:%.*]] =  IERT.ConcatView inputs([[VAR4]], [[VAR7]] : 
// CHECK: [[VAR9:%.*]] =  IERT.Copy inputs([[VAR8]] : memref<1x32x112x112xf16, #NHWC, @DDR>)
