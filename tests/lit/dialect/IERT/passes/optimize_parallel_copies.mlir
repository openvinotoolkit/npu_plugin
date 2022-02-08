// RUN: vpux-opt --split-input-file --optimize-parallel-copies %s | FileCheck %s

func @OptimizeParallelNonConstCopies(
        %input: memref<1x16x112x112xf16>,
        %output1: memref<1x16x112x112xf16, @DDR>,
        %output2: memref<1x16x112x112xf16, @DDR>)
         -> (memref<1x16x112x112xf16, @DDR>, memref<1x16x112x112xf16, @DDR>){
    %0 = memref.alloc() : memref<1x16x112x112xf16, @DDR>

    %1 = IERT.Convert
        inputs(%input : memref<1x16x112x112xf16>)
        outputs(%0 : memref<1x16x112x112xf16, @DDR>)
        -> memref<1x16x112x112xf16, @DDR>
    %2 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %3 = IERT.Copy
            inputs(%1 : memref<1x16x112x112xf16, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, @CMX_NN>)
             -> memref<1x16x112x112xf16, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %5 = IERT.ReLU
            inputs(%3 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%4 : memref<1x16x112x112xf16, @CMX_NN>)
            -> memref<1x16x112x112xf16, @CMX_NN>
    %6 = IERT.Copy
            inputs(%5 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, @DDR>)
            -> memref<1x16x112x112xf16, @DDR>

    %7 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %8 = IERT.Copy
            inputs(%1 : memref<1x16x112x112xf16, @DDR>)
            outputs(%7 : memref<1x16x112x112xf16, @CMX_NN>)
             -> memref<1x16x112x112xf16, @CMX_NN>
    %9 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %10 = IERT.ReLU
            inputs(%8 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%9 : memref<1x16x112x112xf16, @CMX_NN>)
            -> memref<1x16x112x112xf16, @CMX_NN>
    %11 = IERT.Copy
            inputs(%10 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, @DDR>)
            -> memref<1x16x112x112xf16, @DDR>

    return %6, %11 : memref<1x16x112x112xf16, @DDR>, memref<1x16x112x112xf16, @DDR>

}

// CHECK-LABEL: func @OptimizeParallelNonConstCopies

// CHECK: [[VAR0:%.*]] =  IERT.Convert inputs(%arg0 : memref<1x16x112x112xf16>)
// CHECK: [[VAR1:%.*]] =  IERT.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, @DDR>)
// CHECK: [[VAR2:%.*]] =  IERT.ReLU inputs([[VAR1]] : memref<1x16x112x112xf16, @CMX_NN>)
// CHECK: [[VAR3:%.*]] =  IERT.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, @CMX_NN>)

// CHECK-NOT: IERT.COPY
// CHECK: [[VAR4:%.*]] =  IERT.ReLU inputs([[VAR1]] : memref<1x16x112x112xf16, @CMX_NN>)
// CHECK: [[VAR5:%.*]] =  IERT.Copy inputs([[VAR4]] : memref<1x16x112x112xf16, @CMX_NN>)

func @OptimizeParallelConstCopies(
        %output1: memref<1x16x112x112xf16, @DDR>,
        %output2: memref<1x16x112x112xf16, @DDR>)
         -> (memref<1x16x112x112xf16, @DDR>, memref<1x16x112x112xf16, @DDR>){
    %0 = const.Declare memref<1x16x112x112xf16, @DDR> = #const.Content<dense<1.000000e+00> : tensor<1x16x112x112xf16>>
    %1 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %2 = IERT.Copy
            inputs(%0 : memref<1x16x112x112xf16, @DDR>)
            outputs(%1 : memref<1x16x112x112xf16, @CMX_NN>)
             -> memref<1x16x112x112xf16, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %5 = IERT.ReLU
            inputs(%2 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%4 : memref<1x16x112x112xf16, @CMX_NN>)
            -> memref<1x16x112x112xf16, @CMX_NN>
    %6 = IERT.Copy
            inputs(%5 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, @DDR>)
            -> memref<1x16x112x112xf16, @DDR>

    %7 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %8 = IERT.Copy
            inputs(%0 : memref<1x16x112x112xf16, @DDR>)
            outputs(%7 : memref<1x16x112x112xf16, @CMX_NN>)
             -> memref<1x16x112x112xf16, @CMX_NN>
    %9 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %10 = IERT.ReLU
            inputs(%8 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%9 : memref<1x16x112x112xf16, @CMX_NN>)
            -> memref<1x16x112x112xf16, @CMX_NN>
    %11 = IERT.Copy
            inputs(%10 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, @DDR>)
            -> memref<1x16x112x112xf16, @DDR>

    return %6, %11 : memref<1x16x112x112xf16, @DDR>, memref<1x16x112x112xf16, @DDR>

}

// CHECK-LABEL: func @OptimizeParallelConstCopies

// CHECK: [[VAR0:%.*]] =  const.Declare memref<1x16x112x112xf16, @DDR>
// CHECK: [[VAR1:%.*]] =  IERT.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, @DDR>)
// CHECK: [[VAR2:%.*]] =  IERT.ReLU inputs([[VAR1]] : memref<1x16x112x112xf16, @CMX_NN>)
// CHECK: [[VAR3:%.*]] =  IERT.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, @CMX_NN>)

// CHECK: [[VAR4:%.*]] =  IERT.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, @DDR>)
// CHECK: [[VAR5:%.*]] =  IERT.ReLU inputs([[VAR4]] : memref<1x16x112x112xf16, @CMX_NN>)
// CHECK: [[VAR6:%.*]] =  IERT.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, @CMX_NN>)

func @OptimizeParallelSubViewPatternCopies(
        %input: memref<1x16x112x113xf16>,
        %output1: memref<1x16x112x112xf16, @DDR>,
        %output2: memref<1x16x112x112xf16, @DDR>)
         -> (memref<1x16x112x112xf16, @DDR>, memref<1x16x112x112xf16, @DDR>){
    %0 = memref.alloc() : memref<1x16x112x113xf16, @DDR>

    %1 = IERT.Convert
        inputs(%input : memref<1x16x112x113xf16>)
        outputs(%0 : memref<1x16x112x113xf16, @DDR>)
        -> memref<1x16x112x113xf16, @DDR>
    %2 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %3 = IERT.SubView %1 [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, @DDR> to memref<1x16x112x112xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [202496, 12656, 113, 1]}, @DDR>
    %4 = IERT.Copy
            inputs(%3 : memref<1x16x112x112xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [202496, 12656, 113, 1]}, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, @CMX_NN>)
             -> memref<1x16x112x112xf16, @CMX_NN>
    %5 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %6 = IERT.ReLU
            inputs(%4 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%5 : memref<1x16x112x112xf16, @CMX_NN>)
            -> memref<1x16x112x112xf16, @CMX_NN>
    %7 = IERT.Copy
            inputs(%6 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, @DDR>)
            -> memref<1x16x112x112xf16, @DDR>

    %8 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %9 = IERT.SubView %1 [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, @DDR> to memref<1x16x112x112xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [202496, 12656, 113, 1]}, @DDR>
    %10 = IERT.Copy
            inputs(%9 : memref<1x16x112x112xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [202496, 12656, 113, 1]}, @DDR>)
            outputs(%8 : memref<1x16x112x112xf16, @CMX_NN>)
             -> memref<1x16x112x112xf16, @CMX_NN>
    %11 = memref.alloc() : memref<1x16x112x112xf16, @CMX_NN>
    %12 = IERT.ReLU
            inputs(%10 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%11 : memref<1x16x112x112xf16, @CMX_NN>)
            -> memref<1x16x112x112xf16, @CMX_NN>
    %13 = IERT.Copy
            inputs(%12 : memref<1x16x112x112xf16, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, @DDR>)
            -> memref<1x16x112x112xf16, @DDR>

    return %7, %13 : memref<1x16x112x112xf16, @DDR>, memref<1x16x112x112xf16, @DDR>

}

// CHECK-LABEL: func @OptimizeParallelSubViewPatternCopies

// CHECK: [[VAR0:%.*]] =  IERT.Convert inputs(%arg0 : memref<1x16x112x113xf16>)
// CHECK: [[VAR1:%.*]] =  IERT.SubView [[VAR0]] [0, 0, 0, 0] [1, 16, 112, 112]
// CHECK: [[VAR2:%.*]] =  IERT.Copy
// CHECK-SAME inputs([[VAR1]] : memref<1x16x112x112xf16, {order = #NCHW, strides = [202496, 12656, 113, 1]}, @DDR>)
// CHECK: [[VAR3:%.*]] =  IERT.ReLU inputs([[VAR2]] : memref<1x16x112x112xf16, @CMX_NN>)
// CHECK: [[VAR4:%.*]] =  IERT.Copy inputs([[VAR3]] : memref<1x16x112x112xf16, @CMX_NN>)

// CHECK-NOT: IERT.SubView
// CHECK-NOT: IERT.COPY
// CHECK: [[VAR5:%.*]] =  IERT.ReLU inputs([[VAR2]] : memref<1x16x112x112xf16, @CMX_NN>)
// CHECK: [[VAR6:%.*]] =  IERT.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, @CMX_NN>)
