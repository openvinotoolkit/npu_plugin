// RUN: vpux-opt --cmx-tiling  %s | FileCheck %s

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 1048576 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

//
// This conv is expected to have 8 tiles with this CMX config... will do check directives for first 2 and last 2 tiles
//

func @conv1(%arg0: memref<1x32x101x101xf32>, %arg1: memref<64x32x3x3xf32>, %arg2: memref<1x64x1x1xf32>, %arg3: memref<1x64x99x99xf32>)
        -> memref<1x64x99x99xf32> {
    %0 = memref.alloc() : memref<1x64x99x99xf32>
    %1 = IERT.Convolution
        { strides = [1:i32, 1:i32], pads_begin = [0:i32, 0:i32], pads_end = [0:i32, 0:i32], dilations = [1:i32, 1:i32] }
        inputs(%arg0: memref<1x32x101x101xf32>, %arg1: memref<64x32x3x3xf32>, %arg2 : memref<1x64x1x1xf32>)
        outputs(%0 : memref<1x64x99x99xf32>) -> memref<1x64x99x99xf32>
    %2 = IERT.Copy inputs(%1 : memref<1x64x99x99xf32>) outputs(%arg3: memref<1x64x99x99xf32>) -> memref<1x64x99x99xf32>
    return %2 : memref<1x64x99x99xf32>
}

// CHECK:       [[VAR0:%.*]] = memref.alloc() : memref<1x64x99x99xf32>

//
// Tile #0
//

// CHECK:       [[VAR1:%.*]] = memref.subview %arg0[0, 0, 0, 0] [1, 32, 35, 51] [1, 1, 1, 1]
// CHECK:       [[VAR5:%.*]] = memref.alloc() : memref<1x32x35x51xf32{{.*}}>
// CHECK:       [[VAR9:%.*]] = IERT.Copy inputs([[VAR1]] : memref<1x32x35x51xf32{{.*}}>) outputs([[VAR5]] : memref<1x32x35x51xf32{{.*}}>)

// CHECK:       [[VAR2:%.*]] = memref.subview %arg1[0, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
// CHECK:       [[VAR6:%.*]] = memref.alloc() : memref<32x32x3x3xf32{{.*}}>
// CHECK:       [[VAR10:%.*]] = IERT.Copy inputs([[VAR2]] : memref<32x32x3x3xf32{{.*}}>) outputs([[VAR6]] : memref<32x32x3x3xf32{{.*}}>)

// CHECK:       [[VAR3:%.*]] = memref.subview %arg2[0, 0, 0, 0] [1, 32, 1, 1] [1, 1, 1, 1]
// CHECK:       [[VAR7:%.*]] = memref.alloc() : memref<1x32x1x1xf32{{.*}}>
// CHECK:       [[VAR11:%.*]] = IERT.Copy inputs([[VAR3]] : memref<1x32x1x1xf32{{.*}}>) outputs([[VAR7]] : memref<1x32x1x1xf32{{.*}}>)

// CHECK:       [[VAR8:%.*]] = memref.alloc() : memref<1x32x33x49xf32{{.*}}>
// CHECK:       [[VAR12:%.*]] = IERT.Convolution
// CHECK-SAME:      dilations = [1 : i32, 1 : i32]
// CHECK-SAME:      pads_begin = [0 : i32, 0 : i32]
// CHECK-SAME:      pads_end = [0 : i32, 0 : i32]
// CHECK-SAME:      strides = [1 : i32, 1 : i32]}
// CHECK-SAME:      inputs([[VAR9]] : memref<1x32x35x51xf32{{.*}}>, [[VAR10]] : memref<32x32x3x3xf32{{.*}}>, [[VAR11]] : memref<1x32x1x1xf32{{.*}}>) outputs([[VAR8]] : memref<1x32x33x49xf32{{.*}}>

// CHECK:       [[VAR4:%.*]] = memref.subview [[VAR0]][0, 0, 0, 0] [1, 32, 33, 49] [1, 1, 1, 1]
// CHECK:       [[VAR13:%.*]] = IERT.Copy inputs([[VAR12]] : memref<1x32x33x49xf32{{.*}}>) outputs([[VAR4]] : memref<1x32x33x49xf32{{.*}}>)

//
// Tile #1
//

// CHECK:       [[VAR21:%.*]] = memref.subview %arg0[0, 0, 0, 0] [1, 32, 35, 51] [1, 1, 1, 1]
// CHECK:       [[VAR25:%.*]] = memref.alloc() : memref<1x32x35x51xf32{{.*}}>
// CHECK:       [[VAR29:%.*]] = IERT.Copy inputs([[VAR21]] : memref<1x32x35x51xf32{{.*}}>) outputs([[VAR25]] : memref<1x32x35x51xf32{{.*}}>)

// CHECK:       [[VAR22:%.*]] = memref.subview %arg1[32, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
// CHECK:       [[VAR26:%.*]] = memref.alloc() : memref<32x32x3x3xf32{{.*}}>
// CHECK:       [[VAR30:%.*]] = IERT.Copy inputs([[VAR22]] : memref<32x32x3x3xf32{{.*}}>) outputs([[VAR26]] : memref<32x32x3x3xf32{{.*}}>)

// CHECK:       [[VAR23:%.*]] = memref.subview %arg2[0, 32, 0, 0] [1, 32, 1, 1] [1, 1, 1, 1]
// CHECK:       [[VAR27:%.*]] = memref.alloc() : memref<1x32x1x1xf32{{.*}}>
// CHECK:       [[VAR31:%.*]] = IERT.Copy inputs([[VAR23]] : memref<1x32x1x1xf32{{.*}}>) outputs([[VAR27]] : memref<1x32x1x1xf32{{.*}}>)

// CHECK:       [[VAR28:%.*]] = memref.alloc() : memref<1x32x33x49xf32{{.*}}>
// CHECK:       [[VAR32:%.*]] = IERT.Convolution
// CHECK-SAME:      dilations = [1 : i32, 1 : i32]
// CHECK-SAME:      pads_begin = [0 : i32, 0 : i32]
// CHECK-SAME:      pads_end = [0 : i32, 0 : i32]
// CHECK-SAME:      strides = [1 : i32, 1 : i32]}
// CHECK-SAME:      inputs([[VAR29]] : memref<1x32x35x51xf32{{.*}}>, [[VAR30]] : memref<32x32x3x3xf32{{.*}}>, [[VAR31]] : memref<1x32x1x1xf32{{.*}}>) outputs([[VAR28]] : memref<1x32x33x49xf32{{.*}}>

// CHECK:       [[VAR24:%.*]] = memref.subview [[VAR0]][0, 32, 0, 0] [1, 32, 33, 49] [1, 1, 1, 1]
// CHECK:       [[VAR33:%.*]] = IERT.Copy inputs([[VAR32]] : memref<1x32x33x49xf32{{.*}}>) outputs([[VAR24]] : memref<1x32x33x49xf32{{.*}}>)

//
// Tile #6
//

// CHECK:       [[VAR41:%.*]] = memref.subview %arg0[0, 0, 65, 48] [1, 32, 35, 52] [1, 1, 1, 1]
// CHECK:       [[VAR45:%.*]] = memref.alloc() : memref<1x32x35x52xf32{{.*}}>
// CHECK:       [[VAR49:%.*]] = IERT.Copy inputs([[VAR41]] : memref<1x32x35x52xf32{{.*}}>) outputs([[VAR45]] : memref<1x32x35x52xf32{{.*}}>)

// CHECK:       [[VAR42:%.*]] = memref.subview %arg1[0, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
// CHECK:       [[VAR46:%.*]] = memref.alloc() : memref<32x32x3x3xf32{{.*}}>
// CHECK:       [[VAR50:%.*]] = IERT.Copy inputs([[VAR42]] : memref<32x32x3x3xf32{{.*}}>) outputs([[VAR46]] : memref<32x32x3x3xf32{{.*}}>)

// CHECK:       [[VAR43:%.*]] = memref.subview %arg2[0, 0, 0, 0] [1, 32, 1, 1] [1, 1, 1, 1]
// CHECK:       [[VAR47:%.*]] = memref.alloc() : memref<1x32x1x1xf32{{.*}}>
// CHECK:       [[VAR51:%.*]] = IERT.Copy inputs([[VAR43]] : memref<1x32x1x1xf32{{.*}}>) outputs([[VAR47]] : memref<1x32x1x1xf32{{.*}}>)

// CHECK:       [[VAR48:%.*]] = memref.alloc() : memref<1x32x33x50xf32{{.*}}>
// CHECK:       [[VAR52:%.*]] = IERT.Convolution
// CHECK-SAME:      dilations = [1 : i32, 1 : i32]
// CHECK-SAME:      pads_begin = [0 : i32, 0 : i32]
// CHECK-SAME:      pads_end = [0 : i32, 0 : i32]
// CHECK-SAME:      strides = [1 : i32, 1 : i32]}
// CHECK-SAME:      inputs([[VAR49]] : memref<1x32x35x52xf32{{.*}}>, [[VAR50]] : memref<32x32x3x3xf32{{.*}}>, [[VAR51]] : memref<1x32x1x1xf32{{.*}}>) outputs([[VAR48]] : memref<1x32x33x50xf32{{.*}}>

// CHECK:       [[VAR44:%.*]] = memref.subview [[VAR0]][0, 0, 66, 49] [1, 32, 33, 50] [1, 1, 1, 1]
// CHECK:       [[VAR53:%.*]] = IERT.Copy inputs([[VAR52]] : memref<1x32x33x50xf32{{.*}}>) outputs([[VAR44]] : memref<1x32x33x50xf32{{.*}}>)

//
// Tile #7
//

// CHECK:       [[VAR61:%.*]] = memref.subview %arg0[0, 0, 65, 48] [1, 32, 35, 52] [1, 1, 1, 1]
// CHECK:       [[VAR65:%.*]] = memref.alloc() : memref<1x32x35x52xf32{{.*}}>
// CHECK:       [[VAR69:%.*]] = IERT.Copy inputs([[VAR61]] : memref<1x32x35x52xf32{{.*}}>) outputs([[VAR65]] : memref<1x32x35x52xf32{{.*}}>)

// CHECK:       [[VAR62:%.*]] = memref.subview %arg1[32, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
// CHECK:       [[VAR66:%.*]] = memref.alloc() : memref<32x32x3x3xf32{{.*}}>
// CHECK:       [[VAR70:%.*]] = IERT.Copy inputs([[VAR62]] : memref<32x32x3x3xf32{{.*}}>) outputs([[VAR66]] : memref<32x32x3x3xf32{{.*}}>)

// CHECK:       [[VAR63:%.*]] = memref.subview %arg2[0, 32, 0, 0] [1, 32, 1, 1] [1, 1, 1, 1]
// CHECK:       [[VAR67:%.*]] = memref.alloc() : memref<1x32x1x1xf32{{.*}}>
// CHECK:       [[VAR71:%.*]] = IERT.Copy inputs([[VAR63]] : memref<1x32x1x1xf32{{.*}}>) outputs([[VAR67]] : memref<1x32x1x1xf32{{.*}}>)

// CHECK:       [[VAR68:%.*]] = memref.alloc() : memref<1x32x33x50xf32{{.*}}>
// CHECK:       [[VAR72:%.*]] = IERT.Convolution
// CHECK-SAME:      dilations = [1 : i32, 1 : i32]
// CHECK-SAME:      pads_begin = [0 : i32, 0 : i32]
// CHECK-SAME:      pads_end = [0 : i32, 0 : i32]
// CHECK-SAME:      strides = [1 : i32, 1 : i32]}
// CHECK-SAME:      inputs([[VAR69]] : memref<1x32x35x52xf32{{.*}}>, [[VAR70]] : memref<32x32x3x3xf32{{.*}}>, [[VAR71]] : memref<1x32x1x1xf32{{.*}}>) outputs([[VAR68]] : memref<1x32x33x50xf32{{.*}}>

// CHECK:       [[VAR64:%.*]] = memref.subview [[VAR0]][0, 32, 66, 49] [1, 32, 33, 50] [1, 1, 1, 1]
// CHECK:       [[VAR73:%.*]] = IERT.Copy inputs([[VAR72]] : memref<1x32x33x50xf32{{.*}}>) outputs([[VAR64]] : memref<1x32x33x50xf32{{.*}}>)

//
// Concat
//

// CHECK:       [[VAR80:%.*]] = IERT.ConcatView
// CHECK-SAME:      inputs
// CHECK-SAME:          [[VAR13]]
// CHECK-SAME:          [[VAR33]]
// CHECK-SAME:          [[VAR53]]
// CHECK-SAME:          [[VAR73]]
// CHECK-SAME:      outputs([[VAR0]] : memref<1x64x99x99xf32>)

// CHECK:       [[VAR81:%.*]] = IERT.Copy inputs([[VAR80]] : memref<1x64x99x99xf32>) outputs(%arg3 : memref<1x64x99x99xf32>)
// CHECK:       return [[VAR81]] : memref<1x64x99x99xf32>
