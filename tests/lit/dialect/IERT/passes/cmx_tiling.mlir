// RUN: vpux-opt --cmx-tiling --split-input-file  %s | FileCheck %s

module {

  IERT.RunTimeResources availableMemory :  {
    IERT.MemoryResource 1048576 bytes
    IERT.MemoryResource 1048576 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
  } usedMemory :  {
  } executors :  {
    IERT.ExecutorResource 4 of "NCE_Cluster"  {
    IERT.ExecutorResource 5 of "NCE_PerClusterDPU"
    }
  }

//this conv is expected to have 8 tiles with this CMX config... will do check directives for first 2 and last 2 tiles
  //CHECK-LABEL: @conv1
  func @conv1(%arg0: memref<1x32x101x101xf32>, %arg1: memref<64x32x3x3xf32>, %arg2: memref<64xf32>, %arg3: memref<1x64x99x99xf32>) -> memref<1x64x99x99xf32>
  {
    %0 = memref.alloc() : memref<1x64x99x99xf32>
    %1 = IERT.Convolution {strides = [1:i32,1:i32], pads_begin = [0:i32,0:i32], pads_end = [0:i32,0:i32], dilations = [1:i32,1:i32]}
                          inputs(%arg0: memref<1x32x101x101xf32>, %arg1: memref<64x32x3x3xf32>, %arg2 : memref<64xf32>)
                          outputs(%0 : memref<1x64x99x99xf32>) -> memref<1x64x99x99xf32>
    %2 = IERT.Copy inputs(%1 : memref<1x64x99x99xf32>) outputs(%arg3: memref<1x64x99x99xf32>) -> memref<1x64x99x99xf32>
    return %2 : memref<1x64x99x99xf32>
  }

  //CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x64x99x99xf32>

  //CHECK-DAG: [[VAR1:%.*]] = memref.subview %arg0[0, 0, 0, 0] [1, 32, 35, 51] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR2:%.*]] = memref.subview %arg1[0, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR3:%.*]] = memref.subview %arg2[0] [32] [1]
  //CHECK-NEXT: [[VAR4:%.*]] = memref.subview [[VAR0]][0, 0, 0, 0] [1, 32, 33, 49] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR5:%.*]] = memref.alloc() : memref<1x32x35x51xf32{{.*}}>
  //CHECK-NEXT: [[VAR6:%.*]] = memref.alloc() : memref<32x32x3x3xf32{{.*}}>
  //CHECK-NEXT: [[VAR7:%.*]] = memref.alloc() : memref<32xf32{{.*}}>
  //CHECK-NEXT: [[VAR8:%.*]] = memref.alloc() : memref<1x32x33x49xf32{{.*}}>
  //CHECK-NEXT: [[VAR9:%.*]] = IERT.Copy inputs([[VAR1]] : memref<1x32x35x51xf32{{.*}}>) outputs([[VAR5]] : memref<1x32x35x51xf32{{.*}}>)
  //CHECK-NEXT: [[VAR10:%.*]] = IERT.Copy inputs([[VAR2]] : memref<32x32x3x3xf32{{.*}}>) outputs([[VAR6]] : memref<32x32x3x3xf32{{.*}}>)
  //CHECK-NEXT: [[VAR11:%.*]] = IERT.Copy inputs([[VAR3]] : memref<32xf32{{.*}}>) outputs([[VAR7]] : memref<32xf32{{.*}}>)
  //CHECK-NEXT: [[VAR12:%.*]] = IERT.Convolution
  //CHECK-SAME: dilations = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_begin = [0 : i32, 0 : i32]
  //CHECK-SAME: pads_end = [0 : i32, 0 : i32]
  //CHECK-SAME: strides = [1 : i32, 1 : i32]}
  //CHECK-SAME: inputs([[VAR9]] : memref<1x32x35x51xf32{{.*}}>, [[VAR10]] : memref<32x32x3x3xf32{{.*}}>, [[VAR11]] : memref<32xf32{{.*}}>) outputs([[VAR8]] : memref<1x32x33x49xf32{{.*}}>
  //CHECK-NEXT: [[VAR13:%.*]] = IERT.Copy inputs([[VAR12]] : memref<1x32x33x49xf32{{.*}}>) outputs([[VAR4]] : memref<1x32x33x49xf32{{.*}}>)


  //CHECK-DAG: [[VAR21:%.*]] = memref.subview %arg0[0, 0, 0, 0] [1, 32, 35, 51] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR22:%.*]] = memref.subview %arg1[32, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR23:%.*]] = memref.subview %arg2[32] [32] [1]
  //CHECK-NEXT: [[VAR24:%.*]] = memref.subview [[VAR0]][0, 32, 0, 0] [1, 32, 33, 49] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR25:%.*]] = memref.alloc() : memref<1x32x35x51xf32{{.*}}>
  //CHECK-NEXT: [[VAR26:%.*]] = memref.alloc() : memref<32x32x3x3xf32{{.*}}>
  //CHECK-NEXT: [[VAR27:%.*]] = memref.alloc() : memref<32xf32{{.*}}>
  //CHECK-NEXT: [[VAR28:%.*]] = memref.alloc() : memref<1x32x33x49xf32{{.*}}>
  //CHECK-NEXT: [[VAR29:%.*]] = IERT.Copy inputs([[VAR21]] : memref<1x32x35x51xf32{{.*}}>) outputs([[VAR25]] : memref<1x32x35x51xf32{{.*}}>)
  //CHECK-NEXT: [[VAR30:%.*]] = IERT.Copy inputs([[VAR22]] : memref<32x32x3x3xf32{{.*}}>) outputs([[VAR26]] : memref<32x32x3x3xf32{{.*}}>)
  //CHECK-NEXT: [[VAR31:%.*]] = IERT.Copy inputs([[VAR23]] : memref<32xf32{{.*}}>) outputs([[VAR27]] : memref<32xf32{{.*}}>)
  //CHECK-NEXT: [[VAR32:%.*]] = IERT.Convolution
  //CHECK-SAME: dilations = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_begin = [0 : i32, 0 : i32]
  //CHECK-SAME: pads_end = [0 : i32, 0 : i32]
  //CHECK-SAME: strides = [1 : i32, 1 : i32]}
  //CHECK-SAME: inputs([[VAR29]] : memref<1x32x35x51xf32{{.*}}>, [[VAR30]] : memref<32x32x3x3xf32{{.*}}>, [[VAR31]] : memref<32xf32{{.*}}>) outputs([[VAR28]] : memref<1x32x33x49xf32{{.*}}>
  //CHECK-NEXT: [[VAR33:%.*]] = IERT.Copy inputs([[VAR32]] : memref<1x32x33x49xf32{{.*}}>) outputs([[VAR24]] : memref<1x32x33x49xf32{{.*}}>)


  //CHECK-DAG: [[VAR41:%.*]] = memref.subview %arg0[0, 0, 65, 48] [1, 32, 35, 52] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR42:%.*]] = memref.subview %arg1[0, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR43:%.*]] = memref.subview %arg2[0] [32] [1]
  //CHECK-NEXT: [[VAR44:%.*]] = memref.subview [[VAR0]][0, 0, 66, 49] [1, 32, 33, 50] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR45:%.*]] = memref.alloc() : memref<1x32x35x52xf32{{.*}}>
  //CHECK-NEXT: [[VAR46:%.*]] = memref.alloc() : memref<32x32x3x3xf32{{.*}}>
  //CHECK-NEXT: [[VAR47:%.*]] = memref.alloc() : memref<32xf32{{.*}}>
  //CHECK-NEXT: [[VAR48:%.*]] = memref.alloc() : memref<1x32x33x50xf32{{.*}}>
  //CHECK-NEXT: [[VAR49:%.*]] = IERT.Copy inputs([[VAR41]] : memref<1x32x35x52xf32{{.*}}>) outputs([[VAR45]] : memref<1x32x35x52xf32{{.*}}>)
  //CHECK-NEXT: [[VAR50:%.*]] = IERT.Copy inputs([[VAR42]] : memref<32x32x3x3xf32{{.*}}>) outputs([[VAR46]] : memref<32x32x3x3xf32{{.*}}>)
  //CHECK-NEXT: [[VAR51:%.*]] = IERT.Copy inputs([[VAR43]] : memref<32xf32{{.*}}>) outputs([[VAR47]] : memref<32xf32{{.*}}>)
  //CHECK-NEXT: [[VAR52:%.*]] = IERT.Convolution
  //CHECK-SAME: dilations = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_begin = [0 : i32, 0 : i32]
  //CHECK-SAME: pads_end = [0 : i32, 0 : i32]
  //CHECK-SAME: strides = [1 : i32, 1 : i32]}
  //CHECK-SAME: inputs([[VAR49]] : memref<1x32x35x52xf32{{.*}}>, [[VAR50]] : memref<32x32x3x3xf32{{.*}}>, [[VAR51]] : memref<32xf32{{.*}}>) outputs([[VAR48]] : memref<1x32x33x50xf32{{.*}}>
  //CHECK-NEXT: [[VAR53:%.*]] = IERT.Copy inputs([[VAR52]] : memref<1x32x33x50xf32{{.*}}>) outputs([[VAR44]] : memref<1x32x33x50xf32{{.*}}>)

  //CHECK-DAG: [[VAR61:%.*]] = memref.subview %arg0[0, 0, 65, 48] [1, 32, 35, 52] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR62:%.*]] = memref.subview %arg1[32, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR63:%.*]] = memref.subview %arg2[32] [32] [1]
  //CHECK-NEXT: [[VAR64:%.*]] = memref.subview [[VAR0]][0, 32, 66, 49] [1, 32, 33, 50] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR65:%.*]] = memref.alloc() : memref<1x32x35x52xf32{{.*}}>
  //CHECK-NEXT: [[VAR66:%.*]] = memref.alloc() : memref<32x32x3x3xf32{{.*}}>
  //CHECK-NEXT: [[VAR67:%.*]] = memref.alloc() : memref<32xf32{{.*}}>
  //CHECK-NEXT: [[VAR68:%.*]] = memref.alloc() : memref<1x32x33x50xf32{{.*}}>
  //CHECK-NEXT: [[VAR69:%.*]] = IERT.Copy inputs([[VAR61]] : memref<1x32x35x52xf32{{.*}}>) outputs([[VAR65]] : memref<1x32x35x52xf32{{.*}}>)
  //CHECK-NEXT: [[VAR70:%.*]] = IERT.Copy inputs([[VAR62]] : memref<32x32x3x3xf32{{.*}}>) outputs([[VAR66]] : memref<32x32x3x3xf32{{.*}}>)
  //CHECK-NEXT: [[VAR71:%.*]] = IERT.Copy inputs([[VAR63]] : memref<32xf32{{.*}}>) outputs([[VAR67]] : memref<32xf32{{.*}}>)
  //CHECK-NEXT: [[VAR72:%.*]] = IERT.Convolution
  //CHECK-SAME: dilations = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_begin = [0 : i32, 0 : i32]
  //CHECK-SAME: pads_end = [0 : i32, 0 : i32]
  //CHECK-SAME: strides = [1 : i32, 1 : i32]}
  //CHECK-SAME: inputs([[VAR69]] : memref<1x32x35x52xf32{{.*}}>, [[VAR70]] : memref<32x32x3x3xf32{{.*}}>, [[VAR71]] : memref<32xf32{{.*}}>) outputs([[VAR68]] : memref<1x32x33x50xf32{{.*}}>
  //CHECK-NEXT: [[VAR73:%.*]] = IERT.Copy inputs([[VAR72]] : memref<1x32x33x50xf32{{.*}}>) outputs([[VAR64]] : memref<1x32x33x50xf32{{.*}}>)

  //CHECK: [[VAR80:%.*]] = IERT.ConcatView inputs
  //CHECK-SAME: [[VAR13]]
  //CHECK-SAME: [[VAR33]]
  //CHECK-SAME: [[VAR53]]
  //CHECK-SAME: [[VAR73]]
  //CHECK-SAME: outputs([[VAR0]] : memref<1x64x99x99xf32>)
  //CHECK-NEXT: [[VAR81:%.*]] = IERT.Copy inputs([[VAR80]] : memref<1x64x99x99xf32>) outputs(%arg3 : memref<1x64x99x99xf32>)
  //CHECK-NEXT: return [[VAR81]] : memref<1x64x99x99xf32>
}

// -----

module {

  IERT.RunTimeResources availableMemory :  {
    IERT.MemoryResource 1048576 bytes
    IERT.MemoryResource 1048576 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
  } usedMemory :  {
  } executors :  {
    IERT.ExecutorResource 4 of "NCE_Cluster"  {
    IERT.ExecutorResource 5 of "NCE_PerClusterDPU"
    }
  }

//CHECK-LABEL: @conv2
  func @conv2(%arg0: memref<1x128x16x16xf32>, %arg1: memref<128x128x3x3xf32>, %arg2: memref<128xf32>, %arg3: memref<1x128x16x16xf32>) -> memref<1x128x16x16xf32>
  {
    %0 = memref.alloc() : memref<1x128x16x16xf32>
    %1 = IERT.Convolution {strides = [1:i32,1:i32], pads_begin = [1:i32,1:i32], pads_end = [1:i32,1:i32], dilations = [1:i32,1:i32]}
                          inputs(%arg0: memref<1x128x16x16xf32>, %arg1: memref<128x128x3x3xf32>, %arg2 : memref<128xf32>)
                          outputs(%0 : memref<1x128x16x16xf32>) -> memref<1x128x16x16xf32>
    %2 = IERT.Copy inputs(%1 : memref<1x128x16x16xf32>) outputs(%arg3: memref<1x128x16x16xf32>) -> memref<1x128x16x16xf32>
    return %2 : memref<1x128x16x16xf32>
  }

  //CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x128x16x16xf32>
  //CHECK-DAG: [[VAR1:%.*]] = memref.subview %arg1[0, 0, 0, 0] [64, 128, 3, 3] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR2:%.*]] = memref.subview %arg2[0] [64] [1]
  //CHECK-NEXT: [[VAR3:%.*]] = memref.subview [[VAR0]][0, 0, 0, 0] [1, 64, 16, 16] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR4:%.*]] = memref.alloc() : memref<1x128x16x16xf32{{.*}}>
  //CHECK-NEXT: [[VAR5:%.*]] = memref.alloc() : memref<64x128x3x3xf32{{.*}}>
  //CHECK-NEXT: [[VAR6:%.*]] = memref.alloc() : memref<64xf32{{.*}}>
  //CHECK-NEXT: [[VAR7:%.*]] = memref.alloc() : memref<1x64x16x16xf32{{.*}}>
  //CHECK-NEXT: [[VAR8:%.*]] = IERT.Copy inputs(%arg0 : memref<1x128x16x16xf32{{.*}}>) outputs([[VAR4]] : memref<1x128x16x16xf32{{.*}}>)
  //CHECK-NEXT: [[VAR9:%.*]] = IERT.Copy inputs([[VAR1]] : memref<64x128x3x3xf32{{.*}}>) outputs([[VAR5]] : memref<64x128x3x3xf32{{.*}}>)
  //CHECK-NEXT: [[VAR10:%.*]] = IERT.Copy inputs([[VAR2]] : memref<64xf32{{.*}}>) outputs([[VAR6]] : memref<64xf32{{.*}}>)
  //CHECK-NEXT: [[VAR11:%.*]] = IERT.Convolution
  //CHECK-SAME: dilations = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_begin = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_end = [1 : i32, 1 : i32]
  //CHECK-SAME: strides = [1 : i32, 1 : i32]
  //CHECK-SAME: inputs([[VAR8]] : memref<1x128x16x16xf32{{.*}}>, [[VAR9]] : memref<64x128x3x3xf32{{.*}}>, [[VAR10]] : memref<64xf32{{.*}}>) outputs([[VAR7]] : memref<1x64x16x16xf32{{.*}}>)
  //CHECK-NEXT: [[VAR12:%.*]] = IERT.Copy inputs([[VAR11]] : memref<1x64x16x16xf32{{.*}}>) outputs([[VAR3]] : memref<1x64x16x16xf32{{.*}}>)

  //CHECK-DAG: [[VAR21:%.*]] = memref.subview %arg1[64, 0, 0, 0] [64, 128, 3, 3] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR22:%.*]] = memref.subview %arg2[64] [64] [1]
  //CHECK-NEXT: [[VAR23:%.*]] = memref.subview [[VAR0]][0, 64, 0, 0] [1, 64, 16, 16] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR24:%.*]] = memref.alloc() : memref<1x128x16x16xf32{{.*}}>
  //CHECK-NEXT: [[VAR25:%.*]] = memref.alloc() : memref<64x128x3x3xf32{{.*}}>
  //CHECK-NEXT: [[VAR26:%.*]] = memref.alloc() : memref<64xf32{{.*}}>
  //CHECK-NEXT: [[VAR27:%.*]] = memref.alloc() : memref<1x64x16x16xf32{{.*}}>
  //CHECK-NEXT: [[VAR28:%.*]] = IERT.Copy inputs(%arg0 : memref<1x128x16x16xf32{{.*}}>) outputs([[VAR24]] : memref<1x128x16x16xf32{{.*}}>)
  //CHECK-NEXT: [[VAR29:%.*]] = IERT.Copy inputs([[VAR21]] : memref<64x128x3x3xf32{{.*}}) outputs([[VAR25]] : memref<64x128x3x3xf32{{.*}}>)
  //CHECK-NEXT: [[VAR30:%.*]] = IERT.Copy inputs([[VAR22]] : memref<64xf32{{.*}}>) outputs([[VAR26]] : memref<64xf32{{.*}}>)
  //CHECK-NEXT: [[VAR31:%.*]] = IERT.Convolution
  //CHECK-SAME: dilations = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_begin = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_end = [1 : i32, 1 : i32]
  //CHECK-SAME: strides = [1 : i32, 1 : i32]
  //CHECK-SAME: inputs([[VAR28]] : memref<1x128x16x16xf32{{.*}}>, [[VAR29]] : memref<64x128x3x3xf32{{.*}}>, [[VAR30]] : memref<64xf32{{.*}}>) outputs([[VAR27]] : memref<1x64x16x16xf32{{.*}}>)
  //CHECK-NEXT: [[VAR32:%.*]] = IERT.Copy inputs([[VAR31]] : memref<1x64x16x16xf32{{.*}}>) outputs([[VAR23]] : memref<1x64x16x16xf32{{.*}}>)

  // CHECK: [[VAR40:%.*]] = IERT.ConcatView inputs
  // CHECK-SAME: [[VAR12]]
  // CHECK-SAME: [[VAR32]]
  // CHECK-SAME: outputs([[VAR0]] : memref<{{.*}}>)
  // CHECK-NEXT: [[VAR41:%.*]] = IERT.Copy inputs([[VAR40]] : memref<{{.*}}>) outputs(%arg3 : memref<{{.*}}>)
  // CHECK-NEXT: return [[VAR41]]
}

// -----

module {

  IERT.RunTimeResources availableMemory :  {
    IERT.MemoryResource 1048576 bytes
    IERT.MemoryResource 1048576 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
  } usedMemory :  {
  } executors :  {
    IERT.ExecutorResource 4 of "NCE_Cluster"  {
    IERT.ExecutorResource 5 of "NCE_PerClusterDPU"
    }
  }

//CHECK-LABEL: @conv_nobias
  func @conv_nobias(%arg0: memref<1x128x16x16xf32>, %arg1: memref<128x128x3x3xf32>, %arg2: memref<1x128x16x16xf32>) -> memref<1x128x16x16xf32>
  {
    %0 = memref.alloc() : memref<1x128x16x16xf32>
    %1 = IERT.Convolution {strides = [1:i32,1:i32], pads_begin = [1:i32,1:i32], pads_end = [1:i32,1:i32], dilations = [1:i32,1:i32]}
                          inputs(%arg0: memref<1x128x16x16xf32>, %arg1: memref<128x128x3x3xf32>)
                          outputs(%0 : memref<1x128x16x16xf32>) -> memref<1x128x16x16xf32>
    %2 = IERT.Copy inputs(%1 : memref<1x128x16x16xf32>) outputs(%arg2: memref<1x128x16x16xf32>) -> memref<1x128x16x16xf32>
    return %2 : memref<1x128x16x16xf32>
  }

  //CHECK: [[VAR0:%.*]] = memref.alloc() : memref<1x128x16x16xf32>
  //CHECK-DAG: [[VAR1:%.*]] = memref.subview %arg1[0, 0, 0, 0] [64, 128, 3, 3] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR3:%.*]] = memref.subview [[VAR0]][0, 0, 0, 0] [1, 64, 16, 16] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR4:%.*]] = memref.alloc() : memref<1x128x16x16xf32{{.*}}>
  //CHECK-NEXT: [[VAR5:%.*]] = memref.alloc() : memref<64x128x3x3xf32{{.*}}>
  //CHECK-NEXT: [[VAR7:%.*]] = memref.alloc() : memref<1x64x16x16xf32{{.*}}>
  //CHECK-NEXT: [[VAR8:%.*]] = IERT.Copy inputs(%arg0 : memref<1x128x16x16xf32{{.*}}>) outputs([[VAR4]] : memref<1x128x16x16xf32{{.*}}>)
  //CHECK-NEXT: [[VAR9:%.*]] = IERT.Copy inputs([[VAR1]] : memref<64x128x3x3xf32{{.*}}>) outputs([[VAR5]] : memref<64x128x3x3xf32{{.*}}>)
  //CHECK-NEXT: [[VAR11:%.*]] = IERT.Convolution
  //CHECK-SAME: dilations = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_begin = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_end = [1 : i32, 1 : i32]
  //CHECK-SAME: strides = [1 : i32, 1 : i32]
  //CHECK-SAME: inputs([[VAR8]] : memref<1x128x16x16xf32{{.*}}>, [[VAR9]] : memref<64x128x3x3xf32{{.*}}>) outputs([[VAR7]] : memref<1x64x16x16xf32{{.*}}>)
  //CHECK-NEXT: [[VAR12:%.*]] = IERT.Copy inputs([[VAR11]] : memref<1x64x16x16xf32{{.*}}>) outputs([[VAR3]] : memref<1x64x16x16xf32{{.*}}>)

  //CHECK-DAG: [[VAR21:%.*]] = memref.subview %arg1[64, 0, 0, 0] [64, 128, 3, 3] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR23:%.*]] = memref.subview [[VAR0]][0, 64, 0, 0] [1, 64, 16, 16] [1, 1, 1, 1]
  //CHECK-NEXT: [[VAR24:%.*]] = memref.alloc() : memref<1x128x16x16xf32{{.*}}>
  //CHECK-NEXT: [[VAR25:%.*]] = memref.alloc() : memref<64x128x3x3xf32{{.*}}>
  //CHECK-NEXT: [[VAR27:%.*]] = memref.alloc() : memref<1x64x16x16xf32{{.*}}>
  //CHECK-NEXT: [[VAR28:%.*]] = IERT.Copy inputs(%arg0 : memref<1x128x16x16xf32{{.*}}>) outputs([[VAR24]] : memref<1x128x16x16xf32{{.*}}>)
  //CHECK-NEXT: [[VAR29:%.*]] = IERT.Copy inputs([[VAR21]] : memref<64x128x3x3xf32{{.*}}) outputs([[VAR25]] : memref<64x128x3x3xf32{{.*}}>)
  //CHECK-NEXT: [[VAR31:%.*]] = IERT.Convolution
  //CHECK-SAME: dilations = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_begin = [1 : i32, 1 : i32]
  //CHECK-SAME: pads_end = [1 : i32, 1 : i32]
  //CHECK-SAME: strides = [1 : i32, 1 : i32]
  //CHECK-SAME: inputs([[VAR28]] : memref<1x128x16x16xf32{{.*}}>, [[VAR29]] : memref<64x128x3x3xf32{{.*}}>) outputs([[VAR27]] : memref<1x64x16x16xf32{{.*}}>)
  //CHECK-NEXT: [[VAR32:%.*]] = IERT.Copy inputs([[VAR31]] : memref<1x64x16x16xf32{{.*}}>) outputs([[VAR23]] : memref<1x64x16x16xf32{{.*}}>)

  // CHECK: [[VAR40:%.*]] = IERT.ConcatView inputs
  // CHECK-SAME: [[VAR12]]
  // CHECK-SAME: [[VAR32]]
  // CHECK-SAME: outputs([[VAR0]] : memref<{{.*}}>)
  // CHECK-NEXT: [[VAR41:%.*]] = IERT.Copy inputs([[VAR40]] : memref<{{.*}}>) outputs(%arg2 : memref<{{.*}}>)
  // CHECK-NEXT: return [[VAR41]]
}
