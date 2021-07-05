// RUN: vpux-opt --vpux-edsl-shave-patterns %s | FileCheck %s

// CHECK-LABEL: func @grn
func @grn(%arg0: index, %arg1: memref<1x4x128x24xf16>, %arg2: memref<1x4x128x24xf16>) {
  %cst = constant 1.000000e+00 : f16
  %cst_0 = constant 1.001360e-05 : f16
  %cst_1 = constant 1.600000e+01 : f16
  %cst_2 = constant 0.000000e+00 : f16
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c8 = constant 8 : index
  %c128 = constant 128 : index
  scf.for %arg3 = %c0 to %c4 step %c1 {
    scf.for %arg4 = %c0 to %c128 step %c1 {
      %0 = memref.alloc() : memref<1x1x1x1xf32>
      %1 = fpext %cst_2 : f16 to f32
      memref.store %1, %0[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      scf.for %arg5 = %c0 to %c3 step %c1 {
        %11 = muli %arg5, %c8 : index
        // CHECK: %[[X1:.*]] = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} : memref<1x4x128x24xf16>, vector<8xf16>
        %12 = vector.transfer_read %arg1[%c0, %arg3, %arg4, %11], %cst_2 : memref<1x4x128x24xf16>, vector<8xf16>
        // CHECK-NEXT: %[[ACC:.*]] = call @mvuDot(%[[X1]], %[[X1]]) : (vector<8xf16>, vector<8xf16>) -> f32
        %13 = fpext %12 : vector<8xf16> to vector<8xf32>
        %14 = mulf %13, %13 : vector<8xf32>
        %15 = vector.reduction "add", %14 : vector<8xf32> into f32
        %16 = memref.load %0[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
        %17 = addf %16, %15 : f32
        memref.store %17, %0[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      }
      %2 = memref.load %0[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      %3 = fpext %cst : f16 to f32
      %4 = addf %2, %3 : f32
      %5 = fpext %cst_0 : f16 to f32
      %6 = addf %4, %5 : f32
      %7 = fpext %cst_1 : f16 to f32
      %8 = divf %6, %7 : f32
      // CHECK: %[[X2:.*]] = fptrunc
      %9 = fptrunc %8 : f32 to f16
      %10 = math.sqrt %9 : f16
      scf.for %arg5 = %c0 to %c3 step %c1 {
        %11 = muli %arg5, %c8 : index
        // CHECK: vector.transfer_read
        %12 = vector.transfer_read %arg1[%c0, %arg3, %arg4, %11], %cst_2 : memref<1x4x128x24xf16>, vector<8xf16>
        // CHECK-NEXT: %[[X3:.*]] = math.rsqrt %[[X2]] : f16
        // CHECK-NEXT: %[[X4:.*]] = vector.broadcast %[[X3]] : f16 to vector<8xf16>
        %13 = vector.broadcast %10 : f16 to vector<8xf16>
        // CHECK-NEXT: mulf %{{.*}}, %[[X4]] : vector<8xf16>
        %14 = divf %12, %13 : vector<8xf16>
        vector.transfer_write %14, %arg2[%c0, %arg3, %arg4, %11] : vector<8xf16>, memref<1x4x128x24xf16>
      }
    }
  }
  return
}
