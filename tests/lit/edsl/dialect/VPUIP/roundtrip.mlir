// RUN: vpux-opt %s | vpux-opt | FileCheck %s

#map = affine_map<(d0) -> (d0 * 8192)>

// CHECK-LABEL: @EdslUPATask
func @EdslUPATask(%arg0: memref<1x256x256x16xf16>, %arg1: memref<1x256x256x16xf16>, %arg2: memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16> {
    // CHECK:      VPUIP.EdslUPA {inits = [unit], kernel = @kernel, middles = [], outers = [128], transfers = [
    // CHECK-SAME:   {baseMap = #{{.*}}, dir = "IN", stage = "OUTER"},
    // CHECK-SAME:   {baseMap = #{{.*}}, dir = "IN", stage = "MIDDLE"},
    // CHECK-SAME:   {baseMap = #{{.*}}, dir = "OUT", stage = "MIDDLE"}]}
    // CHECK-SAME:   inputs(%{{.*}}, %{{.*}} : memref<1x256x256x16xf16>, memref<1x256x256x16xf16>)
    // CHECK-SAME:   outputs(%{{.*}} : memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16>
    %0 = VPUIP.EdslUPA {
        kernel=@kernel,
        outers=[128],
        middles=[],
        inits=[unit],
        transfers=[
            {dir="IN", stage="OUTER", baseMap=#map},
            {dir="IN", stage="MIDDLE", baseMap=#map},
            {dir="OUT", stage="MIDDLE", baseMap=#map}
        ]
    }
    inputs(%arg0, %arg1 : memref<1x256x256x16xf16>, memref<1x256x256x16xf16>)
    outputs(%arg2 : memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16>
    return %0 : memref<1x256x256x16xf16>
}
