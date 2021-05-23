// RUN: vpux-translate --export-VPUIP -o %t %s && flatc --raw-binary --json %vpuip_schema_file% -- %t && FileCheck %s --input-file %basename_t.json

#map = affine_map<(d0) -> (d0 * 8192)>

module @EDSL attributes {VPUIP.arch = "VPU3400_A0"} {

IERT.RunTimeResources
    availableMemory :  {
        IERT.MemoryResource 1073741824 bytes
        IERT.MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
        IERT.MemoryResource 2097152 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
    }
    usedMemory :  {}
    executors :  {
        IERT.ExecutorResource 1 of "Leon_RT"
        IERT.ExecutorResource 1 of "Leon_NN"
        IERT.ExecutorResource 1 of "DMA_UPA"
        IERT.ExecutorResource 1 of "SHAVE_NN"
        IERT.ExecutorResource 1 of "SHAVE_UPA"
        IERT.ExecutorResource 1 of "NCE_Cluster"  {
            IERT.ExecutorResource 1 of "NCE_PerClusterDPU"
        }
        IERT.ExecutorResource 2 of "DMA_NN"
    }

VPUIP.Graph
    options : "NONE"
    version : {
        majorV = 3 : i32,
        minorV = 11 : i32,
        patchV = 0 : i32,
        hash = "",
        contextStr = "VPUX Compiler"
    }

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "data0" : memref<1x256x256x16xf16>
        IE.DataInfo "data1" : memref<1x256x256x16xf16>
    }
    outputsInfo : {
        IE.DataInfo "prob" : memref<1x256x256x16xf16>
    }

func @main(%arg0: memref<1x256x256x16xf16>, %arg1: memref<1x256x256x16xf16>, %arg2: memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16> {
    %0 = VPUIP.EdslUPA {
        kernel=@kernels::@eltwise_kernel_0,
        outers=[128],
        middles=[],
        transfers=[
            {dir="IN", stage="MIDDLE", baseMap=#map},
            {dir="IN", stage="MIDDLE", baseMap=#map},
            {dir="OUT", stage="MIDDLE", baseMap=#map}
        ]
    }
    inputs(%arg0, %arg1 : memref<1x256x256x16xf16>, memref<1x256x256x16xf16>)
    outputs(%arg2 : memref<1x256x256x16xf16>) -> memref<1x256x256x16xf16>

    return %0 : memref<1x256x256x16xf16>
}

module @kernels {
    func @eltwise_kernel_0(%arg0: index, %arg1: memref<1x2x256x16xf16>, %arg2: memref<1x2x256x16xf16>, %arg3: memref<1x2x256x16xf16>) {
        %c0 = constant 0 : index
        %c2 = constant 2 : index
        %c1 = constant 1 : index
        scf.for %arg4 = %c0 to %c2 step %c1 {
            %c0_0 = constant 0 : index
            %c256 = constant 256 : index
            %c1_1 = constant 1 : index
            scf.for %arg5 = %c0_0 to %c256 step %c1_1 {
                %c0_2 = constant 0 : index
                %c16 = constant 16 : index
                %c1_3 = constant 1 : index
                scf.for %arg6 = %c0_2 to %c16 step %c1_3 {
                    %c0_4 = constant 0 : index
                    %0 = memref.load %arg1[%c0_4, %arg4, %arg5, %arg6] : memref<1x2x256x16xf16>
                    %c0_5 = constant 0 : index
                    %1 = memref.load %arg2[%c0_5, %arg4, %arg5, %arg6] : memref<1x2x256x16xf16>
                    %2 = addf %0, %1 : f16
                    %c0_6 = constant 0 : index
                    %3 = memref.load %arg3[%c0_6, %arg4, %arg5, %arg6] : memref<1x2x256x16xf16>
                    %c0_7 = constant 0 : index
                    memref.store %2, %arg3[%c0_7, %arg4, %arg5, %arg6] : memref<1x2x256x16xf16>
                }
            }
        }
        return
    }
} // module @kernels

} // module @EDSL

// CHECK: identifier: "EDSL"
// CHECK: task_type: "UPALayerTask",
// CHECK: task: {
// CHECK:   maxShaves: 1,
// CHECK:   softLayerParams_type: "EdslParams",
// CHECK:   softLayerParams: {
// CHECK:     dmaTransfers: [
// CHECK:       {
// CHECK:         fromDDR: true,
// CHECK:         dataTypeSize: 2,
// CHECK:         stage: 2,
// CHECK:         localShape: [
// CHECK:           1,
// CHECK:           2,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         globalShape: [
// CHECK:           1,
// CHECK:           256,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         ranges: [
// CHECK:           1,
// CHECK:           2,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         bases: {
// CHECK:           terms: [
// CHECK:             {
// CHECK:               factor: 8192,
// CHECK:               idx: 0
// CHECK:             }
// CHECK:           ]
// CHECK:         }
// CHECK:       },
// CHECK:       {
// CHECK:         fromDDR: true,
// CHECK:         dataTypeSize: 2,
// CHECK:         bufArg: 1,
// CHECK:         stage: 2,
// CHECK:         localShape: [
// CHECK:           1,
// CHECK:           2,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         globalShape: [
// CHECK:           1,
// CHECK:           256,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         ranges: [
// CHECK:           1,
// CHECK:           2,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         bases: {
// CHECK:           terms: [
// CHECK:             {
// CHECK:               factor: 8192,
// CHECK:               idx: 0
// CHECK:             }
// CHECK:           ]
// CHECK:         }
// CHECK:       },
// CHECK:       {
// CHECK:         dataTypeSize: 2,
// CHECK:         bufArg: 2,
// CHECK:         stage: 2,
// CHECK:         localShape: [
// CHECK:           1,
// CHECK:           2,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         globalShape: [
// CHECK:           1,
// CHECK:           256,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         ranges: [
// CHECK:           1,
// CHECK:           2,
// CHECK:           256,
// CHECK:           16
// CHECK:         ],
// CHECK:         bases: {
// CHECK:           terms: [
// CHECK:             {
// CHECK:               factor: 8192,
// CHECK:               idx: 0
// CHECK:             }
// CHECK:           ]
// CHECK:         }
// CHECK:       }
// CHECK:     ]
// CHECK:   },
