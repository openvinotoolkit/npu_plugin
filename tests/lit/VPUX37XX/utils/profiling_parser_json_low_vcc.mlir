// RUN: vpux-translate %s --export-VPUIP -o %t && prof_parser -b %t -p %profiling_0_37XX_PLL_10_bin% -f json | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#loc0 = loc(unknown)
#loc2 = loc("profiling_result")
module @"torch-jit-export" attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "DefaultHW"} {
  module @UsedMemory {
    IE.MemoryResource 1906688 bytes of @DDR loc(#loc0)
    IE.MemoryResource 1914240 bytes of @CMX_NN loc(#loc0)
  } loc(#loc0)
  module @DmaProfilingReservedMemory {
    IE.MemoryResource 256 bytes of @CMX_NN offset 0 loc(#loc0)
  } loc(#loc0)
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096] loc(#loc0)
  module @VPU.SW {
    func private @builtin_Swish(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, f64) attributes {VPU.kernel_code = "swish_fp16.cpp", VPU.kernel_entry = "swish_fp16"} loc(#loc0)
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"} loc(#loc0)
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"} loc(#loc0)
  } loc(#loc0)
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU  loc(#loc0)
  } loc(#loc0)
  IE.ExecutorResource 2 of @SHAVE_ACT  loc(#loc0)
  IE.ExecutorResource 1 of @SHAVE_NN  loc(#loc0)
  IE.ExecutorResource 2 of @DMA_NN  loc(#loc0)
  IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64} loc(#loc0)
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64} loc(#loc0)
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "result.1" : tensor<1x3x224x224xf16> loc(#loc0)
  } outputsInfo : {
    DataInfo "495" : tensor<1x3x224x224xf16> loc(#loc0)
  } profilingOutputsInfo : {
    DataInfo "0_dpu_160_actshave_208_dma" : tensor<152xui32> loc(#loc1)
  } loc(#loc0)
  func @main(%arg0: memref<1x3x224x224xf16, @DDR> loc(unknown), %arg1: memref<1x3x224x224xf16, @DDR> loc(unknown), %arg2: memref<152xui32> loc("profiling_result")) -> (memref<1x3x224x224xf16, @DDR>, memref<152xui32>) {
    %cst = const.Declare memref<1x13x224x224xf16, #NHWC> = dense<0.000000e+00> : tensor<1x13x224x224xf16, {order = #NHWC}> loc(#loc3)
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier loc(#loc4)
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier loc(#loc4)
    %2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier loc(#loc4)
    %3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier loc(#loc4)
    %4 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier loc(#loc4)
    %5 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier loc(#loc4)
    %6 = VPURT.ConfigureBarrier<6> -> !VPURT.Barrier loc(#loc4)
    %7 = VPURT.ConfigureBarrier<7> -> !VPURT.Barrier loc(#loc4)
    %8 = VPURT.ConfigureBarrier<8> -> !VPURT.Barrier loc(#loc4)
    %9 = VPURT.ConfigureBarrier<9> -> !VPURT.Barrier loc(#loc4)
    %10 = VPURT.ConfigureBarrier<10> -> !VPURT.Barrier loc(#loc4)
    %11 = VPURT.ConfigureBarrier<11> -> !VPURT.Barrier loc(#loc4)
    %12 = VPURT.ConfigureBarrier<12> -> !VPURT.Barrier loc(#loc4)
    %13 = VPURT.ConfigureBarrier<13> -> !VPURT.Barrier loc(#loc4)
    %14 = VPURT.ConfigureBarrier<14> -> !VPURT.Barrier loc(#loc4)
    %15 = VPURT.ConfigureBarrier<15> -> !VPURT.Barrier loc(#loc4)
    %16 = VPURT.ConfigureBarrier<16> -> !VPURT.Barrier loc(#loc4)
    %17 = VPURT.ConfigureBarrier<17> -> !VPURT.Barrier loc(#loc4)
    %18 = VPURT.ConfigureBarrier<18> -> !VPURT.Barrier loc(#loc4)
    %19 = VPURT.ConfigureBarrier<19> -> !VPURT.Barrier loc(#loc4)
    %20 = VPURT.ConfigureBarrier<20> -> !VPURT.Barrier loc(#loc4)
    %21 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x3x224x224xf16, @DDR> loc(#loc3)
    %22 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x3x224x224xf16, @DDR> loc(#loc3)
    %23 = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<10xui64, @DDR> loc(#loc5)
    %24 = VPURT.DeclareBuffer "ProfilingOutput" [0] <80> -> memref<10xui64, @DDR> loc(#loc5)
    %25 = VPURT.DeclareBuffer "ProfilingOutput" [0] <160> -> memref<12xui32> loc(#loc3)
    %26 = VPURT.DeclareBuffer "CMX_NN" [0] <602496> -> memref<12xui32, [@CMX_NN, 0]> loc(#loc0)
    %27 = VPURT.DeclareBuffer "CMX_NN" [0] <301312> -> memref<10xui64, [@CMX_NN, 0]> loc(#loc5)
    %28 = VPURT.DeclareBuffer "CMX_NN" [1] <301312> -> memref<10xui64, [@CMX_NN, 1]> loc(#loc5)
    %29 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x3x224x224xf16, [@CMX_NN, 0]> loc(#loc4)
    %30 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<3x50176xf16, [@CMX_NN, 0]> loc(#loc4)
    %31 = VPURT.DeclareBuffer "CMX_NN" [0] <301312> -> memref<50176x3xf16, [@CMX_NN, 0]> loc(#loc4)
    %32 = VPURT.DeclareBuffer "DDR" <1605632> -> memref<1x16x98x96xf16, #NHWC, @DDR> loc(#loc4)
    %33 = VPURT.DeclareBuffer "DDR" <1605632> -> memref<1x16x49x96xf16, #NHWC, @DDR> loc(#loc4)
    %34 = VPURT.DeclareBuffer "DDR" <1756160> -> memref<1x16x49x96xf16, #NHWC, @DDR> loc(#loc4)
    %35 = VPURT.DeclareBuffer "CMX_NN" <256> -> !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> loc(#loc4)
    %36 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc6)
    %37 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc7)
    %38 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc8)
    %39 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc9)
    %40 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc4)
    %41 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc4)
    %42 = VPURT.DeclareBuffer "CMX_NN" <150784> -> !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> loc(#loc4)
    %43 = VPURT.DeclareBuffer "CMX_NN" [0] <150784> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc10)
    %44 = VPURT.DeclareBuffer "CMX_NN" [1] <150784> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc11)
    %45 = VPURT.DeclareBuffer "CMX_NN" [0] <301440> -> memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc12)
    %46 = VPURT.DeclareBuffer "CMX_NN" [0] <301440> -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc4)
    %47 = VPURT.DeclareBuffer "CMX_NN" [0] <451968> -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc4)
    %48 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc12)
    %49 = VPURT.DeclareBuffer "DDR" <1605632> -> memref<1x16x98x96xf16, #NHWC, @DDR> loc(#loc12)
    %50 = VPURT.DeclareBuffer "DDR" <1605632> -> memref<1x16x49x96xf16, #NHWC, @DDR> loc(#loc13)
    %51 = VPURT.DeclareBuffer "DDR" <1756160> -> memref<1x16x49x96xf16, #NHWC, @DDR> loc(#loc13)
    %52 = VPURT.DeclareBuffer "CMX_NN" <256> -> !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> loc(#loc13)
    %53 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc14)
    %54 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc15)
    %55 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc16)
    %56 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc17)
    %57 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc13)
    %58 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc13)
    %59 = VPURT.DeclareBuffer "CMX_NN" <150784> -> !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> loc(#loc13)
    %60 = VPURT.DeclareBuffer "CMX_NN" [0] <150784> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc18)
    %61 = VPURT.DeclareBuffer "CMX_NN" [1] <150784> -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc19)
    %62 = VPURT.DeclareBuffer "CMX_NN" [0] <301440> -> memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc20)
    %63 = VPURT.DeclareBuffer "CMX_NN" [0] <301440> -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc13)
    %64 = VPURT.DeclareBuffer "CMX_NN" [0] <451968> -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc13)
    %65 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc20)
    %66 = VPURT.DeclareBuffer "CMX_NN" <301440> -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc21)
    %67 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <301440> -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc21)
    %68 = VPURT.DeclareBuffer "CMX_NN" [0] <839040> -> memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc22)
    %69 = VPURT.DeclareBuffer "CMX_NN" [1] <839040> -> memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 1]> loc(#loc23)
    %70 = VPURT.DeclareBuffer "CMX_NN" [0] <839040> -> memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc24)
    %71 = VPURT.DeclareBuffer "CMX_NN" [1] <839040> -> memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 1]> loc(#loc25)
    %72 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <839040> -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc21)
    %73 = VPURT.DeclareBuffer "CMX_NN" <1376640> -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc21)
    %74 = VPURT.DeclareBuffer "CMX_NN" [0] <1376640> -> memref<1x16x75x224xf16, [@CMX_NN, 0]> loc(#loc26)
    %75 = VPURT.DeclareBuffer "CMX_NN" [1] <1376640> -> memref<1x16x75x224xf16, [@CMX_NN, 1]> loc(#loc27)
    %76 = VPURT.DeclareBuffer "CMX_NN" <301440> -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc28)
    %77 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <301440> -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc28)
    %78 = VPURT.DeclareBuffer "CMX_NN" [0] <839040> -> memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc29)
    %79 = VPURT.DeclareBuffer "CMX_NN" [1] <839040> -> memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 1]> loc(#loc30)
    %80 = VPURT.DeclareBuffer "CMX_NN" [0] <839040> -> memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc31)
    %81 = VPURT.DeclareBuffer "CMX_NN" [1] <839040> -> memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 1]> loc(#loc32)
    %82 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <839040> -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc28)
    %83 = VPURT.DeclareBuffer "CMX_NN" <1376640> -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc28)
    %84 = VPURT.DeclareBuffer "CMX_NN" [0] <1376640> -> memref<1x16x75x224xf16, [@CMX_NN, 0]> loc(#loc33)
    %85 = VPURT.DeclareBuffer "CMX_NN" [1] <1376640> -> memref<1x16x75x224xf16, [@CMX_NN, 1]> loc(#loc34)
    %86 = VPURT.DeclareBuffer "CMX_NN" <301440> -> !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc35)
    %87 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <301440> -> !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc35)
    %88 = VPURT.DeclareBuffer "CMX_NN" [0] <831872> -> memref<1x16x74x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc36)
    %89 = VPURT.DeclareBuffer "CMX_NN" [1] <831872> -> memref<1x16x74x224xf16, #NHWC, [@CMX_NN, 1]> loc(#loc37)
    %90 = VPURT.DeclareBuffer "CMX_NN" [0] <831872> -> memref<1x16x74x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc38)
    %91 = VPURT.DeclareBuffer "CMX_NN" [1] <831872> -> memref<1x16x74x224xf16, #NHWC, [@CMX_NN, 1]> loc(#loc39)
    %92 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <831872> -> !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc35)
    %93 = VPURT.DeclareBuffer "CMX_NN" <1362304> -> !VPUIP.DistributedBuffer<1x16x74x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc35)
    %94 = VPURT.DeclareBuffer "CMX_NN" [0] <1362304> -> memref<1x16x74x224xf16, [@CMX_NN, 0]> loc(#loc40)
    %95 = VPURT.DeclareBuffer "CMX_NN" [1] <1362304> -> memref<1x16x74x224xf16, [@CMX_NN, 1]> loc(#loc41)
    %96 = VPURT.DeclareBuffer "DDR" <1605632> -> memref<1x3x224x224xf16, @DDR> loc(#loc21)
    %97 = VPURT.DeclareBuffer "CMX_NN" [0] <320> -> memref<1x3x224x224xf16, [@CMX_NN, 0]> loc(#loc42)
    %98 = VPURT.DeclareBuffer "CMX_NN" [0] <301376> -> memref<1x3x224x224xf16, [@CMX_NN, 0]> loc(#loc42)
    %99 = VPURT.DeclareBuffer "DDR" <1605632> -> memref<12xui32, @DDR> loc(#loc43)
    %100 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<12xui32, [@CMX_NN, 0]> loc(#loc44)
    %101 = VPURT.DeclareBuffer "DDR" <6> -> memref<1x13x224x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc45)
    %102 = VPURT.DeclareBuffer "CMX_NN" [0] <301312> -> memref<1x16x98x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc4)
    %103 = VPURT.DeclareBuffer "CMX_NN" [0] <301312> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc46)
    %104 = VPURT.DeclareBuffer "CMX_NN" [1] <301312> -> memref<2xui64, [@CMX_NN, 1]> loc(#loc47)
    %105 = VPURT.DeclareBuffer "CMX_NN" [0] <150784> -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc4)
    %106 = VPURT.DeclareBuffer "CMX_NN" [1] <150784> -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 1]> loc(#loc4)
    %107 = VPURT.DeclareBuffer "CMX_NN" [0] <602496> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc48)
    %108 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x16x98x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc13)
    %109 = VPURT.DeclareBuffer "CMX_NN" [0] <301328> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc49)
    %110 = VPURT.DeclareBuffer "CMX_NN" [1] <301328> -> memref<2xui64, [@CMX_NN, 1]> loc(#loc50)
    %111 = VPURT.DeclareBuffer "CMX_NN" [0] <150784> -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc13)
    %112 = VPURT.DeclareBuffer "CMX_NN" [1] <150784> -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 1]> loc(#loc13)
    %113 = VPURT.DeclareBuffer "CMX_NN" [0] <602512> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc48)
    %114 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x3x224x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc51)
    %115 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x75x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc52)
    %116 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x75x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc52)
    %117 = VPURT.DeclareBuffer "CMX_NN" [0] <301344> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc53)
    %118 = VPURT.DeclareBuffer "CMX_NN" [1] <301344> -> memref<2xui64, [@CMX_NN, 1]> loc(#loc54)
    %119 = VPURT.DeclareBuffer "CMX_NN" [0] <1376640> -> memref<1x3x75x224xf16, {order = #NCHW, strides = [268800, 16800, 224, 1]}, [@CMX_NN, 0]> loc(#loc21)
    %120 = VPURT.DeclareBuffer "DDR" <1605632> -> memref<1x3x75x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR> loc(#loc51)
    %121 = VPURT.DeclareBuffer "DDR" <537600> -> memref<1x16x75x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc55)
    %122 = VPURT.DeclareBuffer "DDR" <537600> -> memref<1x16x75x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc55)
    %123 = VPURT.DeclareBuffer "CMX_NN" [0] <301360> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc56)
    %124 = VPURT.DeclareBuffer "CMX_NN" [1] <301360> -> memref<2xui64, [@CMX_NN, 1]> loc(#loc57)
    %125 = VPURT.DeclareBuffer "CMX_NN" [0] <1376640> -> memref<1x3x75x224xf16, {order = #NCHW, strides = [268800, 16800, 224, 1]}, [@CMX_NN, 0]> loc(#loc28)
    %126 = VPURT.DeclareBuffer "DDR" <1639232> -> memref<1x3x75x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR> loc(#loc51)
    %127 = VPURT.DeclareBuffer "DDR" <1075200> -> memref<1x16x74x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc58)
    %128 = VPURT.DeclareBuffer "DDR" <1075200> -> memref<1x16x74x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc58)
    %129 = VPURT.DeclareBuffer "CMX_NN" [0] <301376> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc59)
    %130 = VPURT.DeclareBuffer "CMX_NN" [1] <301376> -> memref<2xui64, [@CMX_NN, 1]> loc(#loc60)
    %131 = VPURT.DeclareBuffer "CMX_NN" [0] <1362304> -> memref<1x3x74x224xf16, {order = #NCHW, strides = [265216, 16576, 224, 1]}, [@CMX_NN, 0]> loc(#loc35)
    %132 = VPURT.DeclareBuffer "DDR" <1672832> -> memref<1x3x74x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR> loc(#loc51)
    %133 = VPURT.DeclareBuffer "CMX_NN" [0] <288> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc48)
    %134 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc61)
    %135 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc61)
    %136 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc62)
    %137 = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc62)
    %138 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc63)
    %139 = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc63)
    %140 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc64)
    %141 = VPURT.DeclareBuffer "CMX_NN" [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc64)
    %142 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc65)
    %143 = VPURT.DeclareBuffer "CMX_NN" [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc65)
    %144 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc66)
    %145 = VPURT.DeclareBuffer "CMX_NN" [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc66)
    %146 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc67)
    %147 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc67)
    %148 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc68)
    %149 = VPURT.DeclareBuffer "CMX_NN" [0] <136> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc68)
    %150 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc69)
    %151 = VPURT.DeclareBuffer "CMX_NN" [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc69)
    %152 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc70)
    %153 = VPURT.DeclareBuffer "CMX_NN" [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc70)
    %154 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc71)
    %155 = VPURT.DeclareBuffer "CMX_NN" [0] <144> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc71)
    %156 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc72)
    %157 = VPURT.DeclareBuffer "CMX_NN" [0] <152> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc72)
    %158 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc69)
    %159 = VPURT.DeclareBuffer "CMX_NN" [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc69)
    %160 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc73)
    %161 = VPURT.DeclareBuffer "CMX_NN" [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc73)
    %162 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc71)
    %163 = VPURT.DeclareBuffer "CMX_NN" [0] <160> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc71)
    %164 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc74)
    %165 = VPURT.DeclareBuffer "CMX_NN" [0] <168> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc74)
    %166 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc75)
    %167 = VPURT.DeclareBuffer "CMX_NN" [0] <80> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc75)
    %168 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc76)
    %169 = VPURT.DeclareBuffer "CMX_NN" [0] <88> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc76)
    %170 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc77)
    %171 = VPURT.DeclareBuffer "CMX_NN" [0] <96> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc77)
    %172 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc78)
    %173 = VPURT.DeclareBuffer "CMX_NN" [0] <104> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc78)
    %174 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc79)
    %175 = VPURT.DeclareBuffer "CMX_NN" [0] <176> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc79)
    %176 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc80)
    %177 = VPURT.DeclareBuffer "CMX_NN" [0] <184> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc80)
    %178 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc77)
    %179 = VPURT.DeclareBuffer "CMX_NN" [0] <112> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc77)
    %180 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc81)
    %181 = VPURT.DeclareBuffer "CMX_NN" [0] <120> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc81)
    %182 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xui64, [@CMX_NN, 0]> loc(#loc82)
    %183 = VPURT.DeclareBuffer "ProfilingOutput" [0] <208> -> memref<16xui64> loc(#loc82)
    %184 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc79)
    %185 = VPURT.DeclareBuffer "CMX_NN" [0] <192> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc79)
    %186 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc83)
    %187 = VPURT.DeclareBuffer "CMX_NN" [0] <200> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc83)
    %188 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc84)
    %189 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc84)
    %190 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc85)
    %191 = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc85)
    %192 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc86)
    %193 = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc86)
    %194 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc87)
    %195 = VPURT.DeclareBuffer "CMX_NN" [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc87)
    %196 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc86)
    %197 = VPURT.DeclareBuffer "CMX_NN" [0] <208> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc86)
    %198 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc88)
    %199 = VPURT.DeclareBuffer "CMX_NN" [0] <216> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc88)
    %200 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc89)
    %201 = VPURT.DeclareBuffer "CMX_NN" [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc89)
    %202 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc90)
    %203 = VPURT.DeclareBuffer "CMX_NN" [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc90)
    %204 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc91)
    %205 = VPURT.DeclareBuffer "CMX_NN" [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    %206 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc92)
    %207 = VPURT.DeclareBuffer "CMX_NN" [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    %208 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc91)
    %209 = VPURT.DeclareBuffer "CMX_NN" [0] <224> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    %210 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc93)
    %211 = VPURT.DeclareBuffer "CMX_NN" [0] <232> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc93)
    %212 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc94)
    %213 = VPURT.DeclareBuffer "CMX_NN" [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc94)
    %214 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc95)
    %215 = VPURT.DeclareBuffer "CMX_NN" [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    %216 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc96)
    %217 = VPURT.DeclareBuffer "CMX_NN" [0] <80> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc96)
    %218 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc97)
    %219 = VPURT.DeclareBuffer "CMX_NN" [0] <88> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    %220 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc96)
    %221 = VPURT.DeclareBuffer "CMX_NN" [0] <240> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc96)
    %222 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc98)
    %223 = VPURT.DeclareBuffer "CMX_NN" [0] <248> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc98)
    %224 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<16xui64, [@CMX_NN, 0]> loc(#loc99)
    %225 = VPURT.DeclareBuffer "ProfilingOutput" [0] <464> -> memref<16xui64> loc(#loc99)
    %226 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc100)
    %227 = VPURT.DeclareBuffer "CMX_NN" [0] <96> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    %228 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc101)
    %229 = VPURT.DeclareBuffer "CMX_NN" [0] <104> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc101)
    %230 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc102)
    %231 = VPURT.DeclareBuffer "CMX_NN" [0] <112> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc102)
    %232 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc103)
    %233 = VPURT.DeclareBuffer "CMX_NN" [0] <120> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc103)
    %234 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xui64, [@CMX_NN, 0]> loc(#loc104)
    %235 = VPURT.DeclareBuffer "ProfilingOutput" [0] <336> -> memref<16xui64> loc(#loc104)
    %236 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc105)
    %237 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc105)
    %238 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc106)
    %239 = VPURT.DeclareBuffer "CMX_NN" [0] <136> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc106)
    %240 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc107)
    %241 = VPURT.DeclareBuffer "ProfilingOutput" [0] <592> -> memref<2xui64> loc(#loc107)
    %242 = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<20xui64> loc(#loc108)
    %243 = VPURT.DeclareBuffer "ProfilingOutput" [0] <160> -> memref<12xui32> loc(#loc108)
    %244 = VPURT.DeclareBuffer "ProfilingOutput" [0] <208> -> memref<50xui64> loc(#loc108)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%134 : memref<1xui64, @Register>) outputs(%135 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc61)
    } loc(#loc61)
    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 15446 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%21 : memref<1x3x224x224xf16, @DDR>) outputs(%29 : memref<1x3x224x224xf16, [@CMX_NN, 0]>) -> memref<1x3x224x224xf16, [@CMX_NN, 0]> loc(#loc4)
    } loc(#loc4)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%136 : memref<1xui64, @Register>) outputs(%137 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc62)
    } loc(#loc62)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%138 : memref<1xui64, @Register>) outputs(%139 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc63)
    } loc(#loc63)
    VPURT.Task attributes {cycleBegin = 15446 : i64, cycleEnd = 79209 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x13x224x224xf16, #NHWC>) outputs(%101 : memref<1x13x224x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x13x224x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc109)
    } loc(#loc109)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%140 : memref<1xui64, @Register>) outputs(%141 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc64)
    } loc(#loc64)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%142 : memref<1xui64, @Register>) outputs(%143 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc65)
    } loc(#loc65)
    VPURT.Task attributes {cycleBegin = 15446 : i64, cycleEnd = 30892 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.PermuteDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 6 : i64, dstWidth = 2 : i64, len = 100352 : i64, numPlanes = 3 : i64, srcPlaneStride = 100352 : i64, srcStride = 2 : i64, srcWidth = 100352 : i64}, port = 0 : i64} inputs(%30 : memref<3x50176xf16, [@CMX_NN, 0]>) outputs(%31 : memref<50176x3xf16, [@CMX_NN, 0]>) -> memref<50176x3xf16, [@CMX_NN, 0]> loc(#loc110)
    } loc(#loc110)
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%144 : memref<1xui64, @Register>) outputs(%145 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc66)
    } loc(#loc66)
    VPURT.Task waits(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%146 : memref<1xui64, @Register>) outputs(%147 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc67)
    } loc(#loc67)
    VPURT.Task attributes {cycleBegin = 30892 : i64, cycleEnd = 46338 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%102 : memref<1x16x98x96xf16, #NHWC, [@CMX_NN, 0]>) outputs(%32 : memref<1x16x98x96xf16, #NHWC, @DDR>) -> memref<1x16x98x96xf16, #NHWC, @DDR> loc(#loc4)
    } loc(#loc4)
    VPURT.Task updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%148 : memref<1xui64, @Register>) outputs(%149 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc68)
    } loc(#loc68)
    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%150 : memref<1xui64, @Register>) outputs(%151 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc69)
    } loc(#loc69)
    VPURT.Task attributes {cycleBegin = 79209 : i64, cycleEnd = 94655 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%33 : memref<1x16x49x96xf16, #NHWC, @DDR>) outputs(%40 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc111)
    } loc(#loc111)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%152 : memref<1xui64, @Register>) outputs(%153 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc70)
    } loc(#loc70)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%154 : memref<1xui64, @Register>) outputs(%155 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc71)
    } loc(#loc71)
    VPURT.Task attributes {cycleBegin = 79209 : i64, cycleEnd = 94655 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%34 : memref<1x16x49x96xf16, #NHWC, @DDR>) outputs(%41 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc112)
    } loc(#loc112)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%156 : memref<1xui64, @Register>) outputs(%157 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc72)
    } loc(#loc72)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {cycleBegin = 94655 : i64, cycleEnd = 101184 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_segmented, task_type = "ELTWISE"} input(%38 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>) weights(%36 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%35 : !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%42 : !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%43 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>) profiling_data(%103 : memref<2xui64, [@CMX_NN, 0]>) -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>, memref<2xui64, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [95, 48, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc4)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc4)
      } loc(#loc113)
    } loc(#loc113)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {cycleBegin = 94655 : i64, cycleEnd = 101184 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_segmented, task_type = "ELTWISE"} input(%39 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>) weights(%37 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>) parent_input(%35 : !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%42 : !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%44 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>) profiling_data(%104 : memref<2xui64, [@CMX_NN, 1]>) -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>, memref<2xui64, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [95, 97, 15], outStart = [0, 49, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc4)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc4)
      } loc(#loc114)
    } loc(#loc114)
    VPURT.Task waits(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%158 : memref<1xui64, @Register>) outputs(%159 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc69)
    } loc(#loc69)
    VPURT.Task attributes {cycleBegin = 101184 : i64, cycleEnd = 116630 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%105 : memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]>) outputs(%46 : memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc111)
    } loc(#loc111)
    VPURT.Task updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%160 : memref<1xui64, @Register>) outputs(%161 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc73)
    } loc(#loc73)
    VPURT.Task waits(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%162 : memref<1xui64, @Register>) outputs(%163 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc71)
    } loc(#loc71)
    VPURT.Task attributes {cycleBegin = 101184 : i64, cycleEnd = 116630 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%106 : memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 1]>) outputs(%47 : memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc112)
    } loc(#loc112)
    VPURT.Task updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%164 : memref<1xui64, @Register>) outputs(%165 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc74)
    } loc(#loc74)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 116630 : i64, cycleEnd = 116632 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%45 as %arg3: memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>) outputs(%48 as %arg4: memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>) profiling_data(%107 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>, memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc115)
    } loc(#loc115)
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%166 : memref<1xui64, @Register>) outputs(%167 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc75)
    } loc(#loc75)
    VPURT.Task attributes {cycleBegin = 116632 : i64, cycleEnd = 132078 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%108 : memref<1x16x98x96xf16, #NHWC, [@CMX_NN, 0]>) outputs(%49 : memref<1x16x98x96xf16, #NHWC, @DDR>) -> memref<1x16x98x96xf16, #NHWC, @DDR> loc(#loc12)
    } loc(#loc12)
    VPURT.Task updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%168 : memref<1xui64, @Register>) outputs(%169 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc76)
    } loc(#loc76)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%170 : memref<1xui64, @Register>) outputs(%171 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc77)
    } loc(#loc77)
    VPURT.Task attributes {cycleBegin = 132078 : i64, cycleEnd = 147524 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%50 : memref<1x16x49x96xf16, #NHWC, @DDR>) outputs(%57 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]> loc(#loc116)
    } loc(#loc116)
    VPURT.Task updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%172 : memref<1xui64, @Register>) outputs(%173 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc78)
    } loc(#loc78)
    VPURT.Task waits(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%174 : memref<1xui64, @Register>) outputs(%175 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc79)
    } loc(#loc79)
    VPURT.Task attributes {cycleBegin = 132078 : i64, cycleEnd = 147524 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%51 : memref<1x16x49x96xf16, #NHWC, @DDR>) outputs(%58 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]> loc(#loc117)
    } loc(#loc117)
    VPURT.Task updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%176 : memref<1xui64, @Register>) outputs(%177 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc80)
    } loc(#loc80)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {cycleBegin = 147524 : i64, cycleEnd = 154053 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_segmented, task_type = "ELTWISE"} input(%55 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>) weights(%53 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%52 : !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%59 : !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%60 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>) profiling_data(%109 : memref<2xui64, [@CMX_NN, 0]>) -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 0]>, memref<2xui64, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [95, 48, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc13)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc13)
      } loc(#loc118)
    } loc(#loc118)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {cycleBegin = 147524 : i64, cycleEnd = 154053 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, is_segmented, task_type = "ELTWISE"} input(%56 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>) weights(%54 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>) parent_input(%52 : !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%59 : !VPUIP.DistributedBuffer<1x16x98x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%61 : memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>) profiling_data(%110 : memref<2xui64, [@CMX_NN, 1]>) -> memref<1x16x49x96xf16, #NHWC, [@CMX_NN, 1]>, memref<2xui64, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [95, 97, 15], outStart = [0, 49, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc13)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc13)
      } loc(#loc119)
    } loc(#loc119)
    VPURT.Task waits(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%178 : memref<1xui64, @Register>) outputs(%179 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc77)
    } loc(#loc77)
    VPURT.Task attributes {cycleBegin = 154053 : i64, cycleEnd = 169499 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%111 : memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]>) outputs(%63 : memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc116)
    } loc(#loc116)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%180 : memref<1xui64, @Register>) outputs(%181 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc81)
    } loc(#loc81)
    VPURT.Task updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%182 : memref<16xui64, [@CMX_NN, 0]>) outputs(%183 : memref<16xui64>) -> memref<16xui64> loc(#loc82)
    } loc(#loc82)
    VPURT.Task waits(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%184 : memref<1xui64, @Register>) outputs(%185 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc79)
    } loc(#loc79)
    VPURT.Task attributes {cycleBegin = 154053 : i64, cycleEnd = 169499 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%112 : memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 1]>) outputs(%64 : memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x112x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc117)
    } loc(#loc117)
    VPURT.Task updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%186 : memref<1xui64, @Register>) outputs(%187 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc83)
    } loc(#loc83)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 169499 : i64, cycleEnd = 169501 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%62 as %arg3: memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>) outputs(%65 as %arg4: memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>) profiling_data(%113 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>, memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc120)
    } loc(#loc120)
    VPURT.Task waits(%10 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%188 : memref<1xui64, @Register>) outputs(%189 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc84)
    } loc(#loc84)
    VPURT.Task attributes {cycleBegin = 169501 : i64, cycleEnd = 184947 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%65 : memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>) outputs(%114 : memref<1x3x224x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR> loc(#loc51)
    } loc(#loc51)
    VPURT.Task updates(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%190 : memref<1xui64, @Register>) outputs(%191 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc85)
    } loc(#loc85)
    VPURT.Task attributes {cycleBegin = 184947 : i64, cycleEnd = 185899 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%26 : memref<12xui32, [@CMX_NN, 0]>) outputs(%99 : memref<12xui32, @DDR>) -> memref<12xui32, @DDR> loc(#loc43)
    } loc(#loc43)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%192 : memref<1xui64, @Register>) outputs(%193 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc86)
    } loc(#loc86)
    VPURT.Task attributes {cycleBegin = 185899 : i64, cycleEnd = 212734 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%115 : memref<1x16x75x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) outputs(%67 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc121)
    } loc(#loc121)
    VPURT.Task updates(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%194 : memref<1xui64, @Register>) outputs(%195 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc87)
    } loc(#loc87)
    VPURT.Task waits(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%196 : memref<1xui64, @Register>) outputs(%197 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc86)
    } loc(#loc86)
    VPURT.Task attributes {cycleBegin = 185899 : i64, cycleEnd = 212734 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%116 : memref<1x16x75x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) outputs(%72 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc121)
    } loc(#loc121)
    VPURT.Task updates(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%198 : memref<1xui64, @Register>) outputs(%199 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc88)
    } loc(#loc88)
    VPURT.Task waits(%12 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) attributes {cycleBegin = 212734 : i64, cycleEnd = 235689 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, task_type = "ELTWISE"} input(%70 : memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 0]>) weights(%68 : memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%66 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) parent_output(%73 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%74 : memref<1x16x75x224xf16, [@CMX_NN, 0]>) profiling_data(%117 : memref<2xui64, [@CMX_NN, 0]>) -> memref<1x16x75x224xf16, [@CMX_NN, 0]>, memref<2xui64, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [223, 74, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc21)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc21)
      } loc(#loc122)
    } loc(#loc122)
    VPURT.Task waits(%12 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) attributes {cycleBegin = 212734 : i64, cycleEnd = 235689 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, task_type = "ELTWISE"} input(%71 : memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 1]>) weights(%69 : memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 1]>) parent_input(%66 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) parent_output(%73 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%75 : memref<1x16x75x224xf16, [@CMX_NN, 1]>) profiling_data(%118 : memref<2xui64, [@CMX_NN, 1]>) -> memref<1x16x75x224xf16, [@CMX_NN, 1]>, memref<2xui64, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [223, 74, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc21)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc21)
      } loc(#loc123)
    } loc(#loc123)
    VPURT.Task attributes {cycleBegin = 212734 : i64, cycleEnd = 213686 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%99 : memref<12xui32, @DDR>) outputs(%100 : memref<12xui32, [@CMX_NN, 0]>) -> memref<12xui32, [@CMX_NN, 0]> loc(#loc44)
    } loc(#loc44)
    VPURT.Task waits(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%200 : memref<1xui64, @Register>) outputs(%201 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc89)
    } loc(#loc89)
    VPURT.Task attributes {cycleBegin = 235689 : i64, cycleEnd = 241493 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%119 : memref<1x3x75x224xf16, {order = #NCHW, strides = [268800, 16800, 224, 1]}, [@CMX_NN, 0]>) outputs(%120 : memref<1x3x75x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) -> memref<1x3x75x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR> loc(#loc21)
    } loc(#loc21)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%202 : memref<1xui64, @Register>) outputs(%203 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc90)
    } loc(#loc90)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%204 : memref<1xui64, @Register>) outputs(%205 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    } loc(#loc91)
    VPURT.Task attributes {cycleBegin = 241493 : i64, cycleEnd = 268328 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%122 : memref<1x16x75x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) outputs(%82 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc124)
    } loc(#loc124)
    VPURT.Task updates(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%206 : memref<1xui64, @Register>) outputs(%207 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    } loc(#loc92)
    VPURT.Task waits(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%208 : memref<1xui64, @Register>) outputs(%209 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    } loc(#loc91)
    VPURT.Task attributes {cycleBegin = 241493 : i64, cycleEnd = 268328 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%121 : memref<1x16x75x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) outputs(%77 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc124)
    } loc(#loc124)
    VPURT.Task updates(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%210 : memref<1xui64, @Register>) outputs(%211 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc93)
    } loc(#loc93)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) attributes {cycleBegin = 268328 : i64, cycleEnd = 291283 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, task_type = "ELTWISE"} input(%80 : memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 0]>) weights(%78 : memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%76 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) parent_output(%83 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%84 : memref<1x16x75x224xf16, [@CMX_NN, 0]>) profiling_data(%123 : memref<2xui64, [@CMX_NN, 0]>) -> memref<1x16x75x224xf16, [@CMX_NN, 0]>, memref<2xui64, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [223, 74, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc28)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc28)
      } loc(#loc125)
    } loc(#loc125)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) attributes {cycleBegin = 268328 : i64, cycleEnd = 291283 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, task_type = "ELTWISE"} input(%81 : memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 1]>) weights(%79 : memref<1x16x75x224xf16, #NHWC, [@CMX_NN, 1]>) parent_input(%76 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) parent_output(%83 : !VPUIP.DistributedBuffer<1x16x75x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%85 : memref<1x16x75x224xf16, [@CMX_NN, 1]>) profiling_data(%124 : memref<2xui64, [@CMX_NN, 1]>) -> memref<1x16x75x224xf16, [@CMX_NN, 1]>, memref<2xui64, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [223, 74, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc28)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc28)
      } loc(#loc126)
    } loc(#loc126)
    VPURT.Task waits(%15 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%212 : memref<1xui64, @Register>) outputs(%213 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc94)
    } loc(#loc94)
    VPURT.Task attributes {cycleBegin = 291283 : i64, cycleEnd = 297087 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%125 : memref<1x3x75x224xf16, {order = #NCHW, strides = [268800, 16800, 224, 1]}, [@CMX_NN, 0]>) outputs(%126 : memref<1x3x75x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) -> memref<1x3x75x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR> loc(#loc28)
    } loc(#loc28)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%214 : memref<1xui64, @Register>) outputs(%215 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    } loc(#loc95)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%216 : memref<1xui64, @Register>) outputs(%217 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc96)
    } loc(#loc96)
    VPURT.Task attributes {cycleBegin = 297087 : i64, cycleEnd = 323577 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%128 : memref<1x16x74x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) outputs(%92 : !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc127)
    } loc(#loc127)
    VPURT.Task updates(%16 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%218 : memref<1xui64, @Register>) outputs(%219 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    } loc(#loc97)
    VPURT.Task waits(%15 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%220 : memref<1xui64, @Register>) outputs(%221 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc96)
    } loc(#loc96)
    VPURT.Task attributes {cycleBegin = 297087 : i64, cycleEnd = 323577 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%127 : memref<1x16x74x224xf16, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) outputs(%87 : !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> loc(#loc127)
    } loc(#loc127)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%222 : memref<1xui64, @Register>) outputs(%223 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc98)
    } loc(#loc98)
    VPURT.Task updates(%16 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%224 : memref<16xui64, [@CMX_NN, 0]>) outputs(%225 : memref<16xui64>) -> memref<16xui64> loc(#loc99)
    } loc(#loc99)
    VPURT.Task waits(%16 : !VPURT.Barrier) updates(%17 : !VPURT.Barrier) attributes {cycleBegin = 323577 : i64, cycleEnd = 346267 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, task_type = "ELTWISE"} input(%90 : memref<1x16x74x224xf16, #NHWC, [@CMX_NN, 0]>) weights(%88 : memref<1x16x74x224xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%86 : !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) parent_output(%93 : !VPUIP.DistributedBuffer<1x16x74x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%94 : memref<1x16x74x224xf16, [@CMX_NN, 0]>) profiling_data(%129 : memref<2xui64, [@CMX_NN, 0]>) -> memref<1x16x74x224xf16, [@CMX_NN, 0]>, memref<2xui64, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = "CUBOID_8x16", outEnd = [223, 73, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc35)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc35)
      } loc(#loc128)
    } loc(#loc128)
    VPURT.Task waits(%16 : !VPURT.Barrier) updates(%17 : !VPURT.Barrier) attributes {cycleBegin = 323577 : i64, cycleEnd = 346267 : i64, isTrailingSWLayer = false} {
      %245:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 0 : i64, task_type = "ELTWISE"} input(%91 : memref<1x16x74x224xf16, #NHWC, [@CMX_NN, 1]>) weights(%89 : memref<1x16x74x224xf16, #NHWC, [@CMX_NN, 1]>) parent_input(%86 : !VPUIP.DistributedBuffer<1x16x74x224xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) parent_output(%93 : !VPUIP.DistributedBuffer<1x16x74x224xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) outputs(%95 : memref<1x16x74x224xf16, [@CMX_NN, 1]>) profiling_data(%130 : memref<2xui64, [@CMX_NN, 1]>) -> memref<1x16x74x224xf16, [@CMX_NN, 1]>, memref<2xui64, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, mpe_mode = "CUBOID_8x16", outEnd = [223, 73, 15], outStart = [0, 0, 0], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}} loc(#loc35)
      } PPE : {
        PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]} loc(#loc35)
      } loc(#loc129)
    } loc(#loc129)
    VPURT.Task waits(%17 : !VPURT.Barrier) attributes {cycleBegin = 346267 : i64, cycleEnd = 347219 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%27 : memref<10xui64, [@CMX_NN, 0]>) outputs(%23 : memref<10xui64, @DDR>) -> memref<10xui64, @DDR> loc(#loc130)
    } loc(#loc130)
    VPURT.Task waits(%17 : !VPURT.Barrier) updates(%18 : !VPURT.Barrier) attributes {cycleBegin = 346267 : i64, cycleEnd = 347219 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%28 : memref<10xui64, [@CMX_NN, 1]>) outputs(%24 : memref<10xui64, @DDR>) -> memref<10xui64, @DDR> loc(#loc131)
    } loc(#loc131)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%226 : memref<1xui64, @Register>) outputs(%227 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    } loc(#loc100)
    VPURT.Task attributes {cycleBegin = 347219 : i64, cycleEnd = 352958 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%131 : memref<1x3x74x224xf16, {order = #NCHW, strides = [265216, 16576, 224, 1]}, [@CMX_NN, 0]>) outputs(%132 : memref<1x3x74x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) -> memref<1x3x74x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR> loc(#loc35)
    } loc(#loc35)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%228 : memref<1xui64, @Register>) outputs(%229 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc101)
    } loc(#loc101)
    VPURT.Task waits(%18 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%230 : memref<1xui64, @Register>) outputs(%231 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc102)
    } loc(#loc102)
    VPURT.Task attributes {cycleBegin = 352958 : i64, cycleEnd = 368404 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%96 : memref<1x3x224x224xf16, @DDR>) outputs(%97 : memref<1x3x224x224xf16, [@CMX_NN, 0]>) -> memref<1x3x224x224xf16, [@CMX_NN, 0]> loc(#loc42)
    } loc(#loc42)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%232 : memref<1xui64, @Register>) outputs(%233 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc103)
    } loc(#loc103)
    VPURT.Task updates(%19 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%234 : memref<16xui64, [@CMX_NN, 0]>) outputs(%235 : memref<16xui64>) -> memref<16xui64> loc(#loc104)
    } loc(#loc104)
    VPURT.Task waits(%19 : !VPURT.Barrier) updates(%20 : !VPURT.Barrier) attributes {cycleBegin = 368404 : i64, cycleEnd = 368406 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%97 as %arg3: memref<1x3x224x224xf16, [@CMX_NN, 0]>) outputs(%98 as %arg4: memref<1x3x224x224xf16, [@CMX_NN, 0]>) profiling_data(%133 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x3x224x224xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x3x224x224xf16, [@CMX_NN, 0]>, memref<1x3x224x224xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc132)
    } loc(#loc132)
    VPURT.Task waits(%20 : !VPURT.Barrier) attributes {cycleBegin = 368406 : i64, cycleEnd = 369358 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 0 : i64} inputs(%100 : memref<12xui32, [@CMX_NN, 0]>) outputs(%25 : memref<12xui32>) -> memref<12xui32> loc(#loc133)
    } loc(#loc133)
    VPURT.Task waits(%20 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%236 : memref<1xui64, @Register>) outputs(%237 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc105)
    } loc(#loc105)
    VPURT.Task attributes {cycleBegin = 368406 : i64, cycleEnd = 383852 : i64, isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%98 : memref<1x3x224x224xf16, [@CMX_NN, 0]>) outputs(%22 : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR> loc(#loc134)
    } loc(#loc134)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%238 : memref<1xui64, @Register>) outputs(%239 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc106)
    } loc(#loc106)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %245 = VPUIP.NNDMA {port = 1 : i64} inputs(%240 : memref<2xui64, [@CMX_NN, 0]>) outputs(%241 : memref<2xui64>) -> memref<2xui64> loc(#loc107)
    } loc(#loc107)
    return %arg1, %arg2 : memref<1x3x224x224xf16, @DDR>, memref<152xui32> loc(#loc134)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc("combinedProfilingDataOutputInfo")
#loc3 = loc(fused["Add_2?", "_constant_permute_1_13"])
#loc4 = loc("Add_0?")
#loc5 = loc("dpuProfilingCMX2DDR0")
#loc6 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_weights_cluster_0"])
#loc7 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_weights_cluster_1"])
#loc8 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_input_cluster_0"])
#loc9 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_input_cluster_1"])
#loc10 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_outputBuff_cluster_0"])
#loc11 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_outputBuff_cluster_1"])
#loc12 = loc("Swish_0?")
#loc13 = loc("Add_1?")
#loc14 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_weights_cluster_0"])
#loc15 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_weights_cluster_1"])
#loc16 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_input_cluster_0"])
#loc17 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_input_cluster_1"])
#loc18 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_outputBuff_cluster_0"])
#loc19 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_outputBuff_cluster_1"])
#loc20 = loc("Swish_1?")
#loc21 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]"])
#loc22 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_weights_cluster_0"])
#loc23 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_weights_cluster_1"])
#loc24 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_input_cluster_0"])
#loc25 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_input_cluster_1"])
#loc26 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_outputBuff_cluster_0"])
#loc27 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_outputBuff_cluster_1"])
#loc28 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]"])
#loc29 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_weights_cluster_0"])
#loc30 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_weights_cluster_1"])
#loc31 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_input_cluster_0"])
#loc32 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_input_cluster_1"])
#loc33 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_outputBuff_cluster_0"])
#loc34 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_outputBuff_cluster_1"])
#loc35 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]"])
#loc36 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_weights_cluster_0"])
#loc37 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_weights_cluster_1"])
#loc38 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_input_cluster_0"])
#loc39 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_input_cluster_1"])
#loc40 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_outputBuff_cluster_0"])
#loc41 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_outputBuff_cluster_1"])
#loc42 = loc("Swish_2?")
#loc43 = loc(fused["Swish_1?", "_PROF_1", "spill_write_11"])
#loc44 = loc(fused["Swish_1?", "_PROF_1", "spill_read_11"])
#loc45 = loc(fused["Add_2?", "_expand_subview_1_13"])
#loc46 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_profilingBuff_cluster_0"])
#loc47 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_profilingBuff_cluster_1"])
#loc48 = loc("actshaveProfilingSubview")
#loc49 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_profilingBuff_cluster_0"])
#loc50 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_profilingBuff_cluster_1"])
#loc51 = loc("Add_2?")
#loc52 = loc(fused["Add_2?", "input 1 tile [0, 0, 0, 0]"])
#loc53 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_profilingBuff_cluster_0"])
#loc54 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_profilingBuff_cluster_1"])
#loc55 = loc(fused["Add_2?", "input 1 tile [0, 0, 75, 0]"])
#loc56 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_profilingBuff_cluster_0"])
#loc57 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_profilingBuff_cluster_1"])
#loc58 = loc(fused["Add_2?", "input 1 tile [0, 0, 150, 0]"])
#loc59 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_profilingBuff_cluster_0"])
#loc60 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_profilingBuff_cluster_1"])
#loc61 = loc("Add_0?_PROFBEGIN")
#loc62 = loc("Add_0?_PROFTASKEND_0_1")
#loc63 = loc("Add_2?/_expand_copy_1_13_PROFTASKBEGIN")
#loc64 = loc("Add_2?/_expand_copy_1_13_PROFTASKEND_2_2")
#loc65 = loc("Add_0?/_unrolled_permuteDMA_PROFTASKBEGIN")
#loc66 = loc("Add_0?/_unrolled_permuteDMA_PROFTASKEND_4_3")
#loc67 = loc("Add_0?_PROFTASKBEGIN")
#loc68 = loc("Add_0?_PROFTASKEND_32_17")
#loc69 = loc("Add_0?/_cluster_0_PROFTASKBEGIN")
#loc70 = loc("Add_0?/_cluster_0_PROFTASKEND_6_4")
#loc71 = loc("Add_0?/_cluster_1_PROFTASKBEGIN")
#loc72 = loc("Add_0?/_cluster_1_PROFTASKEND_34_18")
#loc73 = loc("Add_0?/_cluster_0_PROFTASKEND_8_5")
#loc74 = loc("Add_0?/_cluster_1_PROFTASKEND_36_19")
#loc75 = loc("Swish_0?_PROFTASKBEGIN")
#loc76 = loc("Swish_0?_PROFTASKEND_10_6")
#loc77 = loc("Add_1?/_cluster_0_PROFTASKBEGIN")
#loc78 = loc("Add_1?/_cluster_0_PROFTASKEND_12_7")
#loc79 = loc("Add_1?/_cluster_1_PROFTASKBEGIN")
#loc80 = loc("Add_1?/_cluster_1_PROFTASKEND_38_20")
#loc81 = loc("Add_1?/_cluster_0_PROFTASKEND_14_8")
#loc82 = loc("dmaProfilingCMX2DDR0")
#loc83 = loc("Add_1?/_cluster_1_PROFTASKEND_40_21")
#loc84 = loc("Add_2?_PROFTASKBEGIN")
#loc85 = loc("Add_2?_PROFTASKEND_16_9")
#loc86 = loc("Add_2?/output tile [0, 0, 0, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKBEGIN")
#loc87 = loc("Add_2?/output tile [0, 0, 0, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKEND_18_10")
#loc88 = loc("Add_2?/output tile [0, 0, 0, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKEND_42_22")
#loc89 = loc("Add_2?/output tile [0, 0, 0, 0]_PROFTASKBEGIN")
#loc90 = loc("Add_2?/output tile [0, 0, 0, 0]_PROFTASKEND_20_11")
#loc91 = loc("Add_2?/output tile [0, 0, 75, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKBEGIN")
#loc92 = loc("Add_2?/output tile [0, 0, 75, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKEND_22_12")
#loc93 = loc("Add_2?/output tile [0, 0, 75, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKEND_44_23")
#loc94 = loc("Add_2?/output tile [0, 0, 75, 0]_PROFTASKBEGIN")
#loc95 = loc("Add_2?/output tile [0, 0, 75, 0]_PROFTASKEND_24_13")
#loc96 = loc("Add_2?/output tile [0, 0, 150, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKBEGIN")
#loc97 = loc("Add_2?/output tile [0, 0, 150, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKEND_26_14")
#loc98 = loc("Add_2?/output tile [0, 0, 150, 0]/_broadcast_copy_to_CMX[0,1]_PROFTASKEND_46_24")
#loc99 = loc("dmaProfilingCMX2DDR256")
#loc100 = loc("Add_2?/output tile [0, 0, 150, 0]_PROFTASKBEGIN")
#loc101 = loc("Add_2?/output tile [0, 0, 150, 0]_PROFTASKEND_28_15")
#loc102 = loc("Swish_2?_PROFTASKBEGIN")
#loc103 = loc("Swish_2?_PROFTASKEND_30_16")
#loc104 = loc("dmaProfilingCMX2DDR128")
#loc105 = loc("output_PROFTASKBEGIN")
#loc106 = loc("output_PROFTASKEND_48_25")
#loc107 = loc("dmaProfilingCMX2DDR384")
#loc108 = loc("newProfilingBuffer")
#loc109 = loc(fused["Add_2?", "_expand_copy_1_13"])
#loc110 = loc(fused["Add_0?", "_unrolled_permuteDMA"])
#loc111 = loc(fused["Add_0?", "_cluster_0"])
#loc112 = loc(fused["Add_0?", "_cluster_1"])
#loc113 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_cluster_0"])
#loc114 = loc(fused["Add_0?", "_PROF_0_0_2_1-1,1,", "_cluster_1"])
#loc115 = loc(fused["Swish_0?", "_PROF_0_3_0_0"])
#loc116 = loc(fused["Add_1?", "_cluster_0"])
#loc117 = loc(fused["Add_1?", "_cluster_1"])
#loc118 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_cluster_0"])
#loc119 = loc(fused["Add_1?", "_PROF_1_0_2_1-1,1,", "_cluster_1"])
#loc120 = loc(fused["Swish_1?", "_PROF_0_3_1_0"])
#loc121 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_broadcast_copy_to_CMX[0,1]"])
#loc122 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_cluster_0"])
#loc123 = loc(fused["Add_2?", "output tile [0, 0, 0, 0]", "_PROF_2_0_2_1-1,1,", "_cluster_1"])
#loc124 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_broadcast_copy_to_CMX[0,1]"])
#loc125 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_cluster_0"])
#loc126 = loc(fused["Add_2?", "output tile [0, 0, 75, 0]", "_PROF_3_0_2_1-1,1,", "_cluster_1"])
#loc127 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_broadcast_copy_to_CMX[0,1]"])
#loc128 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_cluster_0"])
#loc129 = loc(fused["Add_2?", "output tile [0, 0, 150, 0]", "_PROF_4_0_2_1-1,1,", "_cluster_1"])
#loc130 = loc(fused["dpuProfilingCMX2DDR0", "_cluster_0"])
#loc131 = loc(fused["dpuProfilingCMX2DDR0", "_cluster_1"])
#loc132 = loc(fused["Swish_2?", "_PROF_0_3_2_0"])
#loc133 = loc("actshaveProfilingCMX2DDR0")
#loc134 = loc("output")

//CHECK: {"traceEvents":[
//CHECK: {"name":"Add_0", "cat":"DPU", "ph":"X", "ts":1787.168, "dur":19.267, "pid":1, "tid":3, "args":{}},
//CHECK: {"name":"Add_1", "cat":"DPU", "ph":"X", "ts":5939.932, "dur":19.475, "pid":1, "tid":3, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 0, 0]", "cat":"DPU", "ph":"X", "ts":11166.767, "dur":205.084, "pid":1, "tid":3, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 75, 0]", "cat":"DPU", "ph":"X", "ts":11521.175, "dur":205.048, "pid":1, "tid":3, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 150, 0]", "cat":"DPU", "ph":"X", "ts":11873.984, "dur":205.255, "pid":1, "tid":3, "args":{}},
//CHECK: {"name":"Add_0", "cat":"DMA", "ph":"X", "ts":0.000, "dur":39.036, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/_expand_copy_1_13", "cat":"DMA", "ph":"X", "ts":39.687, "dur":1007.447, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_0?/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":1047.786, "dur":650.911, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_0", "cat":"DMA", "ph":"X", "ts":1699.375, "dur":39.088, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_0?/_cluster_0", "cat":"DMA", "ph":"X", "ts":1739.505, "dur":21.067, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_0?/_cluster_1", "cat":"DMA", "ph":"X", "ts":1739.114, "dur":20.416, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_0?/_cluster_0", "cat":"DMA", "ph":"X", "ts":1783.854, "dur":43.593, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_0?/_cluster_1", "cat":"DMA", "ph":"X", "ts":1783.463, "dur":70.494, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Swish_0", "cat":"DMA", "ph":"X", "ts":5864.166, "dur":39.088, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_1?/_cluster_0", "cat":"DMA", "ph":"X", "ts":5903.906, "dur":20.494, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_1?/_cluster_1", "cat":"DMA", "ph":"X", "ts":5904.296, "dur":20.963, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_1?/_cluster_0", "cat":"DMA", "ph":"X", "ts":5946.302, "dur":43.593, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_1?/_cluster_1", "cat":"DMA", "ph":"X", "ts":5946.692, "dur":69.713, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2", "cat":"DMA", "ph":"X", "ts":10021.015, "dur":1007.057, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 0, 0]/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":11029.192, "dur":131.901, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 0, 0]/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":11028.750, "dur":136.041, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 0, 0]", "cat":"DMA", "ph":"X", "ts":11371.953, "dur":44.661, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 75, 0]/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":11431.562, "dur":88.671, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 75, 0]/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":11372.343, "dur":95.781, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 75, 0]", "cat":"DMA", "ph":"X", "ts":11727.630, "dur":29.166, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 150, 0]/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":11768.697, "dur":105.286, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 150, 0]/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":11727.239, "dur":126.432, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Add_2?/output tile [0, 0, 150, 0]", "cat":"DMA", "ph":"X", "ts":12081.510, "dur":13.750, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Swish_2", "cat":"DMA", "ph":"X", "ts":12095.911, "dur":39.088, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":16139.557, "dur":39.088, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Swish_0", "cat":"SW", "ph":"X", "ts":1856.432, "dur":4005.052, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"Swish_1", "cat":"SW", "ph":"X", "ts":6017.838, "dur":4001.458, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"Swish_2", "cat":"SW", "ph":"X", "ts":12136.848, "dur":4000.989, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"Add_0", "cat":"Layer", "ph":"X", "ts":0.000, "dur":1806.435, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"Add_1", "cat":"Layer", "ph":"X", "ts":5939.932, "dur":19.475, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"Add_2", "cat":"Layer", "ph":"X", "ts":39.687, "dur":12055.573, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"Swish_0", "cat":"Layer", "ph":"X", "ts":1856.432, "dur":4046.822, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"Swish_2", "cat":"Layer", "ph":"X", "ts":12095.911, "dur":4041.926, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":16139.557, "dur":39.088, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"Swish_1", "cat":"Layer", "ph":"X", "ts":6017.838, "dur":4001.458, "pid":1, "tid":5, "args":{}},
//CHECK: {"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name" : "Inference"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name" : "DMA"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 3, "args": {"name" : "DPU"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 4, "args": {"name" : "SW"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 5, "args": {"name" : "Layers"}}
//CHECK: ],
//CHECK: "displayTimeUnit": "ns"
//CHECK: }
