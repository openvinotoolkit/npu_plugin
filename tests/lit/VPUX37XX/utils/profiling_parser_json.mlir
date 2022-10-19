// RUN: vpux-translate --export-VPUIP -o %t %s && prof_parser -b %t -p %profiling_0_37XX_bin% -f json | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#loc0 = loc(unknown)
module @"mul-layer" attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "DefaultHW"}  {
  module @UsedMemory  {
    IE.MemoryResource 896 bytes of @DDR loc(#loc0)
    IE.MemoryResource 1982464 bytes of @CMX_NN loc(#loc0)
  } loc(#loc0)
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096] loc(#loc0)
  module @VPU.SW  {
    func private @builtin_MemPermute(memref<*xf16>, memref<*xf16>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"} loc(#loc0)
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"} loc(#loc0)
  } loc(#loc0)
  IE.ExecutorResource {VPU.processorFrequency = 7.000000e+02 : f64} 2 of @NCE  {
    IE.ExecutorResource 1 of @DPU  loc(#loc0)
  } loc(#loc0)
  IE.ExecutorResource 1 of @SHAVE_ACT  loc(#loc0)
  IE.ExecutorResource 1 of @SHAVE_NN  loc(#loc0)
  IE.ExecutorResource 2 of @DMA_NN  loc(#loc0)
  IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64} loc(#loc0)
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64} loc(#loc0)
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x1x5x5xf16> loc(#loc0)
  } outputsInfo :  {
    DataInfo "input/LogicalAnd" : tensor<1x1x5x5xf16> loc(#loc0)
  } profilingOutputsInfo :  {
    DataInfo "0_dpu_32_actshave_48_dma" : tensor<112xui32> loc(#loc1)
  } loc(#loc0)
  func @main(%arg0: memref<1x1x5x5xf16> loc(unknown), %arg1: memref<1x1x5x5xf16> loc(unknown), %arg2: memref<112xui32> loc(unknown)) -> (memref<1x1x5x5xf16>, memref<112xui32>) {
    %cst = const.Declare memref<1x1x1x16xui8> = #const.Content<dense<[[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xui8>> loc(#loc2)
    %cst_0 = const.Declare memref<16x16x1x1xf16, #NHWC> = #const.Content<dense<"0x0420000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"> : tensor<16x16x1x1xf16, {order = #NHWC}>> loc(#loc2)
    %cst_1 = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<[[[[1981952, 1981632, 1065353216, 0]]], [[[1981984, 1981632, 1065353216, 0]]], [[[1982016, 1981632, 1065353216, 0]]], [[[1982048, 1981632, 1065353216, 0]]], [[[1982080, 1981632, 1065353216, 0]]], [[[1982112, 1981632, 1065353216, 0]]], [[[1982144, 1981632, 1065353216, 0]]], [[[1982176, 1981632, 1065353216, 0]]], [[[1982208, 1981632, 1065353216, 0]]], [[[1982240, 1981632, 1065353216, 0]]], [[[1982272, 1981632, 1065353216, 0]]], [[[1982304, 1981632, 1065353216, 0]]], [[[1982336, 1981632, 1065353216, 0]]], [[[1982368, 1981632, 1065353216, 0]]], [[[1982400, 1981632, 1065353216, 0]]], [[[1982432, 1981632, 1065353216, 0]]]]> : tensor<16x1x1x4xsi32>> loc(#loc2)
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier loc(#loc3)
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier loc(#loc4)
    %2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier loc(#loc3)
    %3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier loc(#loc2)
    %4 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<50xui64, [@CMX_NN, 0]> loc(#loc5)
    %5 = VPURT.DeclareBuffer "CMX_NN" [0] <2112> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc0)
    %6 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x5x5xf16, @DDR> loc(#loc2)
    %7 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x5x5xf16, #NHWC, @DDR> loc(#loc2)
    %8 = VPURT.DeclareBuffer "CMX_NN" [0] <448> -> memref<1x16x5x5xf16, [@CMX_NN, 0]> loc(#loc2)
    %9 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x16x5x5xf16, #NHWC, [@CMX_NN, 0]> loc(#loc2)
    %10 = VPURT.DeclareBuffer "CMX_NN" <448> -> !VPUIP.DistributedBuffer<1x16x5x5xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> loc(#loc2)
    %11 = VPURT.DeclareBuffer "CMX_NN" <960> -> !VPUIP.DistributedBuffer<1x16x5x5xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> loc(#loc2)
    %12 = VPURT.DeclareBuffer "DDR" <832> -> memref<1x1x5x5xf16, #NHWC, @DDR> loc(#loc2)
    %13 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %14 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %15 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc7)
    %16 = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %17 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc8)
    %18 = VPURT.DeclareBuffer "DDR" <50> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %19 = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %20 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %21 = VPURT.DeclareBuffer "CMX_NN" [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %22 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc9)
    %23 = VPURT.DeclareBuffer "DDR" <100> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %24 = VPURT.DeclareBuffer "CMX_NN" [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %25 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %26 = VPURT.DeclareBuffer "CMX_NN" [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %27 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc10)
    %28 = VPURT.DeclareBuffer "DDR" <150> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %29 = VPURT.DeclareBuffer "CMX_NN" [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %30 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %31 = VPURT.DeclareBuffer "CMX_NN" [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %32 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc11)
    %33 = VPURT.DeclareBuffer "DDR" <200> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %34 = VPURT.DeclareBuffer "CMX_NN" [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %35 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %36 = VPURT.DeclareBuffer "CMX_NN" [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %37 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc12)
    %38 = VPURT.DeclareBuffer "DDR" <250> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %39 = VPURT.DeclareBuffer "CMX_NN" [0] <80> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %40 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %41 = VPURT.DeclareBuffer "CMX_NN" [0] <88> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %42 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc13)
    %43 = VPURT.DeclareBuffer "DDR" <300> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %44 = VPURT.DeclareBuffer "CMX_NN" [0] <96> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %45 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %46 = VPURT.DeclareBuffer "CMX_NN" [0] <104> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %47 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc14)
    %48 = VPURT.DeclareBuffer "DDR" <350> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %49 = VPURT.DeclareBuffer "CMX_NN" [0] <112> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %50 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %51 = VPURT.DeclareBuffer "CMX_NN" [0] <120> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %52 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc15)
    %53 = VPURT.DeclareBuffer "DDR" <400> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %54 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %55 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %56 = VPURT.DeclareBuffer "CMX_NN" [0] <136> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %57 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc16)
    %58 = VPURT.DeclareBuffer "DDR" <450> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %59 = VPURT.DeclareBuffer "CMX_NN" [0] <144> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %60 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %61 = VPURT.DeclareBuffer "CMX_NN" [0] <152> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %62 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc17)
    %63 = VPURT.DeclareBuffer "DDR" <500> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %64 = VPURT.DeclareBuffer "CMX_NN" [0] <160> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %65 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %66 = VPURT.DeclareBuffer "CMX_NN" [0] <168> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %67 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc18)
    %68 = VPURT.DeclareBuffer "DDR" <550> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %69 = VPURT.DeclareBuffer "CMX_NN" [0] <176> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %70 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %71 = VPURT.DeclareBuffer "CMX_NN" [0] <184> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %72 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc19)
    %73 = VPURT.DeclareBuffer "DDR" <600> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %74 = VPURT.DeclareBuffer "CMX_NN" [0] <192> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %75 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %76 = VPURT.DeclareBuffer "CMX_NN" [0] <200> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %77 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc20)
    %78 = VPURT.DeclareBuffer "DDR" <650> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %79 = VPURT.DeclareBuffer "CMX_NN" [0] <208> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %80 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %81 = VPURT.DeclareBuffer "CMX_NN" [0] <216> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %82 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc21)
    %83 = VPURT.DeclareBuffer "DDR" <700> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %84 = VPURT.DeclareBuffer "CMX_NN" [0] <224> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %85 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %86 = VPURT.DeclareBuffer "CMX_NN" [0] <232> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %87 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc22)
    %88 = VPURT.DeclareBuffer "DDR" <750> -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    %89 = VPURT.DeclareBuffer "CMX_NN" [0] <240> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %90 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %91 = VPURT.DeclareBuffer "CMX_NN" [0] <248> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %92 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc23)
    %93 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %94 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %95 = VPURT.DeclareBuffer "CMX_NN" [0] <264> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %96 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc24)
    %97 = VPURT.DeclareBuffer "CMX_NN" [0] <272> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %98 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %99 = VPURT.DeclareBuffer "CMX_NN" [0] <280> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %100 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc25)
    %101 = VPURT.DeclareBuffer "CMX_NN" [0] <304> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %102 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %103 = VPURT.DeclareBuffer "CMX_NN" [0] <312> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %104 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc26)
    %105 = VPURT.DeclareBuffer "CMX_NN" [0] <320> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %106 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %107 = VPURT.DeclareBuffer "CMX_NN" [0] <328> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %108 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc27)
    %109 = VPURT.DeclareBuffer "CMX_NN" [0] <336> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %110 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %111 = VPURT.DeclareBuffer "CMX_NN" [0] <344> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %112 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc28)
    %113 = VPURT.DeclareBuffer "CMX_NN" [0] <288> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %114 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %115 = VPURT.DeclareBuffer "CMX_NN" [0] <296> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %116 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc29)
    %117 = VPURT.DeclareBuffer "CMX_NN" [0] <352> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %118 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %119 = VPURT.DeclareBuffer "CMX_NN" [0] <360> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %120 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc30)
    %121 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1x5x5xf16, {order = #NHWC, strides = [400, 1, 80, 16]}, @DDR> loc(#loc2)
    %122 = VPURT.DeclareBuffer "CMX_NN" [0] <368> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %123 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc3)
    %124 = VPURT.DeclareBuffer "CMX_NN" [0] <376> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %125 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc31)
    %126 = VPURT.DeclareBuffer "DDR" <832> -> memref<1x1x5x5xf16, @DDR> loc(#loc2)
    %127 = VPURT.DeclareBuffer "CMX_NN" [0] <384> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %128 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc32)
    %129 = VPURT.DeclareBuffer "CMX_NN" [0] <392> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc6)
    %130 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc33)
    %131 = VPURT.DeclareBuffer "ProfilingOutput" [0] <32> -> memref<4xui32> loc(#loc34)
    %132 = VPURT.DeclareBuffer "ProfilingOutput" [0] <48> -> memref<50xui64> loc(#loc34)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%15 : memref<1xui64, @Register>) outputs(%14 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc7)
    } loc(#loc7)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%13 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%17 : memref<1xui64, @Register>) outputs(%16 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc8)
    } loc(#loc8)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<1xui64, @Register>) outputs(%19 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%18 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%22 : memref<1xui64, @Register>) outputs(%21 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc9)
    } loc(#loc9)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1xui64, @Register>) outputs(%24 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%23 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%27 : memref<1xui64, @Register>) outputs(%26 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc10)
    } loc(#loc10)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%30 : memref<1xui64, @Register>) outputs(%29 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%28 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%32 : memref<1xui64, @Register>) outputs(%31 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc11)
    } loc(#loc11)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%35 : memref<1xui64, @Register>) outputs(%34 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%33 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%37 : memref<1xui64, @Register>) outputs(%36 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc12)
    } loc(#loc12)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%40 : memref<1xui64, @Register>) outputs(%39 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%38 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%42 : memref<1xui64, @Register>) outputs(%41 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc13)
    } loc(#loc13)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%45 : memref<1xui64, @Register>) outputs(%44 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%43 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%47 : memref<1xui64, @Register>) outputs(%46 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc14)
    } loc(#loc14)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%50 : memref<1xui64, @Register>) outputs(%49 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%48 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%52 : memref<1xui64, @Register>) outputs(%51 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc15)
    } loc(#loc15)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%55 : memref<1xui64, @Register>) outputs(%54 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%53 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%57 : memref<1xui64, @Register>) outputs(%56 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc16)
    } loc(#loc16)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%60 : memref<1xui64, @Register>) outputs(%59 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%58 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%62 : memref<1xui64, @Register>) outputs(%61 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc17)
    } loc(#loc17)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%65 : memref<1xui64, @Register>) outputs(%64 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%63 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%67 : memref<1xui64, @Register>) outputs(%66 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc18)
    } loc(#loc18)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%70 : memref<1xui64, @Register>) outputs(%69 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%68 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%72 : memref<1xui64, @Register>) outputs(%71 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc19)
    } loc(#loc19)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%75 : memref<1xui64, @Register>) outputs(%74 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%73 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%77 : memref<1xui64, @Register>) outputs(%76 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc20)
    } loc(#loc20)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%80 : memref<1xui64, @Register>) outputs(%79 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%78 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%82 : memref<1xui64, @Register>) outputs(%81 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc21)
    } loc(#loc21)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%85 : memref<1xui64, @Register>) outputs(%84 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%83 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%87 : memref<1xui64, @Register>) outputs(%86 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc22)
    } loc(#loc22)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%90 : memref<1xui64, @Register>) outputs(%89 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x5x5xf16>) outputs(%88 : memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR>) -> memref<1x1x5x5xf16, {order = #NCHW, strides = [400, 25, 5, 1]}, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%92 : memref<1xui64, @Register>) outputs(%91 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc23)
    } loc(#loc23)
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%94 : memref<1xui64, @Register>) outputs(%93 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%6 : memref<1x16x5x5xf16, @DDR>) outputs(%8 : memref<1x16x5x5xf16, [@CMX_NN, 0]>) -> memref<1x16x5x5xf16, [@CMX_NN, 0]> loc(#loc2)
    } loc(#loc2)
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%96 : memref<1xui64, @Register>) outputs(%95 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc24)
    } loc(#loc24)
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%8 : memref<1x16x5x5xf16, [@CMX_NN, 0]>) outputs(%9 : memref<1x16x5x5xf16, #NHWC, [@CMX_NN, 0]>) profiling_data(%5 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x5x5xf16, #NHWC, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>  {
      ^bb0(%arg3: memref<1x16x5x5xf16, [@CMX_NN, 0]> loc(unknown), %arg4: memref<1x16x5x5xf16, #NHWC, [@CMX_NN, 0]> loc(unknown)):  // no predecessors
        VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg3, %arg4) : memref<1x16x5x5xf16, [@CMX_NN, 0]>, memref<1x16x5x5xf16, #NHWC, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc4)
    } loc(#loc4)
    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%5 : memref<4xui32, [@CMX_NN, 0]>) outputs(%131 : memref<4xui32>) -> memref<4xui32> loc(#loc35)
    } loc(#loc35)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%98 : memref<1xui64, @Register>) outputs(%97 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%9 : memref<1x16x5x5xf16, #NHWC, [@CMX_NN, 0]>) outputs(%7 : memref<1x16x5x5xf16, #NHWC, @DDR>) -> memref<1x16x5x5xf16, #NHWC, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%100 : memref<1xui64, @Register>) outputs(%99 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc25)
    } loc(#loc25)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%102 : memref<1xui64, @Register>) outputs(%101 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    %133 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <1981952> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<16x16x1x1xf16, #NHWC>) outputs(%133 : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc36)
    } loc(#loc36)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%104 : memref<1xui64, @Register>) outputs(%103 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc26)
    } loc(#loc26)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%106 : memref<1xui64, @Register>) outputs(%105 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    %134 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <1981696> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_1 : memref<16x1x1x4xsi32>) outputs(%134 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc36)
    } loc(#loc36)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%108 : memref<1xui64, @Register>) outputs(%107 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc27)
    } loc(#loc27)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%110 : memref<1xui64, @Register>) outputs(%109 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    %135 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <1981632> -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x16xui8>) outputs(%135 : !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc36)
    } loc(#loc36)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%112 : memref<1xui64, @Register>) outputs(%111 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc28)
    } loc(#loc28)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%114 : memref<1xui64, @Register>) outputs(%113 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    %136 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x3x5xf16, #NHWC, @DDR> loc(#loc2)
    %137 = VPURT.DeclareBuffer "CMX_NN" [0] <448> -> memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]> loc(#loc2)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%136 : memref<1x16x3x5xf16, #NHWC, @DDR>) outputs(%137 : memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]> loc(#loc37)
    } loc(#loc37)
    %138 = VPURT.DeclareBuffer "DDR" <480> -> memref<1x16x2x5xf16, #NHWC, @DDR> loc(#loc2)
    %139 = VPURT.DeclareBuffer "CMX_NN" [1] <448> -> memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]> loc(#loc2)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%138 : memref<1x16x2x5xf16, #NHWC, @DDR>) outputs(%139 : memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]> loc(#loc38)
    } loc(#loc38)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%116 : memref<1xui64, @Register>) outputs(%115 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc29)
    } loc(#loc29)
    %140 = VPURT.DeclareBuffer "CMX_NN" [0] <448> -> memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]> loc(#loc39)
    %141 = VPURT.DeclareBuffer "CMX_NN" [1] <448> -> memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]> loc(#loc40)
    %142 = VPURT.DeclareBuffer "CMX_NN" [0] <1981952> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]> loc(#loc41)
    %143 = VPURT.DeclareBuffer "CMX_NN" [1] <1981952> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]> loc(#loc42)
    %144 = VPURT.DeclareBuffer "CMX_NN" [0] <1981696> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]> loc(#loc43)
    %145 = VPURT.DeclareBuffer "CMX_NN" [1] <1981696> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]> loc(#loc44)
    %146 = VPURT.DeclareBuffer "CMX_NN" [0] <1981632> -> memref<1x1x1x16xui8, [@CMX_NN, 0]> loc(#loc45)
    %147 = VPURT.DeclareBuffer "CMX_NN" [1] <1981632> -> memref<1x1x1x16xui8, [@CMX_NN, 1]> loc(#loc46)
    %148 = VPURT.DeclareBuffer "CMX_NN" [0] <960> -> memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]> loc(#loc47)
    %149 = VPURT.DeclareBuffer "CMX_NN" [1] <960> -> memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]> loc(#loc48)
    %150 = VPURT.DeclareBuffer "CMX_NN" [0] <1472> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc49)
    %151 = VPURT.DeclareBuffer "CMX_NN" [1] <1472> -> memref<2xui64, [@CMX_NN, 1]> loc(#loc50)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "DWCONV"} input(%140 : memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]>) weights(%142 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%144 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) activation_window(%146 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x16x5x5xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x16x5x5xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%148 : memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]>) profiling_data(%150 : memref<2xui64, [@CMX_NN, 0]>) -> memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]>, memref<2xui64, [@CMX_NN, 0]> variants :  {
        DPUTask {cluster_id = 0 : i64, end = [4, 2, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]} loc(#loc2)
      } PPE :  {
      } loc(#loc51)
    } loc(#loc51)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160:2 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "DWCONV"} input(%141 : memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]>) weights(%143 : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%145 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) activation_window(%147 : memref<1x1x1x16xui8, [@CMX_NN, 1]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x16x5x5xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) parent_output(%11 : !VPUIP.DistributedBuffer<1x16x5x5xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%149 : memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]>) profiling_data(%151 : memref<2xui64, [@CMX_NN, 1]>) -> memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]>, memref<2xui64, [@CMX_NN, 1]> variants :  {
        DPUTask {cluster_id = 1 : i64, end = [4, 4, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 3, 0]} loc(#loc2)
      } PPE :  {
      } loc(#loc52)
    } loc(#loc52)
    %152 = VPURT.DeclareBuffer "CMX_NN" [0] <1472> -> memref<2xui64, [@CMX_NN, 0]> loc(#loc53)
    %153 = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<2xui64, @DDR> loc(#loc53)
    VPURT.Task waits(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%152 : memref<2xui64, [@CMX_NN, 0]>) outputs(%153 : memref<2xui64, @DDR>) -> memref<2xui64, @DDR> loc(#loc54)
    } loc(#loc54)
    %154 = VPURT.DeclareBuffer "CMX_NN" [1] <1472> -> memref<2xui64, [@CMX_NN, 1]> loc(#loc53)
    %155 = VPURT.DeclareBuffer "ProfilingOutput" [0] <16> -> memref<2xui64, @DDR> loc(#loc53)
    VPURT.Task waits(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%154 : memref<2xui64, [@CMX_NN, 1]>) outputs(%155 : memref<2xui64, @DDR>) -> memref<2xui64, @DDR> loc(#loc55)
    } loc(#loc55)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%118 : memref<1xui64, @Register>) outputs(%117 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    %156 = VPURT.DeclareBuffer "CMX_NN" [0] <960> -> memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]> loc(#loc2)
    %157 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x3x5xf16, #NHWC, @DDR> loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%156 : memref<1x16x3x5xf16, #NHWC, [@CMX_NN, 0]>) outputs(%157 : memref<1x16x3x5xf16, #NHWC, @DDR>) -> memref<1x16x3x5xf16, #NHWC, @DDR> loc(#loc37)
    } loc(#loc37)
    %158 = VPURT.DeclareBuffer "CMX_NN" [1] <960> -> memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]> loc(#loc2)
    %159 = VPURT.DeclareBuffer "DDR" <480> -> memref<1x16x2x5xf16, #NHWC, @DDR> loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%158 : memref<1x16x2x5xf16, #NHWC, [@CMX_NN, 1]>) outputs(%159 : memref<1x16x2x5xf16, #NHWC, @DDR>) -> memref<1x16x2x5xf16, #NHWC, @DDR> loc(#loc38)
    } loc(#loc38)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%120 : memref<1xui64, @Register>) outputs(%119 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc30)
    } loc(#loc30)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%123 : memref<1xui64, @Register>) outputs(%122 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%121 : memref<1x1x5x5xf16, {order = #NHWC, strides = [400, 1, 80, 16]}, @DDR>) outputs(%12 : memref<1x1x5x5xf16, #NHWC, @DDR>) -> memref<1x1x5x5xf16, #NHWC, @DDR> loc(#loc2)
    } loc(#loc2)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%125 : memref<1xui64, @Register>) outputs(%124 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc31)
    } loc(#loc31)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%128 : memref<1xui64, @Register>) outputs(%127 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc32)
    } loc(#loc32)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%126 : memref<1x1x5x5xf16, @DDR>) outputs(%arg1 : memref<1x1x5x5xf16>) -> memref<1x1x5x5xf16> loc(#loc56)
    } loc(#loc56)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%130 : memref<1xui64, @Register>) outputs(%129 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc33)
    } loc(#loc33)
    VPURT.Task attributes {isTrailingSWLayer = false}  {
      %160 = VPUIP.NNDMA {port = 0 : i64} inputs(%4 : memref<50xui64, [@CMX_NN, 0]>) outputs(%132 : memref<50xui64>) -> memref<50xui64> loc(#loc57)
    } loc(#loc57)
    return %arg1, %arg2 : memref<1x1x5x5xf16>, memref<112xui32> loc(#loc56)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc("combinedProfilingDataOutputInfo")
#loc2 = loc("input/LogicalAnd")
#loc3 = loc("input/LogicalAnd_PROFTASKBEGIN")
#loc4 = loc(fused["input/LogicalAnd", "_PROF_0"])
#loc5 = loc("dmaProfilingSubviewBuffer")
#loc6 = loc("dmaProfilingSubview")
#loc7 = loc("input/LogicalAnd_PROFBEGIN")
#loc8 = loc("input/LogicalAnd_PROFTASKEND_0_1")
#loc9 = loc("input/LogicalAnd_PROFTASKEND_2_2")
#loc10 = loc("input/LogicalAnd_PROFTASKEND_4_3")
#loc11 = loc("input/LogicalAnd_PROFTASKEND_6_4")
#loc12 = loc("input/LogicalAnd_PROFTASKEND_8_5")
#loc13 = loc("input/LogicalAnd_PROFTASKEND_10_6")
#loc14 = loc("input/LogicalAnd_PROFTASKEND_12_7")
#loc15 = loc("input/LogicalAnd_PROFTASKEND_14_8")
#loc16 = loc("input/LogicalAnd_PROFTASKEND_16_9")
#loc17 = loc("input/LogicalAnd_PROFTASKEND_18_10")
#loc18 = loc("input/LogicalAnd_PROFTASKEND_20_11")
#loc19 = loc("input/LogicalAnd_PROFTASKEND_22_12")
#loc20 = loc("input/LogicalAnd_PROFTASKEND_24_13")
#loc21 = loc("input/LogicalAnd_PROFTASKEND_26_14")
#loc22 = loc("input/LogicalAnd_PROFTASKEND_28_15")
#loc23 = loc("input/LogicalAnd_PROFTASKEND_30_16")
#loc24 = loc("input/LogicalAnd_PROFTASKEND_32_17")
#loc25 = loc("input/LogicalAnd_PROFTASKEND_34_18")
#loc26 = loc("input/LogicalAnd_PROFTASKEND_38_20")
#loc27 = loc("input/LogicalAnd_PROFTASKEND_40_21")
#loc28 = loc("input/LogicalAnd_PROFTASKEND_42_22")
#loc29 = loc("input/LogicalAnd_PROFTASKEND_36_19")
#loc30 = loc("input/LogicalAnd_PROFTASKEND_44_23")
#loc31 = loc("input/LogicalAnd_PROFTASKEND_46_24")
#loc32 = loc("output_PROFTASKBEGIN")
#loc33 = loc("output_PROFTASKEND_48_25")
#loc34 = loc("newProfilingBuffer")
#loc35 = loc("actshaveProfilingCMX2DDR0")
#loc36 = loc(fused["input/LogicalAnd", "_broadcast_copy_to_CMX[0,1]"])
#loc37 = loc(fused["input/LogicalAnd", "_cluster_0"])
#loc38 = loc(fused["input/LogicalAnd", "_cluster_1"])
#loc39 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_input_cluster_0"])
#loc40 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_input_cluster_1"])
#loc41 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_weights_cluster_0"])
#loc42 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_weights_cluster_1"])
#loc43 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_weightTable_cluster_0"])
#loc44 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_weightTable_cluster_1"])
#loc45 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_activationWindow_cluster_0"])
#loc46 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_activationWindow_cluster_1"])
#loc47 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_outputBuff_cluster_0"])
#loc48 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_outputBuff_cluster_1"])
#loc49 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_profilingBuff_cluster_0"])
#loc50 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_profilingBuff_cluster_1"])
#loc51 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_cluster_0"])
#loc52 = loc(fused["input/LogicalAnd", "_PROF_0_2_1-1,1,", "_cluster_1"])
#loc53 = loc("dpuProfilingCMX2DDR0")
#loc54 = loc(fused["dpuProfilingCMX2DDR0", "_cluster_0"])
#loc55 = loc(fused["dpuProfilingCMX2DDR0", "_cluster_1"])
#loc56 = loc("output")
#loc57 = loc("dmaProfilingCMX2DDR0")

//CHECK: {"traceEvents":[
//CHECK: {"name":"input/LogicalAnd", "cat":"DPU", "ph":"X", "ts":241.495000, "dur":0.128000, "pid":1, "tid":4, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":0.000000, "dur":0.364000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":0.570000, "dur":0.365000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":1.141000, "dur":0.365000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":1.713000, "dur":0.368000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":2.287000, "dur":0.370000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":2.863000, "dur":0.365000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":3.434000, "dur":0.365000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":4.005000, "dur":0.368000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":4.580000, "dur":0.368000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":5.154000, "dur":0.364000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":5.724000, "dur":0.365000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":6.295000, "dur":0.368000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":6.870000, "dur":0.368000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":7.444000, "dur":0.367000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":8.017000, "dur":0.365000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":8.588000, "dur":0.365000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":9.160000, "dur":0.338000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":238.930000, "dur":0.361000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":239.497000, "dur":0.327000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":240.030000, "dur":0.315000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":240.551000, "dur":0.305000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":241.063000, "dur":0.432000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":243.134000, "dur":0.474000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"DMA", "ph":"X", "ts":243.814000, "dur":0.464000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":244.484000, "dur":0.364000, "pid":1, "tid":2, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"SW", "ph":"X", "ts":9.498000, "dur":163.109000, "pid":1, "tid":3, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"Layer", "ph":"X", "ts":0.000000, "dur":172.113000, "pid":1, "tid":5, "tts":3, "args":{}},
//CHECK: {"name":"input/LogicalAnd", "cat":"Layer", "ph":"X", "ts":0.000000, "dur":244.278000, "pid":1, "tid":6, "tts":3, "args":{}},
//CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":244.484000, "dur":0.364000, "pid":1, "tid":5, "tts":3, "args":{}},
//CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":244.484000, "dur":0.364000, "pid":1, "tid":6, "tts":3, "args":{}},
//CHECK: {"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name" : "Inference"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name" : "DMA"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 3, "args": {"name" : "SW"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 4, "args": {"name" : "DPU"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 5, "args": {"name" : "Sum of execution times"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 6, "args": {"name" : "Layer execution time"}}
//CHECK: ]}
