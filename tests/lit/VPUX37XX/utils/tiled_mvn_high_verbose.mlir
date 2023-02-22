// RUN: vpux-translate --export-VPUIP -o %t %s && prof_parser -b %t -p %profiling_0_37XX_MVN_bin% -f json -vv | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#loc0 = loc(unknown)
#loc2 = loc("profiling_result")
module @MVN_case1 attributes {VPU.arch = "VPUX37XX", VPU.compilationMode = "DefaultHW"} {
  module @UsedMemory {
    IE.MemoryResource 4096 bytes of @DDR loc(#loc0)
    IE.MemoryResource 12672 bytes of @CMX_NN loc(#loc0)
  } loc(#loc0)
  module @DmaProfilingReservedMemory {
    IE.MemoryResource 256 bytes of @CMX_NN offset 0 loc(#loc0)
  } loc(#loc0)
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096] loc(#loc0)
  module @VPU.SW {
    func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "tanh_fp16.cpp", VPU.kernel_entry = "tanh_fp16"} loc(#loc0)
    func private @builtin_Swish(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, f64) attributes {VPU.kernel_code = "swish_fp16.cpp", VPU.kernel_entry = "swish_fp16"} loc(#loc0)
    func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"} loc(#loc0)
    func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "single_shave_convert.cpp", VPU.kernel_entry = "single_shave_convert"} loc(#loc0)
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
    DataInfo "input" : tensor<1x4x512xf32> loc(#loc0)
  } outputsInfo : {
    DataInfo "Div_0" : tensor<1x4x512xf32> loc(#loc0)
  } profilingOutputsInfo : {
    DataInfo "0_actshave_384_dma" : tensor<152xui32> loc(#loc1)
  } loc(#loc0)
  func @main(%arg0: memref<1x4x512xf32, @DDR> loc(unknown), %arg1: memref<1x4x512xf32, @DDR> loc(unknown), %arg2: memref<152xui32> loc("profiling_result")) -> (memref<1x4x512xf32, @DDR>, memref<152xui32>) {
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier loc(#loc3)
    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier loc(#loc3)
    %2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier loc(#loc3)
    %3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier loc(#loc3)
    %4 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier loc(#loc3)
    %5 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier loc(#loc3)
    %6 = VPURT.ConfigureBarrier<6> -> !VPURT.Barrier loc(#loc3)
    %7 = VPURT.ConfigureBarrier<7> -> !VPURT.Barrier loc(#loc3)
    %8 = VPURT.ConfigureBarrier<8> -> !VPURT.Barrier loc(#loc3)
    %9 = VPURT.ConfigureBarrier<9> -> !VPURT.Barrier loc(#loc3)
    %10 = VPURT.ConfigureBarrier<10> -> !VPURT.Barrier loc(#loc3)
    %11 = VPURT.ConfigureBarrier<11> -> !VPURT.Barrier loc(#loc3)
    %12 = VPURT.ConfigureBarrier<12> -> !VPURT.Barrier loc(#loc3)
    %13 = VPURT.ConfigureBarrier<13> -> !VPURT.Barrier loc(#loc3)
    %14 = VPURT.ConfigureBarrier<14> -> !VPURT.Barrier loc(#loc3)
    %15 = VPURT.ConfigureBarrier<15> -> !VPURT.Barrier loc(#loc3)
    %16 = VPURT.ConfigureBarrier<16> -> !VPURT.Barrier loc(#loc3)
    %17 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x4x512xf32, @DDR> loc(#loc4)
    %18 = VPURT.DeclareBuffer "CMX_NN" [0] <4352> -> memref<16xui32, [@CMX_NN, 0]> loc(#loc5)
    %19 = VPURT.DeclareBuffer "CMX_NN" [1] <4352> -> memref<16xui32, [@CMX_NN, 1]> loc(#loc5)
    %20 = VPURT.DeclareBuffer "CMX_NN" [0] <12544> -> memref<32xui32, [@CMX_NN, 0]> loc(#loc6)
    %21 = VPURT.DeclareBuffer "CMX_NN" [1] <12544> -> memref<32xui32, [@CMX_NN, 1]> loc(#loc6)
    %22 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x4x1x512xf32, [@CMX_NN, 0]> loc(#loc3)
    %23 = VPURT.DeclareBuffer "CMX_NN" [0] <8448> -> memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc3)
    %24 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x512x1xf16, @DDR> loc(#loc3)
    %25 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc7)
    %26 = VPURT.DeclareBuffer "DDR" <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc7)
    %27 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc7)
    %28 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc7)
    %29 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc8)
    %30 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc8)
    %31 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc9)
    %32 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc8)
    %33 = VPURT.DeclareBuffer "CMX_NN" [0] <4352> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc8)
    %34 = VPURT.DeclareBuffer "CMX_NN" [0] <6400> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc9)
    %35 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x512x1xf16, @DDR> loc(#loc9)
    %36 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc10)
    %37 = VPURT.DeclareBuffer "DDR" <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc10)
    %38 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc10)
    %39 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc10)
    %40 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc11)
    %41 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc11)
    %42 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x512x1xf16, @DDR> loc(#loc11)
    %43 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc11)
    %44 = VPURT.DeclareBuffer "DDR" <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc11)
    %45 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc12)
    %46 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x4x512x1xf16, [@CMX_NN, 1]> loc(#loc13)
    %47 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <256> -> !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc14)
    %48 = VPURT.DeclareBuffer "CMX_NN" [0] <4416> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc15)
    %49 = VPURT.DeclareBuffer "CMX_NN" [1] <4416> -> memref<1x4x512x1xf16, [@CMX_NN, 1]> loc(#loc16)
    %50 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc17)
    %51 = VPURT.DeclareBuffer "CMX_NN" [0] <4416> -> memref<1x4x1x512xf32, [@CMX_NN, 0]> loc(#loc17)
    %52 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x4x1x512xf32, @DDR> loc(#loc3)
    %53 = VPURT.DeclareBuffer "CMX_NN" [0] <12544> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc18)
    %54 = VPURT.DeclareBuffer "CMX_NN" [0] <8448> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc3)
    %55 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc19)
    %56 = VPURT.DeclareBuffer "CMX_NN" [1] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc20)
    %57 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc21)
    %58 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc22)
    %59 = VPURT.DeclareBuffer "CMX_NN" [0] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc23)
    %60 = VPURT.DeclareBuffer "CMX_NN" [1] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc24)
    %61 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc25)
    %62 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc26)
    %63 = VPURT.DeclareBuffer "CMX_NN" [0] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc27)
    %64 = VPURT.DeclareBuffer "CMX_NN" [1] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc28)
    %65 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc29)
    %66 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc30)
    %67 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc31)
    %68 = VPURT.DeclareBuffer "CMX_NN" [1] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc32)
    %69 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc33)
    %70 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc34)
    %71 = VPURT.DeclareBuffer "CMX_NN" [0] <12624> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc35)
    %72 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc36)
    %73 = VPURT.DeclareBuffer "CMX_NN" [1] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc37)
    %74 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc38)
    %75 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc39)
    %76 = VPURT.DeclareBuffer "CMX_NN" [0] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc40)
    %77 = VPURT.DeclareBuffer "CMX_NN" [1] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc41)
    %78 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc42)
    %79 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc43)
    %80 = VPURT.DeclareBuffer "CMX_NN" [0] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc44)
    %81 = VPURT.DeclareBuffer "CMX_NN" [1] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc45)
    %82 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc46)
    %83 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc47)
    %84 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc48)
    %85 = VPURT.DeclareBuffer "CMX_NN" [1] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc49)
    %86 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc50)
    %87 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc51)
    %88 = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<32xui32, @DDR> loc(#loc6)
    %89 = VPURT.DeclareBuffer "ProfilingOutput" [0] <128> -> memref<32xui32, @DDR> loc(#loc6)
    %90 = VPURT.DeclareBuffer "CMX_NN" [0] <4384> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc52)
    %91 = VPURT.DeclareBuffer "CMX_NN" [1] <4384> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc53)
    %92 = VPURT.DeclareBuffer "CMX_NN" [0] <4416> -> memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc14)
    %93 = VPURT.DeclareBuffer "CMX_NN" [0] <4400> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc54)
    %94 = VPURT.DeclareBuffer "ProfilingOutput" [0] <256> -> memref<16xui32, @DDR> loc(#loc5)
    %95 = VPURT.DeclareBuffer "ProfilingOutput" [0] <320> -> memref<16xui32, @DDR> loc(#loc5)
    %96 = VPURT.DeclareBuffer "CMX_NN" [0] <4416> -> memref<1x4x512xf32, [@CMX_NN, 0]> loc(#loc17)
    %97 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc55)
    %98 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc55)
    %99 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc56)
    %100 = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc56)
    %101 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc57)
    %102 = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc57)
    %103 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc58)
    %104 = VPURT.DeclareBuffer "CMX_NN" [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc58)
    %105 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc59)
    %106 = VPURT.DeclareBuffer "CMX_NN" [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc59)
    %107 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc60)
    %108 = VPURT.DeclareBuffer "CMX_NN" [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc60)
    %109 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc61)
    %110 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc61)
    %111 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc62)
    %112 = VPURT.DeclareBuffer "CMX_NN" [0] <136> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc62)
    %113 = VPURT.DeclareBuffer "CMX_NN" [0] <12560> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc63)
    %114 = VPURT.DeclareBuffer "CMX_NN" [1] <12560> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc64)
    %115 = VPURT.DeclareBuffer "CMX_NN" [0] <12576> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc65)
    %116 = VPURT.DeclareBuffer "CMX_NN" [1] <12576> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc66)
    %117 = VPURT.DeclareBuffer "CMX_NN" [0] <12592> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc67)
    %118 = VPURT.DeclareBuffer "CMX_NN" [1] <12592> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc68)
    %119 = VPURT.DeclareBuffer "CMX_NN" [0] <12608> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc69)
    %120 = VPURT.DeclareBuffer "CMX_NN" [1] <12608> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc70)
    %121 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc71)
    %122 = VPURT.DeclareBuffer "CMX_NN" [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc71)
    %123 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc72)
    %124 = VPURT.DeclareBuffer "CMX_NN" [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc72)
    %125 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc73)
    %126 = VPURT.DeclareBuffer "CMX_NN" [0] <144> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc73)
    %127 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc74)
    %128 = VPURT.DeclareBuffer "CMX_NN" [0] <152> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc74)
    %129 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc75)
    %130 = VPURT.DeclareBuffer "CMX_NN" [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc75)
    %131 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc76)
    %132 = VPURT.DeclareBuffer "CMX_NN" [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc76)
    %133 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc77)
    %134 = VPURT.DeclareBuffer "CMX_NN" [0] <80> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc77)
    %135 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc78)
    %136 = VPURT.DeclareBuffer "CMX_NN" [0] <88> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc78)
    %137 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc79)
    %138 = VPURT.DeclareBuffer "CMX_NN" [0] <160> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc79)
    %139 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc80)
    %140 = VPURT.DeclareBuffer "CMX_NN" [0] <168> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc80)
    %141 = VPURT.DeclareBuffer "CMX_NN" [0] <12640> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc81)
    %142 = VPURT.DeclareBuffer "CMX_NN" [1] <12640> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc82)
    %143 = VPURT.DeclareBuffer "CMX_NN" [0] <12656> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc83)
    %144 = VPURT.DeclareBuffer "CMX_NN" [1] <12656> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc84)
    %145 = VPURT.DeclareBuffer "CMX_NN" [0] <4352> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc85)
    %146 = VPURT.DeclareBuffer "CMX_NN" [1] <4352> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc86)
    %147 = VPURT.DeclareBuffer "CMX_NN" [0] <4368> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc87)
    %148 = VPURT.DeclareBuffer "CMX_NN" [1] <4368> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc88)
    %149 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc89)
    %150 = VPURT.DeclareBuffer "CMX_NN" [0] <96> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc89)
    %151 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc90)
    %152 = VPURT.DeclareBuffer "CMX_NN" [0] <104> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc90)
    %153 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc91)
    %154 = VPURT.DeclareBuffer "CMX_NN" [0] <176> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    %155 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc92)
    %156 = VPURT.DeclareBuffer "CMX_NN" [0] <184> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    %157 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<8xui64, [@CMX_NN, 0]> loc(#loc93)
    %158 = VPURT.DeclareBuffer "ProfilingOutput" [0] <544> -> memref<8xui64> loc(#loc93)
    %159 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc94)
    %160 = VPURT.DeclareBuffer "CMX_NN" [0] <112> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc94)
    %161 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc95)
    %162 = VPURT.DeclareBuffer "CMX_NN" [0] <120> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    %163 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xui64, [@CMX_NN, 0]> loc(#loc96)
    %164 = VPURT.DeclareBuffer "ProfilingOutput" [0] <384> -> memref<16xui64> loc(#loc96)
    %165 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc97)
    %166 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    %167 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc98)
    %168 = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc98)
    %169 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc99)
    %170 = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc99)
    %171 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc100)
    %172 = VPURT.DeclareBuffer "CMX_NN" [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    %173 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<4xui64, [@CMX_NN, 0]> loc(#loc101)
    %174 = VPURT.DeclareBuffer "ProfilingOutput" [0] <512> -> memref<4xui64> loc(#loc101)
    %175 = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<96xui32> loc(#loc102)
    %176 = VPURT.DeclareBuffer "ProfilingOutput" [0] <384> -> memref<28xui64> loc(#loc102)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%97 : memref<1xui64, @Register>) outputs(%98 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc55)
    } loc(#loc55)
    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 1148 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%52 : memref<1x4x1x512xf32, @DDR>) outputs(%22 : memref<1x4x1x512xf32, [@CMX_NN, 0]>) -> memref<1x4x1x512xf32, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%99 : memref<1xui64, @Register>) outputs(%100 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc56)
    } loc(#loc56)
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 1148 : i64, cycleEnd = 1150 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%22 as %arg3: memref<1x4x1x512xf32, [@CMX_NN, 0]>) outputs(%23 as %arg4: memref<1x4x1x512xf16, [@CMX_NN, 0]>) profiling_data(%53 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x1x512xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x1x512xf32, [@CMX_NN, 0]>, memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc103)
    } loc(#loc103)
    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%101 : memref<1xui64, @Register>) outputs(%102 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc57)
    } loc(#loc57)
    VPURT.Task attributes {cycleBegin = 1150 : i64, cycleEnd = 2298 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%54 : memref<1x4x512x1xf16, [@CMX_NN, 0]>) outputs(%24 : memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR> loc(#loc3)
    } loc(#loc3)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%103 : memref<1xui64, @Register>) outputs(%104 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc58)
    } loc(#loc58)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%105 : memref<1xui64, @Register>) outputs(%106 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc59)
    } loc(#loc59)
    VPURT.Task attributes {cycleBegin = 2298 : i64, cycleEnd = 3446 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x2x512x1xf16, @DDR>) outputs(%27 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc104)
    } loc(#loc104)
    VPURT.Task updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%107 : memref<1xui64, @Register>) outputs(%108 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc60)
    } loc(#loc60)
    VPURT.Task waits(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%109 : memref<1xui64, @Register>) outputs(%110 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc61)
    } loc(#loc61)
    VPURT.Task attributes {cycleBegin = 2298 : i64, cycleEnd = 3446 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%26 : memref<1x2x512x1xf16, @DDR>) outputs(%28 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc105)
    } loc(#loc105)
    VPURT.Task updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%111 : memref<1xui64, @Register>) outputs(%112 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc62)
    } loc(#loc62)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 3446 : i64, cycleEnd = 3448 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%57 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%61 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%113 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc106)
    } loc(#loc106)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 3446 : i64, cycleEnd = 3448 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%58 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%62 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%114 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc107)
    } loc(#loc107)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 3446 : i64, cycleEnd = 3448 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%55 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%59 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%115 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc108)
    } loc(#loc108)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 3446 : i64, cycleEnd = 3448 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%56 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%60 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%116 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc109)
    } loc(#loc109)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 3448 : i64, cycleEnd = 3450 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%65 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%69 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%117 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc110)
    } loc(#loc110)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 3448 : i64, cycleEnd = 3450 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%66 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%70 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%118 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc111)
    } loc(#loc111)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 3448 : i64, cycleEnd = 3450 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%63 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%67 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%119 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc112)
    } loc(#loc112)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 3448 : i64, cycleEnd = 3450 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%64 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%68 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%120 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc113)
    } loc(#loc113)
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%121 : memref<1xui64, @Register>) outputs(%122 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc71)
    } loc(#loc71)
    VPURT.Task attributes {cycleBegin = 3450 : i64, cycleEnd = 4598 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%29 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) outputs(%32 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc114)
    } loc(#loc114)
    VPURT.Task updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%123 : memref<1xui64, @Register>) outputs(%124 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc72)
    } loc(#loc72)
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%125 : memref<1xui64, @Register>) outputs(%126 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc73)
    } loc(#loc73)
    VPURT.Task attributes {cycleBegin = 3450 : i64, cycleEnd = 4598 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%30 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) outputs(%33 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc115)
    } loc(#loc115)
    VPURT.Task updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%127 : memref<1xui64, @Register>) outputs(%128 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc74)
    } loc(#loc74)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {cycleBegin = 4598 : i64, cycleEnd = 4600 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%31 as %arg3: memref<1x4x512x1xf16, [@CMX_NN, 0]>) outputs(%34 as %arg4: memref<1x4x512x1xf16, [@CMX_NN, 0]>) profiling_data(%71 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x512x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x4x512x1xf16, [@CMX_NN, 0]>, memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc116)
    } loc(#loc116)
    VPURT.Task waits(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%129 : memref<1xui64, @Register>) outputs(%130 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc75)
    } loc(#loc75)
    VPURT.Task attributes {cycleBegin = 4600 : i64, cycleEnd = 5748 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%34 : memref<1x4x512x1xf16, [@CMX_NN, 0]>) outputs(%35 : memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR> loc(#loc9)
    } loc(#loc9)
    VPURT.Task updates(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%131 : memref<1xui64, @Register>) outputs(%132 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc76)
    } loc(#loc76)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%133 : memref<1xui64, @Register>) outputs(%134 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc77)
    } loc(#loc77)
    VPURT.Task attributes {cycleBegin = 5748 : i64, cycleEnd = 6896 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%36 : memref<1x2x512x1xf16, @DDR>) outputs(%38 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc117)
    } loc(#loc117)
    VPURT.Task updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%135 : memref<1xui64, @Register>) outputs(%136 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc78)
    } loc(#loc78)
    VPURT.Task waits(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%137 : memref<1xui64, @Register>) outputs(%138 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc79)
    } loc(#loc79)
    VPURT.Task attributes {cycleBegin = 5748 : i64, cycleEnd = 6896 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%37 : memref<1x2x512x1xf16, @DDR>) outputs(%39 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc118)
    } loc(#loc118)
    VPURT.Task updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%139 : memref<1xui64, @Register>) outputs(%140 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc80)
    } loc(#loc80)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 6896 : i64, cycleEnd = 6898 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%74 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%78 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%141 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc119)
    } loc(#loc119)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 6896 : i64, cycleEnd = 6898 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%75 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%79 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%142 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc120)
    } loc(#loc120)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 6896 : i64, cycleEnd = 6898 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%72 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%76 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%143 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc121)
    } loc(#loc121)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 6896 : i64, cycleEnd = 6898 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%73 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%77 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%144 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc122)
    } loc(#loc122)
    VPURT.Task waits(%10 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 7855 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<32xui32, [@CMX_NN, 0]>) outputs(%88 : memref<32xui32, @DDR>) -> memref<32xui32, @DDR> loc(#loc123)
    } loc(#loc123)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 7855 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%21 : memref<32xui32, [@CMX_NN, 1]>) outputs(%89 : memref<32xui32, @DDR>) -> memref<32xui32, @DDR> loc(#loc124)
    } loc(#loc124)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 6900 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%82 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%86 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%145 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc125)
    } loc(#loc125)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 6900 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%83 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%87 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%146 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc126)
    } loc(#loc126)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 6900 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%80 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%84 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%147 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc127)
    } loc(#loc127)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 6900 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%81 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%85 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%148 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc128)
    } loc(#loc128)
    VPURT.Task waits(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%149 : memref<1xui64, @Register>) outputs(%150 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc89)
    } loc(#loc89)
    VPURT.Task attributes {cycleBegin = 7855 : i64, cycleEnd = 9003 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%40 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) outputs(%43 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc129)
    } loc(#loc129)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%151 : memref<1xui64, @Register>) outputs(%152 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc90)
    } loc(#loc90)
    VPURT.Task waits(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%153 : memref<1xui64, @Register>) outputs(%154 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    } loc(#loc91)
    VPURT.Task attributes {cycleBegin = 7855 : i64, cycleEnd = 9003 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%41 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) outputs(%44 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc130)
    } loc(#loc130)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%155 : memref<1xui64, @Register>) outputs(%156 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    } loc(#loc92)
    VPURT.Task updates(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%157 : memref<8xui64, [@CMX_NN, 0]>) outputs(%158 : memref<8xui64>) -> memref<8xui64> loc(#loc93)
    } loc(#loc93)
    VPURT.Task waits(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%159 : memref<1xui64, @Register>) outputs(%160 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc94)
    } loc(#loc94)
    VPURT.Task attributes {cycleBegin = 9003 : i64, cycleEnd = 10151 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%42 : memref<1x4x512x1xf16, @DDR>) outputs(%47 : !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> loc(#loc131)
    } loc(#loc131)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%161 : memref<1xui64, @Register>) outputs(%162 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    } loc(#loc95)
    VPURT.Task updates(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%163 : memref<16xui64, [@CMX_NN, 0]>) outputs(%164 : memref<16xui64>) -> memref<16xui64> loc(#loc96)
    } loc(#loc96)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) attributes {cycleBegin = 10151 : i64, cycleEnd = 10153 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Tanh inputs(%45 as %arg3: memref<1x4x512x1xf16, [@CMX_NN, 0]>) outputs(%48 as %arg4: memref<1x4x512x1xf16, [@CMX_NN, 0]>) profiling_data(%90 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x512x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x512x1xf16, [@CMX_NN, 0]>, memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc132)
    } loc(#loc132)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) attributes {cycleBegin = 10151 : i64, cycleEnd = 10153 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Tanh inputs(%46 as %arg3: memref<1x4x512x1xf16, [@CMX_NN, 1]>) outputs(%49 as %arg4: memref<1x4x512x1xf16, [@CMX_NN, 1]>) profiling_data(%91 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x4x512x1xf16, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x512x1xf16, [@CMX_NN, 1]>, memref<1x4x512x1xf16, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc133)
    } loc(#loc133)
    VPURT.Task waits(%15 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%165 : memref<1xui64, @Register>) outputs(%166 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    } loc(#loc97)
    VPURT.Task attributes {cycleBegin = 10153 : i64, cycleEnd = 11301 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%92 : memref<1x4x1x512xf16, [@CMX_NN, 0]>) outputs(%50 : memref<1x4x1x512xf16, [@CMX_NN, 0]>) -> memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc14)
    } loc(#loc14)
    VPURT.Task updates(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%167 : memref<1xui64, @Register>) outputs(%168 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc98)
    } loc(#loc98)
    VPURT.Task waits(%11 : !VPURT.Barrier) updates(%16 : !VPURT.Barrier) attributes {cycleBegin = 11301 : i64, cycleEnd = 11303 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%50 as %arg3: memref<1x4x1x512xf16, [@CMX_NN, 0]>) outputs(%51 as %arg4: memref<1x4x1x512xf32, [@CMX_NN, 0]>) profiling_data(%93 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x1x512xf32, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x1x512xf16, [@CMX_NN, 0]>, memref<1x4x1x512xf32, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc134)
    } loc(#loc134)
    VPURT.Task waits(%16 : !VPURT.Barrier) attributes {cycleBegin = 11303 : i64, cycleEnd = 12257 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%18 : memref<16xui32, [@CMX_NN, 0]>) outputs(%94 : memref<16xui32, @DDR>) -> memref<16xui32, @DDR> loc(#loc135)
    } loc(#loc135)
    VPURT.Task waits(%16 : !VPURT.Barrier) attributes {cycleBegin = 11303 : i64, cycleEnd = 12257 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 1 : i64} inputs(%19 : memref<16xui32, [@CMX_NN, 1]>) outputs(%95 : memref<16xui32, @DDR>) -> memref<16xui32, @DDR> loc(#loc136)
    } loc(#loc136)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%169 : memref<1xui64, @Register>) outputs(%170 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc99)
    } loc(#loc99)
    VPURT.Task attributes {cycleBegin = 12257 : i64, cycleEnd = 13405 : i64, isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%96 : memref<1x4x512xf32, [@CMX_NN, 0]>) outputs(%17 : memref<1x4x512xf32, @DDR>) -> memref<1x4x512xf32, @DDR> loc(#loc17)
    } loc(#loc17)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%171 : memref<1xui64, @Register>) outputs(%172 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    } loc(#loc100)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %177 = VPUIP.NNDMA {port = 0 : i64} inputs(%173 : memref<4xui64, [@CMX_NN, 0]>) outputs(%174 : memref<4xui64>) -> memref<4xui64> loc(#loc101)
    } loc(#loc101)
    return %arg1, %arg2 : memref<1x4x512xf32, @DDR>, memref<152xui32> loc(#loc17)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc("combinedProfilingDataOutputInfo")
#loc3 = loc("Reshape_1423?")
#loc4 = loc("2_actProfilingSubviewBuffer_1")
#loc5 = loc("actshaveProfilingCMX2DDR16")
#loc6 = loc("actshaveProfilingCMX2DDR0")
#loc7 = loc("MVN_0?")
#loc8 = loc("MVN_1?")
#loc9 = loc("Swish_0?")
#loc10 = loc("MVN_2?")
#loc11 = loc("MVN_3?")
#loc12 = loc(fused["Tanh_0?", "_PROF_16_4_2_0", "_input_cluster_0"])
#loc13 = loc(fused["Tanh_0?", "_PROF_16_4_2_0", "_input_cluster_1"])
#loc14 = loc("Tanh_0?")
#loc15 = loc(fused["Tanh_0?", "_PROF_16_4_2_0", "_outputBuff_cluster_0"])
#loc16 = loc(fused["Tanh_0?", "_PROF_16_4_2_0", "_outputBuff_cluster_1"])
#loc17 = loc("output")
#loc18 = loc(fused["Reshape_1423?", "_PROF_0_8_0_0", "_view_cast"])
#loc19 = loc(fused["MVN_0?", "_PROF_0_8_2_1", "_input_cluster_0"])
#loc20 = loc(fused["MVN_0?", "_PROF_0_8_2_1", "_input_cluster_1"])
#loc21 = loc(fused["MVN_0?", "_PROF_0_8_1_0", "_input_cluster_0"])
#loc22 = loc(fused["MVN_0?", "_PROF_0_8_1_0", "_input_cluster_1"])
#loc23 = loc(fused["MVN_0?", "_PROF_0_8_2_1", "_outputBuff_cluster_0"])
#loc24 = loc(fused["MVN_0?", "_PROF_0_8_2_1", "_outputBuff_cluster_1"])
#loc25 = loc(fused["MVN_0?", "_PROF_0_8_1_0", "_outputBuff_cluster_0"])
#loc26 = loc(fused["MVN_0?", "_PROF_0_8_1_0", "_outputBuff_cluster_1"])
#loc27 = loc(fused["MVN_1?", "_PROF_0_8_4_1", "_input_cluster_0"])
#loc28 = loc(fused["MVN_1?", "_PROF_0_8_4_1", "_input_cluster_1"])
#loc29 = loc(fused["MVN_1?", "_PROF_0_8_3_0", "_input_cluster_0"])
#loc30 = loc(fused["MVN_1?", "_PROF_0_8_3_0", "_input_cluster_1"])
#loc31 = loc(fused["MVN_1?", "_PROF_0_8_4_1", "_outputBuff_cluster_0"])
#loc32 = loc(fused["MVN_1?", "_PROF_0_8_4_1", "_outputBuff_cluster_1"])
#loc33 = loc(fused["MVN_1?", "_PROF_0_8_3_0", "_outputBuff_cluster_0"])
#loc34 = loc(fused["MVN_1?", "_PROF_0_8_3_0", "_outputBuff_cluster_1"])
#loc35 = loc(fused["Swish_0?", "_PROF_0_8_5_0", "_view_cast"])
#loc36 = loc(fused["MVN_2?", "_PROF_0_8_7_1", "_input_cluster_0"])
#loc37 = loc(fused["MVN_2?", "_PROF_0_8_7_1", "_input_cluster_1"])
#loc38 = loc(fused["MVN_2?", "_PROF_0_8_6_0", "_input_cluster_0"])
#loc39 = loc(fused["MVN_2?", "_PROF_0_8_6_0", "_input_cluster_1"])
#loc40 = loc(fused["MVN_2?", "_PROF_0_8_7_1", "_outputBuff_cluster_0"])
#loc41 = loc(fused["MVN_2?", "_PROF_0_8_7_1", "_outputBuff_cluster_1"])
#loc42 = loc(fused["MVN_2?", "_PROF_0_8_6_0", "_outputBuff_cluster_0"])
#loc43 = loc(fused["MVN_2?", "_PROF_0_8_6_0", "_outputBuff_cluster_1"])
#loc44 = loc(fused["MVN_3?", "_PROF_16_4_1_1", "_input_cluster_0"])
#loc45 = loc(fused["MVN_3?", "_PROF_16_4_1_1", "_input_cluster_1"])
#loc46 = loc(fused["MVN_3?", "_PROF_16_4_0_0", "_input_cluster_0"])
#loc47 = loc(fused["MVN_3?", "_PROF_16_4_0_0", "_input_cluster_1"])
#loc48 = loc(fused["MVN_3?", "_PROF_16_4_1_1", "_outputBuff_cluster_0"])
#loc49 = loc(fused["MVN_3?", "_PROF_16_4_1_1", "_outputBuff_cluster_1"])
#loc50 = loc(fused["MVN_3?", "_PROF_16_4_0_0", "_outputBuff_cluster_0"])
#loc51 = loc(fused["MVN_3?", "_PROF_16_4_0_0", "_outputBuff_cluster_1"])
#loc52 = loc(fused["Tanh_0?", "_PROF_16_4_2_0", "_profilingBuff_cluster_0"])
#loc53 = loc(fused["Tanh_0?", "_PROF_16_4_2_0", "_profilingBuff_cluster_1"])
#loc54 = loc(fused["output", "_PROF_16_4_3_0", "_view_cast"])
#loc55 = loc("Reshape_1423?_PROFBEGIN")
#loc56 = loc("Reshape_1423?_PROFTASKEND_0_1")
#loc57 = loc("Reshape_1423?_PROFTASKBEGIN")
#loc58 = loc("Reshape_1423?_PROFTASKEND_2_2")
#loc59 = loc("MVN_0?/_cluster_0_PROFTASKBEGIN")
#loc60 = loc("MVN_0?/_cluster_0_PROFTASKEND_4_3")
#loc61 = loc("MVN_0?/_cluster_1_PROFTASKBEGIN")
#loc62 = loc("MVN_0?/_cluster_1_PROFTASKEND_20_11")
#loc63 = loc(fused["MVN_0?", "_PROF_0_8_1_0", "_profilingBuff_cluster_0"])
#loc64 = loc(fused["MVN_0?", "_PROF_0_8_1_0", "_profilingBuff_cluster_1"])
#loc65 = loc(fused["MVN_0?", "_PROF_0_8_2_1", "_profilingBuff_cluster_0"])
#loc66 = loc(fused["MVN_0?", "_PROF_0_8_2_1", "_profilingBuff_cluster_1"])
#loc67 = loc(fused["MVN_1?", "_PROF_0_8_3_0", "_profilingBuff_cluster_0"])
#loc68 = loc(fused["MVN_1?", "_PROF_0_8_3_0", "_profilingBuff_cluster_1"])
#loc69 = loc(fused["MVN_1?", "_PROF_0_8_4_1", "_profilingBuff_cluster_0"])
#loc70 = loc(fused["MVN_1?", "_PROF_0_8_4_1", "_profilingBuff_cluster_1"])
#loc71 = loc("MVN_1?/_cluster_0_PROFTASKBEGIN")
#loc72 = loc("MVN_1?/_cluster_0_PROFTASKEND_6_4")
#loc73 = loc("MVN_1?/_cluster_1_PROFTASKBEGIN")
#loc74 = loc("MVN_1?/_cluster_1_PROFTASKEND_22_12")
#loc75 = loc("Swish_0?_PROFTASKBEGIN")
#loc76 = loc("Swish_0?_PROFTASKEND_8_5")
#loc77 = loc("MVN_2?/_cluster_0_PROFTASKBEGIN")
#loc78 = loc("MVN_2?/_cluster_0_PROFTASKEND_10_6")
#loc79 = loc("MVN_2?/_cluster_1_PROFTASKBEGIN")
#loc80 = loc("MVN_2?/_cluster_1_PROFTASKEND_24_13")
#loc81 = loc(fused["MVN_2?", "_PROF_0_8_6_0", "_profilingBuff_cluster_0"])
#loc82 = loc(fused["MVN_2?", "_PROF_0_8_6_0", "_profilingBuff_cluster_1"])
#loc83 = loc(fused["MVN_2?", "_PROF_0_8_7_1", "_profilingBuff_cluster_0"])
#loc84 = loc(fused["MVN_2?", "_PROF_0_8_7_1", "_profilingBuff_cluster_1"])
#loc85 = loc(fused["MVN_3?", "_PROF_16_4_0_0", "_profilingBuff_cluster_0"])
#loc86 = loc(fused["MVN_3?", "_PROF_16_4_0_0", "_profilingBuff_cluster_1"])
#loc87 = loc(fused["MVN_3?", "_PROF_16_4_1_1", "_profilingBuff_cluster_0"])
#loc88 = loc(fused["MVN_3?", "_PROF_16_4_1_1", "_profilingBuff_cluster_1"])
#loc89 = loc("MVN_3?/_cluster_0_PROFTASKBEGIN")
#loc90 = loc("MVN_3?/_cluster_0_PROFTASKEND_12_7")
#loc91 = loc("MVN_3?/_cluster_1_PROFTASKBEGIN")
#loc92 = loc("MVN_3?/_cluster_1_PROFTASKEND_26_14")
#loc93 = loc("dmaProfilingCMX2DDR160")
#loc94 = loc("Tanh_0?/_broadcast_copy_to_CMX[0,1]_PROFTASKBEGIN")
#loc95 = loc("Tanh_0?/_broadcast_copy_to_CMX[0,1]_PROFTASKEND_14_8")
#loc96 = loc("dmaProfilingCMX2DDR0")
#loc97 = loc("Tanh_0?_PROFTASKBEGIN")
#loc98 = loc("Tanh_0?_PROFTASKEND_16_9")
#loc99 = loc("output_PROFTASKBEGIN")
#loc100 = loc("output_PROFTASKEND_18_10")
#loc101 = loc("dmaProfilingCMX2DDR128")
#loc102 = loc("newProfilingBuffer")
#loc103 = loc(fused["Reshape_1423?", "_PROF_0_8_0_0"])
#loc104 = loc(fused["MVN_0?", "_cluster_0"])
#loc105 = loc(fused["MVN_0?", "_cluster_1"])
#loc106 = loc(fused["MVN_0?", "_PROF_0_8_1_0", "_cluster_0"])
#loc107 = loc(fused["MVN_0?", "_PROF_0_8_1_0", "_cluster_1"])
#loc108 = loc(fused["MVN_0?", "_PROF_0_8_2_1", "_cluster_0"])
#loc109 = loc(fused["MVN_0?", "_PROF_0_8_2_1", "_cluster_1"])
#loc110 = loc(fused["MVN_1?", "_PROF_0_8_3_0", "_cluster_0"])
#loc111 = loc(fused["MVN_1?", "_PROF_0_8_3_0", "_cluster_1"])
#loc112 = loc(fused["MVN_1?", "_PROF_0_8_4_1", "_cluster_0"])
#loc113 = loc(fused["MVN_1?", "_PROF_0_8_4_1", "_cluster_1"])
#loc114 = loc(fused["MVN_1?", "_cluster_0"])
#loc115 = loc(fused["MVN_1?", "_cluster_1"])
#loc116 = loc(fused["Swish_0?", "_PROF_0_8_5_0"])
#loc117 = loc(fused["MVN_2?", "_cluster_0"])
#loc118 = loc(fused["MVN_2?", "_cluster_1"])
#loc119 = loc(fused["MVN_2?", "_PROF_0_8_6_0", "_cluster_0"])
#loc120 = loc(fused["MVN_2?", "_PROF_0_8_6_0", "_cluster_1"])
#loc121 = loc(fused["MVN_2?", "_PROF_0_8_7_1", "_cluster_0"])
#loc122 = loc(fused["MVN_2?", "_PROF_0_8_7_1", "_cluster_1"])
#loc123 = loc(fused["actshaveProfilingCMX2DDR0", "_cluster_0"])
#loc124 = loc(fused["actshaveProfilingCMX2DDR0", "_cluster_1"])
#loc125 = loc(fused["MVN_3?", "_PROF_16_4_0_0", "_cluster_0"])
#loc126 = loc(fused["MVN_3?", "_PROF_16_4_0_0", "_cluster_1"])
#loc127 = loc(fused["MVN_3?", "_PROF_16_4_1_1", "_cluster_0"])
#loc128 = loc(fused["MVN_3?", "_PROF_16_4_1_1", "_cluster_1"])
#loc129 = loc(fused["MVN_3?", "_cluster_0"])
#loc130 = loc(fused["MVN_3?", "_cluster_1"])
#loc131 = loc(fused["Tanh_0?", "_broadcast_copy_to_CMX[0,1]"])
#loc132 = loc(fused["Tanh_0?", "_PROF_16_4_2_0", "_cluster_0"])
#loc133 = loc(fused["Tanh_0?", "_PROF_16_4_2_0", "_cluster_1"])
#loc134 = loc(fused["output", "_PROF_16_4_3_0"])
#loc135 = loc(fused["actshaveProfilingCMX2DDR16", "_cluster_0"])
#loc136 = loc(fused["actshaveProfilingCMX2DDR16", "_cluster_1"])

//CHECK: {"traceEvents":[
//CHECK: {"name":"Reshape_1423", "cat":"DMA", "ph":"X", "ts":0.000, "dur":2.369, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Reshape_1423", "cat":"DMA", "ph":"X", "ts":84.166, "dur":1.744, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_0?/_cluster_0", "cat":"DMA", "ph":"X", "ts":86.562, "dur":1.458, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_0?/_cluster_1", "cat":"DMA", "ph":"X", "ts":86.953, "dur":1.458, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_1?/_cluster_0", "cat":"DMA", "ph":"X", "ts":127.942, "dur":1.250, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_1?/_cluster_1", "cat":"DMA", "ph":"X", "ts":128.333, "dur":1.458, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Swish_0", "cat":"DMA", "ph":"X", "ts":166.510, "dur":1.718, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_2?/_cluster_0", "cat":"DMA", "ph":"X", "ts":168.880, "dur":1.484, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_2?/_cluster_1", "cat":"DMA", "ph":"X", "ts":169.271, "dur":1.536, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_3?/_cluster_0", "cat":"DMA", "ph":"X", "ts":194.479, "dur":1.458, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_3?/_cluster_1", "cat":"DMA", "ph":"X", "ts":194.869, "dur":1.484, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Tanh_0?/_broadcast_copy_to_CMX[0,1]", "cat":"DMA", "ph":"X", "ts":197.474, "dur":1.744, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"Tanh_0", "cat":"DMA", "ph":"X", "ts":212.734, "dur":1.744, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":229.661, "dur":2.239, "pid":1, "tid":2, "args":{}},
//CHECK: {"name":"MVN_0", "cat":"SW", "ph":"X", "ts":89.817, "dur":24.661, "pid":1, "tid":7, "args":{}},
//CHECK: {"name":"MVN_1", "cat":"SW", "ph":"X", "ts":117.369, "dur":8.802, "pid":1, "tid":7, "args":{}},
//CHECK: {"name":"MVN_2", "cat":"SW", "ph":"X", "ts":172.213, "dur":8.749, "pid":1, "tid":7, "args":{}},
//CHECK: {"name":"MVN_3", "cat":"SW", "ph":"X", "ts":183.750, "dur":9.010, "pid":1, "tid":7, "args":{}},
//CHECK: {"name":"Reshape_1423", "cat":"SW", "ph":"X", "ts":51.146, "dur":30.338, "pid":1, "tid":7, "args":{}},
//CHECK: {"name":"Swish_0", "cat":"SW", "ph":"X", "ts":131.302, "dur":33.463, "pid":1, "tid":7, "args":{}},
//CHECK: {"name":"Tanh_0", "cat":"SW", "ph":"X", "ts":201.146, "dur":9.895, "pid":1, "tid":7, "args":{}},
//CHECK: {"name":"output", "cat":"SW", "ph":"X", "ts":216.067, "dur":11.380, "pid":1, "tid":7, "args":{}},
//CHECK: {"name":"MVN_0?/cluster_0", "cat":"SW", "ph":"X", "ts":90.182, "dur":24.296, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"MVN_0?/cluster_1", "cat":"SW", "ph":"X", "ts":89.817, "dur":24.296, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"MVN_1?/cluster_0", "cat":"SW", "ph":"X", "ts":117.864, "dur":8.307, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"MVN_1?/cluster_1", "cat":"SW", "ph":"X", "ts":117.369, "dur":8.307, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"MVN_2?/cluster_0", "cat":"SW", "ph":"X", "ts":172.422, "dur":8.229, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"MVN_2?/cluster_1", "cat":"SW", "ph":"X", "ts":172.213, "dur":8.749, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"MVN_3?/cluster_0", "cat":"SW", "ph":"X", "ts":184.036, "dur":8.177, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"MVN_3?/cluster_1", "cat":"SW", "ph":"X", "ts":183.750, "dur":9.010, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"Reshape_1423?/cluster_0", "cat":"SW", "ph":"X", "ts":51.146, "dur":30.338, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"Swish_0?/cluster_0", "cat":"SW", "ph":"X", "ts":131.302, "dur":33.463, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"Tanh_0?/cluster_0", "cat":"SW", "ph":"X", "ts":201.146, "dur":9.713, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"Tanh_0?/cluster_1", "cat":"SW", "ph":"X", "ts":201.354, "dur":9.687, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"output/cluster_0", "cat":"SW", "ph":"X", "ts":216.067, "dur":11.380, "pid":1, "tid":6, "args":{}},
//CHECK: {"name":"Reshape_1423?/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":51.146, "dur":30.338, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_0?/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":90.390, "dur":23.906, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_0?/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":89.817, "dur":24.114, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"MVN_0?/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":90.182, "dur":24.296, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_0?/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":90.000, "dur":24.114, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"MVN_1?/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":117.864, "dur":8.125, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_1?/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":117.369, "dur":8.098, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"MVN_1?/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":118.073, "dur":8.098, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_1?/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":117.552, "dur":8.125, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"Swish_0?/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":131.302, "dur":33.463, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_2?/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":172.422, "dur":8.046, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_2?/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":172.213, "dur":8.072, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"MVN_2?/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":172.604, "dur":8.046, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_2?/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":172.786, "dur":8.177, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"MVN_3?/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":184.036, "dur":7.994, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_3?/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":183.750, "dur":7.994, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"MVN_3?/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":184.218, "dur":7.994, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"MVN_3?/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":184.687, "dur":8.072, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"Tanh_0?/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":201.146, "dur":9.713, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"Tanh_0?/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":201.354, "dur":9.687, "pid":1, "tid":5, "args":{}},
//CHECK: {"name":"output/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":216.067, "dur":11.380, "pid":1, "tid":4, "args":{}},
//CHECK: {"name":"Reshape_1423", "cat":"Layer", "ph":"X", "ts":0.000, "dur":85.910, "pid":1, "tid":8, "args":{}},
//CHECK: {"name":"Swish_0", "cat":"Layer", "ph":"X", "ts":131.302, "dur":36.926, "pid":1, "tid":8, "args":{}},
//CHECK: {"name":"Tanh_0", "cat":"Layer", "ph":"X", "ts":197.474, "dur":17.004, "pid":1, "tid":8, "args":{}},
//CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":216.067, "dur":15.833, "pid":1, "tid":8, "args":{}},
//CHECK: {"name":"MVN_0", "cat":"Layer", "ph":"X", "ts":89.817, "dur":24.661, "pid":1, "tid":8, "args":{}},
//CHECK: {"name":"MVN_1", "cat":"Layer", "ph":"X", "ts":117.369, "dur":8.802, "pid":1, "tid":8, "args":{}},
//CHECK: {"name":"MVN_2", "cat":"Layer", "ph":"X", "ts":172.213, "dur":8.749, "pid":1, "tid":8, "args":{}},
//CHECK: {"name":"MVN_3", "cat":"Layer", "ph":"X", "ts":183.750, "dur":9.010, "pid":1, "tid":8, "args":{}},
//CHECK: {"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name" : "Inference"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name" : "DMA"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 3, "args": {"name" : "DPU"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 4, "args": {"name" : "SW Cluster[0]"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 5, "args": {"name" : "SW Cluster[1]"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 6, "args": {"name" : "SW Clusters"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 7, "args": {"name" : "SW"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid": 1, "tid": 8, "args": {"name" : "Layers"}}
//CHECK: ],
//CHECK: "displayTimeUnit": "ns"
//CHECK: }
