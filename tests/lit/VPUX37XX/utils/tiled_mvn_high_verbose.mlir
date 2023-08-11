// RUN: vpux-translate --export-VPUIP -o %t %s && prof_parser -b %t -p %data_path_37XX%/profiling-0-37XX-MVN.bin -f json -vv | FileCheck %s

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
    func.func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "tanh_fp16.cpp", VPU.kernel_entry = "tanh_fp16"} loc(#loc0)
    func.func private @builtin_Swish(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, f64) attributes {VPU.kernel_code = "swish_fp16.cpp", VPU.kernel_entry = "swish_fp16"} loc(#loc0)
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"} loc(#loc0)
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "single_shave_convert.cpp", VPU.kernel_entry = "single_shave_convert"} loc(#loc0)
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"} loc(#loc0)
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
    DataInfo "0_actshave_384_dma_656_pll" : tensor<166xui32> loc(#loc1)
  } loc(#loc0)
  func.func @main(%arg0: memref<1x4x512xf32, @DDR> loc(unknown), %arg1: memref<1x4x512xf32, @DDR> loc(unknown), %arg2: memref<166xui32> loc("profiling_result")) -> (memref<1x4x512xf32, @DDR>, memref<166xui32>) {
    %pll_in0 = VPURT.DeclareBuffer "Register" <537403424> -> memref<1xui32, @Register> loc(#loc140)
    %pll_out0 = VPURT.DeclareBuffer "ProfilingOutput" [0] <656> -> memref<1xui32> loc(#loc140)
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
    %17 = VPURT.ConfigureBarrier<17> -> !VPURT.Barrier loc(#loc3)
    %18 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x4x512xf32, @DDR> loc(#loc4)
    %19 = VPURT.DeclareBuffer "CMX_NN" [0] <4352> -> memref<16xui32, [@CMX_NN, 0]> loc(#loc5)
    %20 = VPURT.DeclareBuffer "CMX_NN" [1] <4352> -> memref<16xui32, [@CMX_NN, 1]> loc(#loc5)
    %21 = VPURT.DeclareBuffer "CMX_NN" [0] <12544> -> memref<32xui32, [@CMX_NN, 0]> loc(#loc6)
    %22 = VPURT.DeclareBuffer "CMX_NN" [1] <12544> -> memref<32xui32, [@CMX_NN, 1]> loc(#loc6)
    %23 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x4x1x512xf32, [@CMX_NN, 0]> loc(#loc3)
    %24 = VPURT.DeclareBuffer "CMX_NN" [0] <8448> -> memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc3)
    %25 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x512x1xf16, @DDR> loc(#loc3)
    %26 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc7)
    %27 = VPURT.DeclareBuffer "DDR" <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc7)
    %28 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc7)
    %29 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc7)
    %30 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc8)
    %31 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc8)
    %32 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc9)
    %33 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc8)
    %34 = VPURT.DeclareBuffer "CMX_NN" [0] <4352> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc8)
    %35 = VPURT.DeclareBuffer "CMX_NN" [0] <6400> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc9)
    %36 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x512x1xf16, @DDR> loc(#loc9)
    %37 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc10)
    %38 = VPURT.DeclareBuffer "DDR" <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc10)
    %39 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc10)
    %40 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc10)
    %41 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc11)
    %42 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc11)
    %43 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc12)
    %44 = VPURT.DeclareBuffer "DDR" <512> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc12)
    %45 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc11)
    %46 = VPURT.DeclareBuffer "DDR" <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc11)
    %47 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc13)
    %48 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc14)
    %49 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc12)
    %50 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc12)
    %51 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc12)
    %52 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc12)
    %53 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc15)
    %54 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc16)
    %55 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc12)
    %56 = VPURT.DeclareBuffer "DDR" <512> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc12)
    %57 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc17)
    %58 = VPURT.DeclareBuffer "CMX_NN" [0] <4416> -> memref<1x4x1x512xf32, [@CMX_NN, 0]> loc(#loc17)
    %59 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x4x1x512xf32, @DDR> loc(#loc3)
    %60 = VPURT.DeclareBuffer "CMX_NN" [0] <12544> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc18)
    %61 = VPURT.DeclareBuffer "CMX_NN" [0] <8448> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc3)
    %62 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc19)
    %63 = VPURT.DeclareBuffer "CMX_NN" [1] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc20)
    %64 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc21)
    %65 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc22)
    %66 = VPURT.DeclareBuffer "CMX_NN" [0] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc23)
    %67 = VPURT.DeclareBuffer "CMX_NN" [1] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc24)
    %68 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc25)
    %69 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc26)
    %70 = VPURT.DeclareBuffer "CMX_NN" [0] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc27)
    %71 = VPURT.DeclareBuffer "CMX_NN" [1] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc28)
    %72 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc29)
    %73 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc30)
    %74 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc31)
    %75 = VPURT.DeclareBuffer "CMX_NN" [1] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc32)
    %76 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc33)
    %77 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc34)
    %78 = VPURT.DeclareBuffer "CMX_NN" [0] <12624> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc35)
    %79 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc36)
    %80 = VPURT.DeclareBuffer "CMX_NN" [1] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc37)
    %81 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc38)
    %82 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc39)
    %83 = VPURT.DeclareBuffer "CMX_NN" [0] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc40)
    %84 = VPURT.DeclareBuffer "CMX_NN" [1] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc41)
    %85 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc42)
    %86 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc43)
    %87 = VPURT.DeclareBuffer "CMX_NN" [0] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc44)
    %88 = VPURT.DeclareBuffer "CMX_NN" [1] <3328> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc45)
    %89 = VPURT.DeclareBuffer "CMX_NN" [0] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc46)
    %90 = VPURT.DeclareBuffer "CMX_NN" [1] <2304> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc47)
    %91 = VPURT.DeclareBuffer "CMX_NN" [0] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc48)
    %92 = VPURT.DeclareBuffer "CMX_NN" [1] <1280> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc49)
    %93 = VPURT.DeclareBuffer "CMX_NN" [0] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc50)
    %94 = VPURT.DeclareBuffer "CMX_NN" [1] <256> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc51)
    %95 = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<32xui32, @DDR> loc(#loc6)
    %96 = VPURT.DeclareBuffer "ProfilingOutput" [0] <128> -> memref<32xui32, @DDR> loc(#loc6)
    %97 = VPURT.DeclareBuffer "CMX_NN" [0] <4384> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc52)
    %98 = VPURT.DeclareBuffer "CMX_NN" [1] <4384> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc53)
    %99 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x1x512xf16, @DDR> loc(#loc17)
    %100 = VPURT.DeclareBuffer "CMX_NN" [0] <4400> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc54)
    %101 = VPURT.DeclareBuffer "ProfilingOutput" [0] <256> -> memref<16xui32, @DDR> loc(#loc5)
    %102 = VPURT.DeclareBuffer "ProfilingOutput" [0] <320> -> memref<16xui32, @DDR> loc(#loc5)
    %103 = VPURT.DeclareBuffer "CMX_NN" [0] <4416> -> memref<1x4x512xf32, [@CMX_NN, 0]> loc(#loc17)
    %104 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc55)
    %105 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc55)
    %106 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc56)
    %107 = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc56)
    %108 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc55)
    %109 = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc55)
    %110 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc57)
    %111 = VPURT.DeclareBuffer "CMX_NN" [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc57)
    %112 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc58)
    %113 = VPURT.DeclareBuffer "CMX_NN" [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc58)
    %114 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc59)
    %115 = VPURT.DeclareBuffer "CMX_NN" [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc59)
    %116 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc60)
    %117 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc60)
    %118 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc61)
    %119 = VPURT.DeclareBuffer "CMX_NN" [0] <136> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc61)
    %120 = VPURT.DeclareBuffer "CMX_NN" [0] <12560> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc62)
    %121 = VPURT.DeclareBuffer "CMX_NN" [1] <12560> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc63)
    %122 = VPURT.DeclareBuffer "CMX_NN" [0] <12576> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc64)
    %123 = VPURT.DeclareBuffer "CMX_NN" [1] <12576> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc65)
    %124 = VPURT.DeclareBuffer "CMX_NN" [0] <12592> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc66)
    %125 = VPURT.DeclareBuffer "CMX_NN" [1] <12592> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc67)
    %126 = VPURT.DeclareBuffer "CMX_NN" [0] <12608> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc68)
    %127 = VPURT.DeclareBuffer "CMX_NN" [1] <12608> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc69)
    %128 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc70)
    %129 = VPURT.DeclareBuffer "CMX_NN" [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc70)
    %130 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc71)
    %131 = VPURT.DeclareBuffer "CMX_NN" [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc71)
    %132 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc72)
    %133 = VPURT.DeclareBuffer "CMX_NN" [0] <144> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc72)
    %134 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc73)
    %135 = VPURT.DeclareBuffer "CMX_NN" [0] <152> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc73)
    %136 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc74)
    %137 = VPURT.DeclareBuffer "CMX_NN" [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc74)
    %138 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc75)
    %139 = VPURT.DeclareBuffer "CMX_NN" [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc75)
    %140 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc76)
    %141 = VPURT.DeclareBuffer "CMX_NN" [0] <80> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc76)
    %142 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc77)
    %143 = VPURT.DeclareBuffer "CMX_NN" [0] <88> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc77)
    %144 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc78)
    %145 = VPURT.DeclareBuffer "CMX_NN" [0] <160> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc78)
    %146 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc79)
    %147 = VPURT.DeclareBuffer "CMX_NN" [0] <168> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc79)
    %148 = VPURT.DeclareBuffer "CMX_NN" [0] <12640> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc80)
    %149 = VPURT.DeclareBuffer "CMX_NN" [1] <12640> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc81)
    %150 = VPURT.DeclareBuffer "CMX_NN" [0] <12656> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc82)
    %151 = VPURT.DeclareBuffer "CMX_NN" [1] <12656> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc83)
    %152 = VPURT.DeclareBuffer "CMX_NN" [0] <4352> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc84)
    %153 = VPURT.DeclareBuffer "CMX_NN" [1] <4352> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc85)
    %154 = VPURT.DeclareBuffer "CMX_NN" [0] <4368> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc86)
    %155 = VPURT.DeclareBuffer "CMX_NN" [1] <4368> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc87)
    %156 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc88)
    %157 = VPURT.DeclareBuffer "CMX_NN" [0] <96> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc88)
    %158 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc89)
    %159 = VPURT.DeclareBuffer "CMX_NN" [0] <104> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc89)
    %160 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc90)
    %161 = VPURT.DeclareBuffer "CMX_NN" [0] <176> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc90)
    %162 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc91)
    %163 = VPURT.DeclareBuffer "CMX_NN" [0] <184> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    %164 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc92)
    %165 = VPURT.DeclareBuffer "CMX_NN" [0] <112> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    %166 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc93)
    %167 = VPURT.DeclareBuffer "CMX_NN" [0] <120> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc93)
    %168 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16xui64, [@CMX_NN, 0]> loc(#loc94)
    %169 = VPURT.DeclareBuffer "ProfilingOutput" [0] <384> -> memref<16xui64> loc(#loc94)
    %170 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc95)
    %171 = VPURT.DeclareBuffer "CMX_NN" [0] <192> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    %172 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc96)
    %173 = VPURT.DeclareBuffer "CMX_NN" [0] <200> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc96)
    %174 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc92)
    %175 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    %176 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc97)
    %177 = VPURT.DeclareBuffer "CMX_NN" [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    %178 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc95)
    %179 = VPURT.DeclareBuffer "CMX_NN" [0] <208> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    %180 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc98)
    %181 = VPURT.DeclareBuffer "CMX_NN" [0] <216> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc98)
    %182 = VPURT.DeclareBuffer "CMX_NN" [0] <128> -> memref<12xui64, [@CMX_NN, 0]> loc(#loc99)
    %183 = VPURT.DeclareBuffer "ProfilingOutput" [0] <560> -> memref<12xui64> loc(#loc99)
    %184 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc100)
    %185 = VPURT.DeclareBuffer "CMX_NN" [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    %186 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc101)
    %187 = VPURT.DeclareBuffer "CMX_NN" [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc101)
    %188 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc100)
    %189 = VPURT.DeclareBuffer "CMX_NN" [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    %190 = VPURT.DeclareBuffer "Register" <637702144> -> memref<1xui64, @Register> loc(#loc102)
    %191 = VPURT.DeclareBuffer "CMX_NN" [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc102)
    %192 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<6xui64, [@CMX_NN, 0]> loc(#loc103)
    %193 = VPURT.DeclareBuffer "ProfilingOutput" [0] <512> -> memref<6xui64> loc(#loc103)
    %194 = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<96xui32> loc(#loc104)
    %195 = VPURT.DeclareBuffer "ProfilingOutput" [0] <384> -> memref<34xui64> loc(#loc104)
    %pll_in1 = VPURT.DeclareBuffer "Register" <537403424> -> memref<1xui32, @Register> loc(#loc140)
    %pll_out1 = VPURT.DeclareBuffer "ProfilingOutput" [0] <660> -> memref<1xui32> loc(#loc140)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%pll_in0 : memref<1xui32, @Register>) outputs(%pll_out0 : memref<1xui32>) -> memref<1xui32> loc(#loc140)
    } loc(#loc140)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%104 : memref<1xui64, @Register>) outputs(%105 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc55)
    } loc(#loc55)
    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 1148 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%59 : memref<1x4x1x512xf32, @DDR>) outputs(%23 : memref<1x4x1x512xf32, [@CMX_NN, 0]>) -> memref<1x4x1x512xf32, [@CMX_NN, 0]> loc(#loc3)
    } loc(#loc3)
    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%106 : memref<1xui64, @Register>) outputs(%107 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc56)
    } loc(#loc56)
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) attributes {cycleBegin = 1148 : i64, cycleEnd = 1150 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%23 as %arg3: memref<1x4x1x512xf32, [@CMX_NN, 0]>) outputs(%24 as %arg4: memref<1x4x1x512xf16, [@CMX_NN, 0]>) profiling_data(%60 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x1x512xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x1x512xf32, [@CMX_NN, 0]>, memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc105)
    } loc(#loc105)
    VPURT.Task waits(%1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%108 : memref<1xui64, @Register>) outputs(%109 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc55)
    } loc(#loc55)
    VPURT.Task attributes {cycleBegin = 1150 : i64, cycleEnd = 2298 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%61 : memref<1x4x512x1xf16, [@CMX_NN, 0]>) outputs(%25 : memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR> loc(#loc3)
    } loc(#loc3)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%110 : memref<1xui64, @Register>) outputs(%111 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc57)
    } loc(#loc57)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%112 : memref<1xui64, @Register>) outputs(%113 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc58)
    } loc(#loc58)
    VPURT.Task attributes {cycleBegin = 2298 : i64, cycleEnd = 3446 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%26 : memref<1x2x512x1xf16, @DDR>) outputs(%28 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc106)
    } loc(#loc106)
    VPURT.Task updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%114 : memref<1xui64, @Register>) outputs(%115 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc59)
    } loc(#loc59)
    VPURT.Task waits(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%116 : memref<1xui64, @Register>) outputs(%117 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc60)
    } loc(#loc60)
    VPURT.Task attributes {cycleBegin = 2298 : i64, cycleEnd = 3446 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%27 : memref<1x2x512x1xf16, @DDR>) outputs(%29 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc107)
    } loc(#loc107)
    VPURT.Task updates(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%118 : memref<1xui64, @Register>) outputs(%119 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc61)
    } loc(#loc61)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 3446 : i64, cycleEnd = 3448 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%64 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%68 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%120 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc108)
    } loc(#loc108)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 3446 : i64, cycleEnd = 3448 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%65 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%69 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%121 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc109)
    } loc(#loc109)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 3446 : i64, cycleEnd = 3448 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%62 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%66 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%122 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc110)
    } loc(#loc110)
    VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) attributes {cycleBegin = 3446 : i64, cycleEnd = 3448 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%63 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%67 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%123 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc111)
    } loc(#loc111)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 3448 : i64, cycleEnd = 3450 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%72 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%76 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%124 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc112)
    } loc(#loc112)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 3448 : i64, cycleEnd = 3450 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%73 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%77 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%125 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc113)
    } loc(#loc113)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 3448 : i64, cycleEnd = 3450 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%70 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%74 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%126 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc114)
    } loc(#loc114)
    VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) attributes {cycleBegin = 3448 : i64, cycleEnd = 3450 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%71 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%75 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%127 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc115)
    } loc(#loc115)
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%128 : memref<1xui64, @Register>) outputs(%129 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc70)
    } loc(#loc70)
    VPURT.Task attributes {cycleBegin = 3450 : i64, cycleEnd = 4598 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%30 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) outputs(%33 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc116)
    } loc(#loc116)
    VPURT.Task updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%130 : memref<1xui64, @Register>) outputs(%131 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc71)
    } loc(#loc71)
    VPURT.Task waits(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%132 : memref<1xui64, @Register>) outputs(%133 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc72)
    } loc(#loc72)
    VPURT.Task attributes {cycleBegin = 3450 : i64, cycleEnd = 4598 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%31 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) outputs(%34 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc117)
    } loc(#loc117)
    VPURT.Task updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%134 : memref<1xui64, @Register>) outputs(%135 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc73)
    } loc(#loc73)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {cycleBegin = 4598 : i64, cycleEnd = 4600 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%32 as %arg3: memref<1x4x512x1xf16, [@CMX_NN, 0]>) outputs(%35 as %arg4: memref<1x4x512x1xf16, [@CMX_NN, 0]>) profiling_data(%78 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x512x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x4x512x1xf16, [@CMX_NN, 0]>, memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc118)
    } loc(#loc118)
    VPURT.Task waits(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%136 : memref<1xui64, @Register>) outputs(%137 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc74)
    } loc(#loc74)
    VPURT.Task attributes {cycleBegin = 4600 : i64, cycleEnd = 5748 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%35 : memref<1x4x512x1xf16, [@CMX_NN, 0]>) outputs(%36 : memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR> loc(#loc9)
    } loc(#loc9)
    VPURT.Task updates(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%138 : memref<1xui64, @Register>) outputs(%139 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc75)
    } loc(#loc75)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%140 : memref<1xui64, @Register>) outputs(%141 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc76)
    } loc(#loc76)
    VPURT.Task attributes {cycleBegin = 5748 : i64, cycleEnd = 6896 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%37 : memref<1x2x512x1xf16, @DDR>) outputs(%39 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc119)
    } loc(#loc119)
    VPURT.Task updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%142 : memref<1xui64, @Register>) outputs(%143 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc77)
    } loc(#loc77)
    VPURT.Task waits(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%144 : memref<1xui64, @Register>) outputs(%145 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc78)
    } loc(#loc78)
    VPURT.Task attributes {cycleBegin = 5748 : i64, cycleEnd = 6896 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%38 : memref<1x2x512x1xf16, @DDR>) outputs(%40 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc120)
    } loc(#loc120)
    VPURT.Task updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%146 : memref<1xui64, @Register>) outputs(%147 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc79)
    } loc(#loc79)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 6896 : i64, cycleEnd = 6898 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%81 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%85 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%148 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc121)
    } loc(#loc121)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 6896 : i64, cycleEnd = 6898 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%82 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%86 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%149 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc122)
    } loc(#loc122)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 6896 : i64, cycleEnd = 6898 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%79 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%83 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%150 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc123)
    } loc(#loc123)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 6896 : i64, cycleEnd = 6898 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%80 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%84 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%151 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc124)
    } loc(#loc124)
    VPURT.Task waits(%10 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 7855 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%21 : memref<32xui32, [@CMX_NN, 0]>) outputs(%95 : memref<32xui32, @DDR>) -> memref<32xui32, @DDR> loc(#loc125)
    } loc(#loc125)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 7855 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%22 : memref<32xui32, [@CMX_NN, 1]>) outputs(%96 : memref<32xui32, @DDR>) -> memref<32xui32, @DDR> loc(#loc126)
    } loc(#loc126)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 6900 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%89 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%93 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%152 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc127)
    } loc(#loc127)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 6900 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%90 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%94 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%153 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc128)
    } loc(#loc128)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 6900 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%87 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%91 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%154 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc129)
    } loc(#loc129)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%12 : !VPURT.Barrier) attributes {cycleBegin = 6898 : i64, cycleEnd = 6900 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%88 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%92 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%155 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc130)
    } loc(#loc130)
    VPURT.Task waits(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%156 : memref<1xui64, @Register>) outputs(%157 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc88)
    } loc(#loc88)
    VPURT.Task attributes {cycleBegin = 7855 : i64, cycleEnd = 9003 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%41 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) outputs(%45 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc131)
    } loc(#loc131)
    VPURT.Task updates(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%158 : memref<1xui64, @Register>) outputs(%159 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc89)
    } loc(#loc89)
    VPURT.Task waits(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%160 : memref<1xui64, @Register>) outputs(%161 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc90)
    } loc(#loc90)
    VPURT.Task attributes {cycleBegin = 7855 : i64, cycleEnd = 9003 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%42 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) outputs(%46 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc132)
    } loc(#loc132)
    VPURT.Task updates(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%162 : memref<1xui64, @Register>) outputs(%163 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    } loc(#loc91)
    VPURT.Task waits(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%164 : memref<1xui64, @Register>) outputs(%165 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    } loc(#loc92)
    VPURT.Task attributes {cycleBegin = 9003 : i64, cycleEnd = 10151 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%43 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%49 : memref<1x4x256x1xf16, [@CMX_NN, 0]>) -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc133)
    } loc(#loc133)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%166 : memref<1xui64, @Register>) outputs(%167 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc93)
    } loc(#loc93)
    VPURT.Task updates(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%168 : memref<16xui64, [@CMX_NN, 0]>) outputs(%169 : memref<16xui64>) -> memref<16xui64> loc(#loc94)
    } loc(#loc94)
    VPURT.Task waits(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%170 : memref<1xui64, @Register>) outputs(%171 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    } loc(#loc95)
    VPURT.Task attributes {cycleBegin = 9003 : i64, cycleEnd = 10151 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%44 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%50 : memref<1x4x256x1xf16, [@CMX_NN, 1]>) -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc134)
    } loc(#loc134)
    VPURT.Task updates(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%172 : memref<1xui64, @Register>) outputs(%173 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc96)
    } loc(#loc96)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) attributes {cycleBegin = 10151 : i64, cycleEnd = 10153 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Tanh inputs(%47 as %arg3: memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs(%53 as %arg4: memref<1x4x256x1xf16, [@CMX_NN, 0]>) profiling_data(%97 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x256x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x256x1xf16, [@CMX_NN, 0]>, memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc135)
    } loc(#loc135)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) attributes {cycleBegin = 10151 : i64, cycleEnd = 10153 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Tanh inputs(%48 as %arg3: memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs(%54 as %arg4: memref<1x4x256x1xf16, [@CMX_NN, 1]>) profiling_data(%98 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x4x256x1xf16, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x256x1xf16, [@CMX_NN, 1]>, memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc136)
    } loc(#loc136)
    VPURT.Task waits(%15 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%174 : memref<1xui64, @Register>) outputs(%175 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    } loc(#loc92)
    VPURT.Task attributes {cycleBegin = 10153 : i64, cycleEnd = 11301 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%51 : memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs(%55 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc133)
    } loc(#loc133)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%176 : memref<1xui64, @Register>) outputs(%177 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    } loc(#loc97)
    VPURT.Task waits(%15 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%178 : memref<1xui64, @Register>) outputs(%179 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    } loc(#loc95)
    VPURT.Task attributes {cycleBegin = 10153 : i64, cycleEnd = 11301 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%52 : memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs(%56 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc134)
    } loc(#loc134)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%180 : memref<1xui64, @Register>) outputs(%181 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc98)
    } loc(#loc98)
    VPURT.Task updates(%16 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%182 : memref<12xui64, [@CMX_NN, 0]>) outputs(%183 : memref<12xui64>) -> memref<12xui64> loc(#loc99)
    } loc(#loc99)
    VPURT.Task waits(%16 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%184 : memref<1xui64, @Register>) outputs(%185 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    } loc(#loc100)
    VPURT.Task attributes {cycleBegin = 11301 : i64, cycleEnd = 12449 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%99 : memref<1x4x1x512xf16, @DDR>) outputs(%57 : memref<1x4x1x512xf16, [@CMX_NN, 0]>) -> memref<1x4x1x512xf16, [@CMX_NN, 0]> loc(#loc17)
    } loc(#loc17)
    VPURT.Task updates(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%186 : memref<1xui64, @Register>) outputs(%187 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc101)
    } loc(#loc101)
    VPURT.Task waits(%11 : !VPURT.Barrier) updates(%17 : !VPURT.Barrier) attributes {cycleBegin = 12449 : i64, cycleEnd = 12451 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%57 as %arg3: memref<1x4x1x512xf16, [@CMX_NN, 0]>) outputs(%58 as %arg4: memref<1x4x1x512xf32, [@CMX_NN, 0]>) profiling_data(%100 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x1x512xf32, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x1x512xf16, [@CMX_NN, 0]>, memref<1x4x1x512xf32, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc137)
    } loc(#loc137)
    VPURT.Task waits(%17 : !VPURT.Barrier) attributes {cycleBegin = 12451 : i64, cycleEnd = 13405 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%19 : memref<16xui32, [@CMX_NN, 0]>) outputs(%101 : memref<16xui32, @DDR>) -> memref<16xui32, @DDR> loc(#loc138)
    } loc(#loc138)
    VPURT.Task waits(%17 : !VPURT.Barrier) attributes {cycleBegin = 12451 : i64, cycleEnd = 13405 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 1 : i64} inputs(%20 : memref<16xui32, [@CMX_NN, 1]>) outputs(%102 : memref<16xui32, @DDR>) -> memref<16xui32, @DDR> loc(#loc139)
    } loc(#loc139)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%188 : memref<1xui64, @Register>) outputs(%189 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    } loc(#loc100)
    VPURT.Task attributes {cycleBegin = 13405 : i64, cycleEnd = 14553 : i64, isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%103 : memref<1x4x512xf32, [@CMX_NN, 0]>) outputs(%18 : memref<1x4x512xf32, @DDR>) -> memref<1x4x512xf32, @DDR> loc(#loc17)
    } loc(#loc17)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%190 : memref<1xui64, @Register>) outputs(%191 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc102)
    } loc(#loc102)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%192 : memref<6xui64, [@CMX_NN, 0]>) outputs(%193 : memref<6xui64>) -> memref<6xui64> loc(#loc103)
    } loc(#loc103)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %196 = VPUIP.NNDMA {port = 0 : i64} inputs(%pll_in1 : memref<1xui32, @Register>) outputs(%pll_out1 : memref<1xui32>) -> memref<1xui32> loc(#loc140)
    } loc(#loc140)
    return %arg1, %arg2 : memref<1x4x512xf32, @DDR>, memref<166xui32> loc(#loc17)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc("combinedProfilingDataOutputInfo")
#loc3 = loc("Reshape_1423")
#loc4 = loc("2_actProfilingSubviewBuffer_1")
#loc5 = loc("actshaveProfilingCMX2DDR16")
#loc6 = loc("actshaveProfilingCMX2DDR0")
#loc7 = loc("MVN_0")
#loc8 = loc("MVN_1")
#loc9 = loc("Swish_0")
#loc10 = loc("MVN_2")
#loc11 = loc("MVN_3")
#loc12 = loc("Tanh_0")
#loc13 = loc(fused["Tanh_0", "PROF_16_4_2_0", "_input_cluster_0"])
#loc14 = loc(fused["Tanh_0", "PROF_16_4_2_0", "_input_cluster_1"])
#loc15 = loc(fused["Tanh_0", "PROF_16_4_2_0", "_outputBuff_cluster_0"])
#loc16 = loc(fused["Tanh_0", "PROF_16_4_2_0", "_outputBuff_cluster_1"])
#loc17 = loc("output")
#loc18 = loc(fused["Reshape_1423", "PROF_0_8_0_0", "_view_cast"])
#loc19 = loc(fused["MVN_0", "PROF_0_8_2_1", "_input_cluster_0"])
#loc20 = loc(fused["MVN_0", "PROF_0_8_2_1", "_input_cluster_1"])
#loc21 = loc(fused["MVN_0", "PROF_0_8_1_0", "_input_cluster_0"])
#loc22 = loc(fused["MVN_0", "PROF_0_8_1_0", "_input_cluster_1"])
#loc23 = loc(fused["MVN_0", "PROF_0_8_2_1", "_outputBuff_cluster_0"])
#loc24 = loc(fused["MVN_0", "PROF_0_8_2_1", "_outputBuff_cluster_1"])
#loc25 = loc(fused["MVN_0", "PROF_0_8_1_0", "_outputBuff_cluster_0"])
#loc26 = loc(fused["MVN_0", "PROF_0_8_1_0", "_outputBuff_cluster_1"])
#loc27 = loc(fused["MVN_1", "PROF_0_8_4_1", "_input_cluster_0"])
#loc28 = loc(fused["MVN_1", "PROF_0_8_4_1", "_input_cluster_1"])
#loc29 = loc(fused["MVN_1", "PROF_0_8_3_0", "_input_cluster_0"])
#loc30 = loc(fused["MVN_1", "PROF_0_8_3_0", "_input_cluster_1"])
#loc31 = loc(fused["MVN_1", "PROF_0_8_4_1", "_outputBuff_cluster_0"])
#loc32 = loc(fused["MVN_1", "PROF_0_8_4_1", "_outputBuff_cluster_1"])
#loc33 = loc(fused["MVN_1", "PROF_0_8_3_0", "_outputBuff_cluster_0"])
#loc34 = loc(fused["MVN_1", "PROF_0_8_3_0", "_outputBuff_cluster_1"])
#loc35 = loc(fused["Swish_0", "PROF_0_8_5_0", "_view_cast"])
#loc36 = loc(fused["MVN_2", "PROF_0_8_7_1", "_input_cluster_0"])
#loc37 = loc(fused["MVN_2", "PROF_0_8_7_1", "_input_cluster_1"])
#loc38 = loc(fused["MVN_2", "PROF_0_8_6_0", "_input_cluster_0"])
#loc39 = loc(fused["MVN_2", "PROF_0_8_6_0", "_input_cluster_1"])
#loc40 = loc(fused["MVN_2", "PROF_0_8_7_1", "_outputBuff_cluster_0"])
#loc41 = loc(fused["MVN_2", "PROF_0_8_7_1", "_outputBuff_cluster_1"])
#loc42 = loc(fused["MVN_2", "PROF_0_8_6_0", "_outputBuff_cluster_0"])
#loc43 = loc(fused["MVN_2", "PROF_0_8_6_0", "_outputBuff_cluster_1"])
#loc44 = loc(fused["MVN_3", "PROF_16_4_1_1", "_input_cluster_0"])
#loc45 = loc(fused["MVN_3", "PROF_16_4_1_1", "_input_cluster_1"])
#loc46 = loc(fused["MVN_3", "PROF_16_4_0_0", "_input_cluster_0"])
#loc47 = loc(fused["MVN_3", "PROF_16_4_0_0", "_input_cluster_1"])
#loc48 = loc(fused["MVN_3", "PROF_16_4_1_1", "_outputBuff_cluster_0"])
#loc49 = loc(fused["MVN_3", "PROF_16_4_1_1", "_outputBuff_cluster_1"])
#loc50 = loc(fused["MVN_3", "PROF_16_4_0_0", "_outputBuff_cluster_0"])
#loc51 = loc(fused["MVN_3", "PROF_16_4_0_0", "_outputBuff_cluster_1"])
#loc52 = loc(fused["Tanh_0", "PROF_16_4_2_0", "_profilingBuff_cluster_0"])
#loc53 = loc(fused["Tanh_0", "PROF_16_4_2_0", "_profilingBuff_cluster_1"])
#loc54 = loc(fused["output", "PROF_16_4_3_0", "_view_cast"])
#loc55 = loc(fused["Reshape_1423", "PROFTASKBEGIN"])
#loc56 = loc(fused["Reshape_1423", "PROFTASKEND_0"])
#loc57 = loc(fused["Reshape_1423", "PROFTASKEND_1"])
#loc58 = loc(fused["MVN_0", "_cluster_0", "PROFTASKBEGIN"])
#loc59 = loc(fused["MVN_0", "_cluster_0", "PROFTASKEND_2"])
#loc60 = loc(fused["MVN_0", "_cluster_1", "PROFTASKBEGIN"])
#loc61 = loc(fused["MVN_0", "_cluster_1", "PROFTASKEND_11"])
#loc62 = loc(fused["MVN_0", "PROF_0_8_1_0", "_profilingBuff_cluster_0"])
#loc63 = loc(fused["MVN_0", "PROF_0_8_1_0", "_profilingBuff_cluster_1"])
#loc64 = loc(fused["MVN_0", "PROF_0_8_2_1", "_profilingBuff_cluster_0"])
#loc65 = loc(fused["MVN_0", "PROF_0_8_2_1", "_profilingBuff_cluster_1"])
#loc66 = loc(fused["MVN_1", "PROF_0_8_3_0", "_profilingBuff_cluster_0"])
#loc67 = loc(fused["MVN_1", "PROF_0_8_3_0", "_profilingBuff_cluster_1"])
#loc68 = loc(fused["MVN_1", "PROF_0_8_4_1", "_profilingBuff_cluster_0"])
#loc69 = loc(fused["MVN_1", "PROF_0_8_4_1", "_profilingBuff_cluster_1"])
#loc70 = loc(fused["MVN_1", "_cluster_0", "PROFTASKBEGIN"])
#loc71 = loc(fused["MVN_1", "_cluster_0", "PROFTASKEND_3"])
#loc72 = loc(fused["MVN_1", "_cluster_1", "PROFTASKBEGIN"])
#loc73 = loc(fused["MVN_1", "_cluster_1", "PROFTASKEND_12"])
#loc74 = loc(fused["Swish_0", "PROFTASKBEGIN"])
#loc75 = loc(fused["Swish_0", "PROFTASKEND_4"])
#loc76 = loc(fused["MVN_2", "_cluster_0", "PROFTASKBEGIN"])
#loc77 = loc(fused["MVN_2", "_cluster_0", "PROFTASKEND_5"])
#loc78 = loc(fused["MVN_2", "_cluster_1", "PROFTASKBEGIN"])
#loc79 = loc(fused["MVN_2", "_cluster_1", "PROFTASKEND_13"])
#loc80 = loc(fused["MVN_2", "PROF_0_8_6_0", "_profilingBuff_cluster_0"])
#loc81 = loc(fused["MVN_2", "PROF_0_8_6_0", "_profilingBuff_cluster_1"])
#loc82 = loc(fused["MVN_2", "PROF_0_8_7_1", "_profilingBuff_cluster_0"])
#loc83 = loc(fused["MVN_2", "PROF_0_8_7_1", "_profilingBuff_cluster_1"])
#loc84 = loc(fused["MVN_3", "PROF_16_4_0_0", "_profilingBuff_cluster_0"])
#loc85 = loc(fused["MVN_3", "PROF_16_4_0_0", "_profilingBuff_cluster_1"])
#loc86 = loc(fused["MVN_3", "PROF_16_4_1_1", "_profilingBuff_cluster_0"])
#loc87 = loc(fused["MVN_3", "PROF_16_4_1_1", "_profilingBuff_cluster_1"])
#loc88 = loc(fused["MVN_3", "_cluster_0", "PROFTASKBEGIN"])
#loc89 = loc(fused["MVN_3", "_cluster_0", "PROFTASKEND_6"])
#loc90 = loc(fused["MVN_3", "_cluster_1", "PROFTASKBEGIN"])
#loc91 = loc(fused["MVN_3", "_cluster_1", "PROFTASKEND_14"])
#loc92 = loc(fused["Tanh_0", "_cluster_0", "PROFTASKBEGIN"])
#loc93 = loc(fused["Tanh_0", "_cluster_0", "PROFTASKEND_7"])
#loc94 = loc("dmaProfilingCMX2DDR0")
#loc95 = loc(fused["Tanh_0", "_cluster_1", "PROFTASKBEGIN"])
#loc96 = loc(fused["Tanh_0", "_cluster_1", "PROFTASKEND_15"])
#loc97 = loc(fused["Tanh_0", "_cluster_0", "PROFTASKEND_8"])
#loc98 = loc(fused["Tanh_0", "_cluster_1", "PROFTASKEND_16"])
#loc99 = loc("dmaProfilingCMX2DDR176")
#loc100 = loc(fused["output", "PROFTASKBEGIN"])
#loc101 = loc(fused["output", "PROFTASKEND_9"])
#loc102 = loc(fused["output", "PROFTASKEND_10"])
#loc103 = loc("dmaProfilingCMX2DDR128")
#loc104 = loc("newProfilingBuffer")
#loc105 = loc(fused["Reshape_1423", "PROF_0_8_0_0"])
#loc106 = loc(fused["MVN_0", "_cluster_0"])
#loc107 = loc(fused["MVN_0", "_cluster_1"])
#loc108 = loc(fused["MVN_0", "PROF_0_8_1_0", "_cluster_0"])
#loc109 = loc(fused["MVN_0", "PROF_0_8_1_0", "_cluster_1"])
#loc110 = loc(fused["MVN_0", "PROF_0_8_2_1", "_cluster_0"])
#loc111 = loc(fused["MVN_0", "PROF_0_8_2_1", "_cluster_1"])
#loc112 = loc(fused["MVN_1", "PROF_0_8_3_0", "_cluster_0"])
#loc113 = loc(fused["MVN_1", "PROF_0_8_3_0", "_cluster_1"])
#loc114 = loc(fused["MVN_1", "PROF_0_8_4_1", "_cluster_0"])
#loc115 = loc(fused["MVN_1", "PROF_0_8_4_1", "_cluster_1"])
#loc116 = loc(fused["MVN_1", "_cluster_0"])
#loc117 = loc(fused["MVN_1", "_cluster_1"])
#loc118 = loc(fused["Swish_0", "PROF_0_8_5_0"])
#loc119 = loc(fused["MVN_2", "_cluster_0"])
#loc120 = loc(fused["MVN_2", "_cluster_1"])
#loc121 = loc(fused["MVN_2", "PROF_0_8_6_0", "_cluster_0"])
#loc122 = loc(fused["MVN_2", "PROF_0_8_6_0", "_cluster_1"])
#loc123 = loc(fused["MVN_2", "PROF_0_8_7_1", "_cluster_0"])
#loc124 = loc(fused["MVN_2", "PROF_0_8_7_1", "_cluster_1"])
#loc125 = loc(fused["actshaveProfilingCMX2DDR0", "_cluster_0"])
#loc126 = loc(fused["actshaveProfilingCMX2DDR0", "_cluster_1"])
#loc127 = loc(fused["MVN_3", "PROF_16_4_0_0", "_cluster_0"])
#loc128 = loc(fused["MVN_3", "PROF_16_4_0_0", "_cluster_1"])
#loc129 = loc(fused["MVN_3", "PROF_16_4_1_1", "_cluster_0"])
#loc130 = loc(fused["MVN_3", "PROF_16_4_1_1", "_cluster_1"])
#loc131 = loc(fused["MVN_3", "_cluster_0"])
#loc132 = loc(fused["MVN_3", "_cluster_1"])
#loc133 = loc(fused["Tanh_0", "_cluster_0"])
#loc134 = loc(fused["Tanh_0", "_cluster_1"])
#loc135 = loc(fused["Tanh_0", "PROF_16_4_2_0", "_cluster_0"])
#loc136 = loc(fused["Tanh_0", "PROF_16_4_2_0", "_cluster_1"])
#loc137 = loc(fused["output", "PROF_16_4_3_0"])
#loc138 = loc(fused["actshaveProfilingCMX2DDR16", "_cluster_0"])
#loc139 = loc(fused["actshaveProfilingCMX2DDR16", "_cluster_1"])
#loc140 = loc("PROFWORKPOINT_READ")

//CHECK: {"traceEvents":[
//CHECK: {"name":"Reshape_1423", "cat":"DMA", "ph":"X", "ts":0.000, "dur":0.755, "pid":1, "tid":2},
//CHECK: {"name":"Reshape_1423", "cat":"DMA", "ph":"X", "ts":19.401, "dur":0.572, "pid":1, "tid":2},
//CHECK: {"name":"MVN_0", "cat":"DMA", "ph":"X", "ts":20.208, "dur":0.546, "pid":1, "tid":2},
//CHECK: {"name":"MVN_0", "cat":"DMA", "ph":"X", "ts":20.364, "dur":0.546, "pid":1, "tid":2},
//CHECK: {"name":"MVN_1", "cat":"DMA", "ph":"X", "ts":35.260, "dur":0.390, "pid":1, "tid":2},
//CHECK: {"name":"MVN_1", "cat":"DMA", "ph":"X", "ts":35.416, "dur":0.390, "pid":1, "tid":2},
//CHECK: {"name":"Swish_0", "cat":"DMA", "ph":"X", "ts":47.135, "dur":0.572, "pid":1, "tid":2},
//CHECK: {"name":"MVN_2", "cat":"DMA", "ph":"X", "ts":47.942, "dur":0.625, "pid":1, "tid":2},
//CHECK: {"name":"MVN_2", "cat":"DMA", "ph":"X", "ts":48.099, "dur":0.625, "pid":1, "tid":2},
//CHECK: {"name":"MVN_3", "cat":"DMA", "ph":"X", "ts":56.224, "dur":0.494, "pid":1, "tid":2},
//CHECK: {"name":"MVN_3", "cat":"DMA", "ph":"X", "ts":56.380, "dur":0.494, "pid":1, "tid":2},
//CHECK: {"name":"Tanh_0", "cat":"DMA", "ph":"X", "ts":57.109, "dur":0.546, "pid":1, "tid":2},
//CHECK: {"name":"Tanh_0", "cat":"DMA", "ph":"X", "ts":57.265, "dur":0.546, "pid":1, "tid":2},
//CHECK: {"name":"Tanh_0", "cat":"DMA", "ph":"X", "ts":61.953, "dur":0.494, "pid":1, "tid":2},
//CHECK: {"name":"Tanh_0", "cat":"DMA", "ph":"X", "ts":62.109, "dur":0.494, "pid":1, "tid":2},
//CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":63.021, "dur":0.598, "pid":1, "tid":2},
//CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":68.411, "dur":0.703, "pid":1, "tid":2},
//CHECK: {"name":"MVN_0", "cat":"SW", "ph":"X", "ts":21.328, "dur":9.791, "pid":1, "tid":7, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_1", "cat":"SW", "ph":"X", "ts":31.823, "dur":2.916, "pid":1, "tid":7, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_2", "cat":"SW", "ph":"X", "ts":49.140, "dur":3.046, "pid":1, "tid":7, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_3", "cat":"SW", "ph":"X", "ts":52.916, "dur":2.786, "pid":1, "tid":7, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Reshape_1423", "cat":"SW", "ph":"X", "ts":6.172, "dur":11.979, "pid":1, "tid":7, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Swish_0", "cat":"SW", "ph":"X", "ts":36.224, "dur":10.390, "pid":1, "tid":7, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Tanh_0", "cat":"SW", "ph":"X", "ts":58.281, "dur":3.151, "pid":1, "tid":7, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"output", "cat":"SW", "ph":"X", "ts":64.062, "dur":3.671, "pid":1, "tid":7, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_0/cluster_0", "cat":"SW", "ph":"X", "ts":21.588, "dur":9.270, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_0/cluster_1", "cat":"SW", "ph":"X", "ts":21.328, "dur":9.791, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_1/cluster_0", "cat":"SW", "ph":"X", "ts":31.823, "dur":2.656, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_1/cluster_1", "cat":"SW", "ph":"X", "ts":32.083, "dur":2.656, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_2/cluster_0", "cat":"SW", "ph":"X", "ts":49.401, "dur":2.786, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_2/cluster_1", "cat":"SW", "ph":"X", "ts":49.140, "dur":2.552, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_3/cluster_0", "cat":"SW", "ph":"X", "ts":53.177, "dur":2.526, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_3/cluster_1", "cat":"SW", "ph":"X", "ts":52.916, "dur":2.447, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Reshape_1423/cluster_0", "cat":"SW", "ph":"X", "ts":6.172, "dur":11.979, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Swish_0/cluster_0", "cat":"SW", "ph":"X", "ts":36.224, "dur":10.390, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Tanh_0/cluster_0", "cat":"SW", "ph":"X", "ts":58.411, "dur":3.020, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Tanh_0/cluster_1", "cat":"SW", "ph":"X", "ts":58.281, "dur":3.020, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"output/cluster_0", "cat":"SW", "ph":"X", "ts":64.062, "dur":3.671, "pid":1, "tid":6, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Reshape_1423/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":6.172, "dur":11.979, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_0/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":21.719, "dur":9.140, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_0/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":21.328, "dur":9.661, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_0/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":21.588, "dur":9.140, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_0/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":21.458, "dur":9.661, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_1/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":31.953, "dur":2.395, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_1/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":32.083, "dur":2.526, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_1/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":31.823, "dur":2.656, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_1/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":32.344, "dur":2.395, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Swish_0/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":36.224, "dur":10.390, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_2/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":49.531, "dur":2.656, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_2/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":49.271, "dur":2.421, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_2/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":49.401, "dur":2.656, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_2/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":49.140, "dur":2.265, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_3/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":53.177, "dur":2.317, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_3/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":53.047, "dur":2.317, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_3/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":53.463, "dur":2.239, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"MVN_3/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":52.916, "dur":2.291, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Tanh_0/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":58.411, "dur":3.020, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Tanh_0/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":58.281, "dur":3.020, "pid":1, "tid":5, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"output/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":64.062, "dur":3.671, "pid":1, "tid":4, "args":{"Stall cycles": "0"}},
//CHECK: {"name":"Reshape_1423", "cat":"Layer", "ph":"X", "ts":0.000, "dur":19.973, "pid":1, "tid":8, "args":{"Layer type": ""}},
//CHECK: {"name":"MVN_0", "cat":"Layer", "ph":"X", "ts":20.208, "dur":10.911, "pid":1, "tid":8, "args":{"Layer type": ""}},
//CHECK: {"name":"MVN_1", "cat":"Layer", "ph":"X", "ts":31.823, "dur":3.983, "pid":1, "tid":8, "args":{"Layer type": ""}},
//CHECK: {"name":"Swish_0", "cat":"Layer", "ph":"X", "ts":36.224, "dur":11.483, "pid":1, "tid":8, "args":{"Layer type": ""}},
//CHECK: {"name":"MVN_2", "cat":"Layer", "ph":"X", "ts":47.942, "dur":4.244, "pid":1, "tid":8, "args":{"Layer type": ""}},
//CHECK: {"name":"MVN_3", "cat":"Layer", "ph":"X", "ts":52.916, "dur":3.958, "pid":1, "tid":8, "args":{"Layer type": ""}},
//CHECK: {"name":"Tanh_0", "cat":"Layer", "ph":"X", "ts":57.109, "dur":5.494, "pid":1, "tid":8, "args":{"Layer type": ""}},
//CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":63.021, "dur":6.093, "pid":1, "tid":8, "args":{"Layer type": ""}},
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
