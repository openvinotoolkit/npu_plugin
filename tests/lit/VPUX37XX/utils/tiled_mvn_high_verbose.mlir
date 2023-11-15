//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=VPUX37XX --export-VPUIP -o %t %s && prof_parser -b %t -p %data_path_37XX%/profiling-0-37XX-MVN.bin -f json -vv | FileCheck %s
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#loc0 = loc(unknown)
#loc2 = loc("profiling_result")
module @MVN_case1 attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  module @UsedMemory {
    IE.MemoryResource 4096 bytes of @DDR loc(#loc0)
  } loc(#loc0)
  module @ReservedMemory {
    module @DmaProfilingReservedMemory {
      IE.MemoryResource 512 bytes of @CMX_NN offset 0 loc(#loc0)
    } loc(#loc0)
  } loc(#loc0)
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096] loc(#loc0)
  module @VPU.SW {
    func.func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "tanh_fp16.cpp", VPU.kernel_entry = "tanh_fp16"} loc(#loc0)
    func.func private @builtin_Swish(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, f64) attributes {VPU.kernel_code = "swish_fp16.cpp", VPU.kernel_entry = "swish_fp16"} loc(#loc0)
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN"} loc(#loc0)
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "single_shave_convert.cpp", VPU.kernel_entry = "single_shave_convert"} loc(#loc0)
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"} loc(#loc0)
  } loc(#loc0)
  IE.ExecutorResource 2 of @NCE at 1.300000e+03 MHz {
    builtin.module @UsedMemory {
      IE.MemoryResource 15104 bytes of @CMX_NN loc(#loc0)
    } loc(#loc0)
    IE.ExecutorResource 2 of @SHAVE_ACT  loc(#loc0)
    IE.ExecutorResource 1 of @SHAVE_NN  loc(#loc0)
    IE.ExecutorResource 1 of @DPU  loc(#loc0)
    IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware loc(#loc0)
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64} loc(#loc0)
  } loc(#loc0)
  IE.ExecutorResource 2 of @DMA_NN  loc(#loc0)
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64} loc(#loc0)
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x4x512xf32> loc(#loc0)
  } outputsInfo : {
    DataInfo "Div_0" : tensor<1x4x512xf32> loc(#loc0)
  } profilingOutputsInfo : {
    DataInfo "profilingOutput" {
      VPUIP.ProfilingSection type 3 : 416 bytes from 0 loc(#loc0)
      VPUIP.ProfilingSection type 4 : 192 bytes from 448 loc(#loc0)
      VPUIP.ProfilingSection type 5 : 64 bytes from 640 loc(#loc0)
    } : tensor<176xui32> loc(#loc1)
  } loc(#loc0)
  func.func @main(%arg0: memref<1x4x512xf32, @DDR> loc(unknown), %arg1: memref<1x4x512xf32, @DDR> loc(unknown), %arg2: memref<176xui32> loc("profiling_result")) -> (memref<1x4x512xf32, @DDR>, memref<176xui32>) {
    %0 = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register> loc(#loc3)
    %1 = VPURT.DeclareBuffer <ProfilingOutput> [0] <640> -> memref<1xui32> loc(#loc3)
    %2 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier loc(#loc4)
    %3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier loc(#loc5)
    %4 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier loc(#loc4)
    %5 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier loc(#loc6)
    %6 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier loc(#loc6)
    %7 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier loc(#loc7)
    %8 = VPURT.ConfigureBarrier<6> -> !VPURT.Barrier loc(#loc8)
    %9 = VPURT.ConfigureBarrier<7> -> !VPURT.Barrier loc(#loc9)
    %10 = VPURT.ConfigureBarrier<8> -> !VPURT.Barrier loc(#loc10)
    %11 = VPURT.ConfigureBarrier<9> -> !VPURT.Barrier loc(#loc10)
    %12 = VPURT.ConfigureBarrier<10> -> !VPURT.Barrier loc(#loc11)
    %13 = VPURT.ConfigureBarrier<11> -> !VPURT.Barrier loc(#loc11)
    %14 = VPURT.ConfigureBarrier<12> -> !VPURT.Barrier loc(#loc11)
    %15 = VPURT.ConfigureBarrier<13> -> !VPURT.Barrier loc(#loc12)
    %16 = VPURT.ConfigureBarrier<14> -> !VPURT.Barrier loc(#loc13)
    %17 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x4x512xf32, @DDR> loc(#loc14)
    %18 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<24xui32, [@CMX_NN, 0]> loc(#loc15)
    %19 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<24xui32, [@CMX_NN, 1]> loc(#loc15)
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<28xui32, [@CMX_NN, 0]> loc(#loc9)
    %21 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<28xui32, [@CMX_NN, 1]> loc(#loc9)
    %22 = VPURT.DeclareBuffer <CMX_NN> [0] <4736> -> memref<1x1x4x512xf32, [@CMX_NN, 0]> loc(#loc4)
    %23 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x4x512xf16, [@CMX_NN, 0]> loc(#loc4)
    %24 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x512x1xf16, @DDR> loc(#loc4)
    %25 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc6)
    %26 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc6)
    %27 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc6)
    %28 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc6)
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc10)
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc10)
    %31 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc11)
    %32 = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc11)
    %33 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc10)
    %34 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc10)
    %35 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc16)
    %36 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc17)
    %37 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc11)
    %38 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc11)
    %39 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc11)
    %40 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc11)
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc18)
    %42 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc19)
    %43 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc11)
    %44 = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc11)
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <11008> -> memref<1x1x4x512xf16, [@CMX_NN, 0]> loc(#loc12)
    %46 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x1x4x512xf32, [@CMX_NN, 0]> loc(#loc12)
    %47 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x4x512xf32, @DDR> loc(#loc4)
    %48 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc20)
    %49 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x4x512x1xf16, [@CMX_NN, 0]> loc(#loc4)
    %50 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc21)
    %51 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc22)
    %52 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc23)
    %53 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc24)
    %54 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc25)
    %55 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc26)
    %56 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc27)
    %57 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc28)
    %58 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc29)
    %59 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc30)
    %60 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc31)
    %61 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc32)
    %62 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc33)
    %63 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc34)
    %64 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc35)
    %65 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc36)
    %66 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x1x512x1xf16, [@CMX_NN, 0]> loc(#loc37)
    %67 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x1x512x1xf16, [@CMX_NN, 1]> loc(#loc38)
    %68 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x512x1xf16, [@CMX_NN, 0]> loc(#loc39)
    %69 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x512x1xf16, [@CMX_NN, 1]> loc(#loc40)
    %70 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc41)
    %71 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc42)
    %72 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc43)
    %73 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc44)
    %74 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<28xui32, @DDR> loc(#loc9)
    %75 = VPURT.DeclareBuffer <ProfilingOutput> [0] <112> -> memref<28xui32, @DDR> loc(#loc9)
    %76 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc45)
    %77 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc46)
    %78 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc47)
    %79 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc48)
    %80 = VPURT.DeclareBuffer <CMX_NN> [0] <3840> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc49)
    %81 = VPURT.DeclareBuffer <CMX_NN> [1] <3840> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc50)
    %82 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc51)
    %83 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc52)
    %84 = VPURT.DeclareBuffer <CMX_NN> [0] <3840> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc53)
    %85 = VPURT.DeclareBuffer <CMX_NN> [1] <3840> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc54)
    %86 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc55)
    %87 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc56)
    %88 = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc57)
    %89 = VPURT.DeclareBuffer <CMX_NN> [1] <1536> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc58)
    %90 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc59)
    %91 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc60)
    %92 = VPURT.DeclareBuffer <CMX_NN> [0] <2752> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc61)
    %93 = VPURT.DeclareBuffer <CMX_NN> [1] <2752> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc62)
    %94 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x4x512xf16, @DDR> loc(#loc12)
    %95 = VPURT.DeclareBuffer <CMX_NN> [0] <2768> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc63)
    %96 = VPURT.DeclareBuffer <ProfilingOutput> [0] <224> -> memref<24xui32, @DDR> loc(#loc15)
    %97 = VPURT.DeclareBuffer <ProfilingOutput> [0] <320> -> memref<24xui32, @DDR> loc(#loc15)
    %98 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<1x4x512xf32, [@CMX_NN, 0]> loc(#loc12)
    %99 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc64)
    %100 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc64)
    %101 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc65)
    %102 = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc65)
    %103 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc64)
    %104 = VPURT.DeclareBuffer <CMX_NN> [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc64)
    %105 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc66)
    %106 = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc66)
    %107 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc67)
    %108 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc67)
    %109 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc68)
    %110 = VPURT.DeclareBuffer <CMX_NN> [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc68)
    %111 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc69)
    %112 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc69)
    %113 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc70)
    %114 = VPURT.DeclareBuffer <CMX_NN> [0] <264> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc70)
    %115 = VPURT.DeclareBuffer <CMX_NN> [0] <528> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc71)
    %116 = VPURT.DeclareBuffer <CMX_NN> [1] <528> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc72)
    %117 = VPURT.DeclareBuffer <CMX_NN> [0] <544> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc73)
    %118 = VPURT.DeclareBuffer <CMX_NN> [1] <544> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc74)
    %119 = VPURT.DeclareBuffer <CMX_NN> [0] <560> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc75)
    %120 = VPURT.DeclareBuffer <CMX_NN> [1] <560> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc76)
    %121 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc77)
    %122 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc78)
    %123 = VPURT.DeclareBuffer <CMX_NN> [0] <592> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc79)
    %124 = VPURT.DeclareBuffer <CMX_NN> [1] <592> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc80)
    %125 = VPURT.DeclareBuffer <CMX_NN> [0] <608> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc81)
    %126 = VPURT.DeclareBuffer <CMX_NN> [1] <608> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc82)
    %127 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc83)
    %128 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc84)
    %129 = VPURT.DeclareBuffer <CMX_NN> [0] <2704> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc85)
    %130 = VPURT.DeclareBuffer <CMX_NN> [1] <2704> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc86)
    %131 = VPURT.DeclareBuffer <CMX_NN> [0] <2720> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc87)
    %132 = VPURT.DeclareBuffer <CMX_NN> [1] <2720> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc88)
    %133 = VPURT.DeclareBuffer <CMX_NN> [0] <2736> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc89)
    %134 = VPURT.DeclareBuffer <CMX_NN> [1] <2736> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc90)
    %135 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc91)
    %136 = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    %137 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc92)
    %138 = VPURT.DeclareBuffer <CMX_NN> [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    %139 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc93)
    %140 = VPURT.DeclareBuffer <CMX_NN> [0] <272> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc93)
    %141 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc94)
    %142 = VPURT.DeclareBuffer <CMX_NN> [0] <280> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc94)
    %143 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc95)
    %144 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    %145 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc96)
    %146 = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc96)
    %147 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc97)
    %148 = VPURT.DeclareBuffer <CMX_NN> [0] <288> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    %149 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc98)
    %150 = VPURT.DeclareBuffer <CMX_NN> [0] <296> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc98)
    %151 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc95)
    %152 = VPURT.DeclareBuffer <CMX_NN> [0] <80> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    %153 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc99)
    %154 = VPURT.DeclareBuffer <CMX_NN> [0] <88> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc99)
    %155 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc97)
    %156 = VPURT.DeclareBuffer <CMX_NN> [0] <304> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    %157 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc100)
    %158 = VPURT.DeclareBuffer <CMX_NN> [0] <312> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    %159 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<8xui64, [@CMX_NN, 0]> loc(#loc101)
    %160 = VPURT.DeclareBuffer <ProfilingOutput> [0] <576> -> memref<8xui64> loc(#loc101)
    %161 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc102)
    %162 = VPURT.DeclareBuffer <CMX_NN> [0] <96> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc102)
    %163 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc103)
    %164 = VPURT.DeclareBuffer <CMX_NN> [0] <104> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc103)
    %165 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc102)
    %166 = VPURT.DeclareBuffer <CMX_NN> [0] <112> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc102)
    %167 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc104)
    %168 = VPURT.DeclareBuffer <CMX_NN> [0] <120> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc104)
    %169 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16xui64, [@CMX_NN, 0]> loc(#loc105)
    %170 = VPURT.DeclareBuffer <ProfilingOutput> [0] <448> -> memref<16xui64> loc(#loc105)
    %171 = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register> loc(#loc3)
    %172 = VPURT.DeclareBuffer <ProfilingOutput> [0] <644> -> memref<1xui32> loc(#loc3)
    %173 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<104xui32> loc(#loc106)
    %174 = VPURT.DeclareBuffer <ProfilingOutput> [0] <448> -> memref<24xui64> loc(#loc106)
    %175 = VPURT.DeclareBuffer <ProfilingOutput> [0] <640> -> memref<16xui32> loc(#loc106)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1xui32, @Register>) outputs(%1 : memref<1xui32>) -> memref<1xui32> loc(#loc3)
    } loc(#loc3)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%99 : memref<1xui64, @Register>) outputs(%100 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc64)
    } loc(#loc64)
    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 1148 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%47 : memref<1x1x4x512xf32, @DDR>) outputs(%22 : memref<1x1x4x512xf32, [@CMX_NN, 0]>) -> memref<1x1x4x512xf32, [@CMX_NN, 0]> loc(#loc4)
    } loc(#loc4)
    VPURT.Task updates(%2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%101 : memref<1xui64, @Register>) outputs(%102 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc65)
    } loc(#loc65)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) attributes {cycleBegin = 1148 : i64, cycleEnd = 1149 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%22 as %arg3: memref<1x1x4x512xf32, [@CMX_NN, 0]>) outputs(%23 as %arg4: memref<1x1x4x512xf16, [@CMX_NN, 0]>) profiling_data(%48 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x4x512xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x4x512xf32, [@CMX_NN, 0]>, memref<1x1x4x512xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc5)
    } loc(#loc5)
    VPURT.Task waits(%3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%103 : memref<1xui64, @Register>) outputs(%104 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc64)
    } loc(#loc64)
    VPURT.Task attributes {cycleBegin = 1149 : i64, cycleEnd = 2297 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%49 : memref<1x4x512x1xf16, [@CMX_NN, 0]>) outputs(%24 : memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR> loc(#loc4)
    } loc(#loc4)
    VPURT.Task updates(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%105 : memref<1xui64, @Register>) outputs(%106 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc66)
    } loc(#loc66)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%107 : memref<1xui64, @Register>) outputs(%108 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc67)
    } loc(#loc67)
    VPURT.Task attributes {cycleBegin = 2297 : i64, cycleEnd = 3445 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x2x512x1xf16, @DDR>) outputs(%27 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc107)
    } loc(#loc107)
    VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%109 : memref<1xui64, @Register>) outputs(%110 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc68)
    } loc(#loc68)
    VPURT.Task waits(%4 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%111 : memref<1xui64, @Register>) outputs(%112 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc69)
    } loc(#loc69)
    VPURT.Task attributes {cycleBegin = 2297 : i64, cycleEnd = 3445 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%26 : memref<1x2x512x1xf16, @DDR>) outputs(%28 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc108)
    } loc(#loc108)
    VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%113 : memref<1xui64, @Register>) outputs(%114 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc70)
    } loc(#loc70)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {cycleBegin = 3445 : i64, cycleEnd = 8722 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%52 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%56 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%115 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc109)
    } loc(#loc109)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {cycleBegin = 3445 : i64, cycleEnd = 8722 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%53 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%57 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%116 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc110)
    } loc(#loc110)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {cycleBegin = 3445 : i64, cycleEnd = 8722 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%50 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%54 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%117 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc111)
    } loc(#loc111)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {cycleBegin = 3445 : i64, cycleEnd = 8722 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%51 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%55 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%118 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc112)
    } loc(#loc112)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {cycleBegin = 8722 : i64, cycleEnd = 13999 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%60 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%64 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%119 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc113)
    } loc(#loc113)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {cycleBegin = 8722 : i64, cycleEnd = 13999 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%61 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%65 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%120 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc114)
    } loc(#loc114)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {cycleBegin = 8722 : i64, cycleEnd = 13999 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%58 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%62 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%121 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc115)
    } loc(#loc115)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {cycleBegin = 8722 : i64, cycleEnd = 13999 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%59 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%63 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%122 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc116)
    } loc(#loc116)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {cycleBegin = 13999 : i64, cycleEnd = 14000 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%68 as %arg3: memref<1x1x512x1xf16, [@CMX_NN, 0]>) outputs(%72 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%123 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x1x512x1xf16, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc117)
    } loc(#loc117)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {cycleBegin = 13999 : i64, cycleEnd = 14000 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%69 as %arg3: memref<1x1x512x1xf16, [@CMX_NN, 1]>) outputs(%73 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%124 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x1x512x1xf16, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc118)
    } loc(#loc118)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {cycleBegin = 13999 : i64, cycleEnd = 14000 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%66 as %arg3: memref<1x1x512x1xf16, [@CMX_NN, 0]>) outputs(%70 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%125 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x1x512x1xf16, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc119)
    } loc(#loc119)
    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {cycleBegin = 13999 : i64, cycleEnd = 14000 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Swish inputs(%67 as %arg3: memref<1x1x512x1xf16, [@CMX_NN, 1]>) outputs(%71 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%126 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x1x512x1xf16, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc120)
    } loc(#loc120)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {cycleBegin = 14000 : i64, cycleEnd = 14956 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<28xui32, [@CMX_NN, 0]>) outputs(%74 : memref<28xui32, @DDR>) -> memref<28xui32, @DDR> loc(#loc121)
    } loc(#loc121)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {cycleBegin = 14000 : i64, cycleEnd = 14956 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%21 : memref<28xui32, [@CMX_NN, 1]>) outputs(%75 : memref<28xui32, @DDR>) -> memref<28xui32, @DDR> loc(#loc122)
    } loc(#loc122)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {cycleBegin = 14000 : i64, cycleEnd = 19277 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%78 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%82 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%127 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc123)
    } loc(#loc123)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {cycleBegin = 14000 : i64, cycleEnd = 19277 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%79 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%83 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%128 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc124)
    } loc(#loc124)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {cycleBegin = 14000 : i64, cycleEnd = 19277 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%76 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%80 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%129 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc125)
    } loc(#loc125)
    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {cycleBegin = 14000 : i64, cycleEnd = 19277 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%77 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%81 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%130 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc126)
    } loc(#loc126)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 19277 : i64, cycleEnd = 24554 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%86 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%90 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%131 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc127)
    } loc(#loc127)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 19277 : i64, cycleEnd = 24554 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%87 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%91 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%132 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc128)
    } loc(#loc128)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 19277 : i64, cycleEnd = 24554 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%84 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%88 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%133 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc129)
    } loc(#loc129)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) attributes {cycleBegin = 19277 : i64, cycleEnd = 24554 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_MVN inputs(%85 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%89 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%134 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc130)
    } loc(#loc130)
    VPURT.Task waits(%10 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%135 : memref<1xui64, @Register>) outputs(%136 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc91)
    } loc(#loc91)
    VPURT.Task attributes {cycleBegin = 24554 : i64, cycleEnd = 25702 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%29 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) outputs(%33 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc131)
    } loc(#loc131)
    VPURT.Task updates(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%137 : memref<1xui64, @Register>) outputs(%138 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc92)
    } loc(#loc92)
    VPURT.Task waits(%10 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%139 : memref<1xui64, @Register>) outputs(%140 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc93)
    } loc(#loc93)
    VPURT.Task attributes {cycleBegin = 24554 : i64, cycleEnd = 25702 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%30 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) outputs(%34 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc132)
    } loc(#loc132)
    VPURT.Task updates(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%141 : memref<1xui64, @Register>) outputs(%142 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc94)
    } loc(#loc94)
    VPURT.Task waits(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%143 : memref<1xui64, @Register>) outputs(%144 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    } loc(#loc95)
    VPURT.Task attributes {cycleBegin = 25702 : i64, cycleEnd = 26850 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%31 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%37 : memref<1x4x256x1xf16, [@CMX_NN, 0]>) -> memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc133)
    } loc(#loc133)
    VPURT.Task updates(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%145 : memref<1xui64, @Register>) outputs(%146 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc96)
    } loc(#loc96)
    VPURT.Task waits(%11 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%147 : memref<1xui64, @Register>) outputs(%148 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    } loc(#loc97)
    VPURT.Task attributes {cycleBegin = 25702 : i64, cycleEnd = 26850 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%32 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%38 : memref<1x4x256x1xf16, [@CMX_NN, 1]>) -> memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc134)
    } loc(#loc134)
    VPURT.Task updates(%12 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%149 : memref<1xui64, @Register>) outputs(%150 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc98)
    } loc(#loc98)
    VPURT.Task waits(%12 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) attributes {cycleBegin = 26850 : i64, cycleEnd = 36802 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Tanh inputs(%35 as %arg3: memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs(%41 as %arg4: memref<1x4x256x1xf16, [@CMX_NN, 0]>) profiling_data(%92 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x256x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x256x1xf16, [@CMX_NN, 0]>, memref<1x4x256x1xf16, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc135)
    } loc(#loc135)
    VPURT.Task waits(%12 : !VPURT.Barrier) updates(%13 : !VPURT.Barrier) attributes {cycleBegin = 26850 : i64, cycleEnd = 36802 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Tanh inputs(%36 as %arg3: memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs(%42 as %arg4: memref<1x4x256x1xf16, [@CMX_NN, 1]>) profiling_data(%93 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x4x256x1xf16, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x256x1xf16, [@CMX_NN, 1]>, memref<1x4x256x1xf16, [@CMX_NN, 1]> loc(#loc0)
      } loc(#loc136)
    } loc(#loc136)
    VPURT.Task waits(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%151 : memref<1xui64, @Register>) outputs(%152 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc95)
    } loc(#loc95)
    VPURT.Task attributes {cycleBegin = 36802 : i64, cycleEnd = 37950 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%39 : memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs(%43 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc133)
    } loc(#loc133)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%153 : memref<1xui64, @Register>) outputs(%154 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc99)
    } loc(#loc99)
    VPURT.Task waits(%13 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%155 : memref<1xui64, @Register>) outputs(%156 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc97)
    } loc(#loc97)
    VPURT.Task attributes {cycleBegin = 36802 : i64, cycleEnd = 37950 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%40 : memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs(%44 : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc134)
    } loc(#loc134)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%157 : memref<1xui64, @Register>) outputs(%158 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc100)
    } loc(#loc100)
    VPURT.Task updates(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%159 : memref<8xui64, [@CMX_NN, 0]>) outputs(%160 : memref<8xui64>) -> memref<8xui64> loc(#loc101)
    } loc(#loc101)
    VPURT.Task waits(%14 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%161 : memref<1xui64, @Register>) outputs(%162 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc102)
    } loc(#loc102)
    VPURT.Task attributes {cycleBegin = 37950 : i64, cycleEnd = 39098 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%94 : memref<1x1x4x512xf16, @DDR>) outputs(%45 : memref<1x1x4x512xf16, [@CMX_NN, 0]>) -> memref<1x1x4x512xf16, [@CMX_NN, 0]> loc(#loc12)
    } loc(#loc12)
    VPURT.Task updates(%15 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%163 : memref<1xui64, @Register>) outputs(%164 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc103)
    } loc(#loc103)
    VPURT.Task waits(%15 : !VPURT.Barrier) updates(%16 : !VPURT.Barrier) attributes {cycleBegin = 39098 : i64, cycleEnd = 39099 : i64, isTrailingSWLayer = false} {
      %results, %profiling_output = VPUIP.SW.Kernel {result_segment_sizes = dense<1> : vector<2xi32>} @VPU.SW::@builtin_Convert inputs(%45 as %arg3: memref<1x1x4x512xf16, [@CMX_NN, 0]>) outputs(%46 as %arg4: memref<1x1x4x512xf32, [@CMX_NN, 0]>) profiling_data(%95 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x4x512xf32, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x4x512xf16, [@CMX_NN, 0]>, memref<1x1x4x512xf32, [@CMX_NN, 0]> loc(#loc0)
      } loc(#loc13)
    } loc(#loc13)
    VPURT.Task waits(%16 : !VPURT.Barrier) attributes {cycleBegin = 39099 : i64, cycleEnd = 40054 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%18 : memref<24xui32, [@CMX_NN, 0]>) outputs(%96 : memref<24xui32, @DDR>) -> memref<24xui32, @DDR> loc(#loc137)
    } loc(#loc137)
    VPURT.Task waits(%16 : !VPURT.Barrier) attributes {cycleBegin = 39099 : i64, cycleEnd = 40054 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 1 : i64} inputs(%19 : memref<24xui32, [@CMX_NN, 1]>) outputs(%97 : memref<24xui32, @DDR>) -> memref<24xui32, @DDR> loc(#loc138)
    } loc(#loc138)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%165 : memref<1xui64, @Register>) outputs(%166 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc102)
    } loc(#loc102)
    VPURT.Task attributes {cycleBegin = 40054 : i64, cycleEnd = 41202 : i64, isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%98 : memref<1x4x512xf32, [@CMX_NN, 0]>) outputs(%17 : memref<1x4x512xf32, @DDR>) -> memref<1x4x512xf32, @DDR> loc(#loc12)
    } loc(#loc12)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%167 : memref<1xui64, @Register>) outputs(%168 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc104)
    } loc(#loc104)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%169 : memref<16xui64, [@CMX_NN, 0]>) outputs(%170 : memref<16xui64>) -> memref<16xui64> loc(#loc105)
    } loc(#loc105)
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %176 = VPUIP.NNDMA {port = 0 : i64} inputs(%171 : memref<1xui32, @Register>) outputs(%172 : memref<1xui32>) -> memref<1xui32> loc(#loc3)
    } loc(#loc3)
    return %arg1, %arg2 : memref<1x4x512xf32, @DDR>, memref<176xui32> loc(#loc12)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc("combinedProfilingDataOutputInfo")
#loc3 = loc("PROFWORKPOINT_READ")
#loc4 = loc(fused["Reshape_1423", "t_Reshape"])
#loc5 = loc(fused["Reshape_1423", "t_Reshape", "PROF_0_7_0_0"])
#loc6 = loc(fused["MVN_0", "t_MVN"])
#loc7 = loc(fused["MVN_1", "t_MVN"])
#loc8 = loc(fused["Swish_0", "t_Swish"])
#loc9 = loc("actshaveProfilingCMX2DDR0")
#loc10 = loc(fused["MVN_3", "t_MVN"])
#loc11 = loc(fused["Tanh_0", "t_Tanh"])
#loc12 = loc("output")
#loc13 = loc(fused["output", "PROF_14_6_5_0"])
#loc14 = loc("2_actProfilingSubviewBuffer_1")
#loc15 = loc("actshaveProfilingCMX2DDR14")
#loc16 = loc(fused["Tanh_0", "t_Tanh", "PROF_14_6_4_0", "_input_cluster_0"])
#loc17 = loc(fused["Tanh_0", "t_Tanh", "PROF_14_6_4_0", "_input_cluster_1"])
#loc18 = loc(fused["Tanh_0", "t_Tanh", "PROF_14_6_4_0", "_outputBuff_cluster_0"])
#loc19 = loc(fused["Tanh_0", "t_Tanh", "PROF_14_6_4_0", "_outputBuff_cluster_1"])
#loc20 = loc(fused["Reshape_1423", "t_Reshape", "PROF_0_7_0_0", "_view_cast"])
#loc21 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_2_1", "_input_cluster_0"])
#loc22 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_2_1", "_input_cluster_1"])
#loc23 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_1_0", "_input_cluster_0"])
#loc24 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_1_0", "_input_cluster_1"])
#loc25 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_2_1", "_outputBuff_cluster_0"])
#loc26 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_2_1", "_outputBuff_cluster_1"])
#loc27 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_1_0", "_outputBuff_cluster_0"])
#loc28 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_1_0", "_outputBuff_cluster_1"])
#loc29 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_4_1", "_input_cluster_0"])
#loc30 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_4_1", "_input_cluster_1"])
#loc31 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_3_0", "_input_cluster_0"])
#loc32 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_3_0", "_input_cluster_1"])
#loc33 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_4_1", "_outputBuff_cluster_0"])
#loc34 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_4_1", "_outputBuff_cluster_1"])
#loc35 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_3_0", "_outputBuff_cluster_0"])
#loc36 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_3_0", "_outputBuff_cluster_1"])
#loc37 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_6_1", "_input_cluster_0"])
#loc38 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_6_1", "_input_cluster_1"])
#loc39 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_5_0", "_input_cluster_0"])
#loc40 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_5_0", "_input_cluster_1"])
#loc41 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_6_1", "_outputBuff_cluster_0"])
#loc42 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_6_1", "_outputBuff_cluster_1"])
#loc43 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_5_0", "_outputBuff_cluster_0"])
#loc44 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_5_0", "_outputBuff_cluster_1"])
#loc45 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_1_1", "_input_cluster_0"])
#loc46 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_1_1", "_input_cluster_1"])
#loc47 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_0_0", "_input_cluster_0"])
#loc48 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_0_0", "_input_cluster_1"])
#loc49 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_1_1", "_outputBuff_cluster_0"])
#loc50 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_1_1", "_outputBuff_cluster_1"])
#loc51 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_0_0", "_outputBuff_cluster_0"])
#loc52 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_0_0", "_outputBuff_cluster_1"])
#loc53 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_3_1", "_input_cluster_0"])
#loc54 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_3_1", "_input_cluster_1"])
#loc55 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_2_0", "_input_cluster_0"])
#loc56 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_2_0", "_input_cluster_1"])
#loc57 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_3_1", "_outputBuff_cluster_0"])
#loc58 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_3_1", "_outputBuff_cluster_1"])
#loc59 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_2_0", "_outputBuff_cluster_0"])
#loc60 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_2_0", "_outputBuff_cluster_1"])
#loc61 = loc(fused["Tanh_0", "t_Tanh", "PROF_14_6_4_0", "_profilingBuff_cluster_0"])
#loc62 = loc(fused["Tanh_0", "t_Tanh", "PROF_14_6_4_0", "_profilingBuff_cluster_1"])
#loc63 = loc(fused["output", "PROF_14_6_5_0", "_view_cast"])
#loc64 = loc(fused["Reshape_1423", "t_Reshape", "PROFTASKBEGIN"])
#loc65 = loc(fused["Reshape_1423", "t_Reshape", "PROFTASKEND_0"])
#loc66 = loc(fused["Reshape_1423", "t_Reshape", "PROFTASKEND_1"])
#loc67 = loc(fused["MVN_0", "t_MVN", "_cluster_0", "PROFTASKBEGIN"])
#loc68 = loc(fused["MVN_0", "t_MVN", "_cluster_0", "PROFTASKEND_2"])
#loc69 = loc(fused["MVN_0", "t_MVN", "_cluster_1", "PROFTASKBEGIN"])
#loc70 = loc(fused["MVN_0", "t_MVN", "_cluster_1", "PROFTASKEND_8"])
#loc71 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_1_0", "_profilingBuff_cluster_0"])
#loc72 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_1_0", "_profilingBuff_cluster_1"])
#loc73 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_2_1", "_profilingBuff_cluster_0"])
#loc74 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_2_1", "_profilingBuff_cluster_1"])
#loc75 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_3_0", "_profilingBuff_cluster_0"])
#loc76 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_3_0", "_profilingBuff_cluster_1"])
#loc77 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_4_1", "_profilingBuff_cluster_0"])
#loc78 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_4_1", "_profilingBuff_cluster_1"])
#loc79 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_5_0", "_profilingBuff_cluster_0"])
#loc80 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_5_0", "_profilingBuff_cluster_1"])
#loc81 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_6_1", "_profilingBuff_cluster_0"])
#loc82 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_6_1", "_profilingBuff_cluster_1"])
#loc83 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_0_0", "_profilingBuff_cluster_0"])
#loc84 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_0_0", "_profilingBuff_cluster_1"])
#loc85 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_1_1", "_profilingBuff_cluster_0"])
#loc86 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_1_1", "_profilingBuff_cluster_1"])
#loc87 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_2_0", "_profilingBuff_cluster_0"])
#loc88 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_2_0", "_profilingBuff_cluster_1"])
#loc89 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_3_1", "_profilingBuff_cluster_0"])
#loc90 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_3_1", "_profilingBuff_cluster_1"])
#loc91 = loc(fused["MVN_3", "t_MVN", "_cluster_0", "PROFTASKBEGIN"])
#loc92 = loc(fused["MVN_3", "t_MVN", "_cluster_0", "PROFTASKEND_3"])
#loc93 = loc(fused["MVN_3", "t_MVN", "_cluster_1", "PROFTASKBEGIN"])
#loc94 = loc(fused["MVN_3", "t_MVN", "_cluster_1", "PROFTASKEND_9"])
#loc95 = loc(fused["Tanh_0", "t_Tanh", "_cluster_0", "PROFTASKBEGIN"])
#loc96 = loc(fused["Tanh_0", "t_Tanh", "_cluster_0", "PROFTASKEND_4"])
#loc97 = loc(fused["Tanh_0", "t_Tanh", "_cluster_1", "PROFTASKBEGIN"])
#loc98 = loc(fused["Tanh_0", "t_Tanh", "_cluster_1", "PROFTASKEND_10"])
#loc99 = loc(fused["Tanh_0", "t_Tanh", "_cluster_0", "PROFTASKEND_5"])
#loc100 = loc(fused["Tanh_0", "t_Tanh", "_cluster_1", "PROFTASKEND_11"])
#loc101 = loc("dmaProfilingCMX2DDR128")
#loc102 = loc(fused["output", "PROFTASKBEGIN"])
#loc103 = loc(fused["output", "PROFTASKEND_6"])
#loc104 = loc(fused["output", "PROFTASKEND_7"])
#loc105 = loc("dmaProfilingCMX2DDR0")
#loc106 = loc("newProfilingBuffer")
#loc107 = loc(fused["MVN_0", "t_MVN", "_cluster_0"])
#loc108 = loc(fused["MVN_0", "t_MVN", "_cluster_1"])
#loc109 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_1_0", "_cluster_0"])
#loc110 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_1_0", "_cluster_1"])
#loc111 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_2_1", "_cluster_0"])
#loc112 = loc(fused["MVN_0", "t_MVN", "PROF_0_7_2_1", "_cluster_1"])
#loc113 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_3_0", "_cluster_0"])
#loc114 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_3_0", "_cluster_1"])
#loc115 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_4_1", "_cluster_0"])
#loc116 = loc(fused["MVN_1", "t_MVN", "PROF_0_7_4_1", "_cluster_1"])
#loc117 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_5_0", "_cluster_0"])
#loc118 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_5_0", "_cluster_1"])
#loc119 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_6_1", "_cluster_0"])
#loc120 = loc(fused["Swish_0", "t_Swish", "PROF_0_7_6_1", "_cluster_1"])
#loc121 = loc(fused["actshaveProfilingCMX2DDR0", "_cluster_0"])
#loc122 = loc(fused["actshaveProfilingCMX2DDR0", "_cluster_1"])
#loc123 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_0_0", "_cluster_0"])
#loc124 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_0_0", "_cluster_1"])
#loc125 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_1_1", "_cluster_0"])
#loc126 = loc(fused["MVN_2", "t_MVN", "PROF_14_6_1_1", "_cluster_1"])
#loc127 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_2_0", "_cluster_0"])
#loc128 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_2_0", "_cluster_1"])
#loc129 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_3_1", "_cluster_0"])
#loc130 = loc(fused["MVN_3", "t_MVN", "PROF_14_6_3_1", "_cluster_1"])
#loc131 = loc(fused["MVN_3", "t_MVN", "_cluster_0"])
#loc132 = loc(fused["MVN_3", "t_MVN", "_cluster_1"])
#loc133 = loc(fused["Tanh_0", "t_Tanh", "_cluster_0"])
#loc134 = loc(fused["Tanh_0", "t_Tanh", "_cluster_1"])
#loc135 = loc(fused["Tanh_0", "t_Tanh", "PROF_14_6_4_0", "_cluster_0"])
#loc136 = loc(fused["Tanh_0", "t_Tanh", "PROF_14_6_4_0", "_cluster_1"])
#loc137 = loc(fused["actshaveProfilingCMX2DDR14", "_cluster_0"])
#loc138 = loc(fused["actshaveProfilingCMX2DDR14", "_cluster_1"])

// CHECK: {"traceEvents":[
// CHECK: {"name": "process_name", "ph": "M", "pid":0, "args": {"name" : "DMA"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":0, "args": {"sort_index" : "0"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":0, "tid":0, "args": {"name" : "DMA"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":0, "tid":1, "args": {"name" : "DMA"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":1, "args": {"name" : "Cluster (0)"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":1, "args": {"sort_index" : "1"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":1, "tid":0, "args": {"name" : "SW / Shave"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":1, "tid":1, "args": {"name" : "SW / Shave"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":2, "args": {"name" : "Cluster (1)"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":2, "args": {"sort_index" : "2"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":2, "tid":0, "args": {"name" : "SW / Shave"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":2, "tid":1, "args": {"name" : "SW / Shave"}},
// CHECK: {"name": "process_name", "ph": "M", "pid":3, "args": {"name" : "Layers"}},
// CHECK: {"name": "process_sort_index", "ph": "M", "pid":3, "args": {"sort_index" : "3"}},
// CHECK: {"name": "thread_name", "ph": "M", "pid":3, "tid":0, "args": {"name" : "Layers"}},
// CHECK: {"name":"Reshape_1423?t_Reshape", "cat":"DMA", "ph":"X", "ts":0.000, "dur":1.041, "pid":0, "tid":0},
// CHECK: {"name":"Reshape_1423?t_Reshape", "cat":"DMA", "ph":"X", "ts":20.833, "dur":0.651, "pid":0, "tid":0},
// CHECK: {"name":"MVN_0?t_MVN", "cat":"DMA", "ph":"X", "ts":21.744, "dur":0.625, "pid":0, "tid":0},
// CHECK: {"name":"MVN_0?t_MVN", "cat":"DMA", "ph":"X", "ts":21.927, "dur":0.781, "pid":0, "tid":1},
// CHECK: {"name":"MVN_3?t_MVN", "cat":"DMA", "ph":"X", "ts":51.614, "dur":0.572, "pid":0, "tid":0},
// CHECK: {"name":"MVN_3?t_MVN", "cat":"DMA", "ph":"X", "ts":51.796, "dur":0.572, "pid":0, "tid":1},
// CHECK: {"name":"Tanh_0?t_Tanh", "cat":"DMA", "ph":"X", "ts":52.630, "dur":0.598, "pid":0, "tid":0},
// CHECK: {"name":"Tanh_0?t_Tanh", "cat":"DMA", "ph":"X", "ts":52.812, "dur":0.598, "pid":0, "tid":1},
// CHECK: {"name":"Tanh_0?t_Tanh", "cat":"DMA", "ph":"X", "ts":57.968, "dur":0.546, "pid":0, "tid":0},
// CHECK: {"name":"Tanh_0?t_Tanh", "cat":"DMA", "ph":"X", "ts":58.151, "dur":0.546, "pid":0, "tid":1},
// CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":59.166, "dur":0.729, "pid":0, "tid":0},
// CHECK: {"name":"output", "cat":"DMA", "ph":"X", "ts":65.182, "dur":0.833, "pid":0, "tid":0},
// CHECK: {"name":"Reshape_1423?t_Reshape/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":6.484, "dur":13.151, "pid":1, "tid":0},
// CHECK: {"name":"MVN_0?t_MVN/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":23.281, "dur":10.078, "pid":1, "tid":0},
// CHECK: {"name":"MVN_0?t_MVN/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":23.411, "dur":10.208, "pid":1, "tid":1},
// CHECK: {"name":"MVN_1?t_MVN/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":34.609, "dur":2.682, "pid":1, "tid":0},
// CHECK: {"name":"MVN_1?t_MVN/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":34.947, "dur":2.838, "pid":1, "tid":1},
// CHECK: {"name":"Swish_0?t_Swish/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":38.880, "dur":3.645, "pid":1, "tid":0},
// CHECK: {"name":"Swish_0?t_Swish/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":39.140, "dur":3.515, "pid":1, "tid":1},
// CHECK: {"name":"MVN_2?t_MVN/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":43.750, "dur":2.812, "pid":1, "tid":0},
// CHECK: {"name":"MVN_2?t_MVN/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":43.880, "dur":2.812, "pid":1, "tid":1},
// CHECK: {"name":"MVN_3?t_MVN/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":47.812, "dur":2.760, "pid":1, "tid":0},
// CHECK: {"name":"MVN_3?t_MVN/cluster_0/tile_1", "cat":"SW", "ph":"X", "ts":47.942, "dur":2.760, "pid":1, "tid":1},
// CHECK: {"name":"Tanh_0?t_Tanh/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":53.880, "dur":3.359, "pid":1, "tid":0},
// CHECK: {"name":"output/cluster_0/tile_0", "cat":"SW", "ph":"X", "ts":60.390, "dur":3.984, "pid":1, "tid":0},
// CHECK: {"name":"MVN_0?t_MVN/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":23.151, "dur":10.338, "pid":2, "tid":0},
// CHECK: {"name":"MVN_0?t_MVN/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":23.541, "dur":10.208, "pid":2, "tid":1},
// CHECK: {"name":"MVN_1?t_MVN/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":34.739, "dur":2.786, "pid":2, "tid":0},
// CHECK: {"name":"MVN_1?t_MVN/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":35.104, "dur":2.968, "pid":2, "tid":1},
// CHECK: {"name":"Swish_0?t_Swish/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":39.010, "dur":3.776, "pid":2, "tid":0},
// CHECK: {"name":"Swish_0?t_Swish/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":39.375, "dur":3.541, "pid":2, "tid":1},
// CHECK: {"name":"MVN_2?t_MVN/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":44.010, "dur":2.838, "pid":2, "tid":0},
// CHECK: {"name":"MVN_2?t_MVN/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":44.296, "dur":2.682, "pid":2, "tid":1},
// CHECK: {"name":"MVN_3?t_MVN/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":48.072, "dur":2.812, "pid":2, "tid":0},
// CHECK: {"name":"MVN_3?t_MVN/cluster_1/tile_1", "cat":"SW", "ph":"X", "ts":48.359, "dur":2.656, "pid":2, "tid":1},
// CHECK: {"name":"Tanh_0?t_Tanh/cluster_1/tile_0", "cat":"SW", "ph":"X", "ts":54.010, "dur":3.359, "pid":2, "tid":0},
// CHECK: {"name":"Reshape_1423", "cat":"Layer", "ph":"X", "ts":0.000, "dur":21.484, "pid":3, "tid":0, "args":{"Layer type": "Reshape"}},
// CHECK: {"name":"MVN_0", "cat":"Layer", "ph":"X", "ts":21.744, "dur":12.005, "pid":3, "tid":0, "args":{"Layer type": "MVN"}},
// CHECK: {"name":"MVN_1", "cat":"Layer", "ph":"X", "ts":34.609, "dur":3.463, "pid":3, "tid":0, "args":{"Layer type": "MVN"}},
// CHECK: {"name":"Swish_0", "cat":"Layer", "ph":"X", "ts":38.880, "dur":4.036, "pid":3, "tid":0, "args":{"Layer type": "Swish"}},
// CHECK: {"name":"MVN_2", "cat":"Layer", "ph":"X", "ts":43.750, "dur":3.229, "pid":3, "tid":0, "args":{"Layer type": "MVN"}},
// CHECK: {"name":"MVN_3", "cat":"Layer", "ph":"X", "ts":47.812, "dur":4.556, "pid":3, "tid":0, "args":{"Layer type": "MVN"}},
// CHECK: {"name":"Tanh_0", "cat":"Layer", "ph":"X", "ts":52.630, "dur":6.067, "pid":3, "tid":0, "args":{"Layer type": "Tanh"}},
// CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":59.166, "dur":6.849, "pid":3, "tid":0, "args":{"Layer type": ""}}
// CHECK: ],
// CHECK: "displayTimeUnit": "ns"
// CHECK: }
