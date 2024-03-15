//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %s && prof_parser -b %t -p %data_path_npu%/profiling-0-37XX-MVN.bin -f json -vv | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#loc = loc(unknown)
#loc1 = loc("profiling_result")
module @MVN_case1 attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  module @UsedMemory {
    IE.MemoryResource 4096 bytes of @DDR loc(#loc)
  } loc(#loc)
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096] loc(#loc)
  module @VPU.SW {
    func.func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "tanh_fp16.cpp", VPU.kernel_entry = "tanh_fp16", VPU.task_type = @COMPUTE} loc(#loc)
    func.func private @builtin_Swish(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, f64) attributes {VPU.kernel_code = "swish_fp16.cpp", VPU.kernel_entry = "swish_fp16", VPU.task_type = @COMPUTE} loc(#loc)
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "singleShaveMVN.cpp", VPU.kernel_entry = "singleShaveMVN", VPU.task_type = @COMPUTE} loc(#loc)
    func.func private @builtin_Convert(memref<*xf32, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "single_shave_convert.cpp", VPU.kernel_entry = "single_shave_convert", VPU.task_type = @COMPUTE} loc(#loc)
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"} loc(#loc)
  } loc(#loc)
  IE.TileResource {activity_factor = 0.010853512323388196 : f64} 2 of @NCE at 1.300000e+03 MHz {
    builtin.module @UsedMemory {
      IE.MemoryResource 6784 bytes of @CMX_NN loc(#loc)
    } loc(#loc)
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0 loc(#loc)
      } loc(#loc)
    } loc(#loc)
    IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware loc(#loc)
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64} loc(#loc)
    IE.ExecutorResource 2 of @SHAVE_ACT loc(#loc)
    IE.ExecutorResource 1 of @SHAVE_NN loc(#loc)
    IE.ExecutorResource 1 of @DPU loc(#loc)
  } loc(#loc)
  IE.ExecutorResource 2 of @DMA_NN loc(#loc)
  IE.MemoryResource 2306867200 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64} loc(#loc)
  IE.CNNNetwork {inferenceTiming = 90110 : i64} entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x4x512xf32> loc(#loc)
  } outputsInfo : {
    DataInfo "Div_0" : tensor<1x4x512xf32> loc(#loc)
  } profilingOutputsInfo : {
    DataInfo "profilingOutput" {
      VPUIP.ProfilingSection type 3 : 512 bytes from 0 loc(#loc)
      VPUIP.ProfilingSection type 4 : 512 bytes from 512 loc(#loc)
      VPUIP.ProfilingSection type 5 : 64 bytes from 1024 loc(#loc)
    } : tensor<272xui32> loc(#loc)
  } loc(#loc)
  func.func @main(%arg0: memref<1x4x512xf32, @DDR> loc(unknown), %arg1: memref<1x4x512xf32, @DDR> loc(unknown), %arg2: memref<272xui32> loc("profiling_result")) -> (memref<1x4x512xf32, @DDR>, memref<272xui32>) {
    %0 = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register> loc(#loc2)
    %1 = VPURT.DeclareBuffer <ProfilingOutput> [0] <1024> -> memref<1xui32> loc(#loc2)
    %2 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier loc(#loc34)
    %3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier loc(#loc34)
    %4 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier loc(#loc34)
    %5 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier loc(#loc35)
    %6 = VPURT.ConfigureBarrier<4> -> !VPURT.Barrier loc(#loc35)
    %7 = VPURT.ConfigureBarrier<5> -> !VPURT.Barrier loc(#loc36)
    %8 = VPURT.ConfigureBarrier<6> -> !VPURT.Barrier loc(#loc36)
    %9 = VPURT.ConfigureBarrier<7> -> !VPURT.Barrier loc(#loc37)
    %10 = VPURT.ConfigureBarrier<8> -> !VPURT.Barrier loc(#loc37)
    %11 = VPURT.ConfigureBarrier<9> -> !VPURT.Barrier loc(#loc37)
    %12 = VPURT.ConfigureBarrier<10> -> !VPURT.Barrier loc(#loc37)
    %13 = VPURT.ConfigureBarrier<11> -> !VPURT.Barrier loc(#loc10)
    %14 = VPURT.ConfigureBarrier<12> -> !VPURT.Barrier loc(#loc38)
    %15 = VPURT.ConfigureBarrier<13> -> !VPURT.Barrier loc(#loc39)
    %16 = VPURT.ConfigureBarrier<14> -> !VPURT.Barrier loc(#loc39)
    %17 = VPURT.ConfigureBarrier<15> -> !VPURT.Barrier loc(#loc40)
    %18 = VPURT.ConfigureBarrier<16> -> !VPURT.Barrier loc(#loc40)
    %19 = VPURT.ConfigureBarrier<17> -> !VPURT.Barrier loc(#loc40)
    %20 = VPURT.ConfigureBarrier<18> -> !VPURT.Barrier loc(#loc40)
    %21 = VPURT.ConfigureBarrier<19> -> !VPURT.Barrier loc(#loc41)
    %22 = VPURT.ConfigureBarrier<20> -> !VPURT.Barrier loc(#loc41)
    %23 = VPURT.ConfigureBarrier<21> {isFinalBarrier} -> !VPURT.Barrier loc(#loc17)
    %24 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<32xui32, [@CMX_NN, 0]> loc(#loc18)
    %25 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<32xui32, [@CMX_NN, 1]> loc(#loc18)
    %26 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<32xui32, [@CMX_NN, 0]> loc(#loc10)
    %27 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<32xui32, [@CMX_NN, 1]> loc(#loc10)
    %28 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc34)
    %29 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc34)
    %30 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc34)
    %31 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc34)
    %32 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc34)
    %33 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc34)
    %34 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc35)
    %35 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc35)
    %36 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc36)
    %37 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc36)
    %38 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc36)
    %39 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc36)
    %40 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc42)
    %41 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc43)
    %42 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc37)
    %43 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc37)
    %44 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc44)
    %45 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc45)
    %46 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc37)
    %47 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc37)
    %48 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc37)
    %49 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc37)
    %50 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc46)
    %51 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc47)
    %52 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc37)
    %53 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc37)
    %54 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc48)
    %55 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc49)
    %56 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc38)
    %57 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc38)
    %58 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc38)
    %59 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc38)
    %60 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc39)
    %61 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc39)
    %62 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc39)
    %63 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc39)
    %64 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc50)
    %65 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc51)
    %66 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc40)
    %67 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc40)
    %68 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc52)
    %69 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc53)
    %70 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc40)
    %71 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc40)
    %72 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc40)
    %73 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc40)
    %74 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc54)
    %75 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc55)
    %76 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc40)
    %77 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc40)
    %78 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc56)
    %79 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc57)
    %80 = VPURT.DeclareBuffer <CMX_NN> [0] <4736> -> memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc41)
    %81 = VPURT.DeclareBuffer <CMX_NN> [1] <4736> -> memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc41)
    %82 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc41)
    %83 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc41)
    %84 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc34)
    %85 = VPURT.DeclareBuffer <NetworkInput> [0] <4096> -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc34)
    %86 = VPURT.DeclareBuffer <CMX_NN> [0] <4736> -> memref<1x1x1x512xf32, [@CMX_NN, 0]> loc(#loc58)
    %87 = VPURT.DeclareBuffer <CMX_NN> [1] <4736> -> memref<1x1x1x512xf32, [@CMX_NN, 1]> loc(#loc59)
    %88 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x1x512xf32, [@CMX_NN, 0]> loc(#loc60)
    %89 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x1x512xf32, [@CMX_NN, 1]> loc(#loc61)
    %90 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]> loc(#loc62)
    %91 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]> loc(#loc63)
    %92 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]> loc(#loc64)
    %93 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]> loc(#loc65)
    %94 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x512x1xf16, @DDR> loc(#loc35)
    %95 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x2x512x1xf16, @DDR> loc(#loc35)
    %96 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc66)
    %97 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc67)
    %98 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc68)
    %99 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc69)
    %100 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc70)
    %101 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc71)
    %102 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc72)
    %103 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc73)
    %104 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc74)
    %105 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc75)
    %106 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc76)
    %107 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc77)
    %108 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc78)
    %109 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc79)
    %110 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc80)
    %111 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc81)
    %112 = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc37)
    %113 = VPURT.DeclareBuffer <DDR> <768> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc37)
    %114 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc37)
    %115 = VPURT.DeclareBuffer <DDR> <256> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc37)
    %116 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<32xui32, @DDR> loc(#loc10)
    %117 = VPURT.DeclareBuffer <ProfilingOutput> [0] <128> -> memref<32xui32, @DDR> loc(#loc10)
    %118 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc37)
    %119 = VPURT.DeclareBuffer <DDR> <256> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc37)
    %120 = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc37)
    %121 = VPURT.DeclareBuffer <DDR> <768> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc37)
    %122 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc82)
    %123 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc83)
    %124 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc84)
    %125 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc85)
    %126 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc86)
    %127 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc87)
    %128 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc88)
    %129 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc89)
    %130 = VPURT.DeclareBuffer <CMX_NN> [0] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc90)
    %131 = VPURT.DeclareBuffer <CMX_NN> [1] <1664> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc91)
    %132 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc92)
    %133 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc93)
    %134 = VPURT.DeclareBuffer <CMX_NN> [0] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc94)
    %135 = VPURT.DeclareBuffer <CMX_NN> [1] <3712> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc95)
    %136 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc96)
    %137 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc97)
    %138 = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %139 = VPURT.DeclareBuffer <DDR> <768> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %140 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %141 = VPURT.DeclareBuffer <DDR> <256> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %142 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %143 = VPURT.DeclareBuffer <DDR> <256> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %144 = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %145 = VPURT.DeclareBuffer <DDR> <768> -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc40)
    %146 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc41)
    %147 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc41)
    %148 = VPURT.DeclareBuffer <CMX_NN> [0] <5760> -> memref<1x1x1x512xf16, [@CMX_NN, 0]> loc(#loc98)
    %149 = VPURT.DeclareBuffer <CMX_NN> [1] <5760> -> memref<1x1x1x512xf16, [@CMX_NN, 1]> loc(#loc99)
    %150 = VPURT.DeclareBuffer <CMX_NN> [0] <4736> -> memref<1x1x1x512xf16, [@CMX_NN, 0]> loc(#loc100)
    %151 = VPURT.DeclareBuffer <CMX_NN> [1] <4736> -> memref<1x1x1x512xf16, [@CMX_NN, 1]> loc(#loc101)
    %152 = VPURT.DeclareBuffer <CMX_NN> [0] <2688> -> memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]> loc(#loc102)
    %153 = VPURT.DeclareBuffer <CMX_NN> [1] <2688> -> memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]> loc(#loc103)
    %154 = VPURT.DeclareBuffer <CMX_NN> [0] <640> -> memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]> loc(#loc104)
    %155 = VPURT.DeclareBuffer <CMX_NN> [1] <640> -> memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]> loc(#loc105)
    %156 = VPURT.DeclareBuffer <ProfilingOutput> [0] <256> -> memref<32xui32, @DDR> loc(#loc18)
    %157 = VPURT.DeclareBuffer <ProfilingOutput> [0] <384> -> memref<32xui32, @DDR> loc(#loc18)
    %158 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc41)
    %159 = VPURT.DeclareBuffer <NetworkOutput> [0] <4096> -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc41)
    %160 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc106)
    %161 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc107)
    %162 = VPURT.DeclareBuffer <CMX_NN> [0] <528> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc108)
    %163 = VPURT.DeclareBuffer <CMX_NN> [1] <528> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc109)
    %164 = VPURT.DeclareBuffer <CMX_NN> [0] <544> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc110)
    %165 = VPURT.DeclareBuffer <CMX_NN> [1] <544> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc111)
    %166 = VPURT.DeclareBuffer <CMX_NN> [0] <560> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc112)
    %167 = VPURT.DeclareBuffer <CMX_NN> [1] <560> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc113)
    %168 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc114)
    %169 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc115)
    %170 = VPURT.DeclareBuffer <CMX_NN> [0] <592> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc116)
    %171 = VPURT.DeclareBuffer <CMX_NN> [1] <592> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc117)
    %172 = VPURT.DeclareBuffer <CMX_NN> [0] <608> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc118)
    %173 = VPURT.DeclareBuffer <CMX_NN> [1] <608> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc119)
    %174 = VPURT.DeclareBuffer <CMX_NN> [0] <624> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc120)
    %175 = VPURT.DeclareBuffer <CMX_NN> [1] <624> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc121)
    %176 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc122)
    %177 = VPURT.DeclareBuffer <CMX_NN> [1] <512> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc123)
    %178 = VPURT.DeclareBuffer <CMX_NN> [0] <528> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc124)
    %179 = VPURT.DeclareBuffer <CMX_NN> [1] <528> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc125)
    %180 = VPURT.DeclareBuffer <CMX_NN> [0] <544> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc126)
    %181 = VPURT.DeclareBuffer <CMX_NN> [1] <544> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc127)
    %182 = VPURT.DeclareBuffer <CMX_NN> [0] <560> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc128)
    %183 = VPURT.DeclareBuffer <CMX_NN> [1] <560> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc129)
    %184 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc130)
    %185 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc131)
    %186 = VPURT.DeclareBuffer <CMX_NN> [0] <592> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc132)
    %187 = VPURT.DeclareBuffer <CMX_NN> [1] <592> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc133)
    %188 = VPURT.DeclareBuffer <CMX_NN> [0] <608> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc134)
    %189 = VPURT.DeclareBuffer <CMX_NN> [1] <608> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc135)
    %190 = VPURT.DeclareBuffer <CMX_NN> [0] <624> -> memref<4xui32, [@CMX_NN, 0]> loc(#loc136)
    %191 = VPURT.DeclareBuffer <CMX_NN> [1] <624> -> memref<4xui32, [@CMX_NN, 1]> loc(#loc137)
    %192 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc138)
    %193 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc138)
    %194 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc138)
    %195 = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc138)
    %196 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc139)
    %197 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc139)
    %198 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc139)
    %199 = VPURT.DeclareBuffer <CMX_NN> [0] <264> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc139)
    %200 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc138)
    %201 = VPURT.DeclareBuffer <CMX_NN> [0] <16> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc138)
    %202 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc138)
    %203 = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc138)
    %204 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc139)
    %205 = VPURT.DeclareBuffer <CMX_NN> [0] <272> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc139)
    %206 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc139)
    %207 = VPURT.DeclareBuffer <CMX_NN> [0] <280> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc139)
    %208 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc140)
    %209 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc140)
    %210 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc140)
    %211 = VPURT.DeclareBuffer <CMX_NN> [0] <40> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc140)
    %212 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc141)
    %213 = VPURT.DeclareBuffer <CMX_NN> [0] <288> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc141)
    %214 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc141)
    %215 = VPURT.DeclareBuffer <CMX_NN> [0] <296> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc141)
    %216 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc142)
    %217 = VPURT.DeclareBuffer <CMX_NN> [0] <48> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc142)
    %218 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc142)
    %219 = VPURT.DeclareBuffer <CMX_NN> [0] <56> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc142)
    %220 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc143)
    %221 = VPURT.DeclareBuffer <CMX_NN> [0] <304> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc143)
    %222 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc143)
    %223 = VPURT.DeclareBuffer <CMX_NN> [0] <312> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc143)
    %224 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc144)
    %225 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    %226 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc144)
    %227 = VPURT.DeclareBuffer <CMX_NN> [0] <72> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    %228 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc145)
    %229 = VPURT.DeclareBuffer <CMX_NN> [0] <320> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    %230 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc145)
    %231 = VPURT.DeclareBuffer <CMX_NN> [0] <328> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    %232 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc144)
    %233 = VPURT.DeclareBuffer <CMX_NN> [0] <80> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    %234 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc144)
    %235 = VPURT.DeclareBuffer <CMX_NN> [0] <88> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    %236 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc145)
    %237 = VPURT.DeclareBuffer <CMX_NN> [0] <336> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    %238 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc145)
    %239 = VPURT.DeclareBuffer <CMX_NN> [0] <344> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    %240 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc144)
    %241 = VPURT.DeclareBuffer <CMX_NN> [0] <96> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    %242 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc144)
    %243 = VPURT.DeclareBuffer <CMX_NN> [0] <104> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    %244 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc145)
    %245 = VPURT.DeclareBuffer <CMX_NN> [0] <352> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    %246 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc145)
    %247 = VPURT.DeclareBuffer <CMX_NN> [0] <360> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    %248 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc144)
    %249 = VPURT.DeclareBuffer <CMX_NN> [0] <112> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    %250 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc144)
    %251 = VPURT.DeclareBuffer <CMX_NN> [0] <120> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    %252 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc145)
    %253 = VPURT.DeclareBuffer <CMX_NN> [0] <368> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    %254 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc145)
    %255 = VPURT.DeclareBuffer <CMX_NN> [0] <376> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    %256 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc146)
    %257 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc146)
    %258 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc146)
    %259 = VPURT.DeclareBuffer <CMX_NN> [0] <136> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc146)
    %260 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc147)
    %261 = VPURT.DeclareBuffer <CMX_NN> [0] <384> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc147)
    %262 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc147)
    %263 = VPURT.DeclareBuffer <CMX_NN> [0] <392> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc147)
    %264 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc148)
    %265 = VPURT.DeclareBuffer <CMX_NN> [0] <144> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc148)
    %266 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc148)
    %267 = VPURT.DeclareBuffer <CMX_NN> [0] <152> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc148)
    %268 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc149)
    %269 = VPURT.DeclareBuffer <CMX_NN> [0] <400> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc149)
    %270 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc149)
    %271 = VPURT.DeclareBuffer <CMX_NN> [0] <408> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc149)
    %272 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc150)
    %273 = VPURT.DeclareBuffer <CMX_NN> [0] <160> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    %274 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc150)
    %275 = VPURT.DeclareBuffer <CMX_NN> [0] <168> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    %276 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc151)
    %277 = VPURT.DeclareBuffer <CMX_NN> [0] <416> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    %278 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc151)
    %279 = VPURT.DeclareBuffer <CMX_NN> [0] <424> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    %280 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc150)
    %281 = VPURT.DeclareBuffer <CMX_NN> [0] <176> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    %282 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc150)
    %283 = VPURT.DeclareBuffer <CMX_NN> [0] <184> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    %284 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc151)
    %285 = VPURT.DeclareBuffer <CMX_NN> [0] <432> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    %286 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc151)
    %287 = VPURT.DeclareBuffer <CMX_NN> [0] <440> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    %288 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc150)
    %289 = VPURT.DeclareBuffer <CMX_NN> [0] <192> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    %290 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc150)
    %291 = VPURT.DeclareBuffer <CMX_NN> [0] <200> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    %292 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc151)
    %293 = VPURT.DeclareBuffer <CMX_NN> [0] <448> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    %294 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc151)
    %295 = VPURT.DeclareBuffer <CMX_NN> [0] <456> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    %296 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc150)
    %297 = VPURT.DeclareBuffer <CMX_NN> [0] <208> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    %298 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc150)
    %299 = VPURT.DeclareBuffer <CMX_NN> [0] <216> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    %300 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc151)
    %301 = VPURT.DeclareBuffer <CMX_NN> [0] <464> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    %302 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc151)
    %303 = VPURT.DeclareBuffer <CMX_NN> [0] <472> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    %304 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc152)
    %305 = VPURT.DeclareBuffer <CMX_NN> [0] <224> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc152)
    %306 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc152)
    %307 = VPURT.DeclareBuffer <CMX_NN> [0] <232> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc152)
    %308 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc153)
    %309 = VPURT.DeclareBuffer <CMX_NN> [0] <480> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc153)
    %310 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc153)
    %311 = VPURT.DeclareBuffer <CMX_NN> [0] <488> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc153)
    %312 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc152)
    %313 = VPURT.DeclareBuffer <CMX_NN> [0] <240> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc152)
    %314 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc152)
    %315 = VPURT.DeclareBuffer <CMX_NN> [0] <248> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc152)
    %316 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32xui64, [@CMX_NN, 0]> loc(#loc29)
    %317 = VPURT.DeclareBuffer <ProfilingOutput> [0] <512> -> memref<32xui64> loc(#loc29)
    %318 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc153)
    %319 = VPURT.DeclareBuffer <CMX_NN> [0] <496> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc153)
    %320 = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register> loc(#loc153)
    %321 = VPURT.DeclareBuffer <CMX_NN> [0] <504> -> memref<1xui64, [@CMX_NN, 0]> loc(#loc153)
    %322 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<32xui64, [@CMX_NN, 0]> loc(#loc30)
    %323 = VPURT.DeclareBuffer <ProfilingOutput> [0] <768> -> memref<32xui64> loc(#loc30)
    %324 = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register> loc(#loc2)
    %325 = VPURT.DeclareBuffer <ProfilingOutput> [0] <1028> -> memref<1xui32> loc(#loc2)
    %326 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<128xui32> loc(#loc31)
    %327 = VPURT.DeclareBuffer <ProfilingOutput> [0] <512> -> memref<64xui64> loc(#loc31)
    %328 = VPURT.DeclareBuffer <ProfilingOutput> [0] <1024> -> memref<16xui32> loc(#loc31)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%0 : memref<1xui32, @Register>) outputs(%1 : memref<1xui32>) -> memref<1xui32> loc(#loc2)
    } loc(#loc2)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%192 : memref<1xui64, @Register>) outputs(%193 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc138)
    } loc(#loc138)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%84 : memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) outputs(%28 : memref<1x1x2x512xf32, [@CMX_NN, 0]>) -> memref<1x1x2x512xf32, [@CMX_NN, 0]> loc(#loc138)
    } loc(#loc138)
    VPURT.Task updates(%2 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 0 : i64>} inputs(%194 : memref<1xui64, @Register>) outputs(%195 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc138)
    } loc(#loc138)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%196 : memref<1xui64, @Register>) outputs(%197 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc139)
    } loc(#loc139)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%85 : memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) outputs(%29 : memref<1x1x2x512xf32, [@CMX_NN, 1]>) -> memref<1x1x2x512xf32, [@CMX_NN, 1]> loc(#loc139)
    } loc(#loc139)
    VPURT.Task updates(%2 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 16 : i64>} inputs(%198 : memref<1xui64, @Register>) outputs(%199 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc139)
    } loc(#loc139)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Convert inputs(%88 as %arg3: memref<1x1x1x512xf32, [@CMX_NN, 0]>) outputs(%92 as %arg4: memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]>) profiling_data(%160 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x512xf32, [@CMX_NN, 0]>, memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc154)
    } loc(#loc154)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Convert inputs(%89 as %arg3: memref<1x1x1x512xf32, [@CMX_NN, 1]>) outputs(%93 as %arg4: memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]>) profiling_data(%161 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x512xf32, [@CMX_NN, 1]>, memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc155)
    } loc(#loc155)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 1 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Convert inputs(%86 as %arg3: memref<1x1x1x512xf32, [@CMX_NN, 0]>) outputs(%90 as %arg4: memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]>) profiling_data(%162 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x512xf32, [@CMX_NN, 0]>, memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc156)
    } loc(#loc156)
    VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 1 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Convert inputs(%87 as %arg3: memref<1x1x1x512xf32, [@CMX_NN, 1]>) outputs(%91 as %arg4: memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]>) profiling_data(%163 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x512xf32, [@CMX_NN, 1]>, memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc157)
    } loc(#loc157)
    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%200 : memref<1xui64, @Register>) outputs(%201 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc138)
    } loc(#loc138)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%30 : memref<1x1x2x512xf16, [@CMX_NN, 0]>) outputs(%32 : memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc138)
    } loc(#loc138)
    VPURT.Task updates(%4 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 1 : i64>} inputs(%202 : memref<1xui64, @Register>) outputs(%203 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc138)
    } loc(#loc138)
    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%204 : memref<1xui64, @Register>) outputs(%205 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc139)
    } loc(#loc139)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%31 : memref<1x1x2x512xf16, [@CMX_NN, 1]>) outputs(%33 : memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) -> memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc139)
    } loc(#loc139)
    VPURT.Task updates(%4 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 17 : i64>} inputs(%206 : memref<1xui64, @Register>) outputs(%207 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc139)
    } loc(#loc139)
    VPURT.Task waits(%4 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%208 : memref<1xui64, @Register>) outputs(%209 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc140)
    } loc(#loc140)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%94 : memref<1x2x512x1xf16, @DDR>) outputs(%34 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc140)
    } loc(#loc140)
    VPURT.Task updates(%5 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 2 : i64>} inputs(%210 : memref<1xui64, @Register>) outputs(%211 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc140)
    } loc(#loc140)
    VPURT.Task waits(%4 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%212 : memref<1xui64, @Register>) outputs(%213 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc141)
    } loc(#loc141)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%95 : memref<1x2x512x1xf16, @DDR>) outputs(%35 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc141)
    } loc(#loc141)
    VPURT.Task updates(%5 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 18 : i64>} inputs(%214 : memref<1xui64, @Register>) outputs(%215 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc141)
    } loc(#loc141)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%98 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%102 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%164 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc158)
    } loc(#loc158)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%99 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%103 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%165 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc159)
    } loc(#loc159)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 3 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%96 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%100 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%166 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc160)
    } loc(#loc160)
    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 3 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%97 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%101 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%167 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc161)
    } loc(#loc161)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%106 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%110 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%168 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc162)
    } loc(#loc162)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%107 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%111 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%169 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc163)
    } loc(#loc163)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%104 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%108 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%170 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc164)
    } loc(#loc164)
    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%105 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%109 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%171 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc165)
    } loc(#loc165)
    VPURT.Task waits(%7 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%216 : memref<1xui64, @Register>) outputs(%217 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc142)
    } loc(#loc142)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%36 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) outputs(%38 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc142)
    } loc(#loc142)
    VPURT.Task updates(%8 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 3 : i64>} inputs(%218 : memref<1xui64, @Register>) outputs(%219 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc142)
    } loc(#loc142)
    VPURT.Task waits(%7 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%220 : memref<1xui64, @Register>) outputs(%221 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc143)
    } loc(#loc143)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%37 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) outputs(%39 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc143)
    } loc(#loc143)
    VPURT.Task updates(%8 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 19 : i64>} inputs(%222 : memref<1xui64, @Register>) outputs(%223 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc143)
    } loc(#loc143)
    VPURT.Task waits(%8 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%224 : memref<1xui64, @Register>) outputs(%225 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%112 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%42 : memref<1x4x128x1xf16, [@CMX_NN, 0]>) -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task updates(%9 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 4 : i64>} inputs(%226 : memref<1xui64, @Register>) outputs(%227 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task waits(%8 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%228 : memref<1xui64, @Register>) outputs(%229 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%113 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%43 : memref<1x4x128x1xf16, [@CMX_NN, 1]>) -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task updates(%9 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 20 : i64>} inputs(%230 : memref<1xui64, @Register>) outputs(%231 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task waits(%8 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%232 : memref<1xui64, @Register>) outputs(%233 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%114 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%46 : memref<1x4x128x1xf16, [@CMX_NN, 0]>) -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task updates(%9 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 5 : i64>} inputs(%234 : memref<1xui64, @Register>) outputs(%235 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task waits(%8 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%236 : memref<1xui64, @Register>) outputs(%237 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%115 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%47 : memref<1x4x128x1xf16, [@CMX_NN, 1]>) -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task updates(%9 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 21 : i64>} inputs(%238 : memref<1xui64, @Register>) outputs(%239 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 6 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Swish inputs(%44 as %arg3: memref<1x4x128x1xf16, [@CMX_NN, 0]>) outputs(%54 as %arg4: memref<1x4x128x1xf16, [@CMX_NN, 0]>) profiling_data(%172 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x128x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x4x128x1xf16, [@CMX_NN, 0]>, memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc166)
    } loc(#loc166)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 6 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Swish inputs(%45 as %arg3: memref<1x4x128x1xf16, [@CMX_NN, 1]>) outputs(%55 as %arg4: memref<1x4x128x1xf16, [@CMX_NN, 1]>) profiling_data(%173 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x4x128x1xf16, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x4x128x1xf16, [@CMX_NN, 1]>, memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc167)
    } loc(#loc167)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 7 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Swish inputs(%40 as %arg3: memref<1x4x128x1xf16, [@CMX_NN, 0]>) outputs(%50 as %arg4: memref<1x4x128x1xf16, [@CMX_NN, 0]>) profiling_data(%174 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x128x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x4x128x1xf16, [@CMX_NN, 0]>, memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc168)
    } loc(#loc168)
    VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 8 : i64, dataIndex = 7 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Swish inputs(%41 as %arg3: memref<1x4x128x1xf16, [@CMX_NN, 1]>) outputs(%51 as %arg4: memref<1x4x128x1xf16, [@CMX_NN, 1]>) profiling_data(%175 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x4x128x1xf16, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [1.000000e+00]}(%arg3, %arg4) : memref<1x4x128x1xf16, [@CMX_NN, 1]>, memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc169)
    } loc(#loc169)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64} inputs(%26 : memref<32xui32, [@CMX_NN, 0]>) outputs(%116 : memref<32xui32, @DDR>) -> memref<32xui32, @DDR> loc(#loc170)
    } loc(#loc170)
    VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64} inputs(%27 : memref<32xui32, [@CMX_NN, 1]>) outputs(%117 : memref<32xui32, @DDR>) -> memref<32xui32, @DDR> loc(#loc171)
    } loc(#loc171)
    VPURT.Task waits(%10 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%240 : memref<1xui64, @Register>) outputs(%241 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%52 : memref<1x4x128x1xf16, [@CMX_NN, 0]>) outputs(%118 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc144)
    } loc(#loc144)
    VPURT.Task updates(%11 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 6 : i64>} inputs(%242 : memref<1xui64, @Register>) outputs(%243 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task waits(%10 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%244 : memref<1xui64, @Register>) outputs(%245 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%53 : memref<1x4x128x1xf16, [@CMX_NN, 1]>) outputs(%119 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc145)
    } loc(#loc145)
    VPURT.Task updates(%11 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 22 : i64>} inputs(%246 : memref<1xui64, @Register>) outputs(%247 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task waits(%11 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%248 : memref<1xui64, @Register>) outputs(%249 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%48 : memref<1x4x128x1xf16, [@CMX_NN, 0]>) outputs(%120 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc144)
    } loc(#loc144)
    VPURT.Task updates(%12 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 7 : i64>} inputs(%250 : memref<1xui64, @Register>) outputs(%251 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc144)
    } loc(#loc144)
    VPURT.Task waits(%11 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%252 : memref<1xui64, @Register>) outputs(%253 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%49 : memref<1x4x128x1xf16, [@CMX_NN, 1]>) outputs(%121 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc145)
    } loc(#loc145)
    VPURT.Task updates(%12 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 23 : i64>} inputs(%254 : memref<1xui64, @Register>) outputs(%255 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc145)
    } loc(#loc145)
    VPURT.Task waits(%12 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%256 : memref<1xui64, @Register>) outputs(%257 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc146)
    } loc(#loc146)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%56 : memref<1x2x512x1xf16, @DDR>) outputs(%58 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) -> memref<1x2x512x1xf16, [@CMX_NN, 0]> loc(#loc146)
    } loc(#loc146)
    VPURT.Task updates(%13 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 8 : i64>} inputs(%258 : memref<1xui64, @Register>) outputs(%259 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc146)
    } loc(#loc146)
    VPURT.Task waits(%12 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%260 : memref<1xui64, @Register>) outputs(%261 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc147)
    } loc(#loc147)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%57 : memref<1x2x512x1xf16, @DDR>) outputs(%59 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) -> memref<1x2x512x1xf16, [@CMX_NN, 1]> loc(#loc147)
    } loc(#loc147)
    VPURT.Task updates(%13 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 24 : i64>} inputs(%262 : memref<1xui64, @Register>) outputs(%263 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc147)
    } loc(#loc147)
    VPURT.Task waits(%13 : !VPURT.Barrier) updates(%14 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%124 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%128 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%176 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc172)
    } loc(#loc172)
    VPURT.Task waits(%13 : !VPURT.Barrier) updates(%14 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%125 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%129 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%177 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc173)
    } loc(#loc173)
    VPURT.Task waits(%13 : !VPURT.Barrier) updates(%14 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 1 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%122 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%126 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%178 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc174)
    } loc(#loc174)
    VPURT.Task waits(%13 : !VPURT.Barrier) updates(%14 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 1 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%123 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%127 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%179 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc175)
    } loc(#loc175)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%132 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%136 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%180 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc176)
    } loc(#loc176)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 2 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%133 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%137 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%181 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc177)
    } loc(#loc177)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 3 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%130 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) outputs(%134 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>) profiling_data(%182 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc178)
    } loc(#loc178)
    VPURT.Task waits(%14 : !VPURT.Barrier) updates(%15 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 3 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_MVN inputs(%131 as %arg3: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) outputs(%135 as %arg4: memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>) profiling_data(%183 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run {attrs = [false, true, 1.0013580322265625E-5]}(%arg3, %arg4) : memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]>, memref<1x1x512x1xf16, {order = #NCHW, strides = [1024, 512, 1, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc179)
    } loc(#loc179)
    VPURT.Task waits(%15 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%264 : memref<1xui64, @Register>) outputs(%265 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc148)
    } loc(#loc148)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%60 : memref<1x2x512x1xf16, [@CMX_NN, 0]>) outputs(%62 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc148)
    } loc(#loc148)
    VPURT.Task updates(%16 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 9 : i64>} inputs(%266 : memref<1xui64, @Register>) outputs(%267 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc148)
    } loc(#loc148)
    VPURT.Task waits(%15 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%268 : memref<1xui64, @Register>) outputs(%269 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc149)
    } loc(#loc149)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%61 : memref<1x2x512x1xf16, [@CMX_NN, 1]>) outputs(%63 : memref<1x2x512x1xf16, @DDR>) -> memref<1x2x512x1xf16, @DDR> loc(#loc149)
    } loc(#loc149)
    VPURT.Task updates(%16 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 25 : i64>} inputs(%270 : memref<1xui64, @Register>) outputs(%271 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc149)
    } loc(#loc149)
    VPURT.Task waits(%16 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%272 : memref<1xui64, @Register>) outputs(%273 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%138 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%66 : memref<1x4x128x1xf16, [@CMX_NN, 0]>) -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task updates(%17 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 10 : i64>} inputs(%274 : memref<1xui64, @Register>) outputs(%275 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task waits(%16 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%276 : memref<1xui64, @Register>) outputs(%277 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%139 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%67 : memref<1x4x128x1xf16, [@CMX_NN, 1]>) -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task updates(%17 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 26 : i64>} inputs(%278 : memref<1xui64, @Register>) outputs(%279 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task waits(%16 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%280 : memref<1xui64, @Register>) outputs(%281 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%140 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%70 : memref<1x4x128x1xf16, [@CMX_NN, 0]>) -> memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task updates(%17 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 11 : i64>} inputs(%282 : memref<1xui64, @Register>) outputs(%283 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task waits(%16 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%284 : memref<1xui64, @Register>) outputs(%285 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%141 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs(%71 : memref<1x4x128x1xf16, [@CMX_NN, 1]>) -> memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task updates(%17 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 27 : i64>} inputs(%286 : memref<1xui64, @Register>) outputs(%287 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task waits(%17 : !VPURT.Barrier) updates(%18 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Tanh inputs(%68 as %arg3: memref<1x4x128x1xf16, [@CMX_NN, 0]>) outputs(%78 as %arg4: memref<1x4x128x1xf16, [@CMX_NN, 0]>) profiling_data(%184 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x128x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x128x1xf16, [@CMX_NN, 0]>, memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc180)
    } loc(#loc180)
    VPURT.Task waits(%17 : !VPURT.Barrier) updates(%18 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 4 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Tanh inputs(%69 as %arg3: memref<1x4x128x1xf16, [@CMX_NN, 1]>) outputs(%79 as %arg4: memref<1x4x128x1xf16, [@CMX_NN, 1]>) profiling_data(%185 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x4x128x1xf16, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x128x1xf16, [@CMX_NN, 1]>, memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc181)
    } loc(#loc181)
    VPURT.Task waits(%17 : !VPURT.Barrier) updates(%18 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Tanh inputs(%64 as %arg3: memref<1x4x128x1xf16, [@CMX_NN, 0]>) outputs(%74 as %arg4: memref<1x4x128x1xf16, [@CMX_NN, 0]>) profiling_data(%186 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x4x128x1xf16, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x128x1xf16, [@CMX_NN, 0]>, memref<1x4x128x1xf16, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc182)
    } loc(#loc182)
    VPURT.Task waits(%17 : !VPURT.Barrier) updates(%18 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 5 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Tanh inputs(%65 as %arg3: memref<1x4x128x1xf16, [@CMX_NN, 1]>) outputs(%75 as %arg4: memref<1x4x128x1xf16, [@CMX_NN, 1]>) profiling_data(%187 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x4x128x1xf16, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x4x128x1xf16, [@CMX_NN, 1]>, memref<1x4x128x1xf16, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc183)
    } loc(#loc183)
    VPURT.Task waits(%18 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%288 : memref<1xui64, @Register>) outputs(%289 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%76 : memref<1x4x128x1xf16, [@CMX_NN, 0]>) outputs(%142 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc150)
    } loc(#loc150)
    VPURT.Task updates(%19 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 12 : i64>} inputs(%290 : memref<1xui64, @Register>) outputs(%291 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task waits(%18 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%292 : memref<1xui64, @Register>) outputs(%293 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%77 : memref<1x4x128x1xf16, [@CMX_NN, 1]>) outputs(%143 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc151)
    } loc(#loc151)
    VPURT.Task updates(%19 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 28 : i64>} inputs(%294 : memref<1xui64, @Register>) outputs(%295 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task waits(%19 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%296 : memref<1xui64, @Register>) outputs(%297 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%72 : memref<1x4x128x1xf16, [@CMX_NN, 0]>) outputs(%144 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc150)
    } loc(#loc150)
    VPURT.Task updates(%20 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 13 : i64>} inputs(%298 : memref<1xui64, @Register>) outputs(%299 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc150)
    } loc(#loc150)
    VPURT.Task waits(%19 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%300 : memref<1xui64, @Register>) outputs(%301 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%73 : memref<1x4x128x1xf16, [@CMX_NN, 1]>) outputs(%145 : memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x128x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR> loc(#loc151)
    } loc(#loc151)
    VPURT.Task updates(%20 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 29 : i64>} inputs(%302 : memref<1xui64, @Register>) outputs(%303 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc151)
    } loc(#loc151)
    VPURT.Task waits(%20 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%304 : memref<1xui64, @Register>) outputs(%305 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc152)
    } loc(#loc152)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%146 : memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) outputs(%80 : memref<1x1x2x512xf16, [@CMX_NN, 0]>) -> memref<1x1x2x512xf16, [@CMX_NN, 0]> loc(#loc152)
    } loc(#loc152)
    VPURT.Task updates(%21 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 14 : i64>} inputs(%306 : memref<1xui64, @Register>) outputs(%307 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc152)
    } loc(#loc152)
    VPURT.Task waits(%20 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%308 : memref<1xui64, @Register>) outputs(%309 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc153)
    } loc(#loc153)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%147 : memref<1x1x2x512xf16, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) outputs(%81 : memref<1x1x2x512xf16, [@CMX_NN, 1]>) -> memref<1x1x2x512xf16, [@CMX_NN, 1]> loc(#loc153)
    } loc(#loc153)
    VPURT.Task updates(%21 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 30 : i64>} inputs(%310 : memref<1xui64, @Register>) outputs(%311 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc153)
    } loc(#loc153)
    VPURT.Task waits(%21 : !VPURT.Barrier) updates(%22 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 6 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Convert inputs(%150 as %arg3: memref<1x1x1x512xf16, [@CMX_NN, 0]>) outputs(%154 as %arg4: memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]>) profiling_data(%188 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x512xf16, [@CMX_NN, 0]>, memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc184)
    } loc(#loc184)
    VPURT.Task waits(%21 : !VPURT.Barrier) updates(%22 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 6 : i64, tileId = 0 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Convert inputs(%151 as %arg3: memref<1x1x1x512xf16, [@CMX_NN, 1]>) outputs(%155 as %arg4: memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]>) profiling_data(%189 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x512xf16, [@CMX_NN, 1]>, memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc185)
    } loc(#loc185)
    VPURT.Task waits(%21 : !VPURT.Barrier) updates(%22 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 7 : i64, tileId = 1 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Convert inputs(%148 as %arg3: memref<1x1x1x512xf16, [@CMX_NN, 0]>) outputs(%152 as %arg4: memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]>) profiling_data(%190 : memref<4xui32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]>, memref<4xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x512xf16, [@CMX_NN, 0]>, memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 0]> loc(#loc)
      } loc(#loc186)
    } loc(#loc186)
    VPURT.Task waits(%21 : !VPURT.Barrier) updates(%22 : !VPURT.Barrier) {
      %results, %profiling_output = VPUIP.SW.Kernel {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 1 : i64, bufferOffset = 16 : i64, clusterSize = 8 : i64, dataIndex = 7 : i64, tileId = 1 : i64, clusterId = 1 : i64>, resultSegmentSizes = array<i32: 1, 1>} @VPU.SW::@builtin_Convert inputs(%149 as %arg3: memref<1x1x1x512xf16, [@CMX_NN, 1]>) outputs(%153 as %arg4: memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]>) profiling_data(%191 : memref<4xui32, [@CMX_NN, 1]>) on tile 1 -> (memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]>, memref<4xui32, [@CMX_NN, 1]>){
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x512xf16, [@CMX_NN, 1]>, memref<1x1x1x512xf32, {order = #NCHW, strides = [1024, 1024, 512, 1]}, [@CMX_NN, 1]> loc(#loc)
      } loc(#loc187)
    } loc(#loc187)
    VPURT.Task waits(%22 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64} inputs(%24 : memref<32xui32, [@CMX_NN, 0]>) outputs(%156 : memref<32xui32, @DDR>) -> memref<32xui32, @DDR> loc(#loc188)
    } loc(#loc188)
    VPURT.Task waits(%22 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64} inputs(%25 : memref<32xui32, [@CMX_NN, 1]>) outputs(%157 : memref<32xui32, @DDR>) -> memref<32xui32, @DDR> loc(#loc189)
    } loc(#loc189)
    VPURT.Task waits(%22 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%312 : memref<1xui64, @Register>) outputs(%313 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc152)
    } loc(#loc152)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%82 : memref<1x1x2x512xf32, [@CMX_NN, 0]>) outputs(%158 : memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc152)
    } loc(#loc152)
    VPURT.Task {
      %329 = VPUIP.NNDMA {port = 0 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 15 : i64>} inputs(%314 : memref<1xui64, @Register>) outputs(%315 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc152)
    } loc(#loc152)
    VPURT.Task updates(%23 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 0 : i64} inputs(%316 : memref<32xui64, [@CMX_NN, 0]>) outputs(%317 : memref<32xui64>) -> memref<32xui64> loc(#loc29)
    } loc(#loc29)
    VPURT.Task waits(%22 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<profBeginFlag unit>} inputs(%318 : memref<1xui64, @Register>) outputs(%319 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc153)
    } loc(#loc153)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 1 : i64} inputs(%83 : memref<1x1x2x512xf32, [@CMX_NN, 1]>) outputs(%159 : memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR>) -> memref<1x1x2x512xf32, {order = #NCHW, strides = [2048, 2048, 512, 1]}, @DDR> loc(#loc153)
    } loc(#loc153)
    VPURT.Task {
      %329 = VPUIP.NNDMA {port = 1 : i64, profilingMetadata = #VPUIP.DmaProfilingMetadataAttr<dataIndex = 31 : i64>} inputs(%320 : memref<1xui64, @Register>) outputs(%321 : memref<1xui64, [@CMX_NN, 0]>) -> memref<1xui64, [@CMX_NN, 0]> loc(#loc153)
    } loc(#loc153)
    VPURT.Task updates(%23 : !VPURT.Barrier) {
      %329 = VPUIP.NNDMA {port = 1 : i64} inputs(%322 : memref<32xui64, [@CMX_NN, 0]>) outputs(%323 : memref<32xui64>) -> memref<32xui64> loc(#loc30)
    } loc(#loc30)
    VPURT.Task {
      %329 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%324 : memref<1xui32, @Register>) outputs(%325 : memref<1xui32>) -> memref<1xui32> loc(#loc2)
    } loc(#loc2)
    return %arg1, %arg2 : memref<1x4x512xf32, @DDR>, memref<272xui32> loc(#loc41)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("PROFWORKPOINT_READ")
#loc3 = loc("Reshape_1423")
#loc4 = loc("t_Reshape")
#loc5 = loc("MVN_0")
#loc6 = loc("t_MVN")
#loc7 = loc("MVN_1")
#loc8 = loc("Swish_0")
#loc9 = loc("t_Swish")
#loc10 = loc("actshaveProfilingCMX2DDR0")
#loc11 = loc("MVN_2")
#loc12 = loc("MVN_3")
#loc13 = loc("Tanh_0")
#loc14 = loc("t_Tanh")
#loc15 = loc("output")
#loc16 = loc("t_Output")
#loc17 = loc("finishing_barrier")
#loc18 = loc("actshaveProfilingCMX2DDR16")
#loc19 = loc("tile_1")
#loc20 = loc("_input_cluster_0")
#loc21 = loc("_input_cluster_1")
#loc22 = loc("tile_0")
#loc23 = loc("_outputBuff_cluster_0")
#loc24 = loc("_outputBuff_cluster_1")
#loc25 = loc("_profilingBuff_cluster_0")
#loc26 = loc("_profilingBuff_cluster_1")
#loc27 = loc("_cluster_0")
#loc28 = loc("_cluster_1")
#loc29 = loc("dmaProfilingCMX2DDR0")
#loc30 = loc("dmaProfilingCMX2DDR256")
#loc31 = loc("newProfilingBuffer")
#loc32 = loc("cluster_0")
#loc33 = loc("cluster_1")
#loc34 = loc(fused[#loc3, #loc4])
#loc35 = loc(fused[#loc5, #loc6])
#loc36 = loc(fused[#loc7, #loc6])
#loc37 = loc(fused[#loc8, #loc9])
#loc38 = loc(fused[#loc11, #loc6])
#loc39 = loc(fused[#loc12, #loc6])
#loc40 = loc(fused[#loc13, #loc14])
#loc41 = loc(fused[#loc15, #loc16])
#loc42 = loc(fused[#loc8, #loc9, #loc19, #loc20])
#loc43 = loc(fused[#loc8, #loc9, #loc19, #loc21])
#loc44 = loc(fused[#loc8, #loc9, #loc22, #loc20])
#loc45 = loc(fused[#loc8, #loc9, #loc22, #loc21])
#loc46 = loc(fused[#loc8, #loc9, #loc19, #loc23])
#loc47 = loc(fused[#loc8, #loc9, #loc19, #loc24])
#loc48 = loc(fused[#loc8, #loc9, #loc22, #loc23])
#loc49 = loc(fused[#loc8, #loc9, #loc22, #loc24])
#loc50 = loc(fused[#loc13, #loc14, #loc19, #loc20])
#loc51 = loc(fused[#loc13, #loc14, #loc19, #loc21])
#loc52 = loc(fused[#loc13, #loc14, #loc22, #loc20])
#loc53 = loc(fused[#loc13, #loc14, #loc22, #loc21])
#loc54 = loc(fused[#loc13, #loc14, #loc19, #loc23])
#loc55 = loc(fused[#loc13, #loc14, #loc19, #loc24])
#loc56 = loc(fused[#loc13, #loc14, #loc22, #loc23])
#loc57 = loc(fused[#loc13, #loc14, #loc22, #loc24])
#loc58 = loc(fused[#loc3, #loc4, #loc19, #loc20])
#loc59 = loc(fused[#loc3, #loc4, #loc19, #loc21])
#loc60 = loc(fused[#loc3, #loc4, #loc22, #loc20])
#loc61 = loc(fused[#loc3, #loc4, #loc22, #loc21])
#loc62 = loc(fused[#loc3, #loc4, #loc19, #loc23])
#loc63 = loc(fused[#loc3, #loc4, #loc19, #loc24])
#loc64 = loc(fused[#loc3, #loc4, #loc22, #loc23])
#loc65 = loc(fused[#loc3, #loc4, #loc22, #loc24])
#loc66 = loc(fused[#loc5, #loc6, #loc19, #loc20])
#loc67 = loc(fused[#loc5, #loc6, #loc19, #loc21])
#loc68 = loc(fused[#loc5, #loc6, #loc22, #loc20])
#loc69 = loc(fused[#loc5, #loc6, #loc22, #loc21])
#loc70 = loc(fused[#loc5, #loc6, #loc19, #loc23])
#loc71 = loc(fused[#loc5, #loc6, #loc19, #loc24])
#loc72 = loc(fused[#loc5, #loc6, #loc22, #loc23])
#loc73 = loc(fused[#loc5, #loc6, #loc22, #loc24])
#loc74 = loc(fused[#loc7, #loc6, #loc19, #loc20])
#loc75 = loc(fused[#loc7, #loc6, #loc19, #loc21])
#loc76 = loc(fused[#loc7, #loc6, #loc22, #loc20])
#loc77 = loc(fused[#loc7, #loc6, #loc22, #loc21])
#loc78 = loc(fused[#loc7, #loc6, #loc19, #loc23])
#loc79 = loc(fused[#loc7, #loc6, #loc19, #loc24])
#loc80 = loc(fused[#loc7, #loc6, #loc22, #loc23])
#loc81 = loc(fused[#loc7, #loc6, #loc22, #loc24])
#loc82 = loc(fused[#loc11, #loc6, #loc19, #loc20])
#loc83 = loc(fused[#loc11, #loc6, #loc19, #loc21])
#loc84 = loc(fused[#loc11, #loc6, #loc22, #loc20])
#loc85 = loc(fused[#loc11, #loc6, #loc22, #loc21])
#loc86 = loc(fused[#loc11, #loc6, #loc19, #loc23])
#loc87 = loc(fused[#loc11, #loc6, #loc19, #loc24])
#loc88 = loc(fused[#loc11, #loc6, #loc22, #loc23])
#loc89 = loc(fused[#loc11, #loc6, #loc22, #loc24])
#loc90 = loc(fused[#loc12, #loc6, #loc19, #loc20])
#loc91 = loc(fused[#loc12, #loc6, #loc19, #loc21])
#loc92 = loc(fused[#loc12, #loc6, #loc22, #loc20])
#loc93 = loc(fused[#loc12, #loc6, #loc22, #loc21])
#loc94 = loc(fused[#loc12, #loc6, #loc19, #loc23])
#loc95 = loc(fused[#loc12, #loc6, #loc19, #loc24])
#loc96 = loc(fused[#loc12, #loc6, #loc22, #loc23])
#loc97 = loc(fused[#loc12, #loc6, #loc22, #loc24])
#loc98 = loc(fused[#loc15, #loc16, #loc19, #loc20])
#loc99 = loc(fused[#loc15, #loc16, #loc19, #loc21])
#loc100 = loc(fused[#loc15, #loc16, #loc22, #loc20])
#loc101 = loc(fused[#loc15, #loc16, #loc22, #loc21])
#loc102 = loc(fused[#loc15, #loc16, #loc19, #loc23])
#loc103 = loc(fused[#loc15, #loc16, #loc19, #loc24])
#loc104 = loc(fused[#loc15, #loc16, #loc22, #loc23])
#loc105 = loc(fused[#loc15, #loc16, #loc22, #loc24])
#loc106 = loc(fused[#loc3, #loc4, #loc22, #loc25])
#loc107 = loc(fused[#loc3, #loc4, #loc22, #loc26])
#loc108 = loc(fused[#loc3, #loc4, #loc19, #loc25])
#loc109 = loc(fused[#loc3, #loc4, #loc19, #loc26])
#loc110 = loc(fused[#loc5, #loc6, #loc22, #loc25])
#loc111 = loc(fused[#loc5, #loc6, #loc22, #loc26])
#loc112 = loc(fused[#loc5, #loc6, #loc19, #loc25])
#loc113 = loc(fused[#loc5, #loc6, #loc19, #loc26])
#loc114 = loc(fused[#loc7, #loc6, #loc22, #loc25])
#loc115 = loc(fused[#loc7, #loc6, #loc22, #loc26])
#loc116 = loc(fused[#loc7, #loc6, #loc19, #loc25])
#loc117 = loc(fused[#loc7, #loc6, #loc19, #loc26])
#loc118 = loc(fused[#loc8, #loc9, #loc22, #loc25])
#loc119 = loc(fused[#loc8, #loc9, #loc22, #loc26])
#loc120 = loc(fused[#loc8, #loc9, #loc19, #loc25])
#loc121 = loc(fused[#loc8, #loc9, #loc19, #loc26])
#loc122 = loc(fused[#loc11, #loc6, #loc22, #loc25])
#loc123 = loc(fused[#loc11, #loc6, #loc22, #loc26])
#loc124 = loc(fused[#loc11, #loc6, #loc19, #loc25])
#loc125 = loc(fused[#loc11, #loc6, #loc19, #loc26])
#loc126 = loc(fused[#loc12, #loc6, #loc22, #loc25])
#loc127 = loc(fused[#loc12, #loc6, #loc22, #loc26])
#loc128 = loc(fused[#loc12, #loc6, #loc19, #loc25])
#loc129 = loc(fused[#loc12, #loc6, #loc19, #loc26])
#loc130 = loc(fused[#loc13, #loc14, #loc22, #loc25])
#loc131 = loc(fused[#loc13, #loc14, #loc22, #loc26])
#loc132 = loc(fused[#loc13, #loc14, #loc19, #loc25])
#loc133 = loc(fused[#loc13, #loc14, #loc19, #loc26])
#loc134 = loc(fused[#loc15, #loc16, #loc22, #loc25])
#loc135 = loc(fused[#loc15, #loc16, #loc22, #loc26])
#loc136 = loc(fused[#loc15, #loc16, #loc19, #loc25])
#loc137 = loc(fused[#loc15, #loc16, #loc19, #loc26])
#loc138 = loc(fused[#loc3, #loc4, #loc27])
#loc139 = loc(fused[#loc3, #loc4, #loc28])
#loc140 = loc(fused[#loc5, #loc6, #loc27])
#loc141 = loc(fused[#loc5, #loc6, #loc28])
#loc142 = loc(fused[#loc7, #loc6, #loc27])
#loc143 = loc(fused[#loc7, #loc6, #loc28])
#loc144 = loc(fused[#loc8, #loc9, #loc27])
#loc145 = loc(fused[#loc8, #loc9, #loc28])
#loc146 = loc(fused[#loc11, #loc6, #loc27])
#loc147 = loc(fused[#loc11, #loc6, #loc28])
#loc148 = loc(fused[#loc12, #loc6, #loc27])
#loc149 = loc(fused[#loc12, #loc6, #loc28])
#loc150 = loc(fused[#loc13, #loc14, #loc27])
#loc151 = loc(fused[#loc13, #loc14, #loc28])
#loc152 = loc(fused[#loc15, #loc16, #loc27])
#loc153 = loc(fused[#loc15, #loc16, #loc28])
#loc154 = loc(fused[#loc3, #loc4, #loc22, #loc32])
#loc155 = loc(fused[#loc3, #loc4, #loc22, #loc33])
#loc156 = loc(fused[#loc3, #loc4, #loc19, #loc32])
#loc157 = loc(fused[#loc3, #loc4, #loc19, #loc33])
#loc158 = loc(fused[#loc5, #loc6, #loc22, #loc32])
#loc159 = loc(fused[#loc5, #loc6, #loc22, #loc33])
#loc160 = loc(fused[#loc5, #loc6, #loc19, #loc32])
#loc161 = loc(fused[#loc5, #loc6, #loc19, #loc33])
#loc162 = loc(fused[#loc7, #loc6, #loc22, #loc32])
#loc163 = loc(fused[#loc7, #loc6, #loc22, #loc33])
#loc164 = loc(fused[#loc7, #loc6, #loc19, #loc32])
#loc165 = loc(fused[#loc7, #loc6, #loc19, #loc33])
#loc166 = loc(fused[#loc8, #loc9, #loc22, #loc32])
#loc167 = loc(fused[#loc8, #loc9, #loc22, #loc33])
#loc168 = loc(fused[#loc8, #loc9, #loc19, #loc32])
#loc169 = loc(fused[#loc8, #loc9, #loc19, #loc33])
#loc170 = loc(fused[#loc10, #loc27])
#loc171 = loc(fused[#loc10, #loc28])
#loc172 = loc(fused[#loc11, #loc6, #loc22, #loc32])
#loc173 = loc(fused[#loc11, #loc6, #loc22, #loc33])
#loc174 = loc(fused[#loc11, #loc6, #loc19, #loc32])
#loc175 = loc(fused[#loc11, #loc6, #loc19, #loc33])
#loc176 = loc(fused[#loc12, #loc6, #loc22, #loc32])
#loc177 = loc(fused[#loc12, #loc6, #loc22, #loc33])
#loc178 = loc(fused[#loc12, #loc6, #loc19, #loc32])
#loc179 = loc(fused[#loc12, #loc6, #loc19, #loc33])
#loc180 = loc(fused[#loc13, #loc14, #loc22, #loc32])
#loc181 = loc(fused[#loc13, #loc14, #loc22, #loc33])
#loc182 = loc(fused[#loc13, #loc14, #loc19, #loc32])
#loc183 = loc(fused[#loc13, #loc14, #loc19, #loc33])
#loc184 = loc(fused[#loc15, #loc16, #loc22, #loc32])
#loc185 = loc(fused[#loc15, #loc16, #loc22, #loc33])
#loc186 = loc(fused[#loc15, #loc16, #loc19, #loc32])
#loc187 = loc(fused[#loc15, #loc16, #loc19, #loc33])
#loc188 = loc(fused[#loc18, #loc27])
#loc189 = loc(fused[#loc18, #loc28])

//CHECK: {"traceEvents":[
//CHECK: {"name": "process_name", "ph": "M", "pid":0, "args": {"name" : "DMA"}},
//CHECK: {"name": "process_sort_index", "ph": "M", "pid":0, "args": {"sort_index" : "0"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid":0, "tid":0, "args": {"name" : "DMA"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid":0, "tid":1, "args": {"name" : "DMA"}},
//CHECK: {"name": "process_name", "ph": "M", "pid":1, "args": {"name" : "Cluster (0)"}},
//CHECK: {"name": "process_sort_index", "ph": "M", "pid":1, "args": {"sort_index" : "1"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid":1, "tid":0, "args": {"name" : "SW / Shave"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid":1, "tid":1, "args": {"name" : "SW / Shave"}},
//CHECK: {"name": "process_name", "ph": "M", "pid":2, "args": {"name" : "Cluster (1)"}},
//CHECK: {"name": "process_sort_index", "ph": "M", "pid":2, "args": {"sort_index" : "2"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid":2, "tid":0, "args": {"name" : "SW / Shave"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid":2, "tid":1, "args": {"name" : "SW / Shave"}},
//CHECK: {"name": "process_name", "ph": "M", "pid":3, "args": {"name" : "Layers"}},
//CHECK: {"name": "process_sort_index", "ph": "M", "pid":3, "args": {"sort_index" : "3"}},
//CHECK: {"name": "thread_name", "ph": "M", "pid":3, "tid":0, "args": {"name" : "Layers"}},
//CHECK: {"name":"Reshape_1423?t_Reshape/_cluster_0", "cat":"DMA", "ph":"X", "ts":0.000, "dur":0.651, "pid":0, "tid":0},
//CHECK: {"name":"Reshape_1423?t_Reshape/_cluster_1", "cat":"DMA", "ph":"X", "ts":4.167, "dur":0.651, "pid":0, "tid":0},
//CHECK: {"name":"Reshape_1423?t_Reshape/_cluster_0", "cat":"DMA", "ph":"X", "ts":13.151, "dur":0.312, "pid":0, "tid":0},
//CHECK: {"name":"Reshape_1423?t_Reshape/_cluster_1", "cat":"DMA", "ph":"X", "ts":13.307, "dur":0.312, "pid":0, "tid":1},
//CHECK: {"name":"MVN_0?t_MVN/_cluster_0", "cat":"DMA", "ph":"X", "ts":13.854, "dur":0.546, "pid":0, "tid":0},
//CHECK: {"name":"MVN_0?t_MVN/_cluster_1", "cat":"DMA", "ph":"X", "ts":14.011, "dur":0.625, "pid":0, "tid":1},
//CHECK: {"name":"MVN_1?t_MVN/_cluster_0", "cat":"DMA", "ph":"X", "ts":31.172, "dur":0.312, "pid":0, "tid":0},
//CHECK: {"name":"MVN_1?t_MVN/_cluster_1", "cat":"DMA", "ph":"X", "ts":31.328, "dur":0.312, "pid":0, "tid":1},
//CHECK: {"name":"Swish_0?t_Swish/_cluster_0", "cat":"DMA", "ph":"X", "ts":31.875, "dur":0.494, "pid":0, "tid":0},
//CHECK: {"name":"Swish_0?t_Swish/_cluster_1", "cat":"DMA", "ph":"X", "ts":32.031, "dur":0.520, "pid":0, "tid":1},
//CHECK: {"name":"Swish_0?t_Swish/_cluster_0", "cat":"DMA", "ph":"X", "ts":32.708, "dur":0.494, "pid":0, "tid":0},
//CHECK: {"name":"Swish_0?t_Swish/_cluster_1", "cat":"DMA", "ph":"X", "ts":32.995, "dur":0.546, "pid":0, "tid":1},
//CHECK: {"name":"Swish_0?t_Swish/_cluster_0", "cat":"DMA", "ph":"X", "ts":37.839, "dur":0.364, "pid":0, "tid":0},
//CHECK: {"name":"Swish_0?t_Swish/_cluster_1", "cat":"DMA", "ph":"X", "ts":37.995, "dur":0.364, "pid":0, "tid":1},
//CHECK: {"name":"Swish_0?t_Swish/_cluster_0", "cat":"DMA", "ph":"X", "ts":38.594, "dur":0.312, "pid":0, "tid":0},
//CHECK: {"name":"Swish_0?t_Swish/_cluster_1", "cat":"DMA", "ph":"X", "ts":38.750, "dur":0.312, "pid":0, "tid":1},
//CHECK: {"name":"MVN_2?t_MVN/_cluster_0", "cat":"DMA", "ph":"X", "ts":39.297, "dur":0.598, "pid":0, "tid":0},
//CHECK: {"name":"MVN_2?t_MVN/_cluster_1", "cat":"DMA", "ph":"X", "ts":39.453, "dur":0.598, "pid":0, "tid":1},
//CHECK: {"name":"MVN_3?t_MVN/_cluster_0", "cat":"DMA", "ph":"X", "ts":48.047, "dur":0.312, "pid":0, "tid":0},
//CHECK: {"name":"MVN_3?t_MVN/_cluster_1", "cat":"DMA", "ph":"X", "ts":48.203, "dur":0.312, "pid":0, "tid":1},
//CHECK: {"name":"Tanh_0?t_Tanh/_cluster_0", "cat":"DMA", "ph":"X", "ts":48.750, "dur":0.494, "pid":0, "tid":0},
//CHECK: {"name":"Tanh_0?t_Tanh/_cluster_1", "cat":"DMA", "ph":"X", "ts":48.906, "dur":0.494, "pid":0, "tid":1},
//CHECK: {"name":"Tanh_0?t_Tanh/_cluster_0", "cat":"DMA", "ph":"X", "ts":49.557, "dur":0.494, "pid":0, "tid":0},
//CHECK: {"name":"Tanh_0?t_Tanh/_cluster_1", "cat":"DMA", "ph":"X", "ts":49.714, "dur":0.494, "pid":0, "tid":1},
//CHECK: {"name":"Tanh_0?t_Tanh/_cluster_0", "cat":"DMA", "ph":"X", "ts":54.818, "dur":0.312, "pid":0, "tid":0},
//CHECK: {"name":"Tanh_0?t_Tanh/_cluster_1", "cat":"DMA", "ph":"X", "ts":54.974, "dur":0.312, "pid":0, "tid":1},
//CHECK: {"name":"Tanh_0?t_Tanh/_cluster_0", "cat":"DMA", "ph":"X", "ts":55.521, "dur":0.312, "pid":0, "tid":0},
//CHECK: {"name":"Tanh_0?t_Tanh/_cluster_1", "cat":"DMA", "ph":"X", "ts":55.677, "dur":0.312, "pid":0, "tid":1},
//CHECK: {"name":"output?t_Output/_cluster_0", "cat":"DMA", "ph":"X", "ts":56.224, "dur":0.625, "pid":0, "tid":0},
//CHECK: {"name":"output?t_Output/_cluster_1", "cat":"DMA", "ph":"X", "ts":56.380, "dur":0.625, "pid":0, "tid":1},
//CHECK: {"name":"output?t_Output/_cluster_0", "cat":"DMA", "ph":"X", "ts":61.719, "dur":0.677, "pid":0, "tid":0},
//CHECK: {"name":"output?t_Output/_cluster_1", "cat":"DMA", "ph":"X", "ts":61.875, "dur":0.364, "pid":0, "tid":1},
//CHECK: {"name":"Reshape_1423?t_Reshape/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":6.276, "dur":5.312, "pid":1, "tid":0},
//CHECK: {"name":"Reshape_1423?t_Reshape/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":6.771, "dur":4.947, "pid":1, "tid":1},
//CHECK: {"name":"MVN_0?t_MVN/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":15.156, "dur":11.328, "pid":1, "tid":0},
//CHECK: {"name":"MVN_0?t_MVN/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":15.287, "dur":11.458, "pid":1, "tid":1},
//CHECK: {"name":"MVN_1?t_MVN/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":27.578, "dur":2.604, "pid":1, "tid":0},
//CHECK: {"name":"MVN_1?t_MVN/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":27.917, "dur":2.734, "pid":1, "tid":1},
//CHECK: {"name":"Swish_0?t_Swish/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":33.932, "dur":2.994, "pid":1, "tid":0},
//CHECK: {"name":"Swish_0?t_Swish/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":34.323, "dur":2.734, "pid":1, "tid":1},
//CHECK: {"name":"MVN_2?t_MVN/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":40.573, "dur":2.656, "pid":1, "tid":0},
//CHECK: {"name":"MVN_2?t_MVN/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":40.833, "dur":2.656, "pid":1, "tid":1},
//CHECK: {"name":"MVN_3?t_MVN/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":44.193, "dur":2.942, "pid":1, "tid":0},
//CHECK: {"name":"MVN_3?t_MVN/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":44.583, "dur":2.942, "pid":1, "tid":1},
//CHECK: {"name":"Tanh_0?t_Tanh/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":50.729, "dur":3.437, "pid":1, "tid":0},
//CHECK: {"name":"Tanh_0?t_Tanh/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":50.859, "dur":3.437, "pid":1, "tid":1},
//CHECK: {"name":"output?t_Output/tile_0/cluster_0", "cat":"SW", "ph":"X", "ts":57.422, "dur":3.750, "pid":1, "tid":0},
//CHECK: {"name":"output?t_Output/tile_1/cluster_0", "cat":"SW", "ph":"X", "ts":57.813, "dur":3.098, "pid":1, "tid":1},
//CHECK: {"name":"Reshape_1423?t_Reshape/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":6.406, "dur":5.572, "pid":2, "tid":0},
//CHECK: {"name":"Reshape_1423?t_Reshape/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":6.953, "dur":4.895, "pid":2, "tid":1},
//CHECK: {"name":"MVN_0?t_MVN/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":15.026, "dur":11.328, "pid":2, "tid":0},
//CHECK: {"name":"MVN_0?t_MVN/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":15.417, "dur":11.197, "pid":2, "tid":1},
//CHECK: {"name":"MVN_1?t_MVN/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":27.448, "dur":2.864, "pid":2, "tid":0},
//CHECK: {"name":"MVN_1?t_MVN/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":27.708, "dur":2.734, "pid":2, "tid":1},
//CHECK: {"name":"Swish_0?t_Swish/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":34.063, "dur":3.125, "pid":2, "tid":0},
//CHECK: {"name":"Swish_0?t_Swish/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":34.193, "dur":3.125, "pid":2, "tid":1},
//CHECK: {"name":"MVN_2?t_MVN/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":40.443, "dur":2.604, "pid":2, "tid":0},
//CHECK: {"name":"MVN_2?t_MVN/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":40.703, "dur":2.656, "pid":2, "tid":1},
//CHECK: {"name":"MVN_3?t_MVN/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":44.323, "dur":3.072, "pid":2, "tid":0},
//CHECK: {"name":"MVN_3?t_MVN/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":44.453, "dur":2.812, "pid":2, "tid":1},
//CHECK: {"name":"Tanh_0?t_Tanh/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":50.599, "dur":3.307, "pid":2, "tid":0},
//CHECK: {"name":"Tanh_0?t_Tanh/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":50.990, "dur":3.046, "pid":2, "tid":1},
//CHECK: {"name":"output?t_Output/tile_1/cluster_1", "cat":"SW", "ph":"X", "ts":57.552, "dur":3.229, "pid":2, "tid":0},
//CHECK: {"name":"output?t_Output/tile_0/cluster_1", "cat":"SW", "ph":"X", "ts":57.682, "dur":3.359, "pid":2, "tid":1},
//CHECK: {"name":"Reshape_1423", "cat":"Layer", "ph":"X", "ts":0.000, "dur":13.619, "pid":3, "tid":0, "args":{"Layer type": "Reshape"}},
//CHECK: {"name":"MVN_0", "cat":"Layer", "ph":"X", "ts":13.854, "dur":12.891, "pid":3, "tid":0, "args":{"Layer type": "MVN"}},
//CHECK: {"name":"MVN_1", "cat":"Layer", "ph":"X", "ts":27.448, "dur":4.192, "pid":3, "tid":0, "args":{"Layer type": "MVN"}},
//CHECK: {"name":"Swish_0", "cat":"Layer", "ph":"X", "ts":31.875, "dur":7.187, "pid":3, "tid":0, "args":{"Layer type": "Swish"}},
//CHECK: {"name":"MVN_2", "cat":"Layer", "ph":"X", "ts":39.297, "dur":4.192, "pid":3, "tid":0, "args":{"Layer type": "MVN"}},
//CHECK: {"name":"MVN_3", "cat":"Layer", "ph":"X", "ts":44.193, "dur":4.322, "pid":3, "tid":0, "args":{"Layer type": "MVN"}},
//CHECK: {"name":"Tanh_0", "cat":"Layer", "ph":"X", "ts":48.750, "dur":7.239, "pid":3, "tid":0, "args":{"Layer type": "Tanh"}},
//CHECK: {"name":"output", "cat":"Layer", "ph":"X", "ts":56.224, "dur":6.172, "pid":3, "tid":0, "args":{"Layer type": "Output"}}
//CHECK: ],
//CHECK: "taskStatistics": {
//CHECK: "total duration":62.396,
//CHECK: "DMA duration":10.434,
//CHECK: "DPU duration":0.000,
//CHECK: "SW duration":37.835,
//CHECK: "DMA-DPU overlap":0.000,
//CHECK: "DMA-SW overlap":0.000,
//CHECK: "SW-DPU overlap":0.000,
//CHECK: "all tasks union":48.269,
//CHECK: "total idle":14.127,
//CHECK: "SW duration without DPU overlap":37.835,
//CHECK: "DMA duration without overlaps":10.434,
//CHECK: "Sum of DMA task durations":14.462,
//CHECK: "Sum of DPU task durations":0.000,
//CHECK: "Sum of SW task durations":137.954
//CHECK: },
//CHECK: "displayTimeUnit": "ns"
//CHECK: }
