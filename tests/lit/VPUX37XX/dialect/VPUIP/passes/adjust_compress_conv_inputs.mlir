//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --adjust-compress-conv-inputs %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = type !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @CompressConvWeightsMoreThan4IC
func @CompressConvWeightsMoreThan4IC(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> {
    %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare memref<64x16x7x7x!qElemType0, #NHWC> = dense<1.0> :
        tensor<64x13x7x7xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]
    %2 = memref.alloc() : memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %weights = VPUIP.Copy
        inputs(%cst_0 : memref<64x16x7x7x!qElemType0, #NHWC>)
        outputs(%2 : memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>)
         -> memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table = VPUIP.Copy
        inputs(%cst : memref<64x1x1x4xsi32>)
        outputs(%3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %output = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %NCEOp = VPUIP.NCEClusterTask {
          kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = "CONV"
      }
      input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      weights(%weights : memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>)
      weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}

            }
      }
      PPE :  {
      }
    return %NCEOp : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:        [[cst:%.*]] = const.Declare memref<64x13x7x7x!qElemType2, #NHWC>
    //CHECK-DAG:        [[cst_0:%.*]] = const.Declare memref<64x1x1x4xsi32>
    //CHECK:        [[VAR0:%.*]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[VAR1:%.*]] = VPUIP.Copy inputs([[cst_0]] : memref<64x1x1x4xsi32>) outputs([[VAR0]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[VAR2:%.*]] = memref.alloc() : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR3:%.*]] = memref.alloc() : memref<64x13x7x7x!qElemType2, {order = #NHWC, strides = [640, 1, 91, 13]}, [@CMX_NN, 0]>
    //CHECK:        [[VAR4:%.*]] = VPUIP.Copy inputs([[cst]] : memref<64x13x7x7x!qElemType2, #NHWC>) outputs([[VAR3]] :
    //CHECK-SAME:           memref<64x13x7x7x!qElemType2, {order = #NHWC, strides = [640, 1, 91, 13]}, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x13x7x7x!qElemType2, {order = #NHWC, strides = [640, 1, 91, 13]}, [@CMX_NN, 0]>
    //CHECK:        [[VAR5:%.*]] = VPUIP.ShapeCast {shape = [64, 16, 7, 7]} inputs([[VAR4]] : memref<64x13x7x7x!qElemType2, {order = #NHWC, strides = [640, 1, 91, 13]}, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x16x7x7x!qElemType2, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR6:%.*]] = VPUIP.NCEClusterTask {cm_sp_pattern = 8191 : i64,
    //CHECK-SAME:           kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}, kernel_size = [7, 7],
    //CHECK-SAME:           kernel_strides = [2, 2], minimumHardwareExecutionCost = 375613 : i64, task_type = "CONV"}
    //CHECK-SAME:           input(%arg0 : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[VAR5]] : memref<64x16x7x7x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[VAR1]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input(%arg0 : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[VAR2]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           outputs([[VAR2]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK-SAME:           variants :  {
    //CHECK:       DPUTask {mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0],
    //CHECK-SAME:           pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}
    //CHECK:       return [[VAR6]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType0 = type !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = type !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @CompressConvWeightsLessThan4IC
func @CompressConvWeightsLessThan4IC(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> {
    %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare memref<64x1x1x208x!qElemType0, #NHWC> = dense<1.0> :
        tensor<64x3x7x7xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>,
        #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>,
        #const.Reshape<[64, 1, 1, 196]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>, #const.Reorder<#NHWC>]
    %2 = memref.alloc() : memref<64x1x1x208x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %weights = VPUIP.Copy
        inputs(%cst_0 : memref<64x1x1x208x!qElemType0, #NHWC>)
        outputs(%2 : memref<64x1x1x208x!qElemType0, #NHWC, [@CMX_NN, 0]>)
         -> memref<64x1x1x208x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table = VPUIP.Copy
        inputs(%cst : memref<64x1x1x4xsi32>)
        outputs(%3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %output = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %NCEOp = VPUIP.NCEClusterTask {
          kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = "CONV"
      }
      input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      weights(%weights : memref<64x1x1x208x!qElemType0, #NHWC, [@CMX_NN, 0]>)
      weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}
            }
      }
      PPE :  {
      }
    return %NCEOp : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:        [[cst:%.*]] = const.Declare memref<64x1x1x208x!qElemType2, #NHWC>
    //CHECK-DAG:        [[cst_0:%.*]] = const.Declare memref<64x1x1x4xsi32>

    //CHECK:        [[VAR0:%.*]] = memref.alloc() : memref<64x1x1x208x!qElemType2, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR1:%.*]] = VPUIP.Copy inputs([[cst]] : memref<64x1x1x208x!qElemType2, #NHWC>) outputs([[VAR0]] :
    //CHECK-SAME:           memref<64x1x1x208x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x208x!qElemType2, #NHWC, [@CMX_NN, 0]>

    //CHECK:        [[VAR2:%.*]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[VAR3:%.*]] = VPUIP.Copy inputs([[cst_0]] : memref<64x1x1x4xsi32>) outputs([[VAR2]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>

    //CHECK:        [[VAR4:%.*]] = memref.alloc() : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR5:%.*]] = VPUIP.ShapeCast {shape = [64, 16, 7, 7]} inputs([[VAR1]] : memref<64x1x1x208x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x16x7x7x!qElemType2, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR6:%.*]] = VPUIP.NCEClusterTask {cm_sp_pattern = 15 : i64,
    //CHECK-SAME:           kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}, kernel_size = [7, 7],
    //CHECK-SAME:           kernel_strides = [2, 2], minimumHardwareExecutionCost = 375613 : i64, task_type = "CONV"}
    //CHECK-SAME:           input(%arg0 : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[VAR5]] : memref<64x16x7x7x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[VAR3]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input(%arg0 : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[VAR4]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           outputs([[VAR4]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK-SAME:           variants : {
    //CHECK:       DPUTask {mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0],
    //CHECK-SAME:           pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}
    //CHECK:       return [[VAR6]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = type !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @DoNotCompressSparseConvWeights
func @DoNotCompressSparseConvWeights(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> {
    %cst_weights = const.Declare memref<64x16x7x7x!qElemType0, #NHWC> = dense<1.0> : tensor<64x3x7x7xf16>,
        [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare memref<64x1x1x896xi1> = dense<1.0> : tensor<64x3x7x7xf16>,
        [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>, #const.GetSparsityMap]
    %weights_sparse_ddr = VPUIP.GroupSparseBuffer(%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType0, #NHWC>, sparsity_map=memref<64x1x1x896xi1>, is_weights>
    %weights_cmx = memref.alloc() : memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %weights_sm_cmx = memref.alloc() : memref<64x1x1x896xi1, [@CMX_NN, 0]>
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer(%weights_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<64x1x1x896xi1, [@CMX_NN, 0]>, is_weights>
    %weights_sparse = VPUIP.Copy
        inputs(%weights_sparse_ddr : !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType0, #NHWC>, sparsity_map=memref<64x1x1x896xi1>, is_weights>)
        outputs(%weights_sparse_cmx : !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<64x1x1x896xi1, [@CMX_NN, 0]>, is_weights>)
        -> !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<64x1x1x896xi1, [@CMX_NN, 0]>, is_weights>

    %weights_data, %weights_sm = VPUIP.UngroupSparseBuffer(%weights_sparse) {result_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}
        -> memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>, memref<64x1x1x896xi1, [@CMX_NN, 0]>

    %cst_weights_table = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %weights_table_cmx = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table = VPUIP.Copy
        inputs(%cst_weights_table : memref<64x1x1x4xsi32>)
        outputs(%weights_table_cmx : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
        -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>

    %output_cmx = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %output = VPUIP.NCEClusterTask {
          kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = "CONV"
      }
      input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      weights(%weights_data : memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>)
      weights_sparsity_map(%weights_sm : memref<64x1x1x896xi1, [@CMX_NN, 0]>)
      weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output_cmx : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output_cmx : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
      variants: {
          DPUTask { outEnd = [111, 111, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}, outStart = [0, 0, 0] }
      }
      PPE : {
      }
    return %output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare memref<64x16x7x7x!qElemType2, #NHWC>
    //CHECK-DAG:   [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<64x1x1x896xi1>
    //CHECK:       [[WEIGHTS_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}

    //CHECK:       [[WEIGHTS_DATA_CMX:%.+]] = memref.alloc() : memref<64x16x7x7x!qElemType2, #NHWC, [@CMX_NN, 0]>
    //CHECK:       [[WEIGHTS_SM_CMX:%.+]] = memref.alloc() : memref<64x1x1x896xi1, [@CMX_NN, 0]>
    //CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[WEIGHTS_DATA_CMX]], [[WEIGHTS_SM_CMX]]) {is_weights}

    //CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPUIP.Copy inputs([[WEIGHTS_SPARSE_DDR]]
    //CHECK-SAME:                                      outputs([[WEIGHTS_SPARSE_CMX]]
    //CHECK:       [[WEIGHTS_DATA:%.+]], [[WEIGHTS_SM:%.+]] = VPUIP.UngroupSparseBuffer([[WEIGHTS_SPARSE]])

    //CHECK:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32>
    //CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.Copy inputs([[CST_WEIGHTS_TABLE]]
    //CHECK-SAME:                                     outputs([[WEIGHTS_TABLE_CMX]]

    //CHECK:       [[OUTPUT_CMX:%.+]] = memref.alloc() : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT:%.+]] = VPUIP.NCEClusterTask
    //CHECK-SAME:          input(%arg0
    //CHECK-SAME:          weights([[WEIGHTS_DATA]]
    //CHECK-SAME:          weights_sparsity_map([[WEIGHTS_SM]]
    //CHECK-SAME:          weight_table([[WEIGHTS_TABLE]]
    //CHECK-SAME:          parent_input(%arg0
    //CHECK-SAME:          parent_output([[OUTPUT_CMX]]
    //CHECK-SAME:          outputs([[OUTPUT_CMX]]
    //CHECK-SAME:          -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = type !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @TiledCompressConvWeights
func @TiledCompressConvWeights(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType3, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare memref<64x16x7x7x!qElemType0, #NHWC> = dense<1.0> :
        tensor<64x3x7x7xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType0, #NHWC, @CMX_NN,
        {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %weights = VPUIP.NCEClusterTiling
        inputs(%cst_0 as %arg2: memref<64x16x7x7x!qElemType0, #NHWC>)
        outputs(%2 as %arg3: memref<64x16x7x7x!qElemType0, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType0, #NHWC, @CMX_NN,
        {mode = "DUPLICATED", num_clusters = 2 : i64}> {
            %29 = VPUIP.Copy
                inputs(%arg2 : memref<64x16x7x7x!qElemType0, #NHWC>)
                outputs(%arg3 : memref<64x16x7x7x!qElemType0, #NHWC, @CMX_NN>)
                -> memref<64x16x7x7x!qElemType0, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x4xsi32,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN,
        {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %weights_table = VPUIP.NCEClusterTiling
        inputs(%cst as %arg2: memref<64x1x1x4xsi32>)
        outputs(%3 as %arg3: memref<64x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
            %29 = VPUIP.Copy
            inputs(%arg2 : memref<64x1x1x4xsi32>)
            outputs(%arg3 : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>
    }
    %output = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, @CMX_NN>
    %NCETilingOp = VPUIP.NCEClusterTiling
        inputs(%arg0 as %arg2: memref<1x16x224x224x!qElemType2, #NHWC, @CMX_NN>,
            %weights as %arg3: memref<64x16x7x7x!qElemType0, #NHWC, @CMX_NN>,
            %weights_table as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
        outputs(%output as %arg5: memref<1x64x112x112x!qElemType3, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %29 = VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
                kernel_size = [7, 7],
                kernel_strides = [2, 2],
                minimumHardwareExecutionCost = 269263 : i64,
                task_type = "CONV"
            }
            input(%arg2 : memref<1x16x224x224x!qElemType2, #NHWC, @CMX_NN>)
            weights(%arg3 : memref<64x16x7x7x!qElemType0, #NHWC, @CMX_NN>)
            weight_table(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>)
            parent_input(%arg2 : memref<1x16x224x224x!qElemType2, #NHWC, @CMX_NN>)
            parent_output(%arg5 : memref<1x64x112x112x!qElemType3, #NHWC, @CMX_NN>)
            outputs(%arg5 : memref<1x64x112x112x!qElemType3, #NHWC, @CMX_NN>) -> memref<1x64x112x112x!qElemType3, #NHWC, @CMX_NN> variants :
            {
                DPUTask {
                    cluster_id = 0 : i64, outEnd = [63, 55, 63], mpe_mode = "CUBOID_16x16",
                    pad = {bottom = 0 : i64, left = 3 : i64, right = 0 : i64, top = 3 : i64},
                    outStart = [0, 0, 0]
                }
                DPUTask {
                    cluster_id = 0 : i64, outEnd = [111, 55, 63], mpe_mode = "CUBOID_16x16",
                    pad = {bottom = 0 : i64, left = 0 : i64, right = 2 : i64, top = 3 : i64},
                    outStart = [64, 0, 0]
                }
                DPUTask {
                    cluster_id = 1 : i64, outEnd = [63, 111, 63], mpe_mode = "CUBOID_16x16",
                    pad = {bottom = 2 : i64, left = 3 : i64, right = 0 : i64, top = 0 : i64},
                    outStart = [0, 56, 0]
                }
                DPUTask {
                    cluster_id = 1 : i64, outEnd = [111, 111, 63], mpe_mode = "CUBOID_16x16",
                    pad = {bottom = 2 : i64, left = 0 : i64, right = 2 : i64, top = 0 : i64},
                    outStart = [64, 56, 0]
                }
            } PPE : {
                PPETask "NOOP" {
                    clamp_high = 255 : i64, clamp_low = 0 : i64,
                    fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64,
                    lrelu_shift = 0 : i64
                }
            }
    }
    return %NCETilingOp : !VPUIP.DistributedBuffer<1x64x112x112x!qElemType3, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK-DAG:        [[CST_WEIGHTS_TABLE:%.*]] = const.Declare memref<64x1x1x4xsi32>
    //CHECK-DAG:        [[CST_WEIGHTS:%.*]] = const.Declare memref<64x3x7x7x!qElemType2, #NHWC>

    //CHECK:        [[WEIGHTS_TABLE_ALLOC:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x4xsi32,
    //CHECK-SAME:           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[CLUSTER_WEIGHTS_TABLE:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:           inputs([[CST_WEIGHTS_TABLE]] as %arg1: memref<64x1x1x4xsi32>)
    //CHECK-SAME:           outputs([[WEIGHTS_TABLE_ALLOC]] as %arg2: memref<64x1x1x4xsi32, @CMX_NN>)

    //CHECK:        [[OUT_BUFFER:%.*]] = memref.alloc() : memref<1x64x112x112x!qElemType1, #NHWC, @CMX_NN>

    //CHECK:        [[WEIGHTS_ALLOC:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x3x7x7x!qElemType2,
    //CHECK-SAME:           {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[CLUSTER_WEIGHTS:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:           inputs([[CST_WEIGHTS]] as %arg1:
    //CHECK-SAME:           outputs([[WEIGHTS_ALLOC]] as %arg2:
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<64x3x7x7x!qElemType2, {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN,
    //CHECK:                        [[WEIGHTS_COPY:%.+]] = VPUIP.Copy
    //CHECK-SAME:                       inputs(%arg1 : memref<64x3x7x7x!qElemType2, #NHWC>)
    //CHECK-SAME:                       outputs(%arg2 : memref<64x3x7x7x!qElemType2, {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN>)
    //CHECK-SAME:                       -> memref<64x3x7x7x!qElemType2, {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN>
    //CHECK:        [[WEIGHTS_SHAPE_CAST:%.*]] = VPUIP.ShapeCast {shape = [64, 16, 7, 7]}
    //CHECK-SAME:       inputs([[CLUSTER_WEIGHTS]] : !VPUIP.DistributedBuffer<64x3x7x7x!qElemType2,
    //CHECK-SAME:           {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType2,
    //CHECK-SAME:           {order = #NHWC, strides = [784, 1, 112, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:       [[CONV_OUT:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:       inputs(%arg0 as %arg1
    //CHECK-SAME:       [[WEIGHTS_SHAPE_CAST]] as %arg2
    //CHECK-SAME:       [[CLUSTER_WEIGHTS_TABLE]] as %arg3
    //CHECK-SAME:       outputs([[OUT_BUFFER]] as %arg4
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:            [[CONV:%.+]] = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64,
    //CHECK-SAME:           kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
    //CHECK-SAME:           kernel_size = [7, 7], kernel_strides = [2, 2], minimumHardwareExecutionCost = 269263 : i64, task_type = "CONV"}
    //CHECK-SAME:           input(%arg1
    //CHECK-SAME:           weights(%arg2
    //CHECK-SAME:           weight_table(%arg3
    //CHECK-SAME:           parent_input(%arg1
    //CHECK-SAME:           parent_output(%arg4
    //CHECK-SAME:           outputs(%arg4
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, #NHWC, @CMX_NN> variants : {
    //CHECK:        return [[CONV_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = type !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @CompressConvActivations
func @CompressConvActivations(%arg0: memref<1x4x224x224x!qElemType2, #NHWC>,
                              %weights: memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>,
                              %weights_table: memref<64x1x1x4xsi32, [@CMX_NN, 0]>) 
                              -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> {
    %output = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %0 = memref.alloc() : memref<1x4x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>
    %input = VPUIP.Copy
        inputs(%arg0 : memref<1x4x224x224x!qElemType2, #NHWC>)
        outputs(%0 : memref<1x4x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
         -> memref<1x4x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>
    %NCEOp = VPUIP.NCEClusterTask {
          kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64},
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = "CONV"
      }
      input(%input : memref<1x4x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      weights(%weights : memref<64x16x7x7x!qElemType0, #NHWC, [@CMX_NN, 0]>)
      weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%input : memref<1x4x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}
            }
      }
      PPE :  {
      }
    return %NCEOp : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //CHECK:        [[VAR0:%.*]] = memref.alloc() : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR1:%.*]] = memref.alloc() : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR2:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x4x224x224x!qElemType0, #NHWC>) outputs([[VAR1]] :
    //CHECK-SAME:           memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR3:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 224, 224]} inputs([[VAR2]] : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR4:%.*]] = VPUIP.NCEClusterTask
    //CHECK-SAME:           kernel_padding = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}, kernel_size = [7, 7],
    //CHECK-SAME:           kernel_strides = [2, 2], minimumHardwareExecutionCost = 375613 : i64, task_type = "CONV"}
    //CHECK-SAME:           input([[VAR3]] : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights(%arg1 : memref<64x16x7x7x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table(%arg2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[VAR3]] : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[VAR0]] : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           outputs([[VAR0]] : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>
    //CHECK-SAME:           variants :  {
    //CHECK:       DPUTask {mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0],
    //CHECK-SAME:           pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}
    //CHECK:       return [[VAR4]] : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.012699142156862745>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0], {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0], {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x16x3x3x!qElemType1, #NHWC, [@CMX_NN, 0], {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!InputBuffer_DDR = type memref<1x4x224x224x!qElemType0, #NHWC>
!InputBuffer = type memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
!WeightsBuffer = type memref<64x16x2x2x!qElemType1, #NHWC, [@CMX_NN, 0]>
!WeightsTableBuffer = type memref<64x1x1x4xsi32, [@CMX_NN, 0]>
!OutputBuffer = type memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>

// CHECK:       func @CompressTiledConvActivations(
// CHECK-SAME:      [[INPUT_DDR:%.+]]: memref<1x4x224x224x!qElemType0, #NHWC>,
// CHECK-SAME:      [[WEIGHTS:%.+]]: memref<64x16x2x2x!qElemType1, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      [[WEIGHTS_TABLE:%.+]]: memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
// CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
func @CompressTiledConvActivations(%input_ddr: !InputBuffer_DDR,
                                   %weights: !WeightsBuffer,
                                   %weights_table: !WeightsTableBuffer) -> !OutputDistributed {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input = VPUIP.NCEClusterTiling inputs(%input_ddr as %arg3: !InputBuffer_DDR)
                                    outputs(%input_cmx as %arg4: !InputBuffer) -> !InputDistributed {
        %0 = VPUIP.Copy inputs(%arg3 : !InputBuffer_DDR) outputs(%arg4 : !InputBuffer) -> !InputBuffer
    }

    %output = VPURT.AllocDistributed -> !OutputDistributed
    %conv_output = VPUIP.NCEClusterTiling
            inputs(%input as %arg3: !InputBuffer,
                   %weights as %arg4: !WeightsBuffer,
                   %weights_table as %arg5: !WeightsTableBuffer)
            outputs(%output as %arg6: !OutputBuffer) -> !OutputDistributed {
        %conv = VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                kernel_size = [2, 2],
                kernel_strides = [2, 2],
                minimumHardwareExecutionCost = 375613 : i64,
                task_type = "CONV"
            }
            input(%arg3 : !InputBuffer)
            weights(%arg4 : !WeightsBuffer)
            weight_table(%arg5 : !WeightsTableBuffer)
            parent_input(%arg3 : !InputBuffer)
            parent_output(%arg6 : !OutputBuffer)
            outputs(%arg6 : !OutputBuffer) -> !OutputBuffer
            variants : {
                DPUTask { mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0], pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}
            }
            PPE :  {
            }
    }
    return %conv_output : !OutputDistributed

    // CHECK:       [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_DDR]] as %arg3: memref<1x4x224x224x!qElemType0, #NHWC>)
    // CHECK-SAME:                                         outputs([[INPUT_CMX]] as %arg4: memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    // CHECK:           %5 = VPUIP.Copy inputs(%arg3 : memref<1x4x224x224x!qElemType0, #NHWC>)
    // CHECK-SAME:                      outputs(%arg4 : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       }
    // CHECK:       [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[INPUT_SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 224, 224]}
    // CHECK-SAME:      inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                      -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CONV_OUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_SHAPE_CAST]] as [[INNER_IN:%[^:]+]]: memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:             [[WEIGHTS]] as [[INNER_W:%[^:]+]]: memref<64x16x2x2x!qElemType1, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:             [[WEIGHTS_TABLE]] as [[INNER_WT:%[^:]+]]: memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT_CMX]] as [[INNER_OUT:%[^:]+]]: memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {

    // CHECK:           [[CONV:%.+]] = VPUIP.NCEClusterTask {
    // CHECK-SAME:              input_channels_compression
    // CHECK-SAME:          }
    // CHECK-SAME:          input([[INNER_IN]]
    // CHECK-SAME:          weights([[INNER_W]]
    // CHECK-SAME:          weight_table([[INNER_WT]]
    // CHECK-SAME:          parent_input([[INNER_IN]]
    // CHECK-SAME:          parent_output([[INNER_OUT]]
    // CHECK-SAME:          outputs([[INNER_OUT]]
    // CHECK-SAME:          -> memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>
    // CHECK-SAME:           variants :  {
    // CHECK:       DPUTask {mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0],
    // CHECK-SAME:           pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}

    // CHECK:       return [[CONV_OUT]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK:       func @CompressConvSparseActivations([[INPUT_DDR:%.+]]: memref<1x4x224x224x!qElemType0, #NHWC>,
// CHECK-SAME:                                      [[INPUT_SM_DDR:%.+]]: memref<1x4x224x224xi1, #NHWC>,
// CHECK-SAME:                                      [[WEIGHTS:%.+]]: memref<64x16x7x7x!qElemType1, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:                                      [[WEIGHTS_TABLE:%.+]]: memref<64x1x1x4xsi32, [@CMX_NN, 0]>
// CHECK-SAME:  -> (memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>, memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>) {
func @CompressConvSparseActivations(%input_ddr: memref<1x4x224x224x!qElemType0, #NHWC>,
                                    %input_sm_ddr: memref<1x4x224x224xi1, #NHWC>,
                                    %weights: memref<64x16x7x7x!qElemType1, #NHWC, [@CMX_NN, 0]>,
                                    %weights_table: memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
        -> (memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>, memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>) {
    %input_cmx = memref.alloc() : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %input = VPUIP.Copy inputs(%input_ddr : memref<1x4x224x224x!qElemType0, #NHWC>)
                        outputs(%input_cmx : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>

    %input_sm_cmx = memref.alloc() : memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>
    %input_sm = VPUIP.Copy inputs(%input_sm_ddr : memref<1x4x224x224xi1, #NHWC>)
                           outputs(%input_sm_cmx : memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>

    %output = memref.alloc() : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>
    %output_sm = memref.alloc() : memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>

    %conv:2 = VPUIP.NCEClusterTask {
          kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
          kernel_size = [3, 3],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = "CONV"
      } input(%input : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
        input_sparsity_map(%input_sm : memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>)
        weights(%weights : memref<64x16x7x7x!qElemType1, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%input : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
        parent_input_sparsity_map(%input_sm : memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>)
        parent_output(%output : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>)
        parent_output_sparsity_map(%output_sm : memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>)
        outputs(%output : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>)
        output_sparsity_map(%output_sm : memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>)
      -> memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>, memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>
      variants : { DPUTask { mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0], pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64} } }
      PPE : {}

    return %conv#0, %conv#1 : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>, memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>

    // CHECK:        [[INPUT_CMX:%.*]] = memref.alloc() : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
    // CHECK:        [[INPUT:%.*]] = VPUIP.Copy inputs([[INPUT_DDR]] : memref<1x4x224x224x!qElemType0, #NHWC>)
    // CHECK-SAME:                              outputs([[INPUT_CMX]] : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           -> memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>

    // CHECK:        [[INPUT_SM_CMX:%.*]] = memref.alloc() : memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>
    // CHECK:        [[INPUT_SM:%.*]] = VPUIP.Copy inputs([[INPUT_SM_DDR]] : memref<1x4x224x224xi1, #NHWC>)
    // CHECK-SAME:                                 outputs([[INPUT_SM_CMX]] : memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           -> memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>

    // CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>
    // CHECK:        [[OUTPUT_SM:%.*]] = memref.alloc() : memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>

    // CHECK:        [[INPUT_SHAPE_CAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 224, 224]}
    // CHECK-SAME:       inputs([[INPUT]] : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:                       -> memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
    // CHECK:        [[INPUT_SM_SHAPE_CAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 224, 224]}
    // CHECK-SAME:       inputs([[INPUT_SM]] : memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:                          -> memref<1x16x224x224xi1, #NHWC, [@CMX_NN, 0]>

    // CHECK:        [[CONV:%.*]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:               input_channels_compression
    // CHECK-SAME:           input([[INPUT_SHAPE_CAST]] : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           input_sparsity_map([[INPUT_SM_SHAPE_CAST]] : memref<1x16x224x224xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           weights([[WEIGHTS]] : memref<64x16x7x7x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           weight_table([[WEIGHTS_TABLE]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:           parent_input([[INPUT_SHAPE_CAST]] : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           parent_input_sparsity_map([[INPUT_SM_SHAPE_CAST]] : memref<1x16x224x224xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           parent_output([[OUTPUT]] : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           parent_output_sparsity_map([[OUTPUT_SM]] : memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           outputs([[OUTPUT]] : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           output_sparsity_map([[OUTPUT_SM]] : memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:           -> memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>, memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>

    // CHECK:       return [[CONV]]#0, [[CONV]]#1 : memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>, memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType1 = type !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.012699142156862745>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0], {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!InputSMDistributed = type !VPUIP.DistributedBuffer<
    1x4x224x224xi1, #NHWC, [@CMX_NN, 0], {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0], {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputSMDistributed = type !VPUIP.DistributedBuffer<
    1x64x112x112xi1, #NHWC, [@CMX_NN, 0], {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    32x16x3x3x!qElemType1, #NHWC, [@CMX_NN, 0], {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!InputBuffer_DDR = type memref<1x4x224x224x!qElemType0, #NHWC>
!InputSMBuffer_DDR = type memref<1x4x224x224xi1, #NHWC>
!InputBuffer = type memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
!InputSMBuffer = type memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>
!WeightsBuffer = type memref<64x16x2x2x!qElemType1, #NHWC, [@CMX_NN, 0]>
!WeightsTableBuffer = type memref<64x1x1x4xsi32, [@CMX_NN, 0]>
!OutputBuffer = type memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>
!OutputSMBuffer = type memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>

// CHECK:       func @CompressTiledConvSparseActivations(
// CHECK-SAME:      [[INPUT_DDR:%.+]]: memref<1x4x224x224x!qElemType0, #NHWC>,
// CHECK-SAME:      [[INPUT_SM_DDR:%.+]]: memref<1x4x224x224xi1, #NHWC>,
// CHECK-SAME:      [[WEIGHTS:%.+]]: memref<64x16x2x2x!qElemType1, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      [[WEIGHTS_TABLE:%.+]]: memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
// CHECK-SAME:      -> (!VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
// CHECK-SAME:          !VPUIP.DistributedBuffer<1x64x112x112xi1, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
func @CompressTiledConvSparseActivations(%input_ddr: !InputBuffer_DDR,
                                         %input_sm_ddr: !InputSMBuffer_DDR,
                                         %weights: !WeightsBuffer,
                                         %weights_table: !WeightsTableBuffer)
        -> (!OutputDistributed, !OutputSMDistributed) {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %input = VPUIP.NCEClusterTiling inputs(%input_ddr as %arg4: !InputBuffer_DDR)
                                    outputs(%input_cmx as %arg5: !InputBuffer) -> !InputDistributed {
        %0 = VPUIP.Copy inputs(%arg4 : !InputBuffer_DDR) outputs(%arg5 : !InputBuffer) -> !InputBuffer
    }

    %input_sm_cmx = VPURT.AllocDistributed -> !InputSMDistributed
    %input_sm = VPUIP.NCEClusterTiling inputs(%input_sm_ddr as %arg4: !InputSMBuffer_DDR)
                                       outputs(%input_sm_cmx as %arg5: !InputSMBuffer) -> !InputSMDistributed {
        %0 = VPUIP.Copy inputs(%arg4 : !InputSMBuffer_DDR) outputs(%arg5 : !InputSMBuffer) -> !InputSMBuffer
    }

    %output = VPURT.AllocDistributed -> !OutputDistributed
    %output_sm = VPURT.AllocDistributed -> !OutputSMDistributed

    %conv:2 = VPUIP.NCEClusterTiling
            inputs(%input as %arg4: !InputBuffer,
                   %input_sm as %arg5: !InputSMBuffer,
                   %weights as %arg6: !WeightsBuffer,
                   %weights_table as %arg7: !WeightsTableBuffer)
            outputs(%output as %arg8: !OutputBuffer,
                    %output_sm as %arg9: !OutputSMBuffer)
            -> (!OutputDistributed, !OutputSMDistributed) {
        %0:2 = VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                kernel_size = [3, 3],
                kernel_strides = [2, 2],
                minimumHardwareExecutionCost = 375613 : i64,
                task_type = "CONV"
            }
            input(%arg4 : !InputBuffer)
            input_sparsity_map(%arg5 : !InputSMBuffer)
            weights(%arg6 : !WeightsBuffer)
            weight_table(%arg7 : !WeightsTableBuffer)
            parent_input(%arg4 : !InputBuffer)
            parent_input_sparsity_map(%arg5 : !InputSMBuffer)
            parent_output(%arg8 : !OutputBuffer)
            parent_output_sparsity_map(%arg9 : !OutputSMBuffer)
            outputs(%arg8 : !OutputBuffer)
            output_sparsity_map(%arg9 : !OutputSMBuffer)
            -> !OutputBuffer, !OutputSMBuffer
            variants : {
                DPUTask { mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0], pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}
            }
            PPE :  {
            }
    }
    return %conv#0, %conv#1 : !OutputDistributed, !OutputSMDistributed

    // CHECK:       [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_DDR]] as [[INNER_ARG0:[^:]+]]: memref<1x4x224x224x!qElemType0, #NHWC>)
    // CHECK-SAME:                                         outputs([[INPUT_CMX]] as [[INNER_ARG1:[^:]+]]: memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    // CHECK:           [[VAR0:%.+]] = VPUIP.Copy inputs([[INNER_ARG0]] : memref<1x4x224x224x!qElemType0, #NHWC>)
    // CHECK-SAME:                                outputs([[INNER_ARG1]] : memref<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       }

    // CHECK:       [[INPUT_SM_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x224x224xi1, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_SM:%.+]] = VPUIP.NCEClusterTiling inputs([[INPUT_SM_DDR]] as [[INNER_ARG0:[^:]+]]: memref<1x4x224x224xi1, #NHWC>)
    // CHECK-SAME:                                         outputs([[INPUT_SM_CMX]] as [[INNER_ARG1:[^:]+]]: memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK:           [[VAR1:%.+]] = VPUIP.Copy inputs([[INNER_ARG0]] : memref<1x4x224x224xi1, #NHWC>)
    // CHECK-SAME:                                outputs([[INNER_ARG1]] : memref<1x4x224x224xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK:       }

    // CHECK:       [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_SM_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x112x112xi1, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[INPUT_SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 224, 224]}
    // CHECK-SAME:      inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x4x224x224x!qElemType0, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                      -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_SM_SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 224, 224]}
    // CHECK-SAME:      inputs([[INPUT_SM]] : !VPUIP.DistributedBuffer<1x4x224x224xi1, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                         -> !VPUIP.DistributedBuffer<1x16x224x224xi1, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CONV:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_SHAPE_CAST]] as [[INNER_IN:%[^:]+]]: memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:             [[INPUT_SM_SHAPE_CAST]] as [[INNER_IN_SM:%[^:]+]]: memref<1x16x224x224xi1, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:             [[WEIGHTS]] as [[INNER_W:%[^:]+]]: memref<64x16x2x2x!qElemType1, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:             [[WEIGHTS_TABLE]] as [[INNER_WT:%[^:]+]]: memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUTPUT_CMX]] as [[INNER_OUT:%[^:]+]]: memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:              [[OUTPUT_SM_CMX]] as [[INNER_OUT_SM:%[^:]+]]: memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      -> (!VPUIP.DistributedBuffer<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x64x112x112xi1, #NHWC, [@CMX_NN, 0], {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {

    // CHECK:           [[VAR2:%.+]] = VPUIP.NCEClusterTask {
    // CHECK-SAME:              input_channels_compression
    // CHECK-SAME:          }
    // CHECK-SAME:          input([[INNER_IN]]
    // CHECK-SAME:          input_sparsity_map([[INNER_IN_SM]]
    // CHECK-SAME:          weights([[INNER_W]]
    // CHECK-SAME:          weight_table([[INNER_WT]]
    // CHECK-SAME:          parent_input([[INNER_IN]]
    // CHECK-SAME:          parent_input_sparsity_map([[INNER_IN_SM]]
    // CHECK-SAME:          parent_output([[INNER_OUT]]
    // CHECK-SAME:          parent_output_sparsity_map([[INNER_OUT_SM]]
    // CHECK-SAME:          outputs([[INNER_OUT]]
    // CHECK-SAME:          output_sparsity_map([[INNER_OUT_SM]]
    // CHECK-SAME:          -> memref<1x64x112x112x!qElemType2, #NHWC, [@CMX_NN, 0]>, memref<1x64x112x112xi1, #NHWC, [@CMX_NN, 0]>
    // CHECK-SAME:           variants :  {
    // CHECK:       DPUTask {mpe_mode = "CUBOID_16x16", outEnd = [111, 111, 63], outStart = [0, 0, 0],
    // CHECK-SAME:           pad = {bottom = 2 : i64, left = 3 : i64, right = 2 : i64, top = 3 : i64}}

    // CHECK:       return [[CONV]]#0, [[CONV]]#1
}
