// RUN: vpux-opt --split-input-file --convert-weights-to-u8 %s | FileCheck %s

!qElemType0 = type !quant.uniform<u8<0:254>:f32:0, {0.010680671751968504:127,0.0081200787401574797:127,0.010596087598425197:127}>
!qElemType1 = type !quant.uniform<i8<-127:127>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>

func @Conv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>> = #const.Content<dense<-1.0> : tensor<3x3x3x3xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!quant.uniform<i8<-127:127>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>>]>
    %1 = "quant.qcast"(%arg0) : (tensor<1x3x16x16xf16>) -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195>>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195>>, tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>> -> tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>
    %3 = "quant.dcast"(%2) : (tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>) -> tensor<1x3x14x14xf16>
    return %3 : tensor<1x3x14x14xf16>

    //CHECK: [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!quant.uniform<u8<0:254>:f16:0, {0.010680671751968504:127,0.0081200787401574797:127,0.010596087598425197:127}>> =
    //CHECK-SAME:                 #const.Content<dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    //CHECK-SAME:                 [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType1>, #const.QuantCast<>,
    //CHECK-SAME:                 #const.ConvertElemType<i32>, #const.Add<1.270000e+02 : f64>,
    //CHECK-SAME:                 #const.ConvertElemType<ui8>, #const.QuantCast<!qElemType0>]>

    //CHECK: [[VAL1:%.*]] = "quant.qcast"(%arg0) : (tensor<1x3x16x16xf16>) -> tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195>>
    //CHECK: [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16x!quant.uniform<u8:f16, 1.1534313725490195>>, tensor<3x3x3x3x!quant.uniform<u8<0:254>:f16:0, {0.010680671751968504:127,0.0081200787401574797:127,0.010596087598425197:127}>> -> tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>
    //CHECK: [[VAL3:%.*]] = "quant.dcast"([[VAL2]]) : (tensor<1x3x14x14x!quant.uniform<u8:f16, 2.4627450980392158>>) -> tensor<1x3x14x14xf16>
    //CHECK: return [[VAL3]]
}
