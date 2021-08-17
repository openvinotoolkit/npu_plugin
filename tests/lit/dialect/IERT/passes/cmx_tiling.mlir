// RUN: vpux-opt --split-input-file --cmx-tiling %s | FileCheck %s

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 1920000 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

func @SplitOverOC(
        %input: memref<1x32x100x100xf16>,
        %filter: memref<64x32x3x3xf16>,
        %bias: memref<1x64x1x1xf16>,
        %output_buff: memref<1x64x100x100xf16>)
            -> memref<1x64x100x100xf16> {
    %1 = IERT.Convolution {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        }
        inputs(%input: memref<1x32x100x100xf16>, %filter: memref<64x32x3x3xf16>, %bias : memref<1x64x1x1xf16>)
        outputs(%output_buff : memref<1x64x100x100xf16>)
        -> memref<1x64x100x100xf16>
    return %1 : memref<1x64x100x100xf16>
}

// CHECK-LABEL: func @SplitOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: memref<1x32x100x100xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: memref<64x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: memref<1x64x1x1xf16>,
// CHECK-SAME:        [[OUTPUT_BUFF:%arg[0-9]]]: memref<1x64x100x100xf16>

// Tile 0

// CHECK:       [[FILTER_TILE0_VIEW:%.+]] = IERT.SubView [[FILTER]] [0, 0, 0, 0] [32, 32, 3, 3] : memref<64x32x3x3xf16>
// CHECK-SAME:      to memref<32x32x3x3xf16, {{#map[0-9]}}>
// CHECK:       [[FILTER_TILE0_BUFF:%.+]] = memref.alloc() : memref<32x32x3x3xf16>
// CHECK:       [[FILTER_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_TILE0_VIEW]] : memref<32x32x3x3xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[FILTER_TILE0_BUFF]] : memref<32x32x3x3xf16>)

// CHECK:       [[BIAS_TILE0_VIEW:%.+]] = IERT.SubView [[BIAS]] [0, 0, 0, 0] [1, 32, 1, 1] : memref<1x64x1x1xf16>
// CHECK-SAME:      to memref<1x32x1x1xf16, {{#map[0-9]}}>
// CHECK:       [[BIAS_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x32x1x1xf16>
// CHECK:       [[BIAS_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[BIAS_TILE0_VIEW]] : memref<1x32x1x1xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[BIAS_TILE0_BUFF]] : memref<1x32x1x1xf16>)

// CHECK:       [[OUTPUT_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x32x100x100xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = IERT.Convolution
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]}
// CHECK-SAME:      inputs([[INPUT]] : memref<1x32x100x100xf16>,
// CHECK-SAME:             [[FILTER_TILE0]] : memref<32x32x3x3xf16>,
// CHECK-SAME:             [[BIAS_TILE0]] : memref<1x32x1x1xf16>)
// CHECK-SAME:      outputs([[OUTPUT_TILE0_BUFF]] : memref<1x32x100x100xf16>

// CHECK:       [[OUTPUT_BUFF_TILE0_VIEW:%.+]] = IERT.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 32, 100, 100] : memref<1x64x100x100xf16>
// CHECK-SAME:      to memref<1x32x100x100xf16, {{#map[0-9]}}>
// CHECK:       [[OUTPUT_BUFF_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE0]] : memref<1x32x100x100xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE0_VIEW]] : memref<1x32x100x100xf16, {{#map[0-9]}}>)

// Tile 1

// CHECK:       [[FILTER_TILE1_VIEW:%.+]] = IERT.SubView [[FILTER]] [32, 0, 0, 0] [32, 32, 3, 3] : memref<64x32x3x3xf16>
// CHECK-SAME:      to memref<32x32x3x3xf16, {{#map[0-9]}}>
// CHECK:       [[FILTER_TILE1_BUFF:%.+]] = memref.alloc() : memref<32x32x3x3xf16>
// CHECK:       [[FILTER_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_TILE1_VIEW]] : memref<32x32x3x3xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[FILTER_TILE1_BUFF]] : memref<32x32x3x3xf16>)

// CHECK:       [[BIAS_TILE1_VIEW:%.+]] = IERT.SubView [[BIAS]] [0, 32, 0, 0] [1, 32, 1, 1] : memref<1x64x1x1xf16>
// CHECK-SAME:      to memref<1x32x1x1xf16, {{#map[0-9]}}>
// CHECK:       [[BIAS_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x32x1x1xf16>
// CHECK:       [[BIAS_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[BIAS_TILE1_VIEW]] : memref<1x32x1x1xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[BIAS_TILE1_BUFF]] : memref<1x32x1x1xf16>)

// CHECK:       [[OUTPUT_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x32x100x100xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = IERT.Convolution
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]}
// CHECK-SAME:      inputs([[INPUT]] : memref<1x32x100x100xf16>,
// CHECK-SAME:             [[FILTER_TILE1]] : memref<32x32x3x3xf16>,
// CHECK-SAME:             [[BIAS_TILE1]] : memref<1x32x1x1xf16>)
// CHECK-SAME:      outputs([[OUTPUT_TILE1_BUFF]] : memref<1x32x100x100xf16>

// CHECK:       [[OUTPUT_BUFF_TILE1_VIEW:%.+]] = IERT.SubView [[OUTPUT_BUFF]] [0, 32, 0, 0] [1, 32, 100, 100] : memref<1x64x100x100xf16>
// CHECK-SAME:      to memref<1x32x100x100xf16, {{#map[0-9]}}>
// CHECK:       [[OUTPUT_BUFF_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE1]] : memref<1x32x100x100xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE1_VIEW]] : memref<1x32x100x100xf16, {{#map[0-9]}}>)

// Concat

// CHECK:       [[OUTPUT:%.+]] = IERT.ConcatView
// CHECK-SAME:      inputs([[OUTPUT_BUFF_TILE0]], [[OUTPUT_BUFF_TILE1]]
// CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x64x100x100xf16>)

// CHECK:       return [[OUTPUT]] : memref<1x64x100x100xf16>

// -----

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 400000 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

func @SplitOverH(
        %input: memref<1x16x100x100xf16>,
        %output_buff: memref<1x16x100x100xf16>)
            -> memref<1x16x100x100xf16> {
    %1 = IERT.MaxPool {
            kernel_size = [3, 3],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        }
        inputs(%input: memref<1x16x100x100xf16>)
        outputs(%output_buff : memref<1x16x100x100xf16>)
        -> memref<1x16x100x100xf16>
    return %1 : memref<1x16x100x100xf16>
}

// CHECK-LABEL: func @SplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: memref<1x16x100x100xf16>,
// CHECK-SAME:        [[OUTPUT_BUFF:%arg[0-9]]]: memref<1x16x100x100xf16>

// Tile 0

// CHECK:       [[INPUT_TILE0_VIEW:%.+]] = IERT.SubView [[INPUT]] [0, 0, 0, 0] [1, 16, 51, 100] : memref<1x16x100x100xf16>
// CHECK-SAME:      to memref<1x16x51x100xf16, {{#map[0-9]}}>
// CHECK:       [[INPUT_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x16x51x100xf16>
// CHECK:       [[INPUT_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT_TILE0_VIEW]] : memref<1x16x51x100xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT_TILE0_BUFF]] : memref<1x16x51x100xf16>)

// CHECK:       [[OUTPUT_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x16x50x100xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = IERT.MaxPool
// CHECK-SAME:          kernel_size = [3, 3]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [0, 1]
// CHECK-SAME:          strides = [1, 1]}
// CHECK-SAME:      inputs([[INPUT_TILE0]] : memref<1x16x51x100xf16>)
// CHECK-SAME:      outputs([[OUTPUT_TILE0_BUFF]] : memref<1x16x50x100xf16>

// CHECK:       [[OUTPUT_BUFF_TILE0_VIEW:%.+]] = IERT.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 16, 50, 100] : memref<1x16x100x100xf16>
// CHECK-SAME:      to memref<1x16x50x100xf16, {{#map[0-9]}}>
// CHECK:       [[OUTPUT_BUFF_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE0]] : memref<1x16x50x100xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE0_VIEW]] : memref<1x16x50x100xf16, {{#map[0-9]}}>)

// Tile 1

// CHECK:       [[INPUT_TILE1_VIEW:%.+]] = IERT.SubView [[INPUT]] [0, 0, 49, 0] [1, 16, 51, 100] : memref<1x16x100x100xf16>
// CHECK-SAME:      to memref<1x16x51x100xf16, {{#map[0-9]}}>
// CHECK:       [[INPUT_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x16x51x100xf16>
// CHECK:       [[INPUT_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT_TILE1_VIEW]] : memref<1x16x51x100xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT_TILE1_BUFF]] : memref<1x16x51x100xf16>)

// CHECK:       [[OUTPUT_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x16x50x100xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = IERT.MaxPool
// CHECK-SAME:          kernel_size = [3, 3]
// CHECK-SAME:          pads_begin = [0, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]}
// CHECK-SAME:      inputs([[INPUT_TILE1]] : memref<1x16x51x100xf16>)
// CHECK-SAME:      outputs([[OUTPUT_TILE1_BUFF]] : memref<1x16x50x100xf16>

// CHECK:       [[OUTPUT_BUFF_TILE1_VIEW:%.+]] = IERT.SubView [[OUTPUT_BUFF]] [0, 0, 50, 0] [1, 16, 50, 100] : memref<1x16x100x100xf16>
// CHECK-SAME:      to memref<1x16x50x100xf16, {{#map[0-9]}}>
// CHECK:       [[OUTPUT_BUFF_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE1]] : memref<1x16x50x100xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE1_VIEW]] : memref<1x16x50x100xf16, {{#map[0-9]}}>)

// Concat

// CHECK:       [[OUTPUT:%.+]] = IERT.ConcatView
// CHECK-SAME:      inputs([[OUTPUT_BUFF_TILE0]], [[OUTPUT_BUFF_TILE1]]
// CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x16x100x100xf16>)

// CHECK:       return [[OUTPUT]] : memref<1x16x100x100xf16>

// -----

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 1000000 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

func @SplitOverC(
        %input1: memref<1x1024x14x14xf16>,
        %input2: memref<1x1024x14x14xf16>,
        %output_buff: memref<1x1024x14x14xf16>)
            -> memref<1x1024x14x14xf16> {
    %1 = IERT.Add { }
        inputs(%input1: memref<1x1024x14x14xf16>,
               %input2: memref<1x1024x14x14xf16>)
        outputs(%output_buff : memref<1x1024x14x14xf16>)
        -> memref<1x1024x14x14xf16>
    return %1 : memref<1x1024x14x14xf16>
}

// CHECK-LABEL: func @SplitOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: memref<1x1024x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: memref<1x1024x14x14xf16>,
// CHECK-SAME:        [[OUTPUT_BUFF:%arg[0-9]]]: memref<1x1024x14x14xf16>

// Tile 0

// CHECK:       [[INPUT0_TILE0_VIEW:%.+]] = IERT.SubView [[INPUT1]] [0, 0, 0, 0] [1, 512, 14, 14] : memref<1x1024x14x14xf16>
// CHECK-SAME:      to memref<1x512x14x14xf16, {{#map[0-9]}}>
// CHECK:       [[INPUT0_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[INPUT0_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT0_TILE0_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT0_TILE0_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[INPUT1_TILE0_VIEW:%.+]] = IERT.SubView [[INPUT2]] [0, 0, 0, 0] [1, 512, 14, 14] : memref<1x1024x14x14xf16>
// CHECK-SAME:      to memref<1x512x14x14xf16, {{#map[0-9]}}>
// CHECK:       [[INPUT1_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[INPUT1_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT1_TILE0_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT1_TILE0_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[OUTPUT_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = IERT.Add
// CHECK-SAME:      inputs([[INPUT0_TILE0]] : memref<1x512x14x14xf16>, [[INPUT1_TILE0]] : memref<1x512x14x14xf16>)
// CHECK-SAME:      outputs([[OUTPUT_TILE0_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[OUTPUT_BUFF_TILE0_VIEW:%.+]] = IERT.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 512, 14, 14] : memref<1x1024x14x14xf16>
// CHECK-SAME:      to memref<1x512x14x14xf16, {{#map[0-9]}}>
// CHECK:       [[OUTPUT_BUFF_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE0]] : memref<1x512x14x14xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE0_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)

// Tile 1

// CHECK:       [[INPUT0_TILE1_VIEW:%.+]] = IERT.SubView [[INPUT1]] [0, 512, 0, 0] [1, 512, 14, 14] : memref<1x1024x14x14xf16>
// CHECK-SAME:      to memref<1x512x14x14xf16, {{#map[0-9]}}>
// CHECK:       [[INPUT0_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[INPUT0_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT0_TILE1_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT0_TILE1_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[INPUT1_TILE1_VIEW:%.+]] = IERT.SubView [[INPUT2]] [0, 512, 0, 0] [1, 512, 14, 14] : memref<1x1024x14x14xf16>
// CHECK-SAME:      to memref<1x512x14x14xf16, {{#map[0-9]}}>
// CHECK:       [[INPUT1_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[INPUT1_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT1_TILE1_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT1_TILE1_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[OUTPUT_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = IERT.Add
// CHECK-SAME:      inputs([[INPUT0_TILE1]] : memref<1x512x14x14xf16>, [[INPUT1_TILE1]] : memref<1x512x14x14xf16>)
// CHECK-SAME:      outputs([[OUTPUT_TILE1_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[OUTPUT_BUFF_TILE1_VIEW:%.+]] = IERT.SubView [[OUTPUT_BUFF]] [0, 512, 0, 0] [1, 512, 14, 14] : memref<1x1024x14x14xf16>
// CHECK-SAME:      to memref<1x512x14x14xf16, {{#map[0-9]}}>
// CHECK:       [[OUTPUT_BUFF_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE1]] : memref<1x512x14x14xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE1_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)

// Concat

// CHECK:       [[OUTPUT:%.+]] = IERT.ConcatView
// CHECK-SAME:      inputs([[OUTPUT_BUFF_TILE0]], [[OUTPUT_BUFF_TILE1]]
// CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x1024x14x14xf16>)

// CHECK:       return [[OUTPUT]] : memref<1x1024x14x14xf16>

// -----

!qElemType0 = type !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = type !quant.uniform<u8<0:254>:f32:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,8.5112804502952756E-4:127,6.7859559547244093E-4:127,6.4687653789370083E-4:127,7.1992648868110238E-4:127,9.9770853838582673E-4:127,8.1075833538385824E-4:127,6.2476931594488186E-4:127,6.8580447219488186E-4:127,6.7042553518700788E-4:127,0.001030388779527559:127,9.2225562869094487E-4:127,8.5881751353346455E-4:127,2.8330885519192912E-4:127,5.1423320620078736E-4:127,4.921259842519685E-4:127,7.3722779281496062E-4:127,7.8192282849409451E-4:127,6.9253275713582673E-4:127,9.9482498769685036E-4:127,7.9922413262795275E-4:127,6.7234790231299212E-4:127,5.6133120078740155E-4:127,8.1460306963582673E-4:127,6.1707984744094487E-4:127,8.4199680118110238E-4:127,3.7870632381889762E-4:127,3.5467673474409451E-4:127,3.1262495386318896E-4:127,3.020519346702756E-4:127,7.1944589689960632E-4:127,4.5992633489173228E-4:127,3.7293922244094487E-4:127,6.7619263656496062E-4:127,7.5452909694881886E-4:127,3.1166377030019684E-4:127,8.4103561761811026E-4:127,4.6521284448818896E-4:127,7.7086921751968508E-4:127,0.0014177457554133859:127,6.065068282480315E-4:127,6.5648837352362207E-4:127,5.0317959522637793E-4:127,6.382258858267716E-4:127,6.5889133243110238E-4:127,5.4066575418307088E-4:127,7.5116495447834642E-4:127,5.5796705831692912E-4:127,6.6946435162401575E-4:127,5.4835522268700788E-4:127,5.8055487204724415E-4:127,7.8048105314960632E-4:127,6.7042553518700788E-4:127,6.7138671875E-4:127,6.3582292691929129E-4:127}>
!qElemType2 = type !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType3 = type !quant.uniform<u8<0:254>:f32:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,8.5112804502952756E-4:127,6.7859559547244093E-4:127,6.4687653789370083E-4:127,7.1992648868110238E-4:127,9.9770853838582673E-4:127,8.1075833538385824E-4:127,6.2476931594488186E-4:127,6.8580447219488186E-4:127,6.7042553518700788E-4:127,0.001030388779527559:127,9.2225562869094487E-4:127,8.5881751353346455E-4:127,2.8330885519192912E-4:127,5.1423320620078736E-4:127,4.921259842519685E-4:127,7.3722779281496062E-4:127,7.8192282849409451E-4:127,6.9253275713582673E-4:127,9.9482498769685036E-4:127,7.9922413262795275E-4:127,6.7234790231299212E-4:127,5.6133120078740155E-4:127}>
!qElemType4 = type !quant.uniform<u8<0:254>:f32:0, {8.1460306963582673E-4:127,6.1707984744094487E-4:127,8.4199680118110238E-4:127,3.7870632381889762E-4:127,3.5467673474409451E-4:127,3.1262495386318896E-4:127,3.020519346702756E-4:127,7.1944589689960632E-4:127,4.5992633489173228E-4:127,3.7293922244094487E-4:127,6.7619263656496062E-4:127,7.5452909694881886E-4:127,3.1166377030019684E-4:127,8.4103561761811026E-4:127,4.6521284448818896E-4:127,7.7086921751968508E-4:127,0.0014177457554133859:127,6.065068282480315E-4:127,6.5648837352362207E-4:127,5.0317959522637793E-4:127,6.382258858267716E-4:127,6.5889133243110238E-4:127,5.4066575418307088E-4:127,7.5116495447834642E-4:127,5.5796705831692912E-4:127,6.6946435162401575E-4:127,5.4835522268700788E-4:127,5.8055487204724415E-4:127,7.8048105314960632E-4:127,6.7042553518700788E-4:127,6.7138671875E-4:127,6.3582292691929129E-4:127}>

IERT.RunTimeResources
    availableMemory : {
        IERT.MemoryResource 900000 bytes of "CMX_NN"
    }
    usedMemory : {
    }
    executors : {
    }

func @SplitQuantOverOC(
        %input: memref<1x32x100x100x!qElemType0>,
        %filter: memref<64x32x3x3x!qElemType1>,
        %output_buff: memref<1x64x100x100x!qElemType2>)
            -> memref<1x64x100x100x!qElemType2> {
    %1 = IERT.Convolution {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        }
        inputs(%input: memref<1x32x100x100x!qElemType0>, %filter: memref<64x32x3x3x!qElemType1>)
        outputs(%output_buff : memref<1x64x100x100x!qElemType2>)
        -> memref<1x64x100x100x!qElemType2>
    return %1 : memref<1x64x100x100x!qElemType2>
}

// CHECK-LABEL: func @SplitQuantOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: memref<1x32x100x100x!qElemType0>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: memref<64x32x3x3x!qElemType1>,
// CHECK-SAME:        [[OUTPUT_BUFF:%arg[0-9]]]: memref<1x64x100x100x!qElemType2>

// Tile 0

// CHECK:       [[FILTER_TILE0_VIEW:%.+]] = IERT.SubView [[FILTER]] [0, 0, 0, 0] [32, 32, 3, 3] : memref<64x32x3x3x!qElemType1>
// CHECK-SAME:      to memref<32x32x3x3x!qElemType1, {{#map[0-9]}}>
// CHECK:       [[FILTER_TILE0_BUFF:%.+]] = memref.alloc() : memref<32x32x3x3x!qElemType3>
// CHECK:       [[FILTER_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_TILE0_VIEW]] : memref<32x32x3x3x!qElemType1, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[FILTER_TILE0_BUFF]] : memref<32x32x3x3x!qElemType3>)

// CHECK:       [[OUTPUT_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x32x100x100x!qElemType2>
// CHECK:       [[OUTPUT_TILE0:%.+]] = IERT.Convolution
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]}
// CHECK-SAME:      inputs([[INPUT]] : memref<1x32x100x100x!qElemType0>,
// CHECK-SAME:             [[FILTER_TILE0]] : memref<32x32x3x3x!qElemType3>)
// CHECK-SAME:      outputs([[OUTPUT_TILE0_BUFF]] : memref<1x32x100x100x!qElemType2>

// CHECK:       [[OUTPUT_BUFF_TILE0_VIEW:%.+]] = IERT.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 32, 100, 100] : memref<1x64x100x100x!qElemType2>
// CHECK-SAME:      to memref<1x32x100x100x!qElemType2, {{#map[0-9]}}>
// CHECK:       [[OUTPUT_BUFF_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE0]] : memref<1x32x100x100x!qElemType2>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE0_VIEW]] : memref<1x32x100x100x!qElemType2, {{#map[0-9]}}>)

// Tile 1

// CHECK:       [[FILTER_TILE1_VIEW:%.+]] = IERT.SubView [[FILTER]] [32, 0, 0, 0] [32, 32, 3, 3] : memref<64x32x3x3x!qElemType1>
// CHECK-SAME:      to memref<32x32x3x3x!qElemType1, {{#map[0-9]}}>
// CHECK:       [[FILTER_TILE1_BUFF:%.+]] = memref.alloc() : memref<32x32x3x3x!qElemType4>
// CHECK:       [[FILTER_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_TILE1_VIEW]] : memref<32x32x3x3x!qElemType1, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[FILTER_TILE1_BUFF]] : memref<32x32x3x3x!qElemType4>)

// CHECK:       [[OUTPUT_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x32x100x100x!qElemType2>
// CHECK:       [[OUTPUT_TILE1:%.+]] = IERT.Convolution
// CHECK-SAME:          dilations = [1, 1]
// CHECK-SAME:          pads_begin = [1, 1]
// CHECK-SAME:          pads_end = [1, 1]
// CHECK-SAME:          strides = [1, 1]}
// CHECK-SAME:      inputs([[INPUT]] : memref<1x32x100x100x!qElemType0>,
// CHECK-SAME:             [[FILTER_TILE1]] : memref<32x32x3x3x!qElemType4>)
// CHECK-SAME:      outputs([[OUTPUT_TILE1_BUFF]] : memref<1x32x100x100x!qElemType2>

// CHECK:       [[OUTPUT_BUFF_TILE1_VIEW:%.+]] = IERT.SubView [[OUTPUT_BUFF]] [0, 32, 0, 0] [1, 32, 100, 100] : memref<1x64x100x100x!qElemType2>
// CHECK-SAME:      to memref<1x32x100x100x!qElemType2, {{#map[0-9]}}>
// CHECK:       [[OUTPUT_BUFF_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE1]] : memref<1x32x100x100x!qElemType2>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE1_VIEW]] : memref<1x32x100x100x!qElemType2, {{#map[0-9]}}>)

// Concat

// CHECK:       [[OUTPUT:%.+]] = IERT.ConcatView
// CHECK-SAME:      inputs([[OUTPUT_BUFF_TILE0]], [[OUTPUT_BUFF_TILE1]]
// CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x64x100x100x!qElemType2>)

// CHECK:       return [[OUTPUT]] : memref<1x64x100x100x!qElemType2>
