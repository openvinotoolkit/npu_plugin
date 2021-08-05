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

// CHECK:       [[FILTER_TILE0_VIEW:%.+]] = memref.subview [[FILTER]][0, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
// CHECK:       [[FILTER_TILE0_BUFF:%.+]] = memref.alloc() : memref<32x32x3x3xf16>
// CHECK:       [[FILTER_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_TILE0_VIEW]] : memref<32x32x3x3xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[FILTER_TILE0_BUFF]] : memref<32x32x3x3xf16>)

// CHECK:       [[BIAS_TILE0_VIEW:%.+]] = memref.subview [[BIAS]][0, 0, 0, 0] [1, 32, 1, 1] [1, 1, 1, 1]
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

// CHECK:       [[OUTPUT_BUFF_TILE0_VIEW:%.+]] = memref.subview [[OUTPUT_BUFF]][0, 0, 0, 0] [1, 32, 100, 100] [1, 1, 1, 1]
// CHECK:       [[OUTPUT_BUFF_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE0]] : memref<1x32x100x100xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE0_VIEW]] : memref<1x32x100x100xf16, {{#map[0-9]}}>)

// Tile 1

// CHECK:       [[FILTER_TILE1_VIEW:%.+]] = memref.subview [[FILTER]][32, 0, 0, 0] [32, 32, 3, 3] [1, 1, 1, 1]
// CHECK:       [[FILTER_TILE1_BUFF:%.+]] = memref.alloc() : memref<32x32x3x3xf16>
// CHECK:       [[FILTER_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[FILTER_TILE1_VIEW]] : memref<32x32x3x3xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[FILTER_TILE1_BUFF]] : memref<32x32x3x3xf16>)

// CHECK:       [[BIAS_TILE1_VIEW:%.+]] = memref.subview [[BIAS]][0, 32, 0, 0] [1, 32, 1, 1] [1, 1, 1, 1]
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

// CHECK:       [[OUTPUT_BUFF_TILE1_VIEW:%.+]] = memref.subview [[OUTPUT_BUFF]][0, 32, 0, 0] [1, 32, 100, 100] [1, 1, 1, 1]
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

// CHECK:       [[INPUT_TILE0_VIEW:%.+]] = memref.subview [[INPUT]][0, 0, 0, 0] [1, 16, 51, 100] [1, 1, 1, 1]
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

// CHECK:       [[OUTPUT_BUFF_TILE0_VIEW:%.+]] = memref.subview [[OUTPUT_BUFF]][0, 0, 0, 0] [1, 16, 50, 100] [1, 1, 1, 1]
// CHECK:       [[OUTPUT_BUFF_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE0]] : memref<1x16x50x100xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE0_VIEW]] : memref<1x16x50x100xf16, {{#map[0-9]}}>)

// Tile 1

// CHECK:       [[INPUT_TILE1_VIEW:%.+]] = memref.subview [[INPUT]][0, 0, 49, 0] [1, 16, 51, 100] [1, 1, 1, 1]
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

// CHECK:       [[OUTPUT_BUFF_TILE1_VIEW:%.+]] = memref.subview [[OUTPUT_BUFF]][0, 0, 50, 0] [1, 16, 50, 100] [1, 1, 1, 1]
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

// CHECK:       [[INPUT0_TILE0_VIEW:%.+]] = memref.subview [[INPUT1]][0, 0, 0, 0] [1, 512, 14, 14] [1, 1, 1, 1]
// CHECK:       [[INPUT0_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[INPUT0_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT0_TILE0_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT0_TILE0_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[INPUT1_TILE0_VIEW:%.+]] = memref.subview [[INPUT2]][0, 0, 0, 0] [1, 512, 14, 14] [1, 1, 1, 1]
// CHECK:       [[INPUT1_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[INPUT1_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT1_TILE0_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT1_TILE0_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[OUTPUT_TILE0_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = IERT.Add
// CHECK-SAME:      inputs([[INPUT0_TILE0]] : memref<1x512x14x14xf16>, [[INPUT1_TILE0]] : memref<1x512x14x14xf16>)
// CHECK-SAME:      outputs([[OUTPUT_TILE0_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[OUTPUT_BUFF_TILE0_VIEW:%.+]] = memref.subview [[OUTPUT_BUFF]][0, 0, 0, 0] [1, 512, 14, 14] [1, 1, 1, 1]
// CHECK:       [[OUTPUT_BUFF_TILE0:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE0]] : memref<1x512x14x14xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE0_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)

// Tile 1

// CHECK:       [[INPUT0_TILE1_VIEW:%.+]] = memref.subview [[INPUT1]][0, 512, 0, 0] [1, 512, 14, 14] [1, 1, 1, 1]
// CHECK:       [[INPUT0_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[INPUT0_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT0_TILE1_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT0_TILE1_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[INPUT1_TILE1_VIEW:%.+]] = memref.subview [[INPUT2]][0, 512, 0, 0] [1, 512, 14, 14] [1, 1, 1, 1]
// CHECK:       [[INPUT1_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[INPUT1_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[INPUT1_TILE1_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)
// CHECK-SAME:      outputs([[INPUT1_TILE1_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[OUTPUT_TILE1_BUFF:%.+]] = memref.alloc() : memref<1x512x14x14xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = IERT.Add
// CHECK-SAME:      inputs([[INPUT0_TILE1]] : memref<1x512x14x14xf16>, [[INPUT1_TILE1]] : memref<1x512x14x14xf16>)
// CHECK-SAME:      outputs([[OUTPUT_TILE1_BUFF]] : memref<1x512x14x14xf16>)

// CHECK:       [[OUTPUT_BUFF_TILE1_VIEW:%.+]] = memref.subview [[OUTPUT_BUFF]][0, 512, 0, 0] [1, 512, 14, 14] [1, 1, 1, 1]
// CHECK:       [[OUTPUT_BUFF_TILE1:%.+]] = IERT.Copy
// CHECK-SAME:      inputs([[OUTPUT_TILE1]] : memref<1x512x14x14xf16>)
// CHECK-SAME:      outputs([[OUTPUT_BUFF_TILE1_VIEW]] : memref<1x512x14x14xf16, {{#map[0-9]}}>)

// Concat

// CHECK:       [[OUTPUT:%.+]] = IERT.ConcatView
// CHECK-SAME:      inputs([[OUTPUT_BUFF_TILE0]], [[OUTPUT_BUFF_TILE1]]
// CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x1024x14x14xf16>)

// CHECK:       return [[OUTPUT]] : memref<1x1024x14x14xf16>
