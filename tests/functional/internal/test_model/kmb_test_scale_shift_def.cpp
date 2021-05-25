//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "kmb_test_scale_shift_def.hpp"
#include "kmb_test_add_def.hpp"
#include "kmb_test_mul_def.hpp"

TestNetwork& ScaleShiftLayerDef::build() {
    return testNet
        .addLayer<MultiplyLayerDef>(name + "_mul")
            .input1(inputPort.layerName, inputPort.index)
            .input2(scalePort.layerName, scalePort.index)
            .build()
        .addLayer<AddLayerDef>(name)
            .input1(name + "_mul")
            .input2(shiftPort.layerName, shiftPort.index)
            .build();
}
