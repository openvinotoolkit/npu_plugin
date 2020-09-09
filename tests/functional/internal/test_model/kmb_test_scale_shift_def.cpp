//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
