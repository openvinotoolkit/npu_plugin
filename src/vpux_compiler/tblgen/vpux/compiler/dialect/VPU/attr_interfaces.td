//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ATTR_INTERFACES
#define VPUX_COMPILER_DIALECT_VPU_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

//
// SEAttrInterface
//

def VPU_SEAttrInterface : AttrInterface<"SEAttr"> {
    let description = [{
        Interface for Storage Elements attributes.
    }];

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<[{
            Infer the output shape that is obtained after applying the transformation
            represented by the attribute over the input shape.
            
            For example, a nearest-neighbor interpolate attribute with scale 2 for data
            which has a [1, 16, 3, 3] shape would return the [1, 16, 6, 6] shape.

            If tensor associated with the attribute was tiled then tile shape will be returned.
            }],
            "vpux::Shape", "inferOutputShape",
            (ins "vpux::ShapeRef":$inputShape)
        >,
        InterfaceMethod<[{
            Get input shape for a given output shape.
            
            For example, a nearest-neighbor interpolate attribute with scale 2 for data
            which has a [1, 16, 3, 3] shape would return the [1, 16, 6, 6] shape. 
            Then this function will return [1, 16, 3, 3].}],
            "vpux::Shape", "backInferShape",
            (ins "vpux::ShapeRef":$outputShape)
        >,
        InterfaceMethod<[{
            This method provides facility to find coordinates of input pixel/value
            that is used to produce output value with given coordinates.
            }],
            "vpux::Shape", "backInferCoord",
            (ins "vpux::ShapeRef":$tileOffset, "vpux::ShapeRef":$inputShape)
        >,
        InterfaceMethod<[{
            This method extracts input and output tile based on output tile offsets and shape.
            It returns updated attribute since tiling may affect attribute parameters.
            }],
            "vpux::VPU::SEAttr", "extractTile",
            (ins "vpux::ShapeRef":$outputTileOffset, "vpux::ShapeRef":$outputTileShape, 
             "vpux::ShapeRef":$inputShape,
             "vpux::Shape&":$inputTileOffset, "vpux::Shape&":$inputTileShape)
        >,
        InterfaceMethod<[{
            Computes the Storage Element offsets for the given input data after applying
            the transformation represented by the current attribute.

            For example, let's take an example of a nearest-neighbor interpolate attribute
            with scale 2 for data which has a 1x16x3x3xui8 type, compact strides and the
            Storage Element size of 16. The addresses of each Storage Element for the data are:
            0x0   0x10  0x20
            0x30  0x40  0x50
            0x60  0x70  0x80

            The resulting offsets would be:
            0x0   0x0   0x10  0x10  0x20  0x20
            0x0   0x0   0x10  0x10  0x20  0x20
            0x30  0x30  0x40  0x40  0x50  0x50
            0x30  0x30  0x40  0x40  0x50  0x50
            0x60  0x60  0x70  0x70  0x80  0x80
            0x60  0x60  0x70  0x70  0x80  0x80
            }], "std::vector<int32_t>", "computeSEOffsets",
            (ins "vpux::ShapeRef":$dataShape, "vpux::StridesRef":$dataStrides, "vpux::Byte":$elemSize, "int64_t":$seSize)
        >
    ];
}

#endif
