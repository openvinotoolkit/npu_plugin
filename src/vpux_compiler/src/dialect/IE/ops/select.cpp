//
// Copyright 2020 Intel Corporation.
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

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <map>

using namespace vpux;

mlir::LogicalResult vpux::IE::SelectOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SelectOpAdaptor select(operands, attrs);
    if (mlir::failed(select.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = select.input1().getType().cast<mlir::ShapedType>();
    const auto in2Type = select.input2().getType().cast<mlir::ShapedType>();
    const auto in3Type = select.input3().getType().cast<mlir::ShapedType>();

    //const auto outShapeRes =
    //         //IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), select.auto_broadcast().getValue(), loc);
    //        IE::broadcastEltwiseShape({in1Type.getShape(), in2Type.getShape(), in3Type.getShape()}, select.auto_broadcast().getValue(), loc);

    //  if (mlir::succeeded(outShapeRes)) {
    //     inferredReturnShapes.emplace_back(outShapeRes.getValue(), in1Type.getElementType());
    //     inferredReturnShapes.emplace_back(outShapeRes.getValue(), in2Type.getElementType());
    //     //inferredReturnShapes.emplace_back(outShapeRes.getValue(), in3Type.getElementType());
    // }
    
    // return outShapeRes;

    //inferredReturnShapes.emplace_back(outShapeRes.getValue(), in3Type.getElementType());
    
    // std::cout << "This is debug mesg - "<<std::endl;
    // std::cout << in1Type.getShape().size() << std::endl;
    // std::cout << in2Type.getShape().size() << std::endl;
    // std::cout << in3Type.getShape().size() << std::endl;

    int input1_counter=0;
    int input2_counter=0;
    int input3_counter=0;

    int64_t expectedOutputTensorSize;

    int64_t input1_size =in1Type.getShape().size();
    int64_t input2_size =in2Type.getShape().size();
    int64_t input3_size =in3Type.getShape().size();

    //int intexpectedOutputTensorSize = std::max((int)in1Type.getShape().size(), (int)in2Type.getShape().size(), (int)in3Type.getShape().size()) ;

    if( input1_size > input2_size){
        if(input1_size >  input3_size){
            expectedOutputTensorSize =input1_size ;
        }
        else { // 1 equal or smaller then 3 
            expectedOutputTensorSize =input3_size ;
        }
    }
    else if ( input1_size < input2_size ){
        if( input2_size >  input3_size ){
            expectedOutputTensorSize =input2_size ;
        }
        else{
            expectedOutputTensorSize =input3_size ;
        }
    }
    else if (input3_size > input1_size ){
         expectedOutputTensorSize =input3_size ;
    }
    else{ 
       expectedOutputTensorSize = input1_size ;
    }
    std::cout << "Expected size - "<< expectedOutputTensorSize <<std::endl;

    SmallVector<int64_t> outputShape ;
    //---
    for (int i=0; i<expectedOutputTensorSize ; i++ ){
        int input1_dim = 0;
        int input2_dim = 0;
        int input3_dim = 0;

        if((long)(expectedOutputTensorSize-i) <= (long)(in1Type.getShape().size()) ){
            input1_dim = in1Type.getShape()[input1_counter];
            input1_counter+=1;
        }

        if((long)(expectedOutputTensorSize-i) <= (long)in2Type.getShape().size()){
            input2_dim = in2Type.getShape()[input2_counter];
            input2_counter+=1;
        }
        if( (long)(expectedOutputTensorSize-i) <= (long)in3Type.getShape().size()){
            input3_dim = in3Type.getShape()[input3_counter];
            input3_counter+=1;
        }
        int dim = 0 ;
        if (input1_dim > input2_dim ){
            if(input3_dim > input1_dim){
                dim = input3_dim;
            }
            else{
                dim = input1_dim;
            }
        }
        else if (input2_dim > input1_dim){
            if(input3_dim > input2_dim){
                dim = input3_dim;
            }
            else{
                dim = input2_dim;
            }
        }
        else if (input1_dim > input3_dim){
            dim = input1_dim;
        }
        else if (input1_dim < input3_dim){
            dim = input3_dim;
        }
        else{
            dim = input1_dim;
        }

        // = std::max (input1_dim, input2_dim, input3_dim) ;
        outputShape.push_back( dim );
    }

    inferredReturnShapes.emplace_back( outputShape , in3Type.getElementType());
   //std::cout << "debug mesg - "<< in2Type.getElementType()[0]<<std::endl;
    // std::cout << "Final " << sizeConstant << std::endl;
    // //
    // int x = 5 ;
    // int y= 6;
    // SmallVector<int64_t> outputShape{x,y};
    // inferredReturnShapes.emplace_back(  outputShape , in2Type.getElementType());
    //std::cout << "End debug mesg - "<< in1Type.getShape()[0] << " + " << in1Type.getShape()[1 ]<<std::endl;

    //inferredReturnShapes.emplace_back( resultType.getShape(), in2Type.getElementType());

    //inferredReturnShapes.emplace_back( in3Type.getShape(), in2Type.getElementType());

    // if(in2Type.getElementType()>in3Type.getElementType())
    //     inferredReturnShapes.emplace_back(in2Type.getShape(), in2Type.getElementType());
    // else{
    //     inferredReturnShapes.emplace_back(in3Type.getShape(), in3Type.getElementType());
    // }
    //inferredReturnShapes.emplace_back(in1Type.getShape(), in3Type.getElementType());
    return mlir::success();
}
