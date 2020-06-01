///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      sippHwDefs.h
/// @copyright All code copyright Movidius Ltd 2015, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     SIPP engine
///

#ifndef __SIPP_HWDEFS_H__
#define __SIPP_HWDEFS_H__

#include <stdint.h>

//===================================================================
//Dma params

/// @defgroup dma DMA
/// @brief DMA In/Out filter.
/// @ingroup SIPP_Input-Output_Filters
/// @par Output data type(s):\n
///      UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, half, fp16, fp32
/// @par Filter function:\n
///      SIPP_DMA_ID
/// @par Inputs:\n
/// 	- datatypes: UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, half, fp16, fp32; kernels: 1x1
/// @{

/// @brief Parameter structure of the @ref dma filter.
typedef struct
{
    /// @brief dmaMode specific config information
    uint32_t      lineStride; // This is equal to
} DmaParam;
/// @}

//===================================================================
// DMA In/Out filter description for the SIPP Graph Designer Eclipse plugin
// DMA filter has to be described in a special way (not as the rest of the filters) as it has special behaviour
/*{
filter({
 id : "dmaIn",
 name : "DMA In",
 description : "DMA in filter",
 image : "51",

 color : "9090b0",
 shape : "rectangle",
 group : "Input-Output Filters",

 type : "hw",
 symbol :"SIPP_DMA_ID",
 preserve : "",
 dataType : "UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, half, fp16, fp32, float",
 structure :"DmaParam",
 flags : "0x00",

 sourceConstraint : "assert(true,'')",
 targetConstraint : "assert(false,'No input')",
 mandatoryInputs : 0,

 parameters : [ {
  id : "internal0",
  name : "ddrAddr",
  description : "Buffer address to DMA into. If defined as 'auto', the buffer will be automatically generated in DDR.",
  value : "auto",
  type : "u32",
  constraint : "",
 },{
  id : "internal1",
  name : "File",
  description : "File name to DMA from. If specified than the image is automatically loaded into the DMA buffer specified by ddrAddr. Can be also left empty, in this case the application is responsible to fill the DMA buffer.",
  value : "",
  type : "file",
  constraint : "",
 } ]
});

}*/
/*{

filter({
 id : "dmaOut",
 name : "Dma Out",
 description : "DMA out filter",
 image : "52",

 color : "9090b0",
 shape : "rectangle",
 group : "Input-Output Filters",

 type : "hw",
 symbol :"SIPP_DMA_ID",
 preserve : "imgSize, numPlanes, dataType",
 dataType : "UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, half, fp16, fp32, float",
 structure :"DmaParam",
 flags : "0x00",

 sourceConstraint : "assert(false,'No output')",
 targetConstraint : "assert(target.getTargetConnections().size() == 0,'Only 1 input allowed')",

 mandatoryInputs : 1,
 inputs : [ {
       name : "",
       dataTypesOptions : ["UInt8", "UInt16", "UInt32", "UInt64", "Int8", "Int16", "Int32", "half", "fp16", "fp32","float"],
       kernelOptions : [[1,1]]
 } ],

 parameters : [  {
  id : "internal0",
  name : "ddrAddr",
  description : "Buffer address to DMA from. If defined as 'auto', the buffer will be automatically generated in DDR.",
  value : "auto",
  type : "u32",
  constraint : "",
 },{
  id : "internal1",
  name : "File",
  description : "File name to DMA into. If specified than the DMA buffer (ddrAddr) content is automatically saved into the specified file. Can be also left empty.",
  value : "output.raw",
  type : "file",
  constraint : "",
 } ],
});

}*/

#endif // __SIPP_HWDEFS_H__
