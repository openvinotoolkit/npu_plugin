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
/// @file      PlgCropNV12.h
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgCropNV12 Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_CROPNV12_H__
#define __PLG_CROPNV12_H__

#include "Flic.h"
#include "Message.h"

// #############################################################################
// The following section is inculded in a seperate file on the device.
// In the future it may be moved to a seperate common file, but for now it will
// be included here directly.
// #############################################################################
typedef enum CropMode
{
	CropOff,	// in frame converted in full resolution from YUV420 to NV12
	CropManual,	// (x,y) offsets and [width x height] set by application
	CropAuto,	// (x,y) offsets derived from horizontal/vertical alignments set by appl.; [width x height] set by application
	CropModeEND
}CropMode;

typedef uint8_t UInt8;
typedef uint16_t UInt16;

// #############################################################################
// Section end.
// #############################################################################

typedef struct PlgCropNV12Cfg
{
	CropMode	cropMode;
	UInt8		CMXStartSlice;
	UInt8		CMXNoOfSlices;
	UInt16		outWidth; // if cropping active it is the width of the cropped window
	UInt16		outHeight; // if cropping active it is the height of the cropped window
	UInt16		cropHor; // for manual crop, they are the top-left X/Y coordinates of the cropped window
	UInt16		cropVert; // for auto crop, they are the horizontal/vertical alignments of the cropped window as per CropAlignments
} PlgCropNV12Cfg;

/** CropNV12 plugin. */
class PlgCropNV12 : public PluginStub
{
  public:
	SReceiver<ImgFramePtr> in;
	MReceiver<ImgFramePtr> inO;
	MSender  <ImgFramePtr> out;

  public:
    /** Constructor. */
    PlgCropNV12() : PluginStub("PlgCropNV12"){};

    /** Create methods. */
    void Create(void);
    void Create(const PlgCropNV12Cfg* cfg);
};

#endif /* __PLG_CROPNV12_H__ */
