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
    PlgCropNV12(uint32_t device_id) : PluginStub("PlgCropNV12", device_id), out{device_id} {};

    /** Create methods. */
    void Create(void);
    void Create(const PlgCropNV12Cfg* cfg);
};

#endif /* __PLG_CROPNV12_H__ */
