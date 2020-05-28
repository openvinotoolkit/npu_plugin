#ifndef __PLG_M2I_TYPES_H__
#define __PLG_M2I_TYPES_H__
#include <vector>
#include "Flic.h"

namespace vpum2i
{

#define NR_TASKS_IN_LIST 100

typedef enum{
    BILIN,
    LANCZOS,
    NEAREST,
    NOSCALE
}ScaleAlgo;

typedef struct{
    ScaleAlgo scaleAlgo;
    float     normFactor[4];
}M2ICfg;

typedef struct{
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
}CvRect;

typedef struct{
     uint16_t   type;      // Values from frameType
     uint16_t   height;    // width in pixels
     uint16_t   width;     // width in pixels
     uint16_t   stride;    // defined as distance in bytes from pix(y,x) to pix(y+1,x)
     uint16_t   bitsPP;    // bits per pixel (for unpacked types set 8 or 16, for NV12 set only luma pixel size)
}frameSpecIsp;

typedef std::vector<CvRect> CvRectV;
typedef std::vector<ImgFramePtr> ImgFramePtrV;

typedef struct
{
    // Input frame
    ImgFramePtr  inFrm;

    // List of ROIs
    CvRectV         inRoi;

    // Output list of ROI buffers
    ImgFramePtrV outFrm;

    // M2I Config
    M2ICfg          cfg;

    uint32_t nrTasks;
}M2IObj;

typedef struct{
	frameSpecIsp fspec_in;
	frameSpecIsp fspec_out;
    uint32_t nrTasks;
    uint32_t use_interleaved;
}InDesc;

}  // namespace vpum2i

#endif  // __PLG_M2I_TYPES_H__
