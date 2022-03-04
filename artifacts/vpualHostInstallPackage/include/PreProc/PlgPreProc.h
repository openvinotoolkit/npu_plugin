#ifndef _PLG_PRE_PROC_H_
#define _PLG_PRE_PROC_H_
//#include <ImgFrame.h>
#include <stdio.h>
#include <vector>
#include "Flic.h"
#include "Message.h"
#include "Pool.h"

#include "PlgPreProcTypes.h"

enum PreProcUserID {
    NN       = 0,
    Display,
    MAX_PP_USER
};

enum IMAGE_FORMAT
{
//----------------------------------------------luma
    FMT_GRAY    = 0,
//----------------------------------------------YUV420
    FMT_YUV420p = 1,    //YUV420 planar
    FMT_NV12    = 2,    //Y is the same as YUV420p while UV interleaved
    FMT_NV21    = 3,    //swap U/V compared to NV12
//----------------------------------------------YUV422
    FMT_YUV422p = 4,    //YUV422 planar
    FMT_YUYV    = 5,    //YUV422(YUYV)
    FMT_UYVY    = 6,    //YUV422(UYVY)
//----------------------------------------------YUV444
    FMT_YUV444p = 7,    //YUV444 planar
    FMT_YUV444i = 8,    //YUV444 interleaved
//----------------------------------------------RGB
    FMT_RGBp    = 9,    //RGB planar
    FMT_RGBi    = 10,   //RGB interleaved
    FMT_BGRp    = 11,   //BGR planar
    FMT_BGRi    = 12,   //BGR interleaved
//----------------------------------------------
    FMT_RGBp_CNN,
    FMT_RGBi_CNN,
    FMT_BGRp_CNN,       //BGR planar for myX NCE
    FMT_BGRi_CNN,       //BGR row major for myX NCE
    FMT_BGRA,
};

typedef struct {
  uint32_t x; //top-left-corner .x
  uint32_t y; //top-left-corner .y
  uint32_t w; //width
  uint32_t h; //height
}Roi;

typedef struct {
  float x;
  float y;
}Point_pp;

typedef struct
{
    Roi inputRoi;
    IMAGE_FORMAT inType;
    Roi outputRoi;
    IMAGE_FORMAT outType;
    int aspect_ratio;
    int align_center;

    int en_perspective;
    Point_pp src_points[5];
    Point_pp dst_points[5];

    int en_keep_input_size;
} PreProcDesc;

class PlgPreProc : public PluginStub{
  public:
    static const unsigned int MAX_INPUTS=4;

    PlgPreProc(uint32_t device_id = 0) : PluginStub("PlgPreProc", device_id), out{device_id} {}

    SReceiver<ImgFramePtr> in[MAX_INPUTS];
    SReceiver<ImgFramePtr> ppResult;
    SReceiver<vpupreproc::PreProcConfigPtr> ppConfig;
    MSender<ImgFramePtr> out;

    void  Create(int32_t i_num, int32_t s_shave, int32_t e_shave, PreProcUserID user);
};
#endif
