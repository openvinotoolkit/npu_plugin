#ifndef __PLG_M2I_TYPES_H__
#define __PLG_M2I_TYPES_H__
#include <iostream>
#include <vector>
#include <swcFrameTypes.h>

#include "Flic.h"
#include "VpuData.h"

#define M2I_MAX_SUPPORTED_INPUT_WIDTH  (1920)
#define M2I_MAX_SUPPORTED_INPUT_HEIGHT (1080)

namespace vpum2i
{

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
     uint16_t   height;    // height in pixels
     uint16_t   width;     // width in pixels
     uint16_t   stride;    // defined as distance in bytes from pix(y,x) to pix(y+1,x)
     uint16_t   bitsPP;    // bits per pixel (for unpacked types set 8 or 16, for NV12 set only luma pixel size)
}FrameSpec;

typedef struct
{
     FrameSpec spec;
     uint64_t p1;  // Addr to first image plane
     uint64_t p2;  // Addr to second image plane (if used)
     uint64_t p3;  // Addr to third image plane  (if used)
     int64_t  ts;  // Timestamp in NS
} ImgFrame; // this matches frameBufferIsp on VPU

// These definitions are used by M2I FLIC proxies for in/out ports
// ##################################################################
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

    uint32_t storeInterleaved;

    uint32_t descPtr;
}M2IObj;
// ##################################################################

typedef struct{
  CvRect   inRoi;
  ImgFrame outFrm;
}M2ITask;

typedef enum{
    SUCCESS,
    INVALID_PARAMS,
    INVALID_ID,
    TOO_MANY,
}M2IStatus;

#ifndef MAX_M2I_TASK_LST_SZ
 #define MAX_M2I_TASK_LST_SZ 32
#endif

class M2IDesc{
  // Input frame
  ImgFrame inFrm;

  // List of tasks
  // The descriptor will be allocated in a vpusmm region
  // that will have its size aligned to 4KB anyway
  M2ITask  taskList[MAX_M2I_TASK_LST_SZ];
  uint32_t nrTasks;

  // Pointer to the underlying VPU region
  VpuData *vpuData;

  // M2I Config
  M2ICfg cfg;

  // PS: This should be removed and replaced by a proper type
  // in the output spec
  bool storeInterleaved;

public:
  M2IDesc():
    nrTasks(0),
    storeInterleaved(false)
    {};

  ~M2IDesc(){};

  M2IStatus SetFrame(const ImgFrame& inFrm)
  {
    // Check fields against limitations
    if(!(inFrm.spec.width <= M2I_MAX_SUPPORTED_INPUT_WIDTH)){
        std::cerr << "ERROR: Maximum supported input width for M2I exceeded: "
                  << inFrm.spec.width << std::endl;
        return INVALID_PARAMS;
    }

    if(!(inFrm.spec.height <= M2I_MAX_SUPPORTED_INPUT_HEIGHT)){
        std::cerr << "ERROR: Maximum supported input height for M2I exceeded: "
                  << inFrm.spec.height << std::endl;
        return INVALID_PARAMS;
    }

    if(!(inFrm.spec.type == NV12)){
        std::cerr << "ERROR: Unsupported input frame type: " << inFrm.spec.type << std::endl;
        return INVALID_PARAMS;
    }

    this->inFrm = inFrm;

    return SUCCESS;
  }

  M2IStatus AddTask(const M2ITask& task)
  {
    // This is a bit confusing, M2I supports only BGR output
    // Currently, there is no available type for BGR
    if(!(task.outFrm.spec.type == RGB888)){
        std::cerr << "ERROR: Unsupported input frame type: " << task.outFrm.spec.type << std::endl;
        return INVALID_PARAMS;
    }
    if(this->nrTasks >= MAX_M2I_TASK_LST_SZ){
        std::cerr << "ERROR: Maximum supported ROIs per frame: " << MAX_M2I_TASK_LST_SZ << std::endl;
        return TOO_MANY;
    }

    this->taskList[nrTasks ++] = task;

    return SUCCESS;
  }

  M2IStatus SetTask(const M2ITask& task , uint32_t index)
  {
    // This is a bit confusing, M2I supports only BGR output
    // Currently, there is no available type for BGR
    if(!(task.outFrm.spec.type == RGB888)){
        std::cerr << "ERROR: Unsupported input frame type: " << task.outFrm.spec.type << std::endl;
        return INVALID_PARAMS;
    }
    if(index >= nrTasks){
        std::cerr << "ERROR: Number of enqueued tasks: " << nrTasks << std::endl;
        return INVALID_ID;
    }

    this->taskList[index] = task;

    return SUCCESS;
  }

  M2IStatus ClearTaskList()
  {
    this->nrTasks = 0;

    return SUCCESS;
  }

  M2IStatus GetTaskById(M2ITask& task, uint32_t id) const
  {
    if(id >= nrTasks){
        std::cerr << "ERROR: Out of bound request for task id: " << id << std::endl;
        return INVALID_ID;
    }
    task = this->taskList[id];

    return SUCCESS;
  }

  int ListSize() const {
    return this->nrTasks;
  }

  M2IStatus SetConfig(const M2ICfg& cfg)
  {
    if(!(cfg.scaleAlgo == BILIN)){
        std::cerr << "ERROR: Unsupported scaling algo: " << cfg.scaleAlgo << std::endl;
        return INVALID_PARAMS;
    }

    this->cfg = cfg;

    return SUCCESS;
  }

  void SetVpuData(VpuData* vpuData)
  {
    this->vpuData = vpuData;
  }

  VpuData* GetVpuData() const
  {
    return vpuData;
  }

  // This should be removed and replaced by a
  // coresponding interleaved output type in the spec
  void SetInterleaved(const bool enabled)
  {
    this->storeInterleaved = enabled;
  }

};

}  // namespace vpum2i

#endif  // __PLG_M2I_TYPES_H__
