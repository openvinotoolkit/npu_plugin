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
/// @file      PlgMvSipp.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgSipp Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_MV_SIPP_H__
#define __PLG_MV_SIPP_H__

#include <stdint.h>
#include "swcFrameTypes.h"

#include "Flic.h"
#include "Message.h"
#include "xlink.h" // for the async channel
#include <thread>

#include "sipp_messages.h"

#define MAX_IO (4)

#define SIPP_ASYNC_CALLBACK_MESSAGE (1)
#define SIPP_ASYNC_PIPELINE_TERMINATE_MESSAGE (2)
#define SIPP_ASYNC_TERMINATE_ALL (-1)

// Remote stub for Custom SIPP Plugin.
class PlgSipp : public PluginStub
{
  public:
    static const unsigned int MAX_OUTPUTS=MAX_IO;
    static const unsigned int MAX_INPUTS=MAX_IO;
  public:
    SReceiver<ImgFramePtr> in[MAX_INPUTS];
    MSender<ImgFramePtr>   out[MAX_OUTPUTS];
    MReceiver<ImgFramePtr> poolsForOut[MAX_OUTPUTS];

    SReceiver<ImgFramePtr> triggerRec;
    MSender<ImgFramePtr>   trigger;
  public:
    PlgSipp() : PluginStub("PlgSipp"),
                firstShave(0),
                lastShave(0),
                callback(NULL),
                userContext(NULL),
                pipelineID(0)
                  {};

  public:
    void Delete(void) override;

    void Create(void);
    // TODO - review when custom filter is enabled.
    //void AddKernel(void *kernel, uint32_t size);
    void SetInitMask(uint32_t initMask);
    void CreateBasicPipeline();
    void CreatePipeline(uint32_t first_slice, uint32_t last_slice, uint8_t* pmBinImg);
    eSIPP_STATUS AddPipeResource(uint32_t sliceFirst, uint32_t numSlices, uint32_t shaveFirst, uint32_t numShaves, uint8_t* pmBinImg);
    SippFilter* CreateFilter(uint32_t flags, uint32_t outputWidth, uint32_t outHeight,
                             uint32_t numPlanes, uint32_t bpp, uint32_t paramSize, FnSvuRun funcSvuRun, const char* name,
                             void* pipeline = NULL);
    void SendFilterConfig(SippFilter* filter, void* cfg, uint32_t size);
    // Slice width is needed here to address the unaligned case and to avoid the assert inside sipp
    void SetFilterFrameSize(SippFilter* filter, uint32_t width, uint32_t height, uint32_t sliceW = 0);
    void FilterSetBufBitsPP(SippFilter* filter, uint32_t oBufIdx, uint32_t bitsPerPixel);

    void LinkFilter(SippFilter* filter, SippFilter* parent, uint32_t kernelHeight, uint32_t kernelWidth);
    void LinkFilterSetOBuf(SippFilter* filter, SippFilter* parent, uint32_t parentOBufIdx);
    void FilterAddOBuf(SippFilter* filter, uint32_t numPlanes, uint32_t bpp);
    void FinalizePipeline();
    void ProcessFrameNB() const;
    void DeletePipeline();
    void RegisterEventCallback(sippEventCallback_t cb);
    void RegisterUserContext(uint32_t * userContext);
    void PipeSetLinesPerIter(uint32_t linesPerIter);

    // Error reporting
    uint32_t PipeGetErrorStatus();
    uint32_t GetErrorHistory(uint32_t* errorList);
    uint32_t GetLastError();

    uint32_t getPortNrFor(SippFilter* filter);
  private:
    // The firstShave and lastShave parameters specify which SHAVE processors
    // are to be assigned to execute the pipeline. They specify and inclusive,
    // contiguous and zero-based set of SHAVES. firstShave must be <= lastShave.
    uint32_t firstShave; // First shave on which to execute. Note must be <= lastShave.
    uint32_t lastShave;  // Last shave on which to execute.

    sippEventCallback_t callback;
    uint32_t *userContext;

    uint64_t pipelineID;

    std::vector<SippFilter*> filters;

    uint64_t getPipelineId() const {return pipelineID;}
    void setCallback(sippEventCallback_t cb){
        this->callback = cb;
    }

    static void* CheckXlinkMessage(void* This);
    void* CheckXlinkMessageFunc(void* info);

    // Static members
    static uint16_t asyncChannelId;
    static std::thread asyncChanThread;
};

#endif // __PLG_MV_SIPP_H__
