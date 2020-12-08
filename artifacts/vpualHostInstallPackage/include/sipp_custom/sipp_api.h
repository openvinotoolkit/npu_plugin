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
/// @file      sipp_api.h
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for Host side SIPP usage over VPUAL.
///

#ifndef _SIPP_API_H_
#define _SIPP_API_H_

#include <iostream>

#include <sipp_messages.h>
#include <swcFrameTypes.h>
#include "Pool.h"

// TODO - need this for DevicePtr, may be better place to put it.
#include <VpuData.h>

typedef struct SippPipeline SippPipeline;
//////////////////////////////////////////////////////////////////
/////////////////////// Pipeline Creation ////////////////////////
//////////////////////////////////////////////////////////////////
/**
 * Set mask to bypass selected SIPP initialisation steps.
 *
 * @param mask - Mask to set desired initialisation steps.
 *               bit0 - clear bit to bypass CMXDMA SW reset.
 */
void sippSetInitMask(uint32_t mask, uint32_t device_id = 0);

/**
 * Create a basic SIPP pipeline without initially assigning any shaves or
 * slices.
 *
 * @return [SippPipeline*] Pointer to new SIPP pipeline.
 */
SippPipeline* sippCreateBasicPipeline(uint32_t device_id);

/**
 * Create a SIPP pipeline.
 *
 * @param first_slice - First slice/SHAVE to be allocated.
 * @param last_slice  - Last slice/SHAVE to be allocated.
 * @param pmBinImg    - Pointer to SHAVE binary.
 *
 * @return [SippPipeline*] Pointer to new SIPP pipeline.
 */
SippPipeline* sippCreatePipeline(uint32_t first_slice, uint32_t last_slice, uint8_t* pmBinImg, uint32_t device_id = 0);

/**
 * Add SHAVE and CMX slice resources to a SIPP pipeline instance.
 *
 * @param  pl         - SIPP pipeline to which resources are to be added.
 * @param  sliceFirst - First slice to be allocated.
 * @param  numSlices  - Number of slices to be allocated.
 * @param  shaveFirst - First SHAVE to be allocated.
 * @param  numShaves  - Number of SHAVEs to be allocated.
 * @param  pmBinImg   - Pointer to SHAVE binary.
 *
 * @return [eSIPP_STATUS] SIPP error status.
 */
eSIPP_STATUS sippAddPipeResource(SippPipeline* pl, uint32_t sliceFirst, uint32_t numSlices, uint32_t shaveFirst, uint32_t numShaves, uint8_t* pmBinImg);

/**
 * Finalize SIPP pipeline including pipeline validation and schedule creation.
 *
 * @param pl - Pipeline to finalize.
 */
void sippFinalizePipeline(SippPipeline* pl);

/**
 * Create a SIPP filter.
 *
 * @param pl          - SIPP pipeline with which the filter is to be associated..
 * @param flags       - Filter flags.
 * @param outputWidth - Output width.
 * @param outHeight   - Output height.
 * @param numPlanes   - Number of planes.
 * @param bpp         - Bits per pixel.
 * @param paramSize   - Parameter structure size.
 * @param funcSvuRun  - Filter type identifier.
 * @param name        - Filter name (for debug purposes).
 *
 * @return [SippFilter*] Pointer to new SIPP filter.
 */
SippFilter* sippCreateFilter(SippPipeline* pl, uint32_t flags, uint32_t outputWidth, uint32_t outHeight, uint32_t numPlanes, uint32_t bpp, uint32_t paramSize, FnSvuRun funcSvuRun, const char* name);

/**
 * Set IO filter frame size where frame size is different than IO filter dimensions and stride is intended to be used
 *
 * @param filter      - SIPP pipeline with which the IO filter is to be associated..
 * @param outputWidth - Output width.
 * @param outHeight   - Output height.
 * @param sliceW      - Output width for each shave - workaround for unaligned slice width assert
 *
 * @return void
 */
void sippSetFilterFrameSize(SippFilter* filter, uint32_t outputWidth, uint32_t outputHeight, uint32_t sliceW);

/**
 * Set number of bits per pixel for the specified filter.
 *
 * @param filter       - SIPP filter.
 * @param oBufIdx      - Output buffer ID.
 * @param bitsPerPixel - Number of bits per pixel.
 */
void sippFilterSetBufBitsPP(SippFilter* filter, uint32_t oBufIdx, uint32_t bitsPerPixel);

/**
 * Link two SIPP filters.
 *
 * @param filter       - New filter to link.
 * @param parent       - Parent filter.
 * @param kernelHeight - Kernel height (of child filter).
 * @param kernelWidth  - Kernel width (of child filter).
 */
void sippLinkFilter(SippFilter* filter, SippFilter* parent, uint32_t kernelHeight, uint32_t kernelWidth);

/**
 * Modify the parent output buffer which a consumer filter uses as its input buffer.
 * Note: The two filters must have been previously linked through a call to
 * sippLinkFilter.
 *
 * @param filter        - SIPP filter.
 * @param parent        - Parent filter.
 * @param parentOBufIdx - Parent output buffer ID.
 */
void sippLinkFilterSetOBuf(SippFilter* filter, SippFilter* parent, uint32_t parentOBufIdx);

/**
 * Add an output buffer to the specified filter.
 *
 * @param filter    - SIPP filter.
 * @param numPlanes - Number of planes.
 * @param bpp       - Number of buts per pixel.
 */
void sippFilterAddOBuf(SippFilter* filter, uint32_t numPlanes, uint32_t bpp);

/**
 * Send filter configuration to VPU.
 *
 * @filter - Filter to be configured.
 * @cfg    - Filter configuration structure.
 */
void sippSendFilterConfig(SippFilter* filter, void* cfg, uint32_t size);

/**
 * Set up an address update for a dma filter
 *
 * @param filter - dma filter to update
 * @param data   - Memory to write
 * @param spec   - frame spec
 */
void sippDmaWriteBuffer(SippFilter* filter, DevicePtr data, const frameSpec* spec);

/**
 * Set the output address for a dma filter
 *
 * @param filter  - dma filter to update
 * @param buffer  - Memory to write to
 * @param length  - length of buffer
 */
void sippDmaSetOutputBuffer(SippFilter* filter, DevicePtr buffer, uint32_t length);

/**
 * Delete a SIPP pipeline.
 *
 * @param pl - Pipeline to be deleted.
 */
void sippDeletePipeline(SippPipeline* pl);

/**
 * Register a handler for reporting SIPP pipeline events.
 *
 * @param pl - SIPP pipeline with which to register the callback.
 * @param cb - Pointer to function to be executed when a SIPP pipeline event occurs.
 */
void sippRegisterEventCallback(SippPipeline* pl, sippEventCallback_t cb);

/**
 * Register a user context for reporting SIPP pipeline events.
 *
 * @param pl - SIPP pipeline with which to register the callback.
 * @param cb - Pointer to user context.
 */
void sippRegisterUserContext(SippPipeline* pl, uint32_t * userContext);

//////////////////////////////////////////////////////////////////
///////////////////////// SIPP Functions /////////////////////////
//////////////////////////////////////////////////////////////////

/**
 * Process a single frame using the specified SIPP pipeline (non-blocking).
 *
 * @param pl - SIPP pipeline.
 */
void sippProcessFrameNB(SippPipeline* pl);

/**
 * Set the number of lines per scheduling iteration to run for a specific
 * pipeline.
 *
 * @param pl           - SIPP pipeline.
 * @param linesPerIter - Lines per iteration. Note this must be set to 1, 2, 4, 8 or 16.
 */
void sippPipeSetLinesPerIter(SippPipeline* pl, uint32_t linesPerIter);


//////////////////////////////////////////////////////////////////
//////////////////////// Error Reporting /////////////////////////
//////////////////////////////////////////////////////////////////

/**
 * Return last error recorded.
 *
 * @return [uint32_t] Last error code recorded.
 */
uint32_t sippGetLastError(SippPipeline* pl);

/**
 * Retrieve the last errors recorded. The number of errors returned is determined
 * by the value of SIPP_ERROR_HISTORY_SIZE. This is numbered either from boot or
 * from the last call to this function.
 *
 * @param pl - SIPP pipeline for which to retrieve errors.
 * @param errorList - Pointer to list of recorded errors.
 *
 * @return [uint32_t]
 *         - Number of valid errors in list.
 */
uint32_t sippGetErrorHistory(SippPipeline* pl, uint32_t* errorList);

/**
 * Retrieve all current recorded errors relating to the specified SIPP pipeline.
 *
 * @param pl - SIPP pipeline for which to retrieve errors.
 *
 * @return [uint32_t]
 *     - 0x1 is an error has been recorded.
 */
uint32_t sippPipeGetErrorStatus(SippPipeline* pl);

#endif /* _SIPP_API_H_ */
