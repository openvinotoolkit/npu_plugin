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
/// @file      PlgQuantize.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgQuantize Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_QUANTIZE_H__
#define __PLG_QUANTIZE_H__

#include "Flic.h"
#include "Message.h"

/** Quantization plugin. */
class PlgQuantize : public PluginStub
{
  public:
    /** Input frame. */
    SReceiver<ImgFramePtr> in;
    /** Output frame. */
    MSender<ImgFramePtr> out;
    /** Output Buffer. */
    MReceiver<ImgFramePtr> in0;

  public:
    /** Constructor. */
    PlgQuantize() : PluginStub("PlgQuantize"){};

    /** Create method. */
    void Create();

    /**
     * Set the mask of the plugin.
     *
     * @param new_mask is the new mask for the plugin.
     */
    void SetMask(const char new_mask);
};

#endif // __PLG_QUANTIZE_H__
