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
/// @file      PlgTensorSource.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgTensorSource Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_TENSOR_SOURCE_H__
#define __PLG_TENSOR_SOURCE_H__

#include <stdint.h>

#include "Flic.h"
#include "Message.h"
#include "NN_Types.h"

class PlgTensorSource : public PluginStub
{
  private:
    uint16_t channelID = 0;

  public:
    MSender<TensorMsgPtr> tensorOut;

    /** Constructor. */
    PlgTensorSource() : PluginStub("PlgTensorSource"){};

    /** Destructor. */
    ~PlgTensorSource();

    /**
     * Plugin Create method.
     *
     */
    void Create(uint32_t maxSz, uint16_t chanId, flicTensorDescriptor_t desc);

    /**
     * Plugin Stop method.
     *
     * Issue a stop message to the VPU plugin.
     */
    virtual void Stop();

    /**
     * Plugin Delete method.
     *
     * Close the XLink stream.
     */
    virtual void Delete();

    /**
     * Push a tensor to the plugin.
     * This is not a remote method, it simply performs a blocking write on the
     * plugin's XLink stream.
     *
     * @param pAddr- physical address of start of tensor to send.
     * @param size - size of tensor to send.
     * @retval int - Status of the write (0 is success).
     */
    int Push(uint32_t pAddr, int size);
};

#endif // __PLG_XLINK_IN_H__
