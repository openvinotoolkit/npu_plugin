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
/// @file      NnXlinkPlg.h
/// @copyright All code copyright Movidius Ltd 2020, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for NnXlinkPlg Host FLIC plugin using VPUAL.
///
#ifndef __NN_XLINK_PLG_H__
#define __NN_XLINK_PLG_H__

#include <cstddef>
#include "Flic.h"
#include "Message.h"
#include "NN_Types.h"

class NnXlinkPlg : public PluginStub{
  public:
    MSender<NnExecMsg> requestOut;
    SReceiver<NnExecResponseMsg> resultIn;

    /** Constructor. */
    NnXlinkPlg() : PluginStub("NnXlinkPlg") {};

    ~NnXlinkPlg() { Delete(); }; // Ensure delete is called to close channel.

    int Create(size_t queue_len);

    int RequestInference(const NnExecMsg& request);
    int WaitForResponse(NnExecResponseMsg& response);
  private:
    uint16_t channel_id_ {};
    bool channel_open_ { false };

    void Delete(void) override;
};
#endif // __NN_XLINK_PLG_H__
