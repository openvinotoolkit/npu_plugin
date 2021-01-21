// {% copyright %}
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
    NnXlinkPlg(uint32_t device_id) : PluginStub("NnXlinkPlg",device_id), requestOut{device_id} {};

    ~NnXlinkPlg() { Stop(); }; // Ensure stop is called to close channel.

    int Create(size_t queue_len);

    int RequestInference(const NnExecMsg& request);
    int WaitForResponse(NnExecResponseMsg& response);
  private:
    uint16_t channel_id_ {};
    bool channel_open_ { false };

    void Stop(void) override;
};
#endif // __NN_XLINK_PLG_H__
