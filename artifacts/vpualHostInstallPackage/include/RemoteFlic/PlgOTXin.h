///
/// @file      PlgOTXin.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgOTXin Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_XLINK_OT_IN_H__
#define __PLG_XLINK_OT_IN_H__

#include <stdint.h>
#include "swcFrameTypes.h"

#include "Flic.h"
#include "Message.h"
#include "PlgOT.h"
#ifndef XLINK_INVALID_CHANNEL_ID
#define XLINK_INVALID_CHANNEL_ID (0)
#endif

/**
 * Xlink Input FLIC Plugin Stub Class.
 * This object creates the real plugin on the device and links to it with an
 * XLink stream.
 */
struct InputDescriptor {
    float    delta;
    frameSpec fspec_in;
    uint64_t phys_addr_frame;
    uint32_t nDetects;
    uint64_t phys_addr_detects;
};

class PlgOTXin : public PluginStub
{
  private:
    uint16_t channelID;

  public:
    /** Output message (this is a source plugin). */
//    MSender<InputDescriptorPtr> out;
    MSender<ImgFramePtr> out0;
    MSender<vpuot::DetectedObjects> out1;
    MSender<float> out2;

    /** Constructor. */
    PlgOTXin(uint32_t device_id) : PluginStub("PlgOTXin", device_id),
                 channelID(XLINK_INVALID_CHANNEL_ID),
                 out0{device_id},
                 out1{device_id},
                 out2{device_id}
                 {};

    /** Destructor. */
    ~PlgOTXin();

    /**
     * Plugin Create method.
     *
     * @param maxSz maximum size of the XLink Stream.
     * @param chanId_unused  not used, just for API to be backward-compatible.
     */
    int Create(uint32_t maxSz, uint32_t chanId_unused);


    /**
     * Plugin Delete method.
     *
     * Close the XLink stream.
     */
    virtual void Delete();

    /**
     * Push to the plugin.
     * This is not a remote method, it simply performs a blocking write on the
     * plugin's XLink channel.
     *
     * @param pAddr - Physical address to send.
     * @param size - size to send.
     * @retval int - Status of the write (0 is success).
     */
    int Push(const InputDescriptor* header);
};

#endif // __PLG_XLINK_IN_H__
