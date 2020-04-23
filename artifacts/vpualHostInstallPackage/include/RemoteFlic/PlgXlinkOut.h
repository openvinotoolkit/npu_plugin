///
/// @file      PlgXlinkOut.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgXlinkOut Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_XLINK_OUT_H__
#define __PLG_XLINK_OUT_H__

#include <stdint.h>

#include "Flic.h"
#include "Message.h"

#ifndef XLINK_INVALID_CHANNEL_ID
#define XLINK_INVALID_CHANNEL_ID (0)
#endif

/**
 * Xlink Output FLIC Plugin Stub Class.
 * This object creates the real plugin on the device and links to it with an
 * XLink stream.
 */
class PlgXlinkOut : public PluginStub
{
  private:
    uint16_t channelID;

  public:
    /** Input message (this is a sink plugin). */
    SReceiver<ImgFramePtr> in;

    /** Constructor. */
    PlgXlinkOut(uint32_t device_id) : PluginStub("PlgXlinkOut", device_id),
                    channelID(XLINK_INVALID_CHANNEL_ID)
                    {};

    /** Destructor. */
    ~PlgXlinkOut();

    /**
     * Plugin Create method.
     *
     * @param maxSz maximum size of the XLink Stream.
     * @param channame name of the XLink channel.
     */
    int Create(uint32_t maxSz, uint32_t chanId);

    /**
     * Plugin Delete method.
     *
     * Close the XLink stream.
     */
    virtual void Delete();

    /**
     * Pull a frame from the plugin.
     * This is not a remote method, it simply performs a blocking read on the
     * plugin's XLink stream.
     *
     * @param pAddr - Physical address of received frame.
     * @param length - size of received frame.
     */
    int Pull(uint32_t *pAddr, uint32_t *length) const;
};

#endif // __PLG_XLINK_OUT_H__
