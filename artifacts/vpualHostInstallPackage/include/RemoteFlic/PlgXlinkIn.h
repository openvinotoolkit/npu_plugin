///
/// @file      PlgXlinkIn.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgXlinkIn Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_XLINK_IN_H__
#define __PLG_XLINK_IN_H__

#include <stdint.h>
#include "swcFrameTypes.h"

#include "Flic.h"
#include "Message.h"

#ifndef XLINK_INVALID_CHANNEL_ID
#define XLINK_INVALID_CHANNEL_ID (0)
#endif

/**
 * Xlink Input FLIC Plugin Stub Class.
 * This object creates the real plugin on the device and links to it with an
 * XLink stream.
 */
class PlgXlinkIn : public PluginStub
{
  private:
    uint16_t channelID;

  public:
    /** Output message (this is a source plugin). */
    MSender<ImgFramePtr> out;

    /** Constructor. */
    PlgXlinkIn(uint32_t device_id) : PluginStub("PlgXlinkIn", device_id),
                   channelID(XLINK_INVALID_CHANNEL_ID),
                   out{device_id}
                   {};

    // TODO - Might be gcc bug, but we need this declaration to help with initialisation.
    //        Copy-elision should occur, so we will never use it.
    PlgXlinkIn(const PlgXlinkIn&); // Declare copy ctor, but don't define.

    /** Destructor. */
    ~PlgXlinkIn();

    /**
     * Plugin Create method.
     *
     * @param maxSz maximum size of the XLink Stream.
     * @param chanId id of the XLink channel.
     */
    int Create(uint32_t maxSz, uint32_t chanId);

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
     * Push a frame to the plugin.
     * This is not a remote method, it simply performs a blocking write on the
     * plugin's XLink stream.
     *
     * @param pAddr - Physical address of frame to send.
     * @param size - size of frame to send.
     * @param spec - frame spec of the frame to send.
     * @retval int - Status of the write (0 is success).
     */
    int Push(uint32_t pAddr, int size, const frameSpec* spec) const;
};

#endif // __PLG_XLINK_IN_H__
