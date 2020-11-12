///
/// @file      PlgInM2I.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgInM2I Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_IN_M2I_H__
#define __PLG_IN_M2I_H__

#include <stdint.h>
#include "swcFrameTypes.h"
#include "PlgM2ITypes.h"

#include "Flic.h"
#include "Message.h"

#ifndef XLINK_INVALID_CHANNEL_ID
#define XLINK_INVALID_CHANNEL_ID (0)
#endif

#ifndef DEFAULT_M2I_DESC_QSZ
 #define DEFAULT_M2I_DESC_IN_QSZ 4
#endif

/**
 * M2I Input FLIC Plugin Stub Class.
 * This object creates the real plugin on the device and links to it with an
 * XLink stream.
 */
class PlgInM2I : public PluginStub
{
  private:
    uint16_t channelID;

  public:
    /** Output message (this is a source plugin). */
    MSender<vpum2i::M2IObj> out;

    /** Constructor. */
    PlgInM2I(uint32_t device_id = 0) : PluginStub("PlgInM2I", device_id), out{device_id} {};
                 //channelID(XLINK_INVALID_CHANNEL_ID)

    /** Destructor. */
    ~PlgInM2I();

    /**
     * Plugin Create method.
     *
     * @param maxDesc maximum number of descriptors to be queued
     * @param chanId_unused not used anymore.
     */
    int Create(uint32_t maxDesc = DEFAULT_M2I_DESC_IN_QSZ);

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
    int Push(const vpum2i::M2IDesc* header) const;
};

#endif // __PLG_IN_M2I_H__
