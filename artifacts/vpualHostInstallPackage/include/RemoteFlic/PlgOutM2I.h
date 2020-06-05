///
/// @file      PlgOutM2I.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgOutM2I Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_OUT_M2I_H__
#define __PLG_OUT_M2I_H__

#include <stdint.h>

#include "Flic.h"
#include "Message.h"
#include "PlgM2ITypes.h"

#ifndef XLINK_INVALID_CHANNEL_ID
#define XLINK_INVALID_CHANNEL_ID (0)
#endif

/**
 * M2I Output FLIC Plugin Stub Class.
 * This object creates the real plugin on the device and links to it with an
 * XLink stream.
 */
class PlgOutM2I : public PluginStub
{
  private:
    uint16_t channelID;

  public:
    /** Input message (this is a sink plugin). */
    SReceiver<vpum2i::M2IObj> in;

    /** Constructor. */
    PlgOutM2I() : PluginStub("PlgOutM2I"),
                  channelID(XLINK_INVALID_CHANNEL_ID)
                  {};

    /** Destructor. */
    ~PlgOutM2I();

    /**
     * Plugin Create method.
     *
     * @param maxSz maximum size of the XLink Stream.
     * @param chanId_unused not used anymore.
     */
    int Create(uint32_t maxSz);

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
    int Pull(uint32_t *pAddr) const;
};

#endif // __PLG_OUT_M2I_H__
