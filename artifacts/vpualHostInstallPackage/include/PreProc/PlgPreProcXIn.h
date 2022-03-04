#ifndef __PLG_PRE_PROC_XIN_H__
#define __PLG_PRE_PROC_XIN_H__

#include <stdint.h>
#include "PlgPreProcTypes.h"

// TODO - We only need DevicePtr from here:
#include "VpuData.h"

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
class PlgPreProcXIn : public PluginStub
{
  private:
    uint16_t channelID;

  public:
    /** Output message (this is a source plugin). */
    MSender<vpupreproc::PreProcConfigPtr> out;

    /** Constructor. */
    PlgPreProcXIn(uint32_t device_id = 0) : PluginStub("PlgPreProcXIn", device_id),
                   channelID(XLINK_INVALID_CHANNEL_ID),
                   out{device_id}
                   {};

    // TODO - May be a gcc bug, but we need this declaration to help with initialisation.
    //        Copy-elision should occur, so we will never use it.
    PlgPreProcXIn(const PlgPreProcXIn&); // Declare copy ctor, but don't define.

    /** Destructor. */
    ~PlgPreProcXIn();

    /**
     * Plugin Create method.
     *
     * @param maxSz maximum size of the XLink Stream.
     * @param chanId_unused not used anymore.
     */
    int Create(uint32_t maxSz, uint32_t chanId_unused);

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
    int Push(DevicePtr buff, int size) const;
};

#endif // __PLG_PRE_PROC_XIN_H__

