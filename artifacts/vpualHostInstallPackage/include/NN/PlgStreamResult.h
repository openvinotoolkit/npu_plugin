///
/// @file      PlgStreamResult.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgStreamResult Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_XLINK_OUT_H__
#define __PLG_XLINK_OUT_H__

#include <stdint.h>

#include "Flic.h"
#include "Message.h"
#include "NN_Types.h"

class PlgStreamResult : public PluginStub
{
  private:
    uint32_t channelID = -1; // TODO[OB] - maybe use the XLink type.. it is uint16_t currently...

  public:
    SReceiver<TensorMsgPtr> dataIn;

    /** Constructor. */
    PlgStreamResult() : PluginStub("PlgStreamResult"){};

    /** Destructor. */
    ~PlgStreamResult();

    /**
     * Plugin Create method.
     *
     */
    void Create(uint32_t maxSz, uint16_t chanId, flicTensorDescriptor_t desc);

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
     */
    int Pull(uint32_t *pAddr, uint32_t *length);
};

#endif // __PLG_XLINK_OUT_H__
