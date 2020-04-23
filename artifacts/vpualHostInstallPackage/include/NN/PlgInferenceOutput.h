///
/// @file      PlgStreamResult.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgStreamResult Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_INFERENCE_OUTPUT_H__
#define __PLG_INFERENCE_OUTPUT_H__

#include <stdint.h>

#include "Flic.h"
#include "Message.h"
#include "NN_Types.h"

class PlgInferenceOutput : public PluginStub
{
  private:
    uint16_t channelID = -1; // TODO[OB] - maybe use the XLink type.. it is uint16_t currently...

  public:
    SReceiver<InferenceMsgPtr> inferenceIn;

    /** Constructor. */
    PlgInferenceOutput(uint32_t device_id) : PluginStub("PlgInfOutput", device_id){};

    /** Destructor. */
    ~PlgInferenceOutput();

    /**
     * Plugin Create method.
     *
     */
    void Create(uint32_t maxSz, uint16_t chanId);

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
    int PullInferenceID(uint32_t *pAddr, uint32_t *length);
};

#endif // __PLG_INFERENCE_OUTPUT_H__
