///
/// @file      PlgTensorSource.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgTensorSource Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_INFERENCE_INPUT_H__
#define __PLG_INFERENCE_INPUT_H__

#include <stdint.h>

#include "Flic.h"
#include "Message.h"
#include "NN_Types.h"

class PlgInferenceInput : public PluginStub
{
  private:
    uint16_t channelID;

  public:
    MSender<InferenceMsgPtr> inferenceOut;

    /** Constructor. */
    PlgInferenceInput(uint32_t device_id) : PluginStub("PlgInfInput", device_id), inferenceOut{device_id} {}

    /** Destructor. */
    ~PlgInferenceInput();

    /**
     * Plugin Create method.
     *
     */
    void Create(uint32_t maxSz, uint16_t chanId);

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
     * Push a tensor to the plugin.
     * This is not a remote method, it simply performs a blocking write on the
     * plugin's XLink stream.
     *
     * @param pAddr- physical address of start of tensor to send.
     * @param size - size of tensor to send.
     * @retval int - Status of the write (0 is success).
     */
    int PushInferenceID(uint32_t pAddr, int size);
};

#endif // __PLG_INFERENCE_INPUT_H__
