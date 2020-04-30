///
/// @file      VpualDispatcher.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for the VPUAL Dispatcher.
///
#ifndef __VPUAL_DISPATCHER_H__
#define __VPUAL_DISPATCHER_H__

#include <stdint.h>
#include <thread>

#include "VpualMessage.h"
#include "xlink.h"

// Maximum length of a decoder's type name.
#define DECODER_NAME_MAX_LENGTH (20)
// Ensure the correct resources are opened/closed when needed.
class VpualDispatcherResource {
  public:
    VpualDispatcherResource (uint32_t device_id);
    ~VpualDispatcherResource ();
};

VpualDispatcherResource& initVpualDispatcherResource(uint32_t device_id);  // static initializer for every translation unit*/

/**
 * Get the XLink device handle for VPUAL dispatcher.
 *
 * @return - XLink device handle.
 */
xlink_handle getXlinkDeviceHandle(uint32_t device_id);

/**
 * Base class for all Stubs.
 *
 * Handles construction, destruction, and dispatching messages to a
 * corresponding decoder on the device.
 *
 * TODO maybe delete some operations like copy-constructor.
 */
class VpualStub
{
  private:
  	uint32_t device_id;
  	uint32_t channel;

    // protected: // TODO, should really be protected, some child classes should then be listed as "friends" of each other
  public:
    uint32_t stubID; /*< ID of the stub and matching decoder. */

  public:
    /**
	 * Constructor.
	 * Construct this stub and create a corresponding decoder on the device.
	 *
	 * @param type the string name of the decoder type to create.
	 */
    VpualStub(const char type[DECODER_NAME_MAX_LENGTH], uint32_t device_id);

    /**
	 * Destructor.
	 * Destroy this stub and the corresponding decoder on the device.
	 */
    virtual ~VpualStub();

    /**
	 * Dispatch.
	 * Dispatch a command to the corresponding decoder on the device and wait
	 * for a response.
	 *
	 * @param cmd The "command" message to dispatch to the decoder.
	 * @param rep The "response" message containing the reply from the decoder.
	 */
    // TODO[OB] - Is it alright to call this method const?
    void VpualDispatch(const VpualMessage *const cmd, VpualMessage *rep) const;

    uint32_t getDeviceId() const;
};

// TODO dummy types for now. May never need real type, but might be nice to have.
template <typename T>
class PoPtr
{
};

struct ImgFrame {};
typedef PoPtr<ImgFrame> ImgFramePtr;

struct TensorMsg {};
typedef PoPtr<TensorMsg> TensorMsgPtr;

struct InferenceMsg {};
typedef PoPtr<InferenceMsg> InferenceMsgPtr;

#endif // __VPUAL_DISPATCHER_H__
