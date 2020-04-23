///
/// @file      Pool.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for FLIC Pool Plugins on Host over VPUAL.
///
#ifndef __POOL_H__
#define __POOL_H__

#include "MemAllocator.h"
#include "Flic.h"
#include "Message.h"

/**
 * Pool plugin stub template class.
 */

// Remote methods for the pipeline.
enum PoolMethods : uint32_t
{
    P_UNKNOWN = 0,
    P_CREATE  = 1,
};

template <typename T>
class PlgPool : public PluginStub
{
  public:
    /** Output message. */
    PoolSender<PoPtr<T>> out;

  public:
    /** Constructor declaration (definition is type dependant). */
    PlgPool(uint32_t device_id);

  public:
    /**
     * Create method.
     *
     * @param a Allocator to be used by the pool plugin.
     * @param nFrm Number of frames in the pool.
     * @param fSize Size of frames.
     * @param sh Is the pool shared?
     */
    void Create(IAllocator *a, uint32_t nFrm, uint32_t fSize, bool sh = false)
    {
        assert(a != NULL && "Allocator is NULL");  // Ensure "a" is not null before we try to dereference it.

        // Add the member message to the plugin.
        Add(&out);

        // Command message.
        VpualMessage cmd(128);
        PoolMethods method = P_CREATE;
        cmd.serialize(&method, sizeof(method));

        cmd.serialize(&a->stubID, sizeof(a->stubID));   // Send the allocator by ID.
        cmd.serialize(&nFrm, sizeof(nFrm));
        cmd.serialize(&fSize, sizeof(fSize));
        cmd.serialize(&sh, sizeof(sh));

        // Dispatch command.
        VpualDispatch(&cmd, NULL);
    }
};

#endif // __POOL_H__
