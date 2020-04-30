///
/// @file      XPool.h
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for FLIC XPool Plugins on Host over VPUAL.
///
#ifndef __XPOOL_H__
#define __XPOOL_H__

#include <iostream>
#include <thread>

#include "Flic.h"
#include "Message.h"
#include "xlink.h"

/**
 * XPool plugin stub template class.
 */

/** Remote methods for the plugin. */
enum XPoolMethods : uint32_t
{
    XP_UNKNOWN = 0,
    XP_CREATE  = 1,
};

/* Pointer on the VPU is 32-bit. */
typedef uint32_t DevicePtr;

template <typename T>
class XPool : public PluginStub
{
  private:
    /**
     * Timeout length of the xlink channel.
     *
     * Each attempted XLink transaction will timeout after this time, and return
     * the code X_LINK_TIMEOUT.
     */
    static constexpr uint32_t xlink_timeout_ms { 100 };
    static constexpr DevicePtr xpool_stop_message { 0xDEADBEEF };

    // TODO move this to xlink uapi.
    static constexpr uint16_t kXlinkInvalidChannelId { 0 };

  public:
    /** Output message. */
    PoolSender<PoPtr<T>> out;

    /** Allocation / Free function typedefs. */
    typedef DevicePtr (*AllocatorFunction)(uint32_t);
    typedef void (*FreeFunction)(DevicePtr);

  private:
    /** Allocator function for new buffers. */
    AllocatorFunction Alloc { nullptr };
    /** Free function for released buffers. */
    FreeFunction Free { nullptr };

    /** Threads listening on the XLink channel. */
    std::thread allocThread;
    std::thread freeThread;

    /** Flag to mark the plugin's state. Threads will close when not alive. */
    volatile bool alive { false };

    /** Details on the buffers in our pool. */
    uint32_t buffersInPool;
    uint32_t buffersInUse { 0 };
    uint32_t bufferSize;

    /** XLink channel ID used. */
    uint16_t chanId;

  private:

    /** Method to perform allocations when new buffers are needed. */
    void AllocThread(void) {
        while(alive) {
            // Send buffer if buffercount < buffersInPool
            if (buffersInUse < buffersInPool) {
                auto rc = SendBuffer(Alloc(bufferSize), bufferSize);
                assert(0 == rc); // TODO[OB] Could maybe handle this error better.
            }
            std::this_thread::yield();
        }
    }

    /** Method to free buffers when the VPU releases them. */
    void FreeThread(void) {
        while(alive) {
            DevicePtr paddr { 0 };
            DevicePtr *ptr_paddr { &paddr };
            uint32_t sz { 0 };
            xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
            auto rc = xlink_read_data(&XlinkDeviceHandle, chanId, reinterpret_cast<uint8_t**>(&ptr_paddr), &sz);
            // auto rc = xlink_read_data(&XlinkDeviceHandle, chanId, reinterpret_cast<uint8_t**>(&(&paddr)), &sz);
            if (X_LINK_TIMEOUT == rc){
                // Do nothing
            } else if (X_LINK_SUCCESS == rc) {
                // release this buffer:
                if (Free) {
                    Free(paddr);
                }
                --buffersInUse;
            } else {
                // TODO[OB] our error codes from xlink read aren't currently correct, so we enter here
                // when we timeout. In the future we should handle this error.
                // std::cerr << "Error in xlink read data for xpool: " << rc << std::endl;
            }
            std::this_thread::yield();
        }
    }


  public:

    /** Constructor declaration (definition is type dependant). */
    XPool(uint32_t device_id);

    // TODO - Might be gcc bug, but we need this declaration to help with initialisation.
    //        Copy-elision should occur, so we will never use it.
    XPool(const XPool&); // Declare copy ctor, but don't define.

    /**
     * Destructor.
     *
     * Detatch & delete the threads if they exist and haven't been joined.
     */
    ~XPool() {
        if(allocThread.joinable()) {
            std::cerr << "Warning: Killing joinable thread." << std::endl;
            allocThread.detach();
        }
        if(freeThread.joinable()) {
            std::cerr << "Warning: Killing joinable thread." << std::endl;
            freeThread.detach();
        }
    }

    /**
     * Create method.
     *
     * @param nBuf      Number of buffers in the pool.
     * @param bSize     Size of the buffers.
     * @param xId       Channel ID for the pool.
     * @param freefunc  [optional] Free function for buffers no longer in use.
     */
    int Create(uint32_t nBuf, uint32_t bSize, uint16_t xId, FreeFunction freefunc = NULL)
    {
        // Save the parameters.
        buffersInPool = nBuf;
        bufferSize = bSize;
        chanId = xId;
        Free = freefunc;

        // Open non-blocking each way, with a timeout.
        xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
        xlink_error status = xlink_open_channel(&XlinkDeviceHandle, xId, RXN_TXN, bSize, xlink_timeout_ms);
        if (status) {
            std::cerr << "XPool Open Channel Status: " << status << std::endl;
            return static_cast<int>(status);
        }

        // Command message.
        VpualMessage cmd(128), rep;
        XPoolMethods method = XP_CREATE;
        cmd.serialize(&method, sizeof(method));

        cmd.serialize(&nBuf, sizeof(nBuf));
        cmd.serialize(&bSize, sizeof(bSize));
        cmd.serialize(&xId, sizeof(xId));

        // Dispatch command.
        VpualDispatch(&cmd, &rep);

        // Read the response.
        int32_t retval;
        rep.deserialize(&retval, 4);

        if (retval) {
            std::cerr << "XPool Remote Channel Open failed: " << retval << std::endl;
            xlink_close_channel(&XlinkDeviceHandle, xId); // Close the Host channel.
            return static_cast<int>(retval);
        }

        // Set the plugin's state to alive before starting the thread.
        alive = true;

        // Start the free thread on this XLink channel.
        freeThread = std::thread { &XPool::FreeThread,  this };

        // Add the member message to the plugin.
        Add(&out);

        return 0;
    }

    /**
     * Create method with allocation function.
     *
     * Will spawn a thread which will call the allocation function whenever the
     * pool needs more buffers.
     *
     * @param nBuf      Number of buffers in the pool.
     * @param bSize     Size of the buffers.
     * @param xId       Channel ID for the pool.
     * @param allocfunc Allocation function for new buffers.
     * @param freefunc  [optional] Free function for buffers no longer in use.
     */
    int Create(uint32_t nBuf, uint32_t bSize, uint16_t xId, AllocatorFunction allocfunc, FreeFunction freefunc = NULL)
    {
        assert(NULL != allocfunc);
        Alloc = allocfunc;

        // Call the less specific function:
        int rc = Create(nBuf, bSize, xId, freefunc);

        if (0 == rc) {
            // Start the allocation thread on this XLink channel.
            allocThread = std::thread { &XPool::AllocThread, this };
        }

        return rc;
    }

    /** Stop Method. Indicate to the threads to return. */
    void Stop (void) {
        alive = false;
        if(chanId != kXlinkInvalidChannelId)
        {
            // Send a Stop message to break plugin thread function
            DevicePtr paddr { xpool_stop_message };
            xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
            xlink_error status = xlink_write_data(&XlinkDeviceHandle, chanId, reinterpret_cast<uint8_t *>(&paddr), 0);
            if (status != X_LINK_SUCCESS)
            {
                std::cerr << "XPool: Stop message failed: " << status << std::endl;
            }
        }
    }

    /** Wait for the threads to finish (join). */
    void Wait(void) {
        if (allocThread.joinable()) {
            allocThread.join();
        }
        if (freeThread.joinable()) {
            freeThread.join();
        }
    }

    /** Delete method. Close the channel. */
    void Delete (void) {
        xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
        xlink_error rc = xlink_close_channel(&XlinkDeviceHandle, chanId);
        if (X_LINK_SUCCESS != rc) {
            std::cerr << "XPool close channel status: " << rc << std::endl;
        }
    }

    /** Send a buffer over XLink to the XPool plugin on the VPU. */
    int SendBuffer(DevicePtr paddr, uint32_t size)
    {
        assert(size >= bufferSize); // Ensure this buffer is big enough.
        xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
        auto rc = xlink_write_data(&XlinkDeviceHandle, chanId, reinterpret_cast<uint8_t *>(&paddr), bufferSize);
        if (X_LINK_SUCCESS != rc) {
            std::cerr << "Error in XPool XLinkWrite: " << rc << std::endl;
        } else {
            ++buffersInUse;
        }
        return static_cast<int>(rc);
    }
};

#endif // __XPOOL_H__
