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
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sstream>

#include <unistd.h>

#include "Flic.h"
#include "Message.h"

// TODO - We only need DevicePtr from here:
#include <VpuData.h>

/**
 * XPool plugin stub template class.
 */

/** Remote methods for the plugin. */
enum XPoolMethods : uint32_t
{
    XP_UNKNOWN = 0,
    XP_CREATE  = 1,
};

template <typename T>
class XPool : public PluginStub
{
  public:
    /** Output message. */
    PoolSender<PoPtr<T>> out;

    /** Allocation / Free function typedefs. */
    typedef DevicePtr (*AllocatorFunction)(uint32_t);
    typedef void (*FreeFunction)(DevicePtr);
    typedef void (*FreeFunctionCtx)(DevicePtr, void*);

  private:
    /** Allocator function for new buffers. */
    AllocatorFunction Alloc { nullptr };
    /** Free function for released buffers. */
    FreeFunction Free { nullptr };
    FreeFunctionCtx FreeCtx { nullptr };

    /** Threads listening on the XLink channel. */
    std::thread allocThread;
    std::thread freeThread;

    /** Flag to mark the plugin's state. Threads will close when not alive. */
    volatile bool alive { false };

    /** Details on the buffers in our pool. */
    uint32_t buffersInPool;
    uint32_t bufferSize;

    // TODO check if there will be concurrency issues using std queue naitively.
    // TODO - A fifo may not be best. Buffers may be freed out of order..?
    // Buffers which are on the VPU currently.
    std::queue<DevicePtr> output_buffers;
    std::mutex xpool_mutex_;
    std::condition_variable xpool_cond_;

    /** XLink channel ID used. */
    uint16_t chanId;

    void *user_context_ { nullptr };


  private:

    /** Method to perform allocations when new buffers are needed. */
    void AllocThread(void) {
        while(alive) {
            // TODO verify exit condition?
            auto rc { SendBufferBlocking(Alloc(bufferSize), bufferSize) };
            // TODO more graceful handling of this?
            if (0 != rc) {
                throw std::runtime_error("XPool error in AllocThread-SendBufferBlocking");
            }
        }
    }

    /** Method to free buffers when the VPU releases them. */
    void FreeThread(void) {
        while(alive) {
            // TODO try not to throw as much..?

            // Check if there is an output buffer in our queue
            if (false == output_buffers.empty()) {

#ifdef __REMOTE_HOST__

                DevicePtr &out_data { output_buffers.front() };

                // Read data size
                //  - Could be template specialised to read metadata.
                //      - or still just templated...

                uint8_t msg[128];
                uint32_t size { 0 };
                xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
                auto sc = xlink_read_data_to_buffer(&XlinkDeviceHandle, chanId, msg, &size);
                if (sc != X_LINK_SUCCESS) {
                    throw std::runtime_error("Xpool error xlink_read_data_to_buffer");
                }

                // Read data
                size_t data_size { ((uint32_t*)msg)[0] };

                // Need to ensure data_size is not too big for this buffer
                // Should maybe make the queue from a struct of DevicePtr and size_t

                sc = xlink_read_data(&XlinkDeviceHandle, chanId, &out_data, (uint32_t*)&data_size);
                if (sc) {
                    throw std::runtime_error("Xpool error xlink_read_data");
                }

#else // __REMOTE_HOST__

                DevicePtr &expected_data { output_buffers.front() };
                DevicePtr out_data { 0 };

                // Read data size / metadata
                //  - Could be template specialised to read specific metadata.
                //      - or still just templated...

                uint8_t msg[128];
                uint32_t size { 0 };
                xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
                auto sc { xlink_read_data_to_buffer(&XlinkDeviceHandle, chanId,
                                                    msg, &size) };
                if (sc != X_LINK_SUCCESS) {
                    throw std::runtime_error("Xpool error xlink_read_data_to_buffer");
                }

                // Read data
                size_t expected_size { ((uint32_t*)msg)[0] };
                size_t data_size { 0 };

                // TODO - Fix this api, it's very convoluted.
                DevicePtr *ptr_paddr { &out_data };
                sc = xlink_read_data(&XlinkDeviceHandle, chanId,
                                     reinterpret_cast<uint8_t**>(&ptr_paddr),
                                     (uint32_t*)&data_size);

                if (sc != X_LINK_SUCCESS) {
                    throw std::runtime_error("Xpool error xlink_read_data");
                }

                // TODO - Do we need to do this validation? Remote case?
                if (data_size != expected_size) {
                    std::string error_message {
                        "XPool unexpected size returned. E: "
                        + std::to_string(expected_size)
                        + " A: "
                        + std::to_string(data_size)
                        };
                    throw std::runtime_error(error_message);
                }
                if (out_data != expected_data) {
                    std::stringstream error_message;
                    error_message << "XPool unexpected buffer returned. E: "
                        << std::hex
                        << expected_data
                        << " A: "
                        << out_data;
                    throw std::runtime_error(error_message.str());
                }

#endif // __REMOTE_HOST__

                // Notify user that the buffer is ready.
                if (FreeCtx) {
                    // Prefer the free with user context function.
                    FreeCtx(out_data, user_context_);
                } else if (Free) {
                    Free(out_data);
                }
                // Remove the buffer from the queue.
                std::unique_lock<std::mutex> m_lock(xpool_mutex_);
                output_buffers.pop();
                m_lock.unlock(); // Unlock before notification
                xpool_cond_.notify_one();
            }
        }
    }


  public:

    /** Constructor declaration (definition is type dependant). */
    XPool(uint32_t device_id);

    // TODO - May be gcc bug, but we need this declaration to help with initialisation.
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

    void SetFreeFunction(FreeFunction freefunc)
    {
        Free = freefunc;
    }

    void SetFreeFunction(FreeFunctionCtx freefunc)
    {
        FreeCtx = freefunc;
    }

    void SetUserContext(void *context)
    {
        user_context_ = context;
    }

    /**
     * Create method.
     *
     * @param nBuf      Number of buffers in the pool.
     * @param bSize     Size of the buffers.
     * @param xId_unused  not used anymore.
     * @param freefunc  [optional] Free function for buffers no longer in use.
     */
    int Create(uint32_t nBuf, uint32_t bSize, uint16_t xId_unused, FreeFunction freefunc = NULL)
    {
        uint16_t xId;
        // Save the parameters.
        buffersInPool = nBuf;
        bufferSize = bSize;
        Free = freefunc;

        // Command message.
        VpualMessage cmd(128), rep;
        XPoolMethods method = XP_CREATE;
        cmd.serialize(&method, sizeof(method));

        cmd.serialize(&nBuf, sizeof(nBuf));
        cmd.serialize(&bSize, sizeof(bSize));

        // Dispatch command.
        VpualDispatch(&cmd, &rep);

        // Read the response.
        int32_t retval;
        rep.deserialize(&retval, 4);
        rep.deserialize(&xId, 2);

        if (retval) {
            std::cerr << "XPool Remote Channel Open failed: " << retval << std::endl;
            return static_cast<int>(retval);
        }

        // Open blocking each way, with no timeout.
        xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
        xlink_error status = xlink_open_channel(&XlinkDeviceHandle, xId, RXB_TXB, (nBuf * bSize) * 4, 0);
        if (status) {
            std::cerr << "XPool Open Channel Status: " << status << std::endl;
            return static_cast<int>(status);
        }

        chanId = xId;

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
     * @param xId       Channel ID for the pool.(now being unused).
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
    int SendBuffer(DevicePtr buffer, uint32_t size)
    {
        // Lock this function in order to perform xlink write and output_buffer
        // update together
        static std::mutex send_mutex;
        std::lock_guard<std::mutex> lock(send_mutex);

        assert(size >= bufferSize); // Ensure this buffer is big enough.

        // This would block if we are potentially overflowing, but it could
        // cause a locked state:
        //     while (output_buffers.size() >= buffersInPool) {
        //         sched_yield();
        //     }
        // We will assume the caller knows what they are doing, but display a
        // warning.
        if (output_buffers.size() >= buffersInPool) {
            // mvLog(MVLOG_WARNING, "Warning, sending more buffers than XPool can hold on the VPU.");
            std::cout << "Warning, sending more buffers than XPool can hold on the VPU." << std::endl;
        }

        xlink_handle XlinkDeviceHandle {getXlinkDeviceHandle(getDeviceId())};
#ifdef __REMOTE_HOST__
        // TODO We actually just want to do a sort of:
        //    xlink_allocate_vpu_buffer(chan, bufferSize);
        auto rc { xlink_write_data(&XlinkDeviceHandle, chanId, buffer, bufferSize) };
#else // __REMOTE_HOST__
        auto rc { xlink_write_data(&XlinkDeviceHandle, chanId, reinterpret_cast<uint8_t *>(&buffer), bufferSize) };
#endif // __REMOTE_HOST__
        if (X_LINK_SUCCESS != rc) {
            std::cerr << "Error in XPool XLinkWrite: " << rc << std::endl;
        } else {
            output_buffers.push(buffer);
        }
        return static_cast<int>(rc);
    }

    // Possible blocking versions of the SendBuffer function:
    int SendBufferBlocking(DevicePtr buffer, uint32_t size)
    {
        // Wait until the buffer is needed:
        {
            std::unique_lock<std::mutex> m_lock(xpool_mutex_);
            xpool_cond_.wait(m_lock, [this]{return output_buffers.size() < buffersInPool;});
        }
        int rc { SendBuffer(buffer, size) };
        return rc;
    }

    // int SendBufferBlocking_WaitConsumed(DevicePtr buffer, uint32_t size)
    // {
    //     int rc { SendBuffer(buffer, size) };
    //     std::unique_lock<std::mutex> m_lock(xpool_mutex_);
    //     // Wait until the buffer is consumed:
    //     //   - Assuming 1 buffer only in the queue...
    //     xpool_cond_.wait(m_lock, [this]{return output_buffers.empty();});
    //     return rc;
    // }
    // TODO - provide a method for user callback.
};

#endif // __XPOOL_H__
