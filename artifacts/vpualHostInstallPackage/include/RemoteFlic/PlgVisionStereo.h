
// {% copyright %}
///
/// @file
/// @brief     Header for PlgVisionStereo Host FLIC plugin stub using VPUAL.
///

#ifndef __PLGVISIONSTEREO_H__
#define __PLGVISIONSTEREO_H__

#include <stdint.h>

#include <Flic.h>
#include "Message.h"
#include "PlgVisionStereoTypes.h"

#include "Pool.h"

/** Plugin Vision Stereo Class. */
class PlgVisionStereo : public PluginStub {
public:
    /** Config struct that inglobates the configs used for the warp, NN-descriptor, stereo stage, and scratch buffet. */
    struct Configs {
        CfgWarpStage warp;
        CfgNNStage descriptor;
        CfgStereoStage stereo;
        CfgScratchBuffer scratchBuffer;
    };
    /** Input frame left */
    SReceiver<ImgFramePtr> inLeft;
    /** Input frame right */
    SReceiver<ImgFramePtr> inRight;
    /** Port connect to the stereo output pool */
    MReceiver<ImgFramePtr> outBuffer;
    /** Stereo output port */
    MSender<ImgFramePtr> out;
    /** Slice ID - relevant on THB */
    uint32_t device_id;

    /** Class for scratch buffer allocator. */
    class ScratchBufAllocator {
    public:
        /**
         * Allocate a VpuData shared memory for the scratch buffer.
         *
         * @param physAddr  - Physical address extracted from the allocated VpuData shared memory.
         * @param size      - Desire size to be allocated for the scratch buffer.
         * @param device_id - Slice ID.
         * @retval          - Error code.
         */
        int32_t Create(uint32_t &physAddr, uint32_t &size, uint32_t device_id);
        /**
         * Deallocate the scratch buffer.
         *
         * @retval          - Error code.
         */
        int32_t Delete();

    private:
        /** Enable local allocation */
        bool localAlloc_{};
        /** The address of the VpuData shared memory allocated */
        VpuData *base_{};
    };
    /** ScratchBufAllocator obiect */
    ScratchBufAllocator scratchBufAlloc;

    /** Constructor. */
    PlgVisionStereo(uint32_t device_id = 0)
        : PluginStub("PlgVisionStereo", device_id)
        , out{device_id}
        , device_id(device_id){};
    /** Destructor. */
    ~PlgVisionStereo();

    /**
     * Run the proxys for the plgVisionStereo create function.
     *
     * @param cfgVisionStereo - Plugin config.
     * @retval                - Error code.
     */
    int32_t Create(Configs cfgVisionStereo);
    /**
     * Delete the plgVisionStereo plugin and dealocate the resources used.
     *
     * @retval - Error code.
     */
    void Delete();
    /**
     * Calculate the minimum size needed by for the scratch buffer.
     *
     * @param cfgVisionStereo - Plugin config.
     * @retval                - Error code.
     */
    uint32_t GetMinSizeOfScratchBuf(const Configs &cfgVisionStereo);
};

#endif //__PLGVISIONSTEREO_H__
