///
/// @file      PlgQuantize.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgQuantize Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_QUANTIZE_H__
#define __PLG_QUANTIZE_H__

#include "Flic.h"
#include "Message.h"

/** Quantization plugin. */
class PlgQuantize : public PluginStub
{
  public:
    /** Input frame. */
    SReceiver<ImgFramePtr> in;
    /** Output frame. */
    MSender<ImgFramePtr> out;
    /** Output Buffer. */
    MReceiver<ImgFramePtr> in0;

  public:
    /** Constructor. */
    PlgQuantize() : PluginStub("PlgQuantize"){};

    /** Create method. */
    void Create();

    /**
     * Set the mask of the plugin.
     *
     * @param new_mask is the new mask for the plugin.
     */
    void SetMask(const char new_mask);
};

#endif // __PLG_QUANTIZE_H__
