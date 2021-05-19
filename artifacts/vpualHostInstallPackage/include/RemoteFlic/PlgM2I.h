// {% copyright %}
///
/// @file      PlgM2I.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for PlgM2I Host FLIC plugin stub using VPUAL.
///
#ifndef __PLG_M2I_H__
#define __PLG_M2I_H__

#include "Flic.h"
#include "Message.h"
#include <vector>
#include "PlgM2ITypes.h"

/** M2I plugin. */
class PlgM2I : public PluginStub
{
  public:
	SReceiver<vpum2i::M2IObj> in;
    MSender<vpum2i::M2IObj> out;

	uint32_t device_id;
    /** Constructor. */
    PlgM2I(uint32_t device_id = 0) : PluginStub("PlgM2I", device_id), out{device_id}, device_id(device_id) {};
    /** Destructor. */
    ~PlgM2I();

    /** Create method. */
    /** @param[in]  enableHWResizeFilter - parameter for enabling the resize polyfir HW filter(=1), or disabling it and use the resize SW filter working on SHAVE(=0) */
    /** @param[in] enableBltCSConly - parameter for enabling the Blitter HW block for color space conversion. Scaler will not be used if this flag is set */
    int Create(bool enableHWResizeFilter = true, bool enableBltCSConly = false);

    vpum2i::M2IDesc* GetSharedDesc();
    int ReleaseDesc(vpum2i::M2IDesc* desc);

  private:
    std::vector<vpum2i::M2IDesc*> descList;
};

#endif // __PLG_M2I_H__
