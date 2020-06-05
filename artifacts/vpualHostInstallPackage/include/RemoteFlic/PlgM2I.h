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

    /** Constructor. */
    PlgM2I(uint32_t device_id) : PluginStub("PlgM2I", device_id), out{device_id} {};

    /** Create method. */
    int Create();
};

#endif // __PLG_M2I_H__
