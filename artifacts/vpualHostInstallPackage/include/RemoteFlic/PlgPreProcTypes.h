#ifndef __PLG_PRE_PROC_TYPES_H__
#define __PLG_PRE_PROC_TYPES_H__
#include <iostream>
#include <vector>
#include <swcFrameTypes.h>

#include "Flic.h"
#include "VpuData.h"

namespace vpupreproc
{
    struct PreProcConfig {};
    typedef PoPtr<PreProcConfig> PreProcConfigPtr;
}  // namespace vpupreproc

#endif  // __PLG_PRE_PROC_TYPES_H__
