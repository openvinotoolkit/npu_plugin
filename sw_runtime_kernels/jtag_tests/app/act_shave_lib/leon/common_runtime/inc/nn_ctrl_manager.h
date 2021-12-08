/*
 * {% copyright %}
 */

#pragma once

#include <mv_types.h>

namespace nn {

enum SHVCtrlMessage : uint8_t{
    HWStatsEnable,
    PreemptHaltAndAck,
    EnablePerfStream,
    DisablePerfStream,
    Shutdown // we don't need this -- a NULL work item already does this
};

namespace inference_runtime {}

namespace dpu_runtime {}

namespace as_runtime {}

} // namespace nn
