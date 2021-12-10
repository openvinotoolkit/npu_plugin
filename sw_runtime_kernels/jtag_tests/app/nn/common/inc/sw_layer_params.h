/*
* {% copyright %}
*/
#pragma once

#include <nn_memory.h>
#include <stdint.h>

namespace nn {
namespace shave_lib {
/**
 * These structs define parameters proprietary to each layer.
 * Layer-specific parsers (which have access to the graphfile) are expected
 * to build these data strctures and bind them to Layer* which they provide.
 */
struct NN_CACHE_ALIGNED LayerParams : public memory::cache_aligned_base {
  uint8_t *cmxData{nullptr};
  uint32_t availableCmxBytes{0};
  // FIXME: default this to _true_ once the parsers support this flag
  bool layerRequiresCacheFlushOnCompletion{false};

  LayerParams() = default;
  LayerParams(const LayerParams &) = default;
  LayerParams(LayerParams &&) noexcept = default;
  LayerParams &operator=(const LayerParams &) = default;
  LayerParams &operator=(LayerParams &&) noexcept = default;
  virtual ~LayerParams() = default;
};

struct LayerExecContext {
  // TODO: do we need anything here?
};

// Standin for Shave runtime Work object
struct alignas(64) ExecContext {
    uint32_t kernelFunc;
    void* ddrPerfBuf; //MvNCIShaveCounters
};

} // namespace shave_lib
} // namespace nn
