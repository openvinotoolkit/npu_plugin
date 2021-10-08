/*
 * {% copyright %}
 */
#pragma once

#include <unordered_map>
#include "sw_layer.h"

namespace nn {
namespace shave_lib {

class LayerLoader {
public:
    static LayerLoader &instance();
    ~LayerLoader() = default;

    const SoftKernel &builtinKernels() const
    {
        return builtinUPAKernels;
    }


private:

    SoftKernel builtinUPAKernels;

    LayerLoader();
    void registerParsers();

    LayerLoader(const LayerLoader &) = delete;
    LayerLoader &operator =(const LayerLoader &) = delete;

    LayerLoader(LayerLoader &&) = delete;
    LayerLoader &operator =(LayerLoader &&) = delete;

    static void loadElf(const uint8_t *elfAddr, SoftKernel &kernel);
};

} // namespace shave_lib
} // namespace nn
