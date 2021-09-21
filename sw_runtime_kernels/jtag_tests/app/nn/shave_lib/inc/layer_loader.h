/*
 * {% copyright %}
 */
#pragma once

#include <graphfile_generated.h>
#include <unordered_map>
#include "sw_layer.h"
#include "layer_parser.h"

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

    static bool parseUPALayer(const MVCNN::UPALayerTask *task, Layer *layer);

private:
    typedef bool (*ParserFunc)(const MVCNN::UPALayerTask *task, Layer *layer);

    SoftKernel builtinUPAKernels;
    std::unordered_map<MVCNN::SoftwareLayerParams, ParserFunc> parserMap_;

    LayerLoader();
    void registerParsers();

    LayerLoader(const LayerLoader &) = delete;
    LayerLoader &operator =(const LayerLoader &) = delete;

    LayerLoader(LayerLoader &&) = delete;
    LayerLoader &operator =(LayerLoader &&) = delete;

    static bool parseUPALayer(const MVCNN::UPALayerTask *task, Layer *layer, LayerParser &lp);
    static void loadElf(const uint8_t *elfAddr, SoftKernel &kernel);

    template <typename Parser>
    static bool parse(const MVCNN::UPALayerTask *task, nn::shave_lib::Layer *layer)
    {
        Parser p;
        return parseUPALayer(task, layer, p);
    }
};

} // namespace shave_lib
} // namespace nn
