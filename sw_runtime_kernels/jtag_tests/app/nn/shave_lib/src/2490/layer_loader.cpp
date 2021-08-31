/*
* {% copyright %}
*/
#include "layer_loader.h"
#include "sw_shave_dispatcher.h"
#include "sw_shave_lib_common.h"
#include <elf.h>
#include <dma_leon.h>

#include "layers/parser_permute.h"
#include "layers/parser_softmax.h"
#include "layers/parser_postops.h"
#include "layers/parser_custom_cpp.h"
#include "layers/svuSLKernels_EP.h"
#include <assert.h>
#include <mvLog.h>
#include <nn_cache.h>

namespace nn {
namespace shave_lib {

LayerLoader &LayerLoader::instance()
{
    static LayerLoader singleton;
    return singleton;
}

LayerLoader::LayerLoader() :
    builtinUPAKernels(),
    parserMap_()
{
    loadElf(&svuSLKernels_Base, builtinUPAKernels);
    registerParsers();
}

void LayerLoader::registerParsers()
{
    using namespace MVCNN;

    parserMap_.reserve(SoftwareLayerParams::SoftwareLayerParams_MAX);

    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PermuteParams, &parse<PermuteParser>);
    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_SoftmaxParams, &parse<SoftmaxParser>);
    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PostOpsParams, &parse<PostOpsParser>);
    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_CustomLayerCppParams, &parse<CustomLayerCppParser>);
    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PermuteNDParams, &parse<PermuteNDParser>);
}

bool LayerLoader::parseUPALayer(const MVCNN::UPALayerTask *task, Layer *layer, LayerParser &lp) {
    // This is good enough for IMDemo, with multiple inferences though it probably needs to have some sort of
    // per-parsed-inference state... Not sure how to even get that
    static uint count = 0;
    auto success = lp.parse(task, layer);
    UNUSED(count);

    if (success) {
        // Default 0 == no limit
        if (task->maxShaves() == 0)
            layer->maxShaves = NN_MAX_UPA_SHAVE_POOL_SIZE;
        else
            layer->maxShaves = std::min(task->maxShaves(), (uint8_t)NN_MAX_UPA_SHAVE_POOL_SIZE);
    }

    if (success) {
        // nnLog(MVLOG_INFO, "Loaded UPA NN Layer %p", layer);
        nnLog(MVLOG_PERF, "Parsed Layer %d: `%s` at %p", count++,
              EnumNameSoftwareLayerParams(task->softLayerParams_type()), layer);
    } else {
        nnLog(MVLOG_ERROR, "Failed to parse blob");
    }

    return success;
}

bool LayerLoader::parseUPALayer(const MVCNN::UPALayerTask *task, Layer *layer) {
    auto &loader = LayerLoader::instance();
    auto it = loader.parserMap_.find(task->softLayerParams_type());

    if (it == loader.parserMap_.end())
    {
        nnLog(MVLOG_ERROR, "Cannot find parser function for layer type %u", task->softLayerParams_type());
        return false;
    }

    auto parserFunc = it->second;
    bool success = parserFunc(task, layer);

    if (success)
    {
        cache::flush(*layer);
        cache::flush(layer->params.inputs);
        cache::flush(layer->params.outputs);
    }

    return success;
}

void LayerLoader::loadElf(const uint8_t *elfAddr, SoftKernel &kernel) {
    const Elf32_Ehdr *elfHeader = reinterpret_cast<const Elf32_Ehdr *>(elfAddr);

    // Make sure this is a valid ELF header
    if (elfHeader->e_ident[0] != 0x7F || elfHeader->e_ident[1] != 'E' || elfHeader->e_ident[2] != 'L' ||
        elfHeader->e_ident[3] != 'F') {
        assert(false && "Failed to load unsupported ELF file");
    }

    // Reading section headers table offset
    const uint8_t *phAddr = elfAddr + elfHeader->e_shoff;

    const Elf32_Shdr *strTabSec = (const Elf32_Shdr *)(phAddr + (sizeof(Elf32_Shdr) * elfHeader->e_shstrndx));
    const char *strTab = (const char *)elfAddr + strTabSec->sh_offset;

    // Parse section headers:
    for (int secHdr = 0; secHdr < elfHeader->e_shnum; secHdr++) {
        const Elf32_Shdr *elfSecHeader = (const Elf32_Shdr *)(phAddr + sizeof(Elf32_Shdr) * secHdr);
        const void *srcAddr = (const void *)(elfAddr + elfSecHeader->sh_offset);
        uint32_t secSize = elfSecHeader->sh_size;

        // Only load PROGBITS sections
        // Our generated ELF files only have two sections - 1 code and 1 data
        if ((elfSecHeader->sh_type == SHT_PROGBITS) && (secSize > 0)) {
            // Executable (code) section
            if (elfSecHeader->sh_flags & SHF_EXECINSTR) {
                // nnLog(MVLOG_INFO, "    Setting code base address to %p", (uint32_t)srcAddr - (uint32_t)elfAddr);
                assert(kernel.codeBaseAddress == nullptr && "Expected only one code section");
                kernel.allocCodeSpace(secSize);

                DmaAlLeon dma;
                dma.start(srcAddr, kernel.codeBaseAddress, secSize);
                dma.wait();
            }
            // Writable (data) section
            else if (elfSecHeader->sh_flags & SHF_WRITE) {
                if (strcmp(strTab + elfSecHeader->sh_name, ".dyn.data") != 0) {
                    nnLog(MVLOG_INFO, "Ignoring section named %s\n", (strTab + elfSecHeader->sh_name));
                    continue;
                }

                assert(kernel.dataBaseAddress == nullptr && "Expected only one data section");
                kernel.dataBaseAddress = const_cast<void *>(srcAddr);
                kernel.dataSize = secSize;
            }
        }
    }

    assert(kernel.codeBaseAddress != nullptr && kernel.codeSize > 0);
    assert(kernel.dataBaseAddress != nullptr && kernel.dataSize > 0);

    kernel.kernelEntry = (shaveKernelEntry)SVU_NN_KERNEL_ENTRY;
}

} // namespace shave_lib
} // namespace nn
