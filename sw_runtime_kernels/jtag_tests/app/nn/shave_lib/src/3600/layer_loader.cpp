/*
* {% copyright %}
*/
#include "layer_loader.h"
#include "sw_shave_dispatcher.h"
#include "sw_shave_lib_common.h"

#include <sw_nn_runtime_types_3600.h>
#include <elf.h>
#include <dma_leon.h>

#include "layers/parser_argmax.h"
#include "layers/parser_pooling.h"
#include "layers/parser_conv.h"
#include "layers/parser_correlation.h"
#include "layers/parser_ctcdecoder.h"
#include "layers/parser_custom_ocl.h"
#include "layers/parser_deconv.h"
#include "layers/parser_detout.h"
#include "layers/parser_dummy.h"
#include "layers/parser_edsl.h"
#include "layers/parser_eltwise.h"
#include "layers/parser_fakequantize.h"
#include "layers/parser_fully_connected.h"
#include "layers/parser_grn.h"
#include "layers/parser_interp.h"
#include "layers/parser_mvn.h"
#include "layers/parser_norm.h"
#include "layers/parser_normalize.h"
#include "layers/parser_passthrough.h"
#include "layers/parser_permute.h"
#include "layers/parser_priorbox.h"
#include "layers/parser_proposal.h"
#include "layers/parser_psroipooling.h"
#include "layers/parser_quantizer.h"
#include "layers/parser_regionyolo.h"
#include "layers/parser_reorgyolo.h"
#include "layers/parser_resample.h"
#include "layers/parser_reshape.h"
#include "layers/parser_gather.h"
#include "layers/parser_roipooling.h"
#include "layers/parser_softmax.h"
#include "layers/parser_negative.h"
#include "layers/parser_st.h"
#include "layers/parser_tile.h"
#include "layers/parser_postops.h"
#include "layers/parser_convert.h"
#include "layers/parser_custom_cpp.h"
#include "layers/svuSLKernels_EP.h"
#include "layers/parser_pad.h"
#include "layers/parser_gatherelements.h"
#include "layers/parser_interpolate.h"
#include "layers/parser_ctc_greedy_decoder_seq_len.h"
#include "layers/parser_spacetodepth.h"
#include "layers/parser_depthtospace.h"
#include "layers/parser_strided_slice.h"
#include "layers/parser_lstm_cell.h"
#include "layers/parser_scatter_elements_update.h"
#include "layers/svuSLKernels_EP.h"
#include "layers/parser_reversesequence.h"
#include "layers/parser_sw_conv.h"
#include "layers/parser_gathernd.h"

#include <assert.h>
#include <mvLog.h>
#include <nn_cache.h>



namespace nn {
namespace shave_lib {

SoftParams::~SoftParams() { delete layerParams; }

LayerLoader &LayerLoader::instance()
{
    static LayerLoader singleton;
    return singleton;
}

LayerLoader::LayerLoader() :
    builtinUPAKernels(),
    parserMap_()
{
    //loadElf(&svuSLKernels_Base, builtinUPAKernels);
    registerParsers();
}

void LayerLoader::registerParsers()
{
    parserMap_.reserve(64);

    using namespace MVCNN;
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_DummyParams, &parse<DummyParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_DetectionOutputParams, &parse<DetectionOutputParser>);
//    // SoftwareLayerParams_FlattenParams
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_InterpParams, &parse<InterpParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_NormalizeParams, &parse<NormalizeParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PermuteParams, &parse<PermuteParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PriorboxParams, &parse<PriorboxParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ProposalParams, &parse<ProposalParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_RegionYOLOParams, &parse<RegionYoloParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ReorgYOLOParams, &parse<ReorgYoloParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ReshapeParams, &parse<ReshapeParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_SoftmaxParams, &parse<SoftmaxParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_CustomLayerOclParams, &parse<CustomLayerOclParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PassthroughParams, &parse<PassthroughParser>);
//    // SoftwareLayerParams_LayerRecordParams
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ROIPoolingParams, &parse<ROIPoolingParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_QuantizeParams, &parse<QuantizerParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ArgMaxParams, &parse<ArgMaxParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_NormParams, &parse<NormParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_EltwiseParams, &parse<EltwiseParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ResampleParams, &parse<ResampleParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_CorrelationParams, &parse<CorrelationParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_MVNParams, &parse<MVNParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_GRNParams, &parse<GRNParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_CTCDecoderParams, &parse<CTCDecoderParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_SpatialTransformParams, &parse<SpatialTransformParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_FakeQuantizeParams, &parse<FakeQuantizeParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PoolingParams, &parse<PoolingParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_EdslParams, &parse<EdslParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_TileParams, &parse<TileParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PSROIPoolingParams, &parse<PSROIPoolingParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_DeconvolutionParams, &parse<DeconvolutionParser>);
//    // SoftwareLayerParams_UnaryOpParams
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ConvolutionParams, &parse<ConvolutionParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_GatherParams, &parse<GatherParser>);
    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PostOpsParams, &parse<PostOpsParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_NegativeParams, &parse<NegativeParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ConvertParams, &parse<ConvertParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_CustomLayerCppParams, &parse<CustomLayerCppParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PermuteNDParams, &parse<PermuteNDParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_PadParams, &parse<PadParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_InterpolateParams, &parse<InterpolateParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_CTCGreedyDecoderSeqLenParams, &parse<CTCGreedyDecoderSeqLenParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_SpaceToDepthParams, &parse<SpaceToDepthParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_DepthToSpaceParams, &parse<DepthToSpaceParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_GatherElementsParams, &parse<GatherElementsParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ReversesequenceParams, &parse<ReversesequenceParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_LSTMCellParams, &parse<LSTMCellParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_GatherNDParams, &parse<GatherNDParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_StridedSliceParams, &parse<StridedSliceParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_FullyConnectedParams, &parse<FullyConnectedParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_SWConvolutionParams, &parse<SWConvolutionParser>);
//    parserMap_.emplace(SoftwareLayerParams::SoftwareLayerParams_ScatterElementsUpdateParams, &parse<ScatterElementsUpdateParser>);
}

bool LayerLoader::parseUPALayer(const MVCNN::UPALayerTask *task, Layer *layer, LayerParser &lp) {
    // This is good enough for IMDemo, with multiple inferences though it probably needs to have some sort of
    // per-parsed-inference state... Not sure how to even get that
    static uint count = 0;
    auto success = lp.parse(task, layer);
    UNUSED(count);

    if (success) {
        // Default 0 == no limit
        if (task->maxShaves() == 0) {
            layer->maxShaves = NN_MAX_UPA_SHAVE_POOL_SIZE;
        }else {
            layer->maxShaves = std::min(task->maxShaves(), (uint8_t) NN_MAX_UPA_SHAVE_POOL_SIZE);
        }
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
#if 0
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
#endif
} // namespace shave_lib
} // namespace nn
