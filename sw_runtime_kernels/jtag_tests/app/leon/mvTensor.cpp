// {% copyright %}

#include "mvTensor.h"
#include "mvTensor_cpp.h"

#include <stdio.h>
#include <mvTensorDebug.h>

#include "mvTensorUtil.h"
// #include "mvTensorDma.h"
#include "mvTensorOutputStream.h"
// #include "mvTensorInternal.h"

//#include "mvHwOperation.h"

//#include "cache/mvTensorLeonCacheController.h"
//#include "cache/mvTensorShaveCacheController.h"

#define DEBUG_KERNELS 0

#if DEBUG_KERNELS
#define MVT_DPRINTF(...) printf(__VA_ARGS__)
#else
#define MVT_DPRINTF(...)
#endif

//extern DynamicContext_t MODULE_DATA(mvTensor);

typedef const char* (* SubOpNames)(int);

namespace
{
    const char empty[] = "UNKNOWN";

std::map<eltwise::EltwiseSubOp, std::string> eltwiseSubOpStr = {
    {eltwise::EltwiseSubOp::sum, "sum"},
    {eltwise::EltwiseSubOp::prod, "prod"},
    {eltwise::EltwiseSubOp::max, "max"},
    {eltwise::EltwiseSubOp::div, "div"},
    {eltwise::EltwiseSubOp::min, "min"},
    {eltwise::EltwiseSubOp::sqdiff, "sqdiff"},
    {eltwise::EltwiseSubOp::compareeq, "compareeq"},
    {eltwise::EltwiseSubOp::comparene, "comparene"},
    {eltwise::EltwiseSubOp::comparegt, "comparegt"},
    {eltwise::EltwiseSubOp::comparege, "comparege"},
    {eltwise::EltwiseSubOp::comparelt, "comparelt"},
    {eltwise::EltwiseSubOp::comparele, "comparele"},
    {eltwise::EltwiseSubOp::logicalnot, "logicalnot"},
    {eltwise::EltwiseSubOp::logicaland, "logicaland"},
    {eltwise::EltwiseSubOp::logicalor, "logicalor"},
    {eltwise::EltwiseSubOp::logicalxor, "logicalxor"},
    {eltwise::EltwiseSubOp::pow, "pow"},
    {eltwise::EltwiseSubOp::floormod, "floormod"},
    {eltwise::EltwiseSubOp::select, "select"},
};

std::map<SubOpFamilyOffset, SubOpNames> subOpGetNames = {
        {postopFamily, getPostOpSubOpName},
        {eltwiseFamily, getEltwiseSubOpName},
};

}

const char * getEltwiseSubOpName(int subOp) {
    if (eltwiseSubOpStr.count(static_cast<eltwise::EltwiseSubOp>(subOp)) == 0) {
        return empty;
    } else {
        return eltwiseSubOpStr[static_cast<eltwise::EltwiseSubOp>(subOp)].c_str();
    }
}

const char * getPostOpSubOpName(int subOp) {
    return MVCNN::EnumNamePostOpsNestedParams(static_cast<MVCNN::PostOpsNestedParams>(subOp));
}

const char* getOpName(t_MvTensorOpType op)
{
    if ((int)op < (int)MVCNN::SoftwareLayerParams_MIN) {
        return empty;
    } else if ((int)op > (int)MVCNN::SoftwareLayerParams_MAX) {
        int subop = decodeLocalSubOpCode(op);
        int family = decodeLocalFamily(op);
        if (subOpGetNames.count(static_cast<SubOpFamilyOffset>(family)) == 0) {
            return empty;
        } else {
            return (subOpGetNames[static_cast<SubOpFamilyOffset>(family)])(subop);
        }
    } else {
        return MVCNN::EnumNamesSoftwareLayerParams()[op];
    }
}

namespace mv
{
    namespace tensor
    {
        Processor::Processor(
            const t_MvTensorMyriadResources &myriadResources,
            const t_MvTensorDebugInfo *debugInfo) :
            myriadResources_(myriadResources),
            debugStream_(debugInfo ? debugInfo->debugMsg : nullptr, MV_TENSOR_DBG_MSG_SIZE, OutputStream::Overwrite),
//            dmaConfig_{ MV_TENSOR_DMA_PRIORITY, static_cast<u32>(myriadResources.dmaLinkAgent) },
  //          dmaUser_(dmaConfig_, myriadResources.dmaTransactions),
//            leonCache_(),
//            shaveCache_(myriadResources.dataPartitionNo),
            resources_{ debugStream_},
            prevShaves_(0)
        {
        }
    }
}

void mvTensorInit(unsigned int /*start_shave*/, unsigned int /*number_of_shaves*/, unsigned int /*number_of_cmxslices*/,
                  unsigned int /*start_cnn*/, unsigned int /*number_of_cnns*/, bool /*needTurnOnShaves*/, bool /*isLowPower*/) {
//    saveShaveDynContextData(start_shave, number_of_shaves, &MODULE_DATA(mvTensor));
//
//#ifdef MA2480
//    if(!isLowPower)
//    {
//        for (unsigned int block = start_cnn; block < number_of_cnns; ++block) {
//            hw::powerBlock(block, true);
//        }
//    }
//#else
//    (void) start_cnn;
//    (void) number_of_cnns;
//#endif
//
//    if (number_of_shaves == 0)
//    {
//        return;
//    }
//
//    swcShaveUnit_t svuList[MVTENSOR_MAX_SHAVES] = {};
//    for (unsigned int i = 0; i < number_of_shaves; ++i)
//        svuList[i] = start_shave + i;
//
//    if (needTurnOnShaves || isLowPower)
//    {
//        powerShaves(start_shave, number_of_shaves);
//    }
//
//    openShaves(svuList, number_of_shaves);
//
//    for (unsigned int i = start_shave; i < start_shave + number_of_shaves; ++i)
//        setupShaveDynApp(i);
//
//    if(isLowPower) turnOffShaves(start_shave,number_of_shaves);
//
//
//#ifndef MVNCI_DISABLE_POOL_CLEAR
//
//    //TODO: Is it really needed to set all 0? DMA could be better in this case
//    memset(reinterpret_cast<unsigned char*>(CMX_BASE_ADR) + start_shave * CMX_SLICE_SIZE, 0, number_of_cmxslices * CMX_SLICE_SIZE);
//    cache::LeonController().writeback(
//      reinterpret_cast<unsigned char*>(CMX_BASE_ADR) + start_shave * CMX_SLICE_SIZE,
//      reinterpret_cast<unsigned char*>(CMX_BASE_ADR) + start_shave * CMX_SLICE_SIZE + number_of_cmxslices * CMX_SLICE_SIZE);
//#else
//    UNUSED(number_of_cmxslices);
//#endif
}


void mvTensorClose(unsigned int /*start_shave*/, unsigned int /*number_of_shaves*/,
                   unsigned int /*start_cnn*/, unsigned int /*number_of_cnns*/, bool /*needTurnOffShaves*/, bool /*isLowPower*/)
{
//#ifdef MA2480
//    if(!isLowPower)
//    {
//        for (unsigned int block = start_cnn; block < number_of_cnns; ++block)
//            hw::powerBlock(block, false);
//    }
//#else
//    (void) start_cnn;
//    (void) number_of_cnns;
//#endif
//
//    if (number_of_shaves == 0)
//    {
//        return;
//    }
//
//    for (unsigned int i = start_shave; i < start_shave + number_of_shaves; ++i)
//        cleanShaveDynApp(i);
//
//    if (needTurnOffShaves && !isLowPower)
//    {
//        turnOffShaves(start_shave, number_of_shaves);
//    }
//
//    swcShaveUnit_t svuList[MVTENSOR_MAX_SHAVES] = {};
//    for (unsigned int i = 0; i < number_of_shaves; ++i)
//        svuList[i] = start_shave + i;
//
//    closeShaves(svuList, number_of_shaves);
}
