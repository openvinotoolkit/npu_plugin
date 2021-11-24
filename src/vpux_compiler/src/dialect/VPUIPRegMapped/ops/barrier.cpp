//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

#include "vpux/utils/core/format.hpp"

#include "llvm/Support/Debug.h"

using namespace vpux;

//
// ConfigureBarrierOp
//

/*
// From file .../nn_inference_runtime_types.h
//  From meeting TimiCompiler from Sep 15 2021
    struct BarrierCfg
    {
        unsigned char real_id_;
        short next_same_id_;
        unsigned short producer_count_;
        unsigned short consumer_count_;

        BarrierCfg() :
            real_id_(255),
            next_same_id_(-1),
            producer_count_(0),
            consumer_count_(0)
        {
        }
    };
    //StructAttr VPUIPRM_BarrierCfgAttr ...

// Total size: 1 + 2 + 2 + 2 = 7 bytes. But normally, sizeof(struct BarrierCfg) = 8.
*/
void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize(std::vector<char>& buffer) {
    llvm::dbgs() << "Alex: Entered void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize()\n";

    llvm::dbgs() << "  idAttrName = " << idAttrName().str() << "\n";
    llvm::dbgs() << "  idAttr = " << idAttr().getValue() << "\n";
    llvm::dbgs() << "  next_same_id = " << next_same_id() << "\n";
    llvm::dbgs() << "Alex: Exiting void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize()\n";

    struct __attribute__((packed)) BarrierCfg {
        unsigned char real_id_;
        short next_same_id_;
        unsigned short producer_count_;
        unsigned short consumer_count_;

        BarrierCfg(): real_id_(255), next_same_id_(-1), producer_count_(0), consumer_count_(0) {
        }
    } tmp;

    tmp.next_same_id_ = next_same_id();

    char* ptrCharTmp = (char*)(&tmp);
    for (long unsigned i = 0; i < sizeof(struct BarrierCfg); i++) {
        buffer.push_back(*(ptrCharTmp + i));
    }
}
