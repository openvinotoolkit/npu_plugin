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

#include "llvm/Support/Debug.h"  // Alex

using namespace vpux;

//
// ConfigureBarrierOp
//

/*
// Alex
void vpux::VPUIPRegMapped::ConfigureBarrierOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, int64_t id)
{ build(builder, state, vpux::VPUIPRegMapped::BarrierType::get(builder.getContext()), id, mlir::ValueRange{},
          mlir::ValueRange{});
}
*/

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
// VPUIPRegMapped::BlobWriter::SpecificTask
// vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize(VPUIPRegMapped::BlobWriter& writer) {
/// void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize() {
void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize(std::vector<char>& buffer) {
    /*
    const auto barrier = writer.createBarrier(this->barrier(), this->id());

    MVCNN::BarrierConfigurationTaskBuilder subBuilder(writer);
    subBuilder.add_target(barrier);
    const auto subTask = subBuilder.Finish();

    MVCNN::ControllerTaskBuilder builder(writer);
    builder.add_task_type(MVCNN::ControllerSubTask_BarrierConfigurationTask);
    builder.add_task(subTask.Union());

    return {builder.Finish().Union(), MVCNN::SpecificTask_ControllerTask};
    */

    // printf("Alex: Entered void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize()\n");
    llvm::dbgs() << "Alex: Entered void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize()\n";

    // See https://mlir.llvm.org/doxygen/classmlir_1_1Identifier.html
    llvm::dbgs() << "  idAttrName = " << idAttrName().str() << "\n";
    // See similar https://mlir.llvm.org/doxygen/classmlir_1_1BoolAttr.html
    llvm::dbgs() << "  idAttr = " << idAttr().getValue() << "\n";

    // llvm::dbgs() << "  next_same_id = " << next_same_id().getValue() << "\n";
    llvm::dbgs() << "  next_same_id = " << next_same_id() << "\n";

    // printf("Alex: Exiting void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize()\n");
    llvm::dbgs() << "Alex: Exiting void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize()\n";

    //(void)buffer;
    /*
    for (int i = 0; i < 7; i++) {
        buffer.push_back(i);
    }
    */

    /*
    // unsigned char real_id_
    unsigned char real_id_ = 0;
    buffer.push_back(real_id_);

    // short next_same_id_;
    short next_same_id_ = next_same_id();
    buffer.push_back(next_same_id_ & 0xFF);
    buffer.push_back(next_same_id_ >> 8);

    // unsigned short producer_count_;
    buffer.push_back(0);
    buffer.push_back(0);

    // unsigned short consumer_count_;
    buffer.push_back(0);
    buffer.push_back(0);
    */

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
