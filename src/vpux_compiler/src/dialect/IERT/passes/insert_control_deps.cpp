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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/dialect/IERT/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops_interfaces.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/range.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"

#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/control_edge_set.hpp"
#include "vpux/compiler/core/operation_precedence_dag.hpp"

using namespace vpux;

namespace {

//
// InsertControlDepsPass
//

class InsertControlDepsPass final : public IERT::InsertControlDepsBase<InsertControlDepsPass> {
public:
    explicit InsertControlDepsPass(IERT::AttrCreateFunc memSpaceCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    Byte calculateOffset(mlir::Value val);

private:
    IERT::AttrCreateFunc _memSpaceCb;
    mlir::Attribute _memSpace;
};

InsertControlDepsPass::InsertControlDepsPass(IERT::AttrCreateFunc memSpaceCb, Logger log)
        : _memSpaceCb(std::move(memSpaceCb)) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult InsertControlDepsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    _memSpace = _memSpaceCb(ctx, memSpaceName.getValue());

    if (_memSpace == nullptr) {
        return mlir::failure();
    }

    return mlir::success();
}

// TODO:
// This function was copied from ViewLikeRewrite class.
// Create some common utility function for this
Byte InsertControlDepsPass::calculateOffset(mlir::Value val) {
    Byte offset(0);

    if (auto bufferOp = mlir::dyn_cast_or_null<IERT::StaticAllocOp>(val.getDefiningOp())) {
        offset += Byte(bufferOp.offset());
    } else if (auto subViewOp = mlir::dyn_cast_or_null<IERT::SubViewOp>(val.getDefiningOp())) {
        const auto strides = getStrides(subViewOp.source());
        const auto offsets = parseIntArrayAttr<int64_t>(subViewOp.static_offsets());
        VPUX_THROW_UNLESS(strides.size() == offsets.size(), "SubView offsets '{0}' doesn't match strides '{1}'",
                          offsets, strides);

        for (auto p : zip(strides, offsets)) {
            offset += Byte(std::get<0>(p) * std::get<1>(p));
        }

        offset += calculateOffset(subViewOp.source());
    }

    return offset;
}

// TODO:
// Below defines were taken directly as they were in MCM.
// Maybe this could be refactored
typedef vpux::Scheduled_Op scheduled_op_t;
typedef std::list<scheduled_op_t> scheduled_op_list_t;
typedef vpux::Control_Edge control_edge_t;
typedef vpux::Control_Edge_Set control_edge_set_t;
typedef vpux::Control_Edge_Generator<scheduled_op_t> control_edge_generator_t;

void InsertControlDepsPass::safeRunOnFunc() {
    auto func = getFunction();
    std::cout << "InsertControlDepsPass::safeRunOnFunc start\n";
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();

    scheduled_op_list_t scheduled_ops;

    // Check all AsyncExecuteOp to build scheduled_op_list that will be used
    // by control edge generation algorithm. Gather data like operation index,
    // schedule time and produced resources - start and end address
    func.walk([&](mlir::async::ExecuteOp execOp) {
        // AsyncExecuteOp index attribute will be used as operation identifier
        auto index = depsInfo.getIndex(execOp);

        const auto timeAttr = execOp->getAttrOfType<mlir::IntegerAttr>("schedule-time");
        uint32_t time = std::numeric_limits<uint32_t>::max();
        if (timeAttr != nullptr) {
            time = checked_cast<uint32_t>(timeAttr.getValue().getZExtValue());
        }

        std::cout << " [" << index << "] - t: " << time << " " << stringifyLocation(execOp->getLoc()) << ", ";
        std::cout << "  Operations:";

        // Analyze bode of AsyncExecuteOp to get information about produced resources
        // for contained operations
        auto* bodyBlock = &execOp.body().front();
        for (auto& op : bodyBlock->getOperations()) {
            if (mlir::isa<mlir::ViewLikeOpInterface>(op) && mlir::isa<IERT::LayerOpInterface>(op)) {
                auto layer = mlir::dyn_cast<IERT::LayerOpInterface>(op);
                auto outputs = layer.getOutputs();
                std::cout << " " << op.getName().getStringRef().data();
                for (auto output : outputs) {
                    // Focus only on operations producing resources only in specific
                    // memory space
                    const auto type = output.getType().dyn_cast<mlir::MemRefType>();
                    if (type == nullptr || type.getMemorySpace() != _memSpace) {
                        continue;
                    }

                    // Store information about address space for produced by operation result
                    // for later use by control edge generation algorithm
                    int64_t offset = calculateOffset(output).count();
                    int64_t totalSize = getTypeTotalSize(output.getType().dyn_cast<mlir::MemRefType>()).count();
                    scheduled_ops.push_back(
                            scheduled_op_t(index, time, offset, offset + totalSize - 1, vpux::op_type_e::ORIGINAL_OP));

                    std::cout << "[" << offset << " - " << offset + totalSize - 1 << "] ";
                }
                std::cout << ", ";
            }
        }
        std::cout << "\n";
    });

    std::cout << "Mateusz: scheduled_ops set:\n";
    for (auto& scheduled_op : scheduled_ops) {
        std::cout << " op - " << scheduled_op.op_ << " time[" << scheduled_op.schedule_time_ << " - "
                  << scheduled_op.schedule_end_time_ << "], res[" << scheduled_op.cmx_address_start_ << " - "
                  << scheduled_op.cmx_address_end_ << "]\n";
    }

    std::cout << "Mateusz: Generate control edges for overlapping memory regions\n";

    control_edge_set_t control_edges;
    control_edge_generator_t algo;
    // Generate control edges for overlapping memory regions
    algo.generate_control_edges(scheduled_ops.begin(), scheduled_ops.end(), control_edges);

    std::cout << "Mateusz: Apply dependencies in depsInfo\n";
    std::cout << " Control edges:\n";

    // Apply dependencies from control_edges set in depsInfo and
    // later transfer this to token based dependencies between AsyncExecuteOp
    for (auto itr = control_edges.begin(); itr != control_edges.end(); ++itr) {
        std::cout << "  " << (*itr).source_name() << " -> " << (*itr).sink_name() << "\n";
        auto sourceOp = depsInfo.getExecuteOpAtIndex((*itr).source_);
        auto sinkOp = depsInfo.getExecuteOpAtIndex((*itr).sink_);
        depsInfo.addDependency(sourceOp, sinkOp);
    }
    std::cout << "Mateusz: updateTokenDependencies\n";
    depsInfo.updateTokenDependencies();
    std::cout << "InsertControlDepsPass::safeRunOnFunc end\n";
}

}  // namespace

//
// createInsertControlDeps
//

std::unique_ptr<mlir::Pass> vpux::IERT::createInsertControlDepsPass(AttrCreateFunc memSpaceCb, Logger log) {
    return std::make_unique<InsertControlDepsPass>(std::move(memSpaceCb), log);
}
