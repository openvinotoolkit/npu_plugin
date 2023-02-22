//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <functional>
#include <map>

using namespace vpux;

namespace {

using IsSpecificOpFunc = std::function<bool(mlir::Operation*)>;
using RemainderHandlerFunc = std::function<std::string(mlir::Operation*)>;

class SpecificCategoryCounter {
public:
    using CounterPtr = std::shared_ptr<SpecificCategoryCounter>;
    using CountersVec = std::vector<CounterPtr>;

    SpecificCategoryCounter(const std::string& category, IsSpecificOpFunc predicate,
                            const CountersVec& nestedCounters = {},
                            Optional<RemainderHandlerFunc> maybeRemainderHandler = None)
            : category_(category),
              predicate_(std::move(predicate)),
              nestedCounters_(nestedCounters),
              count_(0),
              maybeRemainderHandler_(std::move(maybeRemainderHandler)) {
    }

    bool count(mlir::Operation* op) {
        if (predicate_(op)) {
            ++count_;
            bool counted = false;
            for (auto& nestedCounter : nestedCounters_) {
                counted |= nestedCounter->count(op);
            }
            if (!counted && maybeRemainderHandler_.hasValue()) {
                ++remainderCounter[maybeRemainderHandler_.getValue()(op)];
            }
            return true;
        }
        return false;
    }

    void printStatistics(vpux::Logger log) {
        log.info("{0} - {1} ops", category_, count_);
        log = log.nest();
        for (auto& nestedCounter : nestedCounters_) {
            if (nestedCounter->count_) {
                nestedCounter->printStatistics(log);
            }
        }
        for (auto& remainder : remainderCounter) {
            log.info("{0} - {1} ops", remainder.first, remainder.second);
        }
        log.unnest();
    }

private:
    std::string category_;
    IsSpecificOpFunc predicate_;
    CountersVec nestedCounters_;
    size_t count_;
    Optional<RemainderHandlerFunc> maybeRemainderHandler_;
    std::map<std::string, size_t> remainderCounter{};
};

using CountersVec = SpecificCategoryCounter::CountersVec;

bool isLocContainsStr(mlir::Location loc, const std::string& substr) {
    if (auto nameLoc = loc.dyn_cast<mlir::NameLoc>()) {
        return nameLoc.getName().str().find(substr) != std::string::npos;
    }
    return false;
}

CountersVec getSpillCounter(const std::string& category) {
    if (category != "DDR2CMX" && category != "CMX2DDR") {
        return {};
    }
    const auto spillCategory = category == "DDR2CMX" ? "SPILL_READ" : "SPILL_WRITE";
    const auto targetSubstr = category == "DDR2CMX" ? SPILL_READ_OP_NAME_SUFFIX : SPILL_WRITE_OP_NAME_SUFFIX;
    return {std::make_shared<SpecificCategoryCounter>(spillCategory, [=](mlir::Operation* op) {
        if (auto fusedLoc = op->getLoc().dyn_cast<mlir::FusedLoc>()) {
            const auto locations = fusedLoc.getLocations();
            for (auto it = std::rbegin(locations); it != std::rend(locations); ++it) {
                const auto loc = *it;
                // analyzing only spill related locs
                if (isLocContainsStr(loc, SPILL_READ_OP_NAME_SUFFIX) ||
                    isLocContainsStr(loc, SPILL_WRITE_OP_NAME_SUFFIX)) {
                    // return result on first from end matched loc
                    return isLocContainsStr(loc, targetSubstr);
                }
            }
        }
        return false;
    })};
}

std::string getProfSuffix(const std::string& profCategory) {
    return profCategory + PROFILING_CMX_2_DDR_OP_NAME;
}

CountersVec getDMANestedCounters() {
    using VPU::MemoryKind;
    const auto checkArgMemSpace = [](mlir::Operation* op, unsigned idx, VPU::MemoryKind memKind) {
        return op->getOperand(idx).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == memKind;
    };
    std::vector<std::pair<std::string, MemoryKind>> configurations = {{
                                                                              "CMX",
                                                                              MemoryKind::CMX_NN,
                                                                      },
                                                                      {"DDR", MemoryKind::DDR},
                                                                      {"REG", MemoryKind::Register}};

    CountersVec counters;
    for (const auto& inputConf : configurations) {
        for (const auto& outputConf : configurations) {
            const auto category = inputConf.first + "2" + outputConf.first;
            counters.push_back(std::make_shared<SpecificCategoryCounter>(
                    category,
                    [=](auto op) {
                        return checkArgMemSpace(op, 0, inputConf.second) && checkArgMemSpace(op, 1, outputConf.second);
                    },
                    getSpillCounter(category)));
        }
    }
    std::vector<std::pair<std::string, IsSpecificOpFunc>> profCounters = {
            {"PROF_DMA",
             [](auto op) {
                 for (const auto dmaSubstr :
                      {PROFILING_DMA_BEGIN_SUFFIX, PROFILING_DMA_TASK_BEGIN_SUFFIX, PROFILING_DMA_TASK_END_SUFFIX}) {
                     if (isLocContainsStr(op->getLoc(), dmaSubstr)) {
                         return true;
                     }
                 }
                 return false;
             }},
            {"DMA PROF BUFFER TO DDR",
             [](auto op) {
                 return isLocContainsStr(op->getLoc(), getProfSuffix("dma"));
             }},
            {"DPU PROF BUFFER TO DDR",
             [](mlir::Operation* op) {
                 const auto DPU_PROF_SUBSTR = getProfSuffix("dpu");
                 if (isLocContainsStr(op->getLoc(), DPU_PROF_SUBSTR)) {
                     return true;
                 }
                 if (auto fusedLoc = op->getLoc().dyn_cast<mlir::FusedLoc>()) {
                     for (auto loc : fusedLoc.getLocations()) {
                         if (isLocContainsStr(loc, DPU_PROF_SUBSTR)) {
                             return true;
                         }
                     }
                 }
                 return false;
             }},
            {"ACT PROF BUFFER TO DDR", [](auto op) {
                 return isLocContainsStr(op->getLoc(), getProfSuffix("actshave"));
             }}};
    for (const auto& p : profCounters) {
        counters.push_back(std::make_shared<SpecificCategoryCounter>(p.first, p.second));
    }

    return counters;
}

template <class... Args, typename = typename std::enable_if<sizeof...(Args) == 0>::type>
void populateDMACounters(CountersVec&) {
}

template <class DMAType, class... DMATypeArgs>
void populateDMACounters(CountersVec& counters) {
    RemainderHandlerFunc dmaRemainderHandler = [](mlir::Operation*) {
        return "Unknown memory space DMA";
    };
    counters.push_back(std::make_shared<SpecificCategoryCounter>(
            DMAType::getOperationName().str(),
            [](mlir::Operation* op) {
                return mlir::isa<DMAType>(op);
            },
            getDMANestedCounters(), dmaRemainderHandler));
    populateDMACounters<DMATypeArgs...>(counters);
}

CountersVec getDMACounters() {
    CountersVec dmaCounters;
    using namespace VPUIP;
    populateDMACounters<NNDMAOp, CompressedDMAOp, DepthToSpaceDMAOp, PermuteDMAOp, ExpandDMAOp>(dmaCounters);
    return dmaCounters;
}

SpecificCategoryCounter populateCounters() {
    RemainderHandlerFunc remaindedOpHandler = [](mlir::Operation* op) {
        return op->getName().getIdentifier().str();
    };
    SpecificCategoryCounter topLevelCounter(
            "VPUIP Tasks",
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::TaskOpInterface>(op);
            },
            getDMACounters(), remaindedOpHandler);
    return topLevelCounter;
}

class CompressionRateCounter {
public:
    CompressionRateCounter()
            : totalConstantsBeforeCompression_(0),
              totalCompressedConstants_(0),
              totalCompressedConstantsBeforeCompression_(0),
              totalUncompressedConstants_(0) {
    }

    std::pair<uint64_t, uint64_t> handleDMA(mlir::Value dmaInput, mlir::Value dmaOutput) {
        auto cstOp = dmaInput.getDefiningOp<Const::DeclareOp>();
        if (!cstOp) {
            return {0, 0};
        }
        if (seenConstants_.find(cstOp) != seenConstants_.end()) {
            return {0, 0};
        }
        if (!dmaOutput.getDefiningOp<VPURT::DeclareBufferOp>()) {
            return {0, 0};
        }

        seenConstants_.insert(cstOp);

        const auto inputSize = getTotalSize(dmaInput).count();
        const auto outputSize = getTotalSize(dmaOutput).count();
        return std::make_pair(inputSize, outputSize);
    }

    void count(mlir::Operation* op) {
        if (auto dmaOp = mlir::dyn_cast<VPUIP::CompressedDMAOp>(op)) {
            const auto inputOutputSize = handleDMA(dmaOp.input(), dmaOp.output_buff());

            totalConstantsBeforeCompression_ += inputOutputSize.second;
            totalCompressedConstantsBeforeCompression_ += inputOutputSize.second;
            totalCompressedConstants_ += inputOutputSize.first;
        } else if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(op)) {
            const auto inputOutputSize = handleDMA(dmaOp.input(), dmaOp.output_buff());

            totalConstantsBeforeCompression_ += inputOutputSize.second;
            totalUncompressedConstants_ += inputOutputSize.first;
        }
    }

    void printStatistics(vpux::Logger log) {
        uint64_t totalConstantsAfterCompression_ = totalCompressedConstants_ + totalUncompressedConstants_;
        double compressedPercentage = (double)totalCompressedConstants_ / totalConstantsAfterCompression_ * 100;
        double uncompressedPercentage = (double)totalUncompressedConstants_ / totalConstantsAfterCompression_ * 100;
        double qualifiedCompressionRate =
                (double)totalCompressedConstants_ / totalCompressedConstantsBeforeCompression_ * 100;
        double totalCompressionRate = (double)totalConstantsAfterCompression_ / totalConstantsBeforeCompression_ * 100;

        log = log.nest();
        log.info("Constants size before compression: {0} bytes", totalConstantsBeforeCompression_);
        log.info("Constants size after compression: {0} bytes", totalConstantsAfterCompression_);
        log.info("Constants that were compressed: {0} bytes ({1}% of total)", totalCompressedConstants_,
                 compressedPercentage);
        log.info("Constants that couldn't be compressed: {0} bytes ({1}% of total)", totalUncompressedConstants_,
                 uncompressedPercentage);
        log.info("Compression rate of compressed constants: {0}%", qualifiedCompressionRate);
        log.info("Total compression rate: {0}%", totalCompressionRate);
        log.unnest();
    }

private:
    uint64_t totalConstantsBeforeCompression_;
    uint64_t totalCompressedConstants_;
    uint64_t totalCompressedConstantsBeforeCompression_;
    uint64_t totalUncompressedConstants_;
    std::set<Const::DeclareOp> seenConstants_{};
};

//
// DumpStatisticsOfTaskOpsPass
//

class DumpStatisticsOfTaskOpsPass final : public VPUIP::DumpStatisticsOfTaskOpsPassBase<DumpStatisticsOfTaskOpsPass> {
public:
    explicit DumpStatisticsOfTaskOpsPass(bool printWeightsCompressBTC, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        _printWeightsCompressBTC = printWeightsCompressBTC;
    }

private:
    void safeRunOnFunc() final;
    bool _printWeightsCompressBTC;
};

void DumpStatisticsOfTaskOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    llvm::DenseSet<mlir::OperationName> dpuOperations{
            mlir::OperationName(VPUIP::ConvolutionUPAOp::getOperationName(), &ctx),
            mlir::OperationName(VPUIP::PoolingUPAOp::getOperationName(), &ctx),
            mlir::OperationName(VPUIP::EltwiseUPAOp::getOperationName(), &ctx)};

    auto opStatisticsCounter = populateCounters();
    CompressionRateCounter compressionCounter;
    func->walk([&](mlir::Operation* op) {
        opStatisticsCounter.count(op);
        compressionCounter.count(op);
        if (VPU::getCompilationMode(func) == VPU::CompilationMode::ReferenceSW) {
            return;
        }
        const auto opName = op->getName();
        if (dpuOperations.contains(opName)) {
            _log.nest().warning("'{0}' was not converted to 'VPUIP.NCETask'", opName);
        }
    });
    _log.info("VPUIP tasks statistics:");
    opStatisticsCounter.printStatistics(_log);

    if (_printWeightsCompressBTC) {
        _log.info("Weights compression statistics:");
        compressionCounter.printStatistics(_log);
    } else {
        _log.info("Weights compression statistics: BitCompactor is disabled");
    }
}

}  // namespace

//
// createDumpStatisticsOfTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDumpStatisticsOfTaskOpsPass(bool printWeightsCompressBTC, Logger log) {
    return std::make_unique<DumpStatisticsOfTaskOpsPass>(printWeightsCompressBTC, log);
}
