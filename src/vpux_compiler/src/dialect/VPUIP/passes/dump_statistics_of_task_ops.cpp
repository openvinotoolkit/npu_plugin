//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/profiling.hpp"

#include <functional>
#include <map>

using namespace vpux;

namespace {

using IsSpecificOpFunc = std::function<bool(mlir::Operation*)>;
using RemainderHandlerFunc = std::function<std::string(mlir::Operation*)>;
const std::string DMA_PROFILING_SPILL_CATEGORY = "CMX2DDR profiling spill";

class SpecificCategoryCounter {
public:
    using CounterPtr = std::shared_ptr<SpecificCategoryCounter>;
    using CountersVec = std::vector<CounterPtr>;

    SpecificCategoryCounter(const std::string& category, IsSpecificOpFunc predicate,
                            const CountersVec& nestedCounters = {},
                            Optional<RemainderHandlerFunc> maybeRemainderHandler = None)
            : _category(category),
              _predicate(std::move(predicate)),
              _nestedCounters(nestedCounters),
              _count(0),
              _maybeRemainderHandler(std::move(maybeRemainderHandler)) {
    }

    bool count(mlir::Operation* op) {
        if (!_predicate(op)) {
            return false;
        }

        ++_count;
        bool counted = false;
        for (auto& nestedCounter : _nestedCounters) {
            counted |= nestedCounter->count(op);
        }
        if (!counted && _maybeRemainderHandler.hasValue()) {
            ++_remainderCounter[_maybeRemainderHandler.getValue()(op)];
        }
        return true;
    }

    void printStatistics(vpux::Logger log) {
        log.info("{0} - {1} ops", _category, _count);
        log = log.nest();
        for (auto& nestedCounter : _nestedCounters) {
            if (nestedCounter->_count) {
                nestedCounter->printStatistics(log);
            }
        }
        for (auto& remainder : _remainderCounter) {
            log.info("{0} - {1} ops", remainder.first, remainder.second);
        }
        log = log.unnest();
    }

private:
    std::string _category;
    IsSpecificOpFunc _predicate;
    CountersVec _nestedCounters;
    size_t _count;
    Optional<RemainderHandlerFunc> _maybeRemainderHandler;
    std::map<std::string, size_t> _remainderCounter{};
};

using CountersVec = SpecificCategoryCounter::CountersVec;

bool isNameLocContainsStr(mlir::Location loc, const std::string& substr) {
    if (auto nameLoc = loc.dyn_cast<mlir::NameLoc>()) {
        return nameLoc.getName().str().find(substr) != std::string::npos;
    }
    return false;
}

bool isLocContainsStr(mlir::Location loc, const std::string& substr) {
    if (isNameLocContainsStr(loc, substr)) {
        return true;
    }
    if (auto fusedLoc = loc.dyn_cast<mlir::FusedLoc>()) {
        for (auto& loc : fusedLoc.getLocations()) {
            if (isNameLocContainsStr(loc, substr)) {
                return true;
            }
        }
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

void addProfilingCounters(CountersVec& counters, const std::string& category) {
    if (category == "REG2CMX") {
        counters.push_back(std::make_shared<SpecificCategoryCounter>("Profiling Timestamp DMA", [](auto op) {
            for (const auto dmaSubstr : {PROFILING_DMA_TASK_BEGIN_PREFIX, PROFILING_DMA_TASK_END_PREFIX}) {
                if (isLocContainsStr(op->getLoc(), dmaSubstr)) {
                    return true;
                }
            }
            return false;
        }));
        return;
    }
    if (category == "REG2DDR") {
        counters.push_back(std::make_shared<SpecificCategoryCounter>("Profiling workpoint", [](auto op) {
            return isLocContainsStr(op->getLoc(), PROFILING_WORKPOINT_READ_ATTR);
        }));
        return;
    }
    if (category == "CMX2DDR") {
        CountersVec nestedProfCounters;
        const static std::vector<std::pair<std::string, IsSpecificOpFunc>> profCounterCfgs = {
                {"DMA",
                 [](auto op) {
                     return isLocContainsStr(op->getLoc(), getProfSuffix("dma"));
                 }},
                {"DPU",
                 [](auto op) {
                     return isLocContainsStr(op->getLoc(), getProfSuffix("dpu"));
                 }},
                {"ActShave", [](auto op) {
                     return isLocContainsStr(op->getLoc(), getProfSuffix("actshave"));
                 }}};
        for (const auto& p : profCounterCfgs) {
            nestedProfCounters.push_back(std::make_shared<SpecificCategoryCounter>(p.first, p.second));
        }
        counters.push_back(std::make_shared<SpecificCategoryCounter>(
                "Profiling buffer management",
                [](auto op) {
                    return isLocContainsStr(op->getLoc(), PROFILING_CMX_2_DDR_OP_NAME);
                },
                nestedProfCounters));
    }
}

CountersVec getInnerCounters(const std::string& category) {
    CountersVec counters = getSpillCounter(category);
    addProfilingCounters(counters, category);
    return counters;
}

template <class DMAType>
CountersVec getDMANestedCounters() {
    using VPU::MemoryKind;

    const auto checkInputOutputMemSpace = [](mlir::Operation* op, VPU::MemoryKind inMemKind,
                                             VPU::MemoryKind outMemKind) {
        VPUX_THROW_WHEN(op == nullptr, "NULL operation provided");

        const auto checkArgMemSpace = [](mlir::Value operand, VPU::MemoryKind memKind) {
            return operand.getType().cast<vpux::NDTypeInterface>().getMemoryKind() == memKind;
        };

        if (auto dmaOp = mlir::dyn_cast<DMAType>(op)) {
            return checkArgMemSpace(dmaOp.input(), inMemKind) && checkArgMemSpace(dmaOp.output_buff(), outMemKind);
        }

        VPUX_THROW("Not supported DMA task");
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
            const auto nestedCounters = getInnerCounters(category);
            counters.push_back(std::make_shared<SpecificCategoryCounter>(
                    category,
                    [=](auto op) {
                        return checkInputOutputMemSpace(op, inputConf.second, outputConf.second);
                    },
                    nestedCounters));
        }
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
            getDMANestedCounters<DMAType>(), dmaRemainderHandler));
    populateDMACounters<DMATypeArgs...>(counters);
}

CountersVec getDMACounters() {
    CountersVec dmaCounters;
    using namespace VPUIP;
    populateDMACounters<NNDMAOp, CompressDMAOp, DecompressDMAOp, DepthToSpaceDMAOp, PermuteDMAOp, ExpandDMAOp>(
            dmaCounters);
    return dmaCounters;
}

SpecificCategoryCounter::CounterPtr getSWKernelsCounter() {
    RemainderHandlerFunc swHandler = [](mlir::Operation* op) -> std::string {
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            return swKernelOp.kernelFunction().getLeafReference().str();
        }
        return "Not SwKernel";
    };

    return std::make_shared<SpecificCategoryCounter>(
            VPUIP::SwKernelOp::getOperationName().str(),
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::SwKernelOp>(op);
            },
            CountersVec{}, swHandler);
}

SpecificCategoryCounter::CounterPtr getSparsityCounter() {
    RemainderHandlerFunc remainedOpHandler = [](mlir::Operation*) {
        return "Dense";
    };

    auto sparseInputCounter = std::make_shared<SpecificCategoryCounter>(
            "Sparse input",
            [](mlir::Operation* op) {
                if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
                    bool hasSparseInput =
                            nceOp.input_sparsity_map() != nullptr || nceOp.input_storage_element_table() != nullptr;
                    if (nceOp.task_type() == VPUIP::NCETaskType::ELTWISE) {
                        hasSparseInput |= nceOp.weights_sparsity_map() != nullptr;
                    }
                    return hasSparseInput;
                }
                return false;
            },
            CountersVec{});

    auto sparseWeightsCounter = std::make_shared<SpecificCategoryCounter>(
            "Sparse weights",
            [](mlir::Operation* op) {
                if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
                    return nceOp.weights_sparsity_map() != nullptr;
                }
                return false;
            },
            CountersVec{});

    auto sparseOutputCounter = std::make_shared<SpecificCategoryCounter>(
            "Sparse output",
            [](mlir::Operation* op) {
                if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
                    return nceOp.output_sparsity_map() != nullptr;
                }
                return false;
            },
            CountersVec{});

    CountersVec counters = {sparseInputCounter, sparseWeightsCounter, sparseOutputCounter};
    return std::make_shared<SpecificCategoryCounter>(
            "NCETask Operations",
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::NCEClusterTaskOp>(op);
            },
            counters, remainedOpHandler);
}

SpecificCategoryCounter populateCounters() {
    RemainderHandlerFunc remainedOpHandler = [](mlir::Operation* op) {
        return op->getName().getIdentifier().str();
    };

    auto counters = getDMACounters();
    counters.push_back(getSWKernelsCounter());
    counters.push_back(getSparsityCounter());

    SpecificCategoryCounter topLevelCounter(
            "VPUIP Tasks",
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::TaskOpInterface>(op);
            },
            counters, remainedOpHandler);
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
        if (auto dmaOp = mlir::dyn_cast<VPUIP::DecompressDMAOp>(op)) {
            // act_compression_size_entry() == nullptr will verify that DecompressDMAOp is not the activation
            // decompression
            if (dmaOp.act_compression_size_entry() == nullptr) {
                const auto inputOutputSize = handleDMA(dmaOp.input(), dmaOp.output_buff());

                totalConstantsBeforeCompression_ += inputOutputSize.second;
                totalCompressedConstantsBeforeCompression_ += inputOutputSize.second;
                totalCompressedConstants_ += inputOutputSize.first;
            }
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

        log = log.nest();
        log.info("Constants size before compression: {0} bytes", totalConstantsBeforeCompression_);
        log.info("Constants size after compression: {0} bytes", totalConstantsAfterCompression_);
        log.info("Constants that were compressed: {0} bytes ({1}% of total)", totalCompressedConstants_,
                 compressedPercentage);
        log.info("Constants that couldn't be compressed: {0} bytes ({1}% of total)", totalUncompressedConstants_,
                 uncompressedPercentage);

        if (totalCompressedConstantsBeforeCompression_ > 0) {
            double qualifiedCompressionRate =
                    (double)totalCompressedConstants_ / totalCompressedConstantsBeforeCompression_ * 100;
            double totalCompressionRate =
                    (double)totalConstantsAfterCompression_ / totalConstantsBeforeCompression_ * 100;

            log.info("Compression rate of compressed constants: {0}%", qualifiedCompressionRate);
            log.info("Total compression rate: {0}%", totalCompressionRate);
        }
        log = log.unnest();
    }

private:
    uint64_t totalConstantsBeforeCompression_;
    uint64_t totalCompressedConstants_;
    uint64_t totalCompressedConstantsBeforeCompression_;
    uint64_t totalUncompressedConstants_;
    std::set<Const::DeclareOp> seenConstants_{};
};

class ConstSwizzlingCounter {
public:
    ConstSwizzlingCounter()
            : numOfSwizzledConsts_{0},
              numOfNotSwizzledConsts_{0},
              totalSizeOfSwizzledConsts_{0},
              totalSizeOfNotSwizzledConsts_{0} {
    }

    void count(mlir::Operation* op) {
        if (auto cstOp = mlir::dyn_cast<Const::DeclareOp>(op)) {
            if (cstOp.output().getUsers().empty()) {
                return;
            }

            auto cstOpType = cstOp.getType().cast<vpux::NDTypeInterface>();
            auto size = cstOpType.getTotalAllocSize().count();

            if (vpux::getSwizzlingSchemeAttr(cstOpType)) {
                numOfSwizzledConsts_++;
                totalSizeOfSwizzledConsts_ += size;
            } else {
                numOfNotSwizzledConsts_++;
                totalSizeOfNotSwizzledConsts_ += size;
            }
        }
    }

    void printStatistics(vpux::Logger log) {
        log = log.nest();
        log.info("Swizzled constants     - number: {0}, size: {1} bytes", numOfSwizzledConsts_,
                 totalSizeOfSwizzledConsts_);
        log.info("Not swizzled constants - number: {0}, size: {1} bytes", numOfNotSwizzledConsts_,
                 totalSizeOfNotSwizzledConsts_);
        log = log.unnest();
    }

private:
    uint64_t numOfSwizzledConsts_;
    uint64_t numOfNotSwizzledConsts_;
    uint64_t totalSizeOfSwizzledConsts_;
    uint64_t totalSizeOfNotSwizzledConsts_;
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
    auto func = getOperation();

    llvm::DenseSet<mlir::OperationName> dpuOperations{
            mlir::OperationName(VPUIP::ConvolutionUPAOp::getOperationName(), &ctx),
            mlir::OperationName(VPUIP::PoolingUPAOp::getOperationName(), &ctx),
            mlir::OperationName(VPUIP::EltwiseUPAOp::getOperationName(), &ctx)};

    auto opStatisticsCounter = populateCounters();
    CompressionRateCounter compressionCounter;
    ConstSwizzlingCounter constSwizzlingCounter;

    func->walk([&](mlir::Operation* op) {
        opStatisticsCounter.count(op);
        compressionCounter.count(op);
        constSwizzlingCounter.count(op);

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
    _log.info("Const swizzling statistics:");
    constSwizzlingCounter.printStatistics(_log);
}

}  // namespace

//
// createDumpStatisticsOfTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDumpStatisticsOfTaskOpsPass(bool printWeightsCompressBTC, Logger log) {
    return std::make_unique<DumpStatisticsOfTaskOpsPass>(printWeightsCompressBTC, log);
}
