//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef PROFILING_PARSER_HPP
#define PROFILING_PARSER_HPP

#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"

#include <schema/graphfile_generated.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace vpux {
namespace profiling {

// Suffix used to create cluster name from task name
const std::string CLUSTER_LEVEL_PROFILING_SUFFIX = "/cluster_";
// Suffix used to create variant name from cluster name
const std::string VARIANT_LEVEL_PROFILING_SUFFIX = "/variant_";
// Suffix used to create variant name from cluster name
const std::string TILE_LEVEL_PROFILING_SUFFIX = "/tile_";

enum class ExecutorType { NONE, DPU, UPA, ACTSHAVE, DMA_SW, WORKPOINT, DMA_HW };

/**
 * @enum TaskType
 * @brief Declares which task types are required in profiling output.
 */
enum TaskType {
    ALL,     ///< Report all tasks for profiling
    DPU_SW,  ///< Only execution tasks profiling
    DMA,     ///< Only DMA tasks profiling
};

/**
 * @enum VerbosityLevel
 * @brief Declares verbosity level of printing information
 */
enum VerbosityLevel {
    LOW = 0,     ///< Default, only DMA/SW/Aggregated DPU info
    MEDIUM = 1,  ///< Extend by cluster level information
    HIGH = 5,    ///< Full information including individual variants timings
};

struct LayerInfo {
    char name[256];
    char layer_type[50];
    enum layer_status_t { NOT_RUN, OPTIMIZED_OUT, EXECUTED };
    layer_status_t status;
    uint64_t start_time_ns;        ///< Absolute start time
    uint64_t duration_ns;          ///< Total duration (from start time until last compute task completed)
    uint32_t layer_id = -1;        ///< Not used
    uint64_t fused_layer_id = -1;  ///< Not used
    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns = 0;
    uint64_t sw_ns = 0;
    uint64_t dma_ns = 0;
};

struct TaskInfo {
    char name[256];
    char layer_type[50];
    enum class ExecType { NONE, DPU, SW, DMA, UPA };
    ExecType exec_type;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    uint32_t active_cycles = 0;
    uint32_t stall_cycles = 0;
    uint32_t task_id = -1;
    uint32_t parent_layer_id = -1;  ///< Not used
};

struct ActShaveData_t {
    uint64_t begin;
    uint32_t duration;
    uint32_t stallCycles;
};

struct UpaData_t {
    uint64_t begin;
    uint64_t end;
    uint32_t stallCycles;
    uint32_t activeCycles;
};

// SW DPU profiling data payload
struct SwDpuData_t {
    uint64_t begin;
    uint64_t end;
};

// HWP DPU profiling data payload
struct HwpDpu27Mode0Data_t {
    uint64_t idu_wl_duration : 28;
    uint64_t idu_tstamp : 28;
    uint64_t sve_id : 5;
    uint64_t reserved3 : 3;
    uint64_t odu_wl_duration : 28;
    uint64_t odu_tstamp : 28;
    uint64_t reserved8 : 8;
};

struct HwpDma40Data_t {
    uint64_t desc_addr;
    uint64_t fetch_time;
    uint64_t ready_time;
    uint64_t start_time;
    uint64_t wdone_time;
    uint64_t finish_time;
    uint8_t la_id;
    uint8_t ch_id;
    uint16_t rsvd;
    uint16_t rstall_cnt;
    uint16_t wstall_cnt;
    uint32_t twbytes_cnt;
    uint32_t chcycle_cnt;
};

struct HwpDpu40Mode3Data_t {
    uint64_t idu_wl_duration : 32;
    uint64_t idu_wl_id : 16;
    uint64_t idu_dpu_id : 16;
    uint64_t idu_tstamp;
    uint64_t odu_wl_duration : 32;
    uint64_t odu_wl_id : 16;
    uint64_t odu_dpu_id : 16;
    uint64_t odu_tstamp;
};

struct DMA20Data_t {
    uint32_t startCycle;
    uint32_t endCycle;
};

struct DMA27Data_t {
    uint64_t startCycle;
    uint64_t endCycle;
};

struct WorkpointConfiguration_t {
    uint16_t pllMultiplier;
    uint16_t configId;
};

struct SummaryInfo {
    uint64_t totalBufferSize;
    struct SectionInfo {
        uint32_t entrySize;
        uint32_t numOfTasks;
        uint32_t bufferOffset;
        uint32_t bufferSize;
    };
    SectionInfo dmaInfo;
    SectionInfo dpuInfo;
    SectionInfo swInfo;
    SectionInfo workpointInfo;
};

struct FrequenciesSetup {
public:
    static constexpr double UNITIALIZED_FREQUENCY_VALUE = -1;
    static constexpr double MIN_FREQ_MHZ = 700.0;

public:
    static FrequenciesSetup get30XXSetup(double nceFreq);
    static FrequenciesSetup get37XXSetup(uint16_t pllMult);

public:
    double vpuClk = UNITIALIZED_FREQUENCY_VALUE;
    double dpuClk = UNITIALIZED_FREQUENCY_VALUE;
    double profClk = UNITIALIZED_FREQUENCY_VALUE;
    double dmaBandwidth = UNITIALIZED_FREQUENCY_VALUE;
    bool hasSharedDmaSwCounter = false;
    bool hasSharedDmaDpuCounter = false;
};

constexpr double Dma20Bandwidth = 700. / 20000.;
constexpr double Dma27Bandwidth = 1300. / 31200.;

constexpr int COL_WIDTH_32 = 11;
constexpr int COL_WIDTH_64 = 19;

class ThrowableAssertMixin {
public:
    virtual void checkDataOrDie() const = 0;
};

class DebugFormattableRecordMixin {
public:
    using ColDesc = std::vector<std::pair<std::string, int>>;

protected:
    DebugFormattableRecordMixin(size_t inMemoryOffset): _inMemoryOffset(inMemoryOffset) {
    }

    virtual ColDesc getColDesc() const = 0;

public:
    void printDebugHeader(std::ostream& os) {
        const auto columns = this->getColDesc();
        os << std::setw(14) << "Global offset" << std::setw(15) << "Section offset" << std::setw(14) << "Engine"
           << std::setw(100) << "Layer name";
        for (const std::pair<std::string, int>& p : columns) {
            os << std::setw(p.second) << p.first;
        }
        os << std::endl;
    }

    size_t getInMemoryOffset() const {
        return _inMemoryOffset;
    }

    virtual size_t getDebugDataSize() const = 0;

    virtual void printDebugInfo(std::ostream& outStream) const = 0;

protected:
    size_t _inMemoryOffset;
};

class RawProfilingRecord {
public:
    using BarrierIdType = uint32_t;
    using TimeType = double;
    using BarriersSet = std::set<BarrierIdType>;

public:
    template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
    static TimeType convertTicksToNs(T cycles, double frequency) {
        VPUX_THROW_WHEN(frequency == FrequenciesSetup::UNITIALIZED_FREQUENCY_VALUE, "Invalid frequency {0}", frequency);
        return static_cast<TimeType>(cycles * 1000. / frequency);
    }

    static auto getBarriersIntersection(const BarriersSet& set1, const BarriersSet& set2) {
        std::vector<BarrierIdType> barriersIntersection;
        std::set_intersection(set1.cbegin(), set1.cend(), set2.cbegin(), set2.cend(),
                              std::back_inserter(barriersIntersection));
        return barriersIntersection;
    }

    struct ParsedTaskName {
        std::string taskName;
        std::string layerName;
        std::string layerType;
        std::string profTag;
        int clusterId;
    };

    // Parses the full task name from GraphFile into ParsedTaskName, extracting task name, profiling marker, layer type
    // and cluster id
    static ParsedTaskName deserializeTaskName(const std::string& gfTaskName,
                                              const llvm::Optional<std::string>& maybeProfPrefix);

private:
    static bool isSetIntersectionEmpty(const BarriersSet& set1, const BarriersSet& set2) {
        std::vector<BarrierIdType> barriersIntersection = getBarriersIntersection(set1, set2);
        VPUX_THROW_UNLESS(barriersIntersection.size() < 2, "Tasks should have at most 1 common barrier, but got {0}",
                          barriersIntersection.size());
        return barriersIntersection.empty();
    }

protected:
    static TaskInfo::ExecType convertToExecEnums(ExecutorType exec) {
        switch (exec) {
        case ExecutorType::NONE:
            return TaskInfo::ExecType::NONE;
        case ExecutorType::DMA_SW:
        case ExecutorType::DMA_HW:
            return TaskInfo::ExecType::DMA;
        case ExecutorType::DPU:
            return TaskInfo::ExecType::DPU;
        case ExecutorType::ACTSHAVE:
            return TaskInfo::ExecType::SW;
        case ExecutorType::UPA:
            return TaskInfo::ExecType::UPA;
        default:
            VPUX_THROW("Unknown ExecutorType value");
        }
    }

    RawProfilingRecord(const std::string& name, const std::string& layerName, const std::string& layerType,
                       ExecutorType executorType, const BarriersSet& wBarriers = {}, const BarriersSet& uBarriers = {})
            : _executorType(executorType),
              _name(name),
              _layerName(layerName),
              _layerType(layerType),
              _waitBarriers(wBarriers),
              _updateBarriers(uBarriers) {
    }

    RawProfilingRecord(ExecutorType executorType, const std::string& cleanName, const std::string& layerName,
                       const std::string& layerType, const MVCNN::Task* task)
            : _executorType(executorType), _name(cleanName), _layerName(layerName), _layerType(layerType) {
        VPUX_THROW_WHEN(task == nullptr, "Invalid task");
        VPUX_THROW_WHEN(task->name() == nullptr, "Invalid task name");
        VPUX_THROW_WHEN(task->associated_barriers() == nullptr, "Task should have associated barriers");

        auto barriers = task->associated_barriers();
        if (auto wBarriers = barriers->wait_barriers()) {
            _waitBarriers = BarriersSet(wBarriers->cbegin(), wBarriers->cend());
        }
        if (auto uBarriers = barriers->update_barriers()) {
            _updateBarriers = BarriersSet(uBarriers->cbegin(), uBarriers->cend());
        }
    }

    template <class ParsedProfType>
    struct ParsedTaskNameProf {
        ParsedTaskName meta;
        ParsedProfType prof;
    };

    struct TokenizedTaskName {
        std::string layerName;
        std::vector<std::string> tokens;
        bool hasMalformedMeta;
    };

    static TokenizedTaskName tokenizeTaskName(const std::string& gfTaskName);

public:
    bool isDirectPredecessor(const RawProfilingRecord& other) const {
        return !isSetIntersectionEmpty(_updateBarriers, other._waitBarriers);
    }

    bool isDirectSuccessor(const RawProfilingRecord& other) const {
        return !isSetIntersectionEmpty(_waitBarriers, other._updateBarriers);
    }

    virtual std::string getRecordTypeName() const = 0;

    ExecutorType getExecutorType() const {
        return _executorType;
    }

    const BarriersSet& getWaitBarriers() const {
        return _waitBarriers;
    }

    const BarriersSet& getUpdateBarriers() const {
        return _updateBarriers;
    }

    virtual std::string getOriginalName() const {
        return _name;
    }

    virtual std::string getTaskName() const {
        return _name;
    }

    virtual std::string getLayerName() const {
        return _layerName;
    }

    std::string getLayerType() const {
        return _layerType;
    }

    virtual TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const {
        TaskInfo taskInfo;
        taskInfo.exec_type = convertToExecEnums(_executorType);
        taskInfo.start_time_ns = static_cast<uint64_t>(getStartTime(frequenciesSetup));
        taskInfo.duration_ns = static_cast<uint64_t>(getDuration(frequenciesSetup));

        const auto nameLen = this->getTaskName().copy(taskInfo.name, sizeof(taskInfo.name) - 1);
        taskInfo.name[nameLen] = 0;

        const auto typeLen = _layerType.copy(taskInfo.layer_type, sizeof(taskInfo.layer_type) - 1);
        taskInfo.layer_type[typeLen] = 0;

        return taskInfo;
    }

    virtual void sanitize(vpux::Logger&, FrequenciesSetup) const {
        // do nothing
    }

    virtual TimeType getStartTime(FrequenciesSetup frequenciesSetup) const = 0;

    virtual TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const = 0;

    virtual TimeType getDuration(FrequenciesSetup frequenciesSetup) const {
        return getFinishTime(frequenciesSetup) - getStartTime(frequenciesSetup);
    }

    static std::string getLayerName(const std::string& taskName);

protected:
    ExecutorType _executorType{ExecutorType::NONE};
    std::string _name;
    std::string _layerName;
    std::string _layerType;
    BarriersSet _waitBarriers;
    BarriersSet _updateBarriers;
};

using RawProfilingRecordPtr = std::shared_ptr<RawProfilingRecord>;
using RawProfilingRecords = std::vector<RawProfilingRecordPtr>;

template <class RecordDataType>
class RawProfilingDMARecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    using ExtendedTimestampType = uint64_t;

protected:
    RawProfilingDMARecord(ExecutorType execType, const RecordDataType& record, const std::string& name,
                          const std::string& layerName, const std::string& layerType, const BarriersSet& waitBarriers,
                          const BarriersSet& updateBarriers, size_t inMemoryOffset)
            : RawProfilingRecord(name, layerName, layerType, execType, waitBarriers, updateBarriers),
              DebugFormattableRecordMixin(inMemoryOffset),
              _record(record) {
    }

public:
    void sanitize(vpux::Logger& log, FrequenciesSetup frequenciesSetup) const override {
        const auto dmaDurationNs = getDuration(frequenciesSetup);
        const auto bandwidth = frequenciesSetup.dmaBandwidth;
        VPUX_THROW_WHEN(bandwidth == FrequenciesSetup::UNITIALIZED_FREQUENCY_VALUE, "DMA bandwidth is uninitialized");
        // Maximum 4MB  transfer
        const uint64_t maxTransferSize = 1024LL * 1024LL * 4LL;
        // guard band (DMA transfers seem to have significant variance in duration probably due to
        // variable DDR latency)
        const uint64_t guardBand = 10;
        // Calculation of DMA ticks taken from vpu cost model (including dpuCyclesCoeff provided
        // per platform taken as input parameter)
        const uint64_t maxTicks = static_cast<ExtendedTimestampType>(guardBand * maxTransferSize * bandwidth);
        if (dmaDurationNs > convertTicksToNs(maxTicks, FrequenciesSetup::MIN_FREQ_MHZ)) {
            log.warning("Too long execution time of DMA task");
        }
    }

    size_t getDebugDataSize() const override {
        return sizeof(RecordDataType);
    }

public:
    struct ParsedProf {
        int16_t curDmaId;
    };

    using ParsedTaskNameProf = RawProfilingRecord::ParsedTaskNameProf<ParsedProf>;

    static ParsedProf parseProfilingTag(const std::string& profTag);
    static bool isTaskBegin(const std::string& fullTaskName);
    static bool isTaskWorkpointRead(const std::string& fullTaskName);
    static ParsedTaskNameProf parseTaskName(const std::string& fullTaskName);

protected:
    RecordDataType _record;
};

template <class RecordDataType>
class RawProfilingSwDmaRecord : public RawProfilingDMARecord<RecordDataType> {
public:
    using RawProfilingDMARecord<RecordDataType>::RawProfilingDMARecord;
    using ColDesc = DebugFormattableRecordMixin::ColDesc;

protected:
    RawProfilingSwDmaRecord(const RecordDataType& record, const std::string& name, const std::string& layerName,
                            const std::string& layerType, const RawProfilingRecord::BarriersSet& waitBarriers,
                            const RawProfilingRecord::BarriersSet& updateBarriers, size_t inMemoryOffset)
            : RawProfilingDMARecord<RecordDataType>(ExecutorType::DMA_SW, record, name, layerName, layerType,
                                                    waitBarriers, updateBarriers, inMemoryOffset) {
    }

    ColDesc getColDesc() const override {
        return {{"Begin tstamp", COL_WIDTH_64}, {"End tstamp", COL_WIDTH_64}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto cols = getColDesc();
        outStream << std::setw(cols[0].second) << this->_record.startCycle << std::setw(cols[1].second)
                  << this->_record.endCycle;
    }
};

class RawProfilingDMA20Record : public RawProfilingSwDmaRecord<DMA20Data_t> {
public:
    using ExtendedTimestampType = typename RawProfilingDMARecord::ExtendedTimestampType;

public:
    explicit RawProfilingDMA20Record(const DMA20Data_t& record, const std::string& name, const std::string& layerName,
                                     const std::string& layerType, const BarriersSet& waitBarriers,
                                     const BarriersSet& updateBarriers, ExtendedTimestampType overflowCorrectionShift,
                                     size_t inMemoryOffset)
            : RawProfilingSwDmaRecord<DMA20Data_t>(record, name, layerName, layerType, waitBarriers, updateBarriers,
                                                   inMemoryOffset),
              _overflowCorrectionShift(overflowCorrectionShift) {
    }

    std::string getRecordTypeName() const override {
        return "DMA 2.0";
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(this->getStartCycle(), frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(this->getEndCycle(), frequenciesSetup.profClk);
    }

protected:
    ExtendedTimestampType getStartCycle() const {
        return static_cast<ExtendedTimestampType>(_record.startCycle) + _overflowCorrectionShift;
    }

    ExtendedTimestampType getEndCycle() const {
        // Use unsigned 32-bit arithmetic to automatically avoid overflow
        const uint32_t durationInCycles = _record.endCycle - _record.startCycle;
        return getStartCycle() + static_cast<uint64_t>(durationInCycles);
    }

private:
    ExtendedTimestampType _overflowCorrectionShift;
};

class RawProfilingDMA27Record : public RawProfilingSwDmaRecord<DMA27Data_t> {
public:
    explicit RawProfilingDMA27Record(const DMA27Data_t& record, const std::string& name, const std::string& layerName,
                                     const std::string& layerType, const BarriersSet& waitBarriers,
                                     const BarriersSet& updateBarriers, size_t inMemoryOffset)
            : RawProfilingSwDmaRecord<DMA27Data_t>(record, name, layerName, layerType, waitBarriers, updateBarriers,
                                                   inMemoryOffset) {
    }

    std::string getRecordTypeName() const override {
        return "DMA 2.7";
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.startCycle, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.endCycle, frequenciesSetup.profClk);
    }
};

class RawProfilingDMA40Record : public RawProfilingDMARecord<HwpDma40Data_t>, public ThrowableAssertMixin {
public:
    explicit RawProfilingDMA40Record(const HwpDma40Data_t& record, const std::string& name,
                                     const std::string& layerName, const std::string& layerType,
                                     const BarriersSet& waitBarriers, const BarriersSet& updateBarriers,
                                     size_t inMemoryOffset)
            : RawProfilingDMARecord<HwpDma40Data_t>(ExecutorType::DMA_HW, record, name, layerName, layerType,
                                                    waitBarriers, updateBarriers, inMemoryOffset) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_UNLESS(_record.rsvd == 0, "Reserved value must contain 0.");
    }

    std::string getRecordTypeName() const override {
        return "HWP DMA";
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.start_time, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.wdone_time, frequenciesSetup.profClk);
    }

protected:
    ColDesc getColDesc() const override {
        return {
                {"JDESC_ADDR", COL_WIDTH_64},
                {"JFETCH_TIME", COL_WIDTH_64},
                {"JREADY_TIME", COL_WIDTH_64},
                {"JSTART_TIME", COL_WIDTH_64},
                {"JWDONE_TIME", COL_WIDTH_64},
                {"JFINISH_TIME", COL_WIDTH_64},
                {"JLA_ID", 7},
                {"JCH_ID", 7},
                {"RSVD", 7},
                {"JRSTALL_CNT", 13},
                {"JWSTALL_CNT", 13},
                {"JTWBYTES_CNT", 14},
                {"JCHCYCLE_CNT", 14},
        };
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto cols = getColDesc();
        // std::ostream recognize uint8_t as char and print character instead of value, so explicitly cast for printing
        // purpose
        const auto to_int = [](uint8_t val) {
            return static_cast<uint16_t>(val);
        };
        outStream << std::setw(cols[0].second) << _record.desc_addr << std::setw(cols[1].second) << _record.fetch_time
                  << std::setw(cols[2].second) << _record.ready_time << std::setw(cols[3].second) << _record.start_time
                  << std::setw(cols[4].second) << _record.wdone_time << std::setw(cols[5].second) << _record.finish_time
                  << std::setw(cols[6].second) << to_int(_record.la_id) << std::setw(cols[7].second)
                  << to_int(_record.ch_id) << std::setw(cols[8].second) << _record.rsvd << std::setw(cols[9].second)
                  << _record.rstall_cnt << std::setw(cols[10].second) << _record.wstall_cnt
                  << std::setw(cols[11].second) << _record.twbytes_cnt << std::setw(cols[12].second)
                  << _record.chcycle_cnt;
    }
};

class ClusteredAndTiledMixin {
public:
    ClusteredAndTiledMixin(size_t clusterId, size_t variantId): _clusterId(clusterId), _variantId(variantId) {
    }

    size_t getClusterId() {
        return _clusterId;
    }

protected:
    size_t _clusterId;
    size_t _variantId;
};

class RawProfilingDPURecord :
        public RawProfilingRecord,
        public ClusteredAndTiledMixin,
        public ThrowableAssertMixin,
        public DebugFormattableRecordMixin {
protected:
    RawProfilingDPURecord(const std::string& name, const std::string& layerName, const std::string& layerType,
                          const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t clusterId,
                          size_t variantId, size_t inMemoryOffset)
            : RawProfilingRecord(name, layerName, layerType, ExecutorType::DPU, wBarriers, uBarriers),
              ClusteredAndTiledMixin(clusterId, variantId),
              DebugFormattableRecordMixin(inMemoryOffset) {
    }

    virtual double chooseFrequency(FrequenciesSetup frequenciesSetup) const = 0;

public:
    std::string getTaskName() const override {
        return _name + CLUSTER_LEVEL_PROFILING_SUFFIX + std::to_string(_clusterId) + VARIANT_LEVEL_PROFILING_SUFFIX +
               std::to_string(_variantId);
    }

    void sanitize(vpux::Logger& log, FrequenciesSetup frequenciesSetup) const override {
        const auto dpuExecutionTime = this->getDuration(frequenciesSetup);
        const uint64_t maxKernel = 11 * 11;
        const uint64_t maxElem = 2ll * 1024ll * 1024ll;
        const uint64_t maxChannels = 8192;
        const uint64_t maxCycles = maxKernel * maxElem * maxChannels / 256;
        // TODO: invalid check. HW DPU profiling also count IDU and ODU time
        const auto frequency = this->chooseFrequency(frequenciesSetup);
        const auto maxNs = convertTicksToNs(maxCycles, frequency);
        if (maxNs < dpuExecutionTime) {
            log.warning("Too long execution time of DPU task");
        }
    }

public:
    struct ParsedProf {
        int16_t taskId;
        int16_t memoryId;
        int32_t maxVariants;
        int16_t numVariants;
        int16_t clusterId;
        int16_t bufferId;
        int16_t numClusters;

        // Profiling data is stored in blob in following order
        // Tasks which correspond to NCEClusterTasks with fewer number of clusters located earlier
        // Within one segment, which correspond to same amount of clusters, buffers located sequentially
        // Within one buffer tasks interleaved, but clusters  sequential, i.g.
        // Task-N cluster 0, Task-(N+1), cluster 0, Task-N cluster 1, Task-(N+1) cluster 1...
        uint64_t getOrderDescriptor() const {
            uint64_t descriptor = 0;
            for (uint16_t subDesc : {numClusters, bufferId, clusterId, memoryId}) {
                descriptor = (descriptor << 16) | subDesc;
            }
            return descriptor;
        }
    };
    using ParsedTaskNameProf = RawProfilingRecord::ParsedTaskNameProf<ParsedProf>;

    static ParsedProf parseProfilingTag(const std::string& profTag, int16_t clusterId, unsigned taskListId);
    static ParsedTaskNameProf parseTaskName(const std::string& fullTaskName, unsigned taskListId);
};

class RawProfilingDPUSWRecord : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUSWRecord(SwDpuData_t timestamps, const std::string& name, const std::string& layerName,
                                     const std::string& layerType, const BarriersSet& wBarriers,
                                     const BarriersSet& uBarriers, size_t clusterId, size_t variantId,
                                     size_t inMemoryOffset)
            : RawProfilingDPURecord(name, layerName, layerType, wBarriers, uBarriers, clusterId, variantId,
                                    inMemoryOffset),
              _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.begin == 0 && _timestamps.end == 0, "Invalid DPU task timestamp");
    }

    std::string getRecordTypeName() const override {
        return "SW DPU";
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.begin, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.end, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(SwDpuData_t);
    }

protected:
    double chooseFrequency(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.profClk;
    }

    ColDesc getColDesc() const override {
        return {{"Begin tstamp", COL_WIDTH_64}, {"End tstamp", COL_WIDTH_64}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto swDpuCol = getColDesc();

        outStream << std::setw(swDpuCol[0].second) << _timestamps.begin << std::setw(swDpuCol[1].second)
                  << _timestamps.end;
    }

private:
    SwDpuData_t _timestamps;
};

class RawProfilingDPUHW27Record : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW27Record(HwpDpu27Mode0Data_t timestamps, const std::string& name,
                                       const std::string& layerName, const std::string& layerType,
                                       const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t clusterId,
                                       size_t variantId, size_t inMemoryOffset)
            : RawProfilingDPURecord(name, layerName, layerType, wBarriers, uBarriers, clusterId, variantId,
                                    inMemoryOffset),
              _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.idu_wl_duration == 0 && _timestamps.odu_wl_duration == 0,
                        "Invalid DPU task duration");
        VPUX_THROW_UNLESS(_timestamps.reserved3 == 0 && _timestamps.reserved8 == 0, "Reserved values must contain 0.");
    }

    std::string getRecordTypeName() const override {
        return "HWP DPU 2.7";
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        const auto max28BitTime = convertTicksToNs(0x0FFFFFFFull, frequenciesSetup.vpuClk);
        const auto noOverflowSubtract = [](TimeType first, TimeType second, TimeType max) -> TimeType {
            return first - second + ((first < second) ? max : 0);
        };
        return noOverflowSubtract(convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.vpuClk),
                                  convertTicksToNs(_timestamps.idu_wl_duration, frequenciesSetup.dpuClk), max28BitTime);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.odu_tstamp, frequenciesSetup.vpuClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(HwpDpu27Mode0Data_t);
    }

protected:
    double chooseFrequency(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.dpuClk;
    }

    ColDesc getColDesc() const override {
        return {{"IDU dur", COL_WIDTH_32}, {"IDU tstamp", COL_WIDTH_32}, {"SWE ID", 7}, {"Res", 4},
                {"ODU dur", COL_WIDTH_32}, {"ODU tstamp", COL_WIDTH_32}, {"Res", 7}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto hwpDpuCol = getColDesc();

        outStream << std::setw(hwpDpuCol[0].second) << _timestamps.idu_wl_duration << std::setw(hwpDpuCol[1].second)
                  << _timestamps.idu_tstamp << std::setw(hwpDpuCol[2].second) << _timestamps.sve_id
                  << std::setw(hwpDpuCol[3].second) << _timestamps.reserved3 << std::setw(hwpDpuCol[4].second)
                  << _timestamps.odu_wl_duration << std::setw(hwpDpuCol[5].second) << _timestamps.odu_tstamp
                  << std::setw(hwpDpuCol[6].second) << _timestamps.reserved8;
    }

private:
    HwpDpu27Mode0Data_t _timestamps;
};

class RawProfilingDPUHW40Record : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW40Record(HwpDpu40Mode3Data_t timestamps, const std::string& name,
                                       const std::string& layerName, const std::string& layerType,
                                       const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t clusterId,
                                       size_t variantId, size_t inMemoryOffset)
            : RawProfilingDPURecord(name, layerName, layerType, wBarriers, uBarriers, clusterId, variantId,
                                    inMemoryOffset),
              _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.idu_wl_duration == 0 && _timestamps.odu_wl_duration == 0,
                        "Invalid DPU task duration");
    }

    std::string getRecordTypeName() const override {
        return "HWP DPU 4.0";
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.profClk) -
               convertTicksToNs(_timestamps.idu_wl_duration, frequenciesSetup.dpuClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.odu_tstamp, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(HwpDpu40Mode3Data_t);
    }

protected:
    double chooseFrequency(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.dpuClk;
    }

    ColDesc getColDesc() const override {
        return {{"IDU dur", COL_WIDTH_32}, {"IDU tstamp", COL_WIDTH_64}, {"IDU WL ID", 11}, {"IDU DPU ID", 12},
                {"ODU dur", COL_WIDTH_32}, {"ODU tstamp", COL_WIDTH_64}, {"ODU WL ID", 11}, {"ODU DPU ID", 12}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto hwpDpuCol = getColDesc();

        outStream << std::setw(hwpDpuCol[0].second) << _timestamps.idu_wl_duration << std::setw(hwpDpuCol[1].second)
                  << _timestamps.idu_tstamp << std::setw(hwpDpuCol[2].second) << _timestamps.idu_wl_id
                  << std::setw(hwpDpuCol[3].second) << _timestamps.idu_dpu_id << std::setw(hwpDpuCol[4].second)
                  << _timestamps.odu_wl_duration << std::setw(hwpDpuCol[5].second) << _timestamps.odu_tstamp
                  << std::setw(hwpDpuCol[6].second) << _timestamps.odu_wl_id << std::setw(hwpDpuCol[7].second)
                  << _timestamps.odu_dpu_id;
    }

private:
    HwpDpu40Mode3Data_t _timestamps;
};

class RawProfilingUPARecord :
        public RawProfilingRecord,
        public ThrowableAssertMixin,
        public DebugFormattableRecordMixin {
public:
    explicit RawProfilingUPARecord(UpaData_t data, const std::string& name, const std::string& layerName,
                                   const std::string& layerType, const BarriersSet& wBarriers,
                                   const BarriersSet& uBarriers, size_t inMemoryOffset)
            : RawProfilingRecord(name, layerName, layerType, ExecutorType::UPA, wBarriers, uBarriers),
              DebugFormattableRecordMixin(inMemoryOffset),
              _data(data) {
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
        profInfoItem.active_cycles = _data.activeCycles;
        profInfoItem.stall_cycles = _data.stallCycles;
        return profInfoItem;
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_data.begin == 0 && _data.end == 0, "Can't process UPA profiling data.");
    }

    std::string getRecordTypeName() const override {
        return "UPA";
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.begin, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.end, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(UpaData_t);
    }

protected:
    ColDesc getColDesc() const override {
        return {{"Begin tstamp", COL_WIDTH_64},
                {"End tstamp", COL_WIDTH_64},
                {"Stall", COL_WIDTH_32},
                {"Active", COL_WIDTH_32}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto upaCol = getColDesc();

        outStream << std::setw(upaCol[0].second) << _data.begin << std::setw(upaCol[1].second) << _data.end
                  << std::setw(upaCol[2].second) << _data.stallCycles << std::setw(upaCol[3].second)
                  << _data.activeCycles;
    }

public:
    struct ParsedProf {
        size_t currentPos;
    };
    using ParsedTaskNameProf = RawProfilingRecord::ParsedTaskNameProf<ParsedProf>;

    static ParsedProf parseProfilingTag(const std::string& profTag);
    static ParsedTaskNameProf parseTaskName(const std::string& fullTaskName);

private:
    UpaData_t _data;
};

class RawProfilingACTRecord :
        public RawProfilingRecord,
        public ClusteredAndTiledMixin,
        public ThrowableAssertMixin,
        public DebugFormattableRecordMixin {
public:
    explicit RawProfilingACTRecord(ActShaveData_t data, const std::string& name, const std::string& layerName,
                                   const std::string& layerType, const BarriersSet& wBarriers,
                                   const BarriersSet& uBarriers, size_t clusterId, size_t tileId, size_t inMemoryOffset)
            : RawProfilingRecord(name, layerName, layerType, ExecutorType::ACTSHAVE, wBarriers, uBarriers),
              ClusteredAndTiledMixin(clusterId, tileId),
              DebugFormattableRecordMixin(inMemoryOffset),
              _data(data) {
    }

    std::string getTaskName() const override {
        return _name + CLUSTER_LEVEL_PROFILING_SUFFIX + std::to_string(_clusterId) + TILE_LEVEL_PROFILING_SUFFIX +
               std::to_string(_variantId);
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
        profInfoItem.stall_cycles = _data.stallCycles;
        return profInfoItem;
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_data.begin == 0 && _data.duration == 0, "Can't process ACT profiling data.");
    }

    std::string getRecordTypeName() const override {
        return "ACT";
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.begin, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return getStartTime(frequenciesSetup) + getDuration(frequenciesSetup);
    }

    TimeType getDuration(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.duration, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(ActShaveData_t);
    }

public:
    struct ParsedProf {
        size_t inDdrOffset;
        size_t clusterSize;
        size_t clusterId;
        size_t inClusterOffset;
        size_t tileId;

        size_t getResultingDDROffset() const {
            return inDdrOffset + clusterSize * clusterId + inClusterOffset;
        }
    };
    using ParsedTaskNameProf = RawProfilingRecord::ParsedTaskNameProf<ParsedProf>;

    static ParsedProf parseProfilingTag(const std::string& profTag, int16_t clusterId);
    static ParsedTaskNameProf parseTaskName(const std::string& fullTaskName);

protected:
    ColDesc getColDesc() const override {
        return {{"Begin", COL_WIDTH_64}, {"Duration", COL_WIDTH_32}, {"Stall", COL_WIDTH_32}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto actShaveCol = getColDesc();

        outStream << std::setw(actShaveCol[0].second) << _data.begin << std::setw(actShaveCol[1].second)
                  << _data.duration << std::setw(actShaveCol[2].second) << _data.stallCycles;
    }

private:
    ActShaveData_t _data;
};

class ArrayRecord : public RawProfilingRecord {
public:
    ArrayRecord(const std::string name, const RawProfilingRecords& variants, ExecutorType execType)
            : RawProfilingRecord(name, variants.front()->getLayerName(), variants.front()->getLayerType(), execType,
                                 variants.front()->getWaitBarriers(), variants.front()->getUpdateBarriers()),
              _variants(variants),
              _recordTypeName(variants.front()->getRecordTypeName()) {
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return std::accumulate(_variants.cbegin(), _variants.cend(), std::numeric_limits<TimeType>::max(),
                               [&](TimeType a, RawProfilingRecordPtr variant) -> TimeType {
                                   return std::min(a, variant->getStartTime(frequenciesSetup));
                               });
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return std::accumulate(_variants.cbegin(), _variants.cend(), std::numeric_limits<TimeType>::min(),
                               [&](TimeType a, RawProfilingRecordPtr variant) -> TimeType {
                                   return std::max(a, variant->getFinishTime(frequenciesSetup));
                               });
    }

    RawProfilingRecords getArray() const {
        return _variants;
    }

    std::string getRecordTypeName() const override {
        return _recordTypeName;
    }

protected:
    RawProfilingRecords _variants;
    std::string _recordTypeName;
};

class InvariantRawRecord final : public ArrayRecord {
public:
    InvariantRawRecord(const std::string name, size_t clusterId, const RawProfilingRecords& variants,
                       ExecutorType execType)
            : ArrayRecord(name, variants, execType), _clusterId(clusterId) {
    }

    std::string getTaskName() const override {
        return _name + CLUSTER_LEVEL_PROFILING_SUFFIX + std::to_string(_clusterId);
    }

private:
    size_t _clusterId;
};

// Container for conjucted storage of tasks of one format: RawProfilingRecordPtr/TaskInfo
struct RawProfilingData {
    RawProfilingRecords dmaTasks;
    RawProfilingRecords dpuTasks;
    RawProfilingRecords swTasks;
    // Vector of [ExecutorType; offset in blob(bytes)]
    std::vector<std::pair<ExecutorType, size_t>> parseOrder;
    // Pair of workpoint and offset
    std::vector<std::pair<WorkpointConfiguration_t, size_t>> workpointsConfiguration;

    bool hasWorkpointConfig() const;

    uint16_t getPllValueChecked(vpux::Logger& log) const;

    RawProfilingRecords getTaskOfType(ExecutorType type) const;
};

struct RawDataLayout {
    // Map of exec. type to segment offset and size
    std::map<ExecutorType, std::pair<uint32_t, uint32_t>> offsets;
    size_t totalPadsSize;
};

struct RawData {
    RawProfilingData rawRecords;
    MVCNN::TargetDevice device;
    llvm::Optional<double> maybe30XXNceFreq;
    RawDataLayout layout;
};

/**
 * @fn getTaskInfo
 * @brief Parse raw profiling output to get per-tasks info.
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param type type of tasks to be profiled
 * @param verbosity amount of DPU info to print, may be LOW|MEDIUM|HIGH
 * @param fpga whether buffer was obtained from FPGA
 * @param ignoreSanitizationErrors to ignore sanitization errors
 * @see TaskType
 * @see VerbosityLevel
 * @return std::vector of TaskInfo structures
 */
std::vector<TaskInfo> getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                  TaskType type, VerbosityLevel verbosity, bool fpga = false,
                                  bool ignoreSanitizationErrors = false);

/**
 * @fn getRawProfilingTasks
 * @brief Show raw counters for debug purpose. Intended for use in prof_parser only
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param type type of tasks to be profiled
 * @param ignoreSanitizationErrors to ignore sanitization errors
 * @return RawProfilingData
 */
RawData getRawProfilingTasks(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                             TaskType type, bool ignoreSanitizationErrors = false);

/**
 * @brief Helper function to parse profiling buffer name and extract buffer offsets and sizes
 *
 * @param profilingOutputName - profiling buffer name (e.g. 0_dpu_6432_actshave_6464_dma_15744_pll)
 * @param profilingOutputSize - total size of the profiling output in bytes
 * @return offsets and sizes per ExecutorType in bytes
 *
 */

SummaryInfo getSummary(const RawData& profData, size_t profSize);

/**
 * @fn getLayerInfo
 * @brief Parse raw profiling output to get per-layer info.
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param fpga whether buffer was obtained from FPGA
 * @param ignoreSanitizationErrors to ignore sanitization errors
 * @return std::vector of LayerInfo structures
 */
std::vector<LayerInfo> getLayerInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                    bool fpga = false, bool ignoreSanitizationErrors = true);

/**
 * @fn getLayerInfo
 * @brief Parse raw profiling output to get per-layer info. Reuses precomputed info about tasks.
 * @param taskInfo output from \b getTaskInfo function.
 * @return std::vector of LayerInfo structures
 * @see getTaskInfo
 */
std::vector<LayerInfo> getLayerInfo(const std::vector<TaskInfo>& taskInfo);

}  // namespace profiling
}  // namespace vpux

#endif
