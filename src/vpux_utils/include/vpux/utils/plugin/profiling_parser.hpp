//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef PROFILING_PARSER_HPP
#define PROFILING_PARSER_HPP

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"

#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile_generated.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
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

enum class ExecutorType { NONE, DPU, UPA, ACTSHAVE, DMA };

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
    layer_status_t status{NOT_RUN};
    uint64_t start_time_ns{};   ///< Absolute start time
    uint64_t duration_ns{};     ///< Total duration (from start time until last compute task completed)
    uint32_t layer_id{};        ///< Not used
    uint64_t fused_layer_id{};  ///< Not used

    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns{};
    uint64_t sw_ns{};
    uint64_t dma_ns{};
};

struct TaskInfo {
    char name[256];
    char layer_type[50];
    enum class ExecType {
        NONE,
        DPU,
        SW,
        DMA,
    };
    ExecType exec_type{ExecType::NONE};
    uint64_t start_time_ns{};
    uint64_t duration_ns{};
    uint32_t active_cycles{};
    uint32_t stall_cycles{};
    uint32_t task_id{};
    uint32_t parent_layer_id{};  ///< Not used
};

enum class RecordType { DMA20, DMA27, DPU_SW, DPU_HWP27, SW_UPA, SW_ACT, NONE };

struct ActShaveData_t {
    uint64_t begin{};
    uint32_t duration{};
    uint32_t stallCycles{};
};

struct UpaData_t {
    uint64_t begin{};
    uint64_t end{};
    uint32_t stallCycles{};
    uint32_t activeCycles{};
};

// SW DPU profiling data payload
struct SwDpuData_t {
    uint64_t begin{};
    uint64_t end{};
};

// HWP DPU profiling data payload
struct HwpDpuMode0Data_t {
    uint64_t idu_wl_duration : 28;
    uint64_t idu_tstamp : 28;
    uint64_t sve_id : 5;
    uint64_t reserved3 : 3;
    uint64_t odu_wl_duration : 28;
    uint64_t odu_tstamp : 28;
    uint64_t reserved8 : 8;
};

struct DebugInfo {
    std::string name;                         ///< Task name
    uint64_t offset{};                        ///< Offset in the raw buffer
    RecordType recordType{RecordType::NONE};  ///< Type of data that stored in "raw" union
    union Raw {
        Raw(){};
        uint32_t dma20{};
        uint64_t dma27;
        SwDpuData_t swDpu;
        HwpDpuMode0Data_t hwpDpu;
        ActShaveData_t actShave;
        UpaData_t upa;
    } raw;
};

struct SummaryInfo {
    uint64_t totalBufferSize{};
    struct SectionInfo {
        uint32_t entrySize{};
        uint32_t numOfTasks{};
        uint32_t bufferOffset{};
        uint32_t bufferSize{};
    };
    SectionInfo dmaInfo;
    SectionInfo dpuInfo;
    SectionInfo swInfo;
};

struct FrequenciesSetup {
public:
    static constexpr double UNITIALIZED_FREQUENCY_VALUE = -1;
    static constexpr double MIN_FREQ_MHZ = 700.0;

public:
    double vpuClk = UNITIALIZED_FREQUENCY_VALUE;
    double dpuClk = UNITIALIZED_FREQUENCY_VALUE;
    double profClk = UNITIALIZED_FREQUENCY_VALUE;
    double dmaBandwidth = UNITIALIZED_FREQUENCY_VALUE;
};

constexpr double DMA20Bandwidth = 700. / 20000.;
constexpr double DMA27Bandwidth = 1300. / 31200.;

class ThrowableAssertMixin {
public:
    virtual void checkDataOrDie() const = 0;
};

class RawProfilingRecord {
public:
    using BarrierIdType = uint32_t;
    using TimeType = double;
    using BarriersSet = std::set<BarrierIdType>;

public:
    template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
    static TimeType convertTicksToNs(T cycles, double frequency) {
        VPUX_THROW_WHEN(frequency == FrequenciesSetup::UNITIALIZED_FREQUENCY_VALUE,
                        "Tried to convert cycles to time using invalid frequency");
        return static_cast<TimeType>(cycles * 1000. / frequency);
    }

    static auto getBarriersIntersection(const BarriersSet& set1, const BarriersSet& set2) {
        std::vector<BarrierIdType> barriersIntersection;
        std::set_intersection(set1.cbegin(), set1.cend(), set2.cbegin(), set2.cend(),
                              std::back_inserter(barriersIntersection));
        return barriersIntersection;
    }

private:
    static bool isSetIntersectionEmpty(const BarriersSet& set1, const BarriersSet& set2) {
        std::vector<BarrierIdType> barriersIntersection = getBarriersIntersection(set1, set2);
        VPUX_THROW_UNLESS(barriersIntersection.size() < 2, "Tasks should have at most 1 common barrier, but got {0}",
                          barriersIntersection.size());
        return barriersIntersection.empty();
    }

    static TaskInfo::ExecType convertToExecEnums(ExecutorType exec) {
        switch (exec) {
        case ExecutorType::NONE:
            return TaskInfo::ExecType::NONE;
        case ExecutorType::DMA:
            return TaskInfo::ExecType::DMA;
        case ExecutorType::DPU:
            return TaskInfo::ExecType::DPU;
        case ExecutorType::ACTSHAVE:
        case ExecutorType::UPA:
            return TaskInfo::ExecType::SW;
        default:
            VPUX_THROW("Unknown ExecutorType value");
        }
    }

protected:
    RawProfilingRecord(const std::string& name, RecordType recordType, ExecutorType executorType,
                       const BarriersSet& wBarriers = {}, const BarriersSet& uBarriers = {})
            : _recordType(recordType),
              _executorType(executorType),
              _name(name),
              _waitBarriers(wBarriers),
              _updateBarriers(uBarriers) {
    }

    RawProfilingRecord(RecordType recordType, ExecutorType executorType, const std::string& cleanName,
                       const MVCNN::Task* task)
            : _recordType(recordType), _executorType(executorType), _name(cleanName) {
        VPUX_THROW_WHEN(task == nullptr, "Task must be non-nullptr value");
        VPUX_THROW_WHEN(task->name() == nullptr, "Task name must be non-nullptr value");
        VPUX_THROW_WHEN(task->associated_barriers() == nullptr, "Task should have associated barriers");

        auto barriers = task->associated_barriers();
        if (auto wBarriers = barriers->wait_barriers()) {
            _waitBarriers = BarriersSet(wBarriers->cbegin(), wBarriers->cend());
        }
        if (auto uBarriers = barriers->update_barriers()) {
            _updateBarriers = BarriersSet(uBarriers->cbegin(), uBarriers->cend());
        }
    }

public:
    bool isDirectPredecessor(const RawProfilingRecord& other) const {
        return !isSetIntersectionEmpty(_updateBarriers, other._waitBarriers);
    }

    bool isDirectSuccessor(const RawProfilingRecord& other) const {
        return !isSetIntersectionEmpty(_waitBarriers, other._updateBarriers);
    }

    RecordType getRecordType() const {
        return _recordType;
    }

    ExecutorType getExecutorType() const {
        return _executorType;
    }

    const BarriersSet& getWaitBarriers() const {
        return _waitBarriers;
    }

    const BarriersSet& getUpdateBarriers() const {
        return _updateBarriers;
    }

    virtual std::string getTaskName() const {
        return _name;
    }

    virtual std::string getRawName() const {
        return _name;
    }

    virtual TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const {
        TaskInfo profInfoItem = TaskInfo();
        profInfoItem.layer_type[0] = '\0';
        profInfoItem.task_id = -1;
        profInfoItem.exec_type = convertToExecEnums(_executorType);

        const auto taskName = this->getTaskName();
        const auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
        const auto length = taskName.copy(profInfoItem.name, typeLen, 0);
        profInfoItem.name[length] = '\0';

        profInfoItem.start_time_ns = static_cast<uint64_t>(getStartTime(frequenciesSetup));
        profInfoItem.duration_ns = static_cast<uint64_t>(getDuration(frequenciesSetup));

        return profInfoItem;
    }

    virtual void sanitize(vpux::Logger&, FrequenciesSetup) const {
        // do nothing
    }

    virtual TimeType getStartTime(FrequenciesSetup frequenciesSetup) const = 0;

    virtual TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const = 0;

    virtual TimeType getDuration(FrequenciesSetup frequenciesSetup) const {
        return getFinishTime(frequenciesSetup) - getStartTime(frequenciesSetup);
    }

protected:
    RecordType _recordType{RecordType::NONE};
    ExecutorType _executorType{ExecutorType::NONE};

    std::string _name;
    BarriersSet _waitBarriers;
    BarriersSet _updateBarriers;
};

using RawProfilingRecordPtr = std::shared_ptr<RawProfilingRecord>;
using RawProfilingRecords = std::vector<RawProfilingRecordPtr>;

template <class TimestampStorageType, RecordType RECORD_TYPE>
class RawProfilingDMARecord : public RawProfilingRecord {
public:
    using TimestampType = TimestampStorageType;
    using ExtendedTimestampType = uint64_t;

protected:
    RawProfilingDMARecord(TimestampType startCycle, TimestampType endCycle, const std::string& name,
                          const BarriersSet& waitBarriers, const BarriersSet& updateBarriers)
            : RawProfilingRecord(name, RECORD_TYPE, ExecutorType::DMA, waitBarriers, updateBarriers),
              _startCycle(startCycle),
              _endCycle(endCycle) {
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

protected:
    TimestampType _startCycle;
    TimestampType _endCycle;
};

class RawProfilingDMA20Record : public RawProfilingDMARecord<uint32_t, RecordType::DMA20> {
public:
    using TimestampType = typename RawProfilingDMARecord::TimestampType;
    using ExtendedTimestampType = typename RawProfilingDMARecord::ExtendedTimestampType;

public:
    explicit RawProfilingDMA20Record(TimestampType startCycle, TimestampType endCycle, const std::string& name,
                                     const BarriersSet& waitBarriers, const BarriersSet& updateBarriers,
                                     ExtendedTimestampType overflowCorrectionShift)
            : RawProfilingDMARecord<uint32_t, RecordType::DMA20>(startCycle, endCycle, name, waitBarriers,
                                                                 updateBarriers),
              _overflowCorrectionShift(overflowCorrectionShift) {
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(this->getStartCycle(), frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(this->getEndCycle(), frequenciesSetup.profClk);
    }

protected:
    ExtendedTimestampType getStartCycle() const {
        return static_cast<ExtendedTimestampType>(_startCycle) + _overflowCorrectionShift;
    }

    ExtendedTimestampType getEndCycle() const {
        // Use unsigned 32-bit arithmetic to automatically avoid overflow
        TimestampType durationInCycles = _endCycle - _startCycle;
        return getStartCycle() + durationInCycles;
    }

private:
    ExtendedTimestampType _overflowCorrectionShift;
};

class RawProfilingDMA27Record : public RawProfilingDMARecord<uint64_t, RecordType::DMA27> {
public:
    explicit RawProfilingDMA27Record(TimestampType startCycle, TimestampType endCycle, const std::string& name,
                                     const BarriersSet& waitBarriers, const BarriersSet& updateBarriers)
            : RawProfilingDMARecord<uint64_t, RecordType::DMA27>(startCycle, endCycle, name, waitBarriers,
                                                                 updateBarriers) {
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_startCycle, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_endCycle, frequenciesSetup.profClk);
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

class RawProfilingDPURecord : public RawProfilingRecord, public ClusteredAndTiledMixin, public ThrowableAssertMixin {
protected:
    explicit RawProfilingDPURecord(const std::string& name, RecordType recordType, const MVCNN::Task* task,
                                   size_t clusterId, size_t variantId)
            : RawProfilingRecord(recordType, ExecutorType::DPU, name, task),
              ClusteredAndTiledMixin(clusterId, variantId) {
    }

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
        const auto frequency = _recordType == RecordType::DPU_SW ? frequenciesSetup.profClk : frequenciesSetup.dpuClk;
        const auto maxNs = convertTicksToNs(maxCycles, frequency);
        if (maxNs < dpuExecutionTime) {
            log.warning("Too long execution time of DPU task");
        }
    }
};

class RawProfilingDPUSWRecord : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUSWRecord(SwDpuData_t timestamps, const std::string& name, const MVCNN::Task* task,
                                     size_t clusterId, size_t variantId)
            : RawProfilingDPURecord(name, RecordType::DPU_SW, task, clusterId, variantId), _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.begin == 0 && _timestamps.end == 0,
                        "Unexpected end of blob or empty SW DPU profiling data");
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.begin, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.end, frequenciesSetup.profClk);
    }

private:
    SwDpuData_t _timestamps;
};

class RawProfilingDPUHW27Record : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW27Record(HwpDpuMode0Data_t timestamps, const std::string& name, const MVCNN::Task* task,
                                       size_t clusterId, size_t variantId)
            : RawProfilingDPURecord(name, RecordType::DPU_HWP27, task, clusterId, variantId), _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.idu_wl_duration == 0 && _timestamps.odu_wl_duration == 0,
                        "Unexpected end of blob or empty DPU HW27 profiling data");
        VPUX_THROW_UNLESS(_timestamps.reserved3 == 0 && _timestamps.reserved8 == 0, "Reserved values must contain 0.");
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

private:
    HwpDpuMode0Data_t _timestamps;
};

class RawProfilingUPARecord : public RawProfilingRecord, public ThrowableAssertMixin {
public:
    explicit RawProfilingUPARecord(UpaData_t data, const std::string& name, const MVCNN::Task* task,
                                   const std::string& layerType)
            : RawProfilingRecord(RecordType::SW_UPA, ExecutorType::UPA, name, task),
              _data(data),
              _layerType(layerType) {
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
        profInfoItem.active_cycles = _data.activeCycles;
        profInfoItem.stall_cycles = _data.stallCycles;
        if (!_layerType.empty()) {
            const auto typeLen = sizeof(profInfoItem.layer_type);
            strncpy(profInfoItem.layer_type, _layerType.c_str(), typeLen - 1);
            profInfoItem.layer_type[typeLen - 1] = '\0';
        }
        return profInfoItem;
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_data.begin == 0 && _data.end == 0, "Can't process UPA profiling data.");
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.begin, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.end, frequenciesSetup.profClk);
    }

private:
    UpaData_t _data;
    std::string _layerType;
};

class RawProfilingACTRecord : public RawProfilingRecord, public ClusteredAndTiledMixin, public ThrowableAssertMixin {
public:
    explicit RawProfilingACTRecord(ActShaveData_t data, const std::string& name, const MVCNN::Task* task,
                                   size_t clusterId, size_t variantId)
            : RawProfilingRecord(RecordType::SW_ACT, ExecutorType::ACTSHAVE, name, task),
              ClusteredAndTiledMixin(clusterId, variantId),
              _data(data) {
    }

    std::string getTaskName() const override {
        return _name + CLUSTER_LEVEL_PROFILING_SUFFIX + std::to_string(_clusterId) + TILE_LEVEL_PROFILING_SUFFIX +
               std::to_string(_variantId);
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
        profInfoItem.active_cycles = 0;
        profInfoItem.stall_cycles = _data.stallCycles;
        return profInfoItem;
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_data.begin == 0 && _data.duration == 0, "Can't process ACT profiling data.");
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

private:
    ActShaveData_t _data;
};

class ArrayRecord : public RawProfilingRecord {
public:
    ArrayRecord(const std::string name, const RawProfilingRecords& variants, ExecutorType execType)
            : RawProfilingRecord(name, variants.front()->getRecordType(), execType, variants.front()->getWaitBarriers(),
                                 variants.front()->getUpdateBarriers()),
              _variants(variants) {
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

protected:
    RawProfilingRecords _variants;
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
 * @see TaskType
 * @see VerbosityLevel
 * @return std::vector of TaskInfo structures
 */
std::vector<TaskInfo> getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                  TaskType type, VerbosityLevel verbosity, bool fpga = false);

/**
 * @fn getTaskInfoInDebugMode
 * @brief Show raw counters for debug purpose. Intended for use in prof_parser only
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param type type of tasks to be profiled
 * @see DebugInfo
 * @return std::vector of DebugInfo structures
 */
std::vector<DebugInfo> getTaskInfoInDebugMode(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                              size_t profSize, TaskType type);

SummaryInfo getSummary(const uint8_t* blobData, size_t profSize);

/**
 * @fn getLayerInfo
 * @brief Parse raw profiling output to get per-layer info.
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param fpga whether buffer was obtained from FPGA
 * @return std::vector of LayerInfo structures
 */
std::vector<LayerInfo> getLayerInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                    bool fpga = false);

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
