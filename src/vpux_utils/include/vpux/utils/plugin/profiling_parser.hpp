//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef PROFILING_PARSER_HPP
#define PROFILING_PARSER_HPP

#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/profiling.hpp"

#include <schema/graphfile_generated.h>
#include <schema/profiling_generated.h>

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
const std::string CLUSTER_LEVEL_PROFILING_SUFFIX = "cluster";
// Suffix used to create variant name from cluster name
const std::string VARIANT_LEVEL_PROFILING_SUFFIX = "variant";

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

struct HwpDpuIduOduData_t {
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
constexpr double Dma40Bandwidth = 1700. / 45000.;

constexpr int COL_WIDTH_32 = 11;
constexpr int COL_WIDTH_64 = 19;

struct TokenizedTaskName {
    std::string layerName;
    std::vector<std::string> tokens;
};

struct ParsedTaskName {
    std::string layerName;
    std::string layerType;
};

TokenizedTaskName tokenizeTaskName(const std::string& taskName);

// Parses the full task nameinto ParsedTaskName, extracting task name, layer type and cluster id
ParsedTaskName deserializeTaskName(const std::string& taskName);

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
        for (const std::pair<std::string, int>& p : columns) {
            os << std::setw(p.second) << p.first;
        }
    }

    size_t getInMemoryOffset() const {
        return _inMemoryOffset;
    }

    virtual size_t getDebugDataSize() const = 0;

    virtual void printDebugInfo(std::ostream& outStream) const = 0;

private:
    size_t _inMemoryOffset;
};

class RawProfilingRecord {
public:
    using BarrierIdType = uint32_t;
    using TimeType = double;
    using BarriersSet = std::set<BarrierIdType>;

    template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
    static TimeType convertTicksToNs(T cycles, double frequency) {
        VPUX_THROW_WHEN(frequency == FrequenciesSetup::UNITIALIZED_FREQUENCY_VALUE, "Invalid frequency {0}", frequency);
        return static_cast<TimeType>(cycles * 1000. / frequency);
    }

    template <typename RawMetadata>
    static BarriersSet getWaitBarriersFromTask(const RawMetadata* task) {
        if (task == nullptr) {
            return {};
        }
        const auto barrierList = task->waitBarriers();
        return BarriersSet(barrierList->cbegin(), barrierList->cend());
    }

    template <typename RawMetadata>
    static BarriersSet getUpdateBarriersFromTask(const RawMetadata* task) {
        if (task == nullptr) {
            return {};
        }
        const auto barrierList = task->updateBarriers();
        return BarriersSet(barrierList->cbegin(), barrierList->cend());
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

    static TaskInfo::ExecType convertToTaskExec(ExecutorType exec) {
        switch (exec) {
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

protected:
    template <typename RawMetadata>
    RawProfilingRecord(const RawMetadata* metadata, const BarriersSet& wBarriers, const BarriersSet& uBarriers) {
        const auto parsedNameMetaData = deserializeTaskName(metadata->name()->str());
        _name = metadata->name()->str();
        _layerType = parsedNameMetaData.layerType;
        _waitBarriers = wBarriers;
        _updateBarriers = uBarriers;
    }

    template <typename RawMetadata>
    RawProfilingRecord(const RawMetadata* metadata)
            : RawProfilingRecord(metadata, getWaitBarriersFromTask(metadata), getUpdateBarriersFromTask(metadata)) {
    }

    RawProfilingRecord(const std::string& name, const std::string& layerType, const BarriersSet& wBarriers = {},
                       const BarriersSet& uBarriers = {})
            : _name(name), _layerType(layerType), _waitBarriers(wBarriers), _updateBarriers(uBarriers) {
    }

private:
    RawProfilingRecord(const std::string& cleanName, const std::string& layerType, const MVCNN::Task* task)
            : _name(cleanName), _layerType(layerType) {
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

protected:
    virtual ~RawProfilingRecord() = default;

public:
    bool isDirectPredecessor(const RawProfilingRecord& other) const {
        return !isSetIntersectionEmpty(_updateBarriers, other._waitBarriers);
    }

    bool isDirectSuccessor(const RawProfilingRecord& other) const {
        return !isSetIntersectionEmpty(_waitBarriers, other._updateBarriers);
    }

    virtual ExecutorType getExecutorType() const = 0;

    const BarriersSet& getWaitBarriers() const {
        return _waitBarriers;
    }

    const BarriersSet& getUpdateBarriers() const {
        return _updateBarriers;
    }

    std::string getOriginalName() const {
        return _name;
    }

    virtual std::string getTaskName() const {
        return _name;
    }

    std::string getLayerType() const {
        return _layerType;
    }

    virtual TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const {
        TaskInfo taskInfo;
        taskInfo.exec_type = convertToTaskExec(getExecutorType());
        taskInfo.start_time_ns = static_cast<uint64_t>(getStartTime(frequenciesSetup));
        taskInfo.duration_ns = static_cast<uint64_t>(getDuration(frequenciesSetup));

        const auto nameLen = getTaskName().copy(taskInfo.name, sizeof(taskInfo.name) - 1);
        taskInfo.name[nameLen] = 0;

        const auto typeLen = getLayerType().copy(taskInfo.layer_type, sizeof(taskInfo.layer_type) - 1);
        taskInfo.layer_type[typeLen] = 0;

        return taskInfo;
    }

    virtual void checkDataOrDie() const {
        VPUX_THROW("checkDataOrDie not implemented");
    }

    virtual void sanitize(vpux::Logger&, FrequenciesSetup) const {
        // do nothing in base
    }

    virtual TimeType getStartTime(FrequenciesSetup frequenciesSetup) const = 0;

    virtual TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const = 0;

    virtual TimeType getDuration(FrequenciesSetup frequenciesSetup) const {
        return getFinishTime(frequenciesSetup) - getStartTime(frequenciesSetup);
    }

private:
    std::string _name;

protected:
    std::string _layerType;

private:
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
    RawProfilingDMARecord(const RecordDataType& record, const ProfilingFB::DMATask* metadata,
                          const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t inMemoryOffset)
            : RawProfilingRecord(metadata, wBarriers, uBarriers),
              DebugFormattableRecordMixin(inMemoryOffset),
              _record(record) {
    }

    RawProfilingDMARecord(const RecordDataType& record, const ProfilingFB::DMATask* metadata, size_t inMemoryOffset)
            : RawProfilingRecord(metadata), DebugFormattableRecordMixin(inMemoryOffset), _record(record) {
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

protected:
    RecordDataType _record;
};

template <class RecordDataType>
class RawProfilingSwDmaRecord : public RawProfilingDMARecord<RecordDataType> {
public:
    using RawProfilingDMARecord<RecordDataType>::RawProfilingDMARecord;
    using ColDesc = DebugFormattableRecordMixin::ColDesc;
    using BarriersSet = RawProfilingRecord::BarriersSet;

protected:
    RawProfilingSwDmaRecord(const RecordDataType& record, const ProfilingFB::DMATask* metadata,
                            const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t inMemoryOffset)
            : RawProfilingDMARecord<RecordDataType>(record, metadata, wBarriers, uBarriers, inMemoryOffset) {
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
    using ExtendedTimestampType = RawProfilingDMARecord::ExtendedTimestampType;

public:
    explicit RawProfilingDMA20Record(const DMA20Data_t& record, const ProfilingFB::DMATask* metadata,
                                     const BarriersSet& wBarriers, const BarriersSet& uBarriers,
                                     ExtendedTimestampType overflowCorrectionShift, size_t inMemoryOffset)
            : RawProfilingSwDmaRecord<DMA20Data_t>(record, metadata, wBarriers, uBarriers, inMemoryOffset),
              _overflowCorrectionShift(overflowCorrectionShift) {
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DMA_SW;
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
    explicit RawProfilingDMA27Record(const DMA27Data_t& record, const ProfilingFB::DMATask* metadata,
                                     const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t inMemoryOffset)
            : RawProfilingSwDmaRecord<DMA27Data_t>(record, metadata, wBarriers, uBarriers, inMemoryOffset) {
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DMA_SW;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.startCycle, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.endCycle, frequenciesSetup.profClk);
    }
};

class RawProfilingDMA40Record : public RawProfilingDMARecord<HwpDma40Data_t> {
public:
    explicit RawProfilingDMA40Record(const HwpDma40Data_t& record, const ProfilingFB::DMATask* metadata,
                                     size_t inMemoryOffset)
            : RawProfilingDMARecord<HwpDma40Data_t>(record, metadata, inMemoryOffset) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_UNLESS(_record.rsvd == 0, "Reserved value must contain 0.");
        VPUX_THROW_WHEN(_record.desc_addr == 0, "Invalid DMA descriptor address.");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DMA_HW;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.start_time, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.finish_time, frequenciesSetup.profClk);
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

class RawProfilingDPURecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
protected:
    RawProfilingDPURecord(const ProfilingFB::DPUTask* metadata, uint32_t variantId, size_t inMemoryOffset,
                          uint32_t inClusterOffset)
            : RawProfilingRecord(metadata),
              DebugFormattableRecordMixin(inMemoryOffset),
              _bufferId(metadata->bufferId()),
              _inClusterIndex(inClusterOffset),
              _clusterId(metadata->clusterId()),
              _variantId(variantId) {
    }

    virtual double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const = 0;

public:
    std::string getTaskName() const override {
        // adding variant suffix as it is not stored in meta data
        return getOriginalName() + "/" + VARIANT_LEVEL_PROFILING_SUFFIX + "_" + std::to_string(_variantId);
    }

    void sanitize(vpux::Logger& log, FrequenciesSetup frequenciesSetup) const override {
        const auto dpuExecutionTime = this->getDuration(frequenciesSetup);
        const uint64_t maxKernel = 11 * 11;
        const uint64_t maxElem = 2ll * 1024ll * 1024ll;
        const uint64_t maxChannels = 8192;
        const uint64_t maxCycles = maxKernel * maxElem * maxChannels / 256;
        const auto frequency = this->getTaskDurationClock(frequenciesSetup);
        const auto maxNs = convertTicksToNs(maxCycles, frequency);
        if (maxNs < dpuExecutionTime) {
            log.warning("Too long execution time of DPU task");
        }
    }

    size_t getClusterId() {
        return _clusterId;
    }

protected:
    uint32_t _bufferId;
    uint32_t _inClusterIndex;
    uint32_t _clusterId;
    uint32_t _variantId;
};

class RawProfilingDPUSWRecord : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUSWRecord(SwDpuData_t timestamps, const ProfilingFB::DPUTask* metadata, uint32_t variantId,
                                     size_t inMemoryOffset, uint32_t inClusterOffset)
            : RawProfilingDPURecord(metadata, variantId, inMemoryOffset, inClusterOffset), _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.begin == 0 && _timestamps.end == 0, "Invalid DPU task timestamp");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DPU;
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
    double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.profClk;
    }

    ColDesc getColDesc() const override {
        return {{"Buffer ID", COL_WIDTH_32},
                {"Cluster ID", COL_WIDTH_64},
                {"Buffer offset", COL_WIDTH_64},
                {"Begin tstamp", COL_WIDTH_64},
                {"End tstamp", COL_WIDTH_64}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto swDpuCol = getColDesc();
        const auto bufferOffsetBytes = _inClusterIndex * getDebugDataSize();

        outStream << std::setw(swDpuCol[0].second) << _bufferId << std::setw(swDpuCol[1].second) << _clusterId
                  << std::setw(swDpuCol[2].second) << bufferOffsetBytes << std::setw(swDpuCol[3].second)
                  << _timestamps.begin << std::setw(swDpuCol[4].second) << _timestamps.end;
    }

private:
    SwDpuData_t _timestamps;
};

class RawProfilingDPUHW27Record : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW27Record(HwpDpu27Mode0Data_t timestamps, const ProfilingFB::DPUTask* metadata,
                                       uint32_t variantId, size_t inMemoryOffset, uint32_t inClusterOffset)
            : RawProfilingDPURecord(metadata, variantId, inMemoryOffset, inClusterOffset), _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.idu_wl_duration == 0 && _timestamps.odu_wl_duration == 0,
                        "Invalid DPU task duration");
        VPUX_THROW_UNLESS(_timestamps.reserved3 == 0 && _timestamps.reserved8 == 0, "Reserved values must contain 0.");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DPU;
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
    double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.dpuClk;
    }

    ColDesc getColDesc() const override {
        return {{"Buffer ID", COL_WIDTH_32},
                {"Cluster ID", COL_WIDTH_64},
                {"Buffer offset", COL_WIDTH_64},
                {"IDU dur", COL_WIDTH_32},
                {"IDU tstamp", COL_WIDTH_32},
                {"SWE ID", 7},
                {"Rvd", 4},
                {"ODU dur", COL_WIDTH_32},
                {"ODU tstamp", COL_WIDTH_32},
                {"Rvd", 7}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto hwpDpuCol = getColDesc();
        const auto bufferOffsetBytes = _inClusterIndex * getDebugDataSize();

        outStream << std::setw(hwpDpuCol[0].second) << _bufferId << std::setw(hwpDpuCol[1].second) << _clusterId
                  << std::setw(hwpDpuCol[2].second) << bufferOffsetBytes << std::setw(hwpDpuCol[3].second)
                  << _timestamps.idu_wl_duration << std::setw(hwpDpuCol[4].second) << _timestamps.idu_tstamp
                  << std::setw(hwpDpuCol[5].second) << _timestamps.sve_id << std::setw(hwpDpuCol[6].second)
                  << _timestamps.reserved3 << std::setw(hwpDpuCol[7].second) << _timestamps.odu_wl_duration
                  << std::setw(hwpDpuCol[8].second) << _timestamps.odu_tstamp << std::setw(hwpDpuCol[9].second)
                  << _timestamps.reserved8;
    }

private:
    HwpDpu27Mode0Data_t _timestamps;
};

class RawProfilingDPUHW40Record : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW40Record(HwpDpuIduOduData_t timestamps, const ProfilingFB::DPUTask* metadata,
                                       uint32_t variantId, size_t inMemoryOffset, uint32_t inClusterOffset)
            : RawProfilingDPURecord(metadata, variantId, inMemoryOffset, inClusterOffset), _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.idu_wl_duration == 0 && _timestamps.odu_wl_duration == 0,
                        "Invalid DPU task duration");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DPU;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.profClk) -
               convertTicksToNs(_timestamps.idu_wl_duration, frequenciesSetup.dpuClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.odu_tstamp, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(HwpDpuIduOduData_t);
    }

protected:
    double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.dpuClk;
    }

    ColDesc getColDesc() const override {
        return {{"Buffer ID", COL_WIDTH_32},
                {"Cluster ID", COL_WIDTH_64},
                {"Buffer offset", COL_WIDTH_64},
                {"IDU dur", COL_WIDTH_32},
                {"IDU tstamp", COL_WIDTH_64},
                {"IDU WL ID", 11},
                {"IDU DPU ID", 12},
                {"ODU dur", COL_WIDTH_32},
                {"ODU tstamp", COL_WIDTH_64},
                {"ODU WL ID", 11},
                {"ODU DPU ID", 12}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto hwpDpuCol = getColDesc();
        const auto bufferOffsetBytes = _inClusterIndex * getDebugDataSize();

        outStream << std::setw(hwpDpuCol[0].second) << _bufferId << std::setw(hwpDpuCol[1].second) << _clusterId
                  << std::setw(hwpDpuCol[2].second) << bufferOffsetBytes << std::setw(hwpDpuCol[3].second)
                  << _timestamps.idu_wl_duration << std::setw(hwpDpuCol[4].second) << _timestamps.idu_tstamp
                  << std::setw(hwpDpuCol[5].second) << _timestamps.idu_wl_id << std::setw(hwpDpuCol[6].second)
                  << _timestamps.idu_dpu_id << std::setw(hwpDpuCol[5].second) << _timestamps.odu_wl_duration
                  << std::setw(hwpDpuCol[7].second) << _timestamps.odu_tstamp << std::setw(hwpDpuCol[8].second)
                  << _timestamps.odu_wl_id << std::setw(hwpDpuCol[9].second) << _timestamps.odu_dpu_id;
    }

protected:
    HwpDpuIduOduData_t _timestamps;
};

class RawProfilingDPUHW50Record : public RawProfilingDPUHW40Record {
public:
    using RawProfilingDPUHW40Record::RawProfilingDPUHW40Record;

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.profClk) -
               convertTicksToNs(_timestamps.idu_wl_duration, frequenciesSetup.profClk);
    }

protected:
    double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.profClk;
    }
};

class RawProfilingUPARecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    explicit RawProfilingUPARecord(UpaData_t data, const ProfilingFB::SWTask* metadata, size_t inMemoryOffset)
            : RawProfilingRecord(metadata), DebugFormattableRecordMixin(inMemoryOffset), _data(data) {
        // TODO: Why we don't derive layer type from the task name for UPA?
        if (metadata->taskType() != nullptr) {
            _layerType = metadata->taskType()->str();
        }
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

    ExecutorType getExecutorType() const override {
        return ExecutorType::UPA;
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

private:
    UpaData_t _data;
};

class RawProfilingACTRecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    explicit RawProfilingACTRecord(ActShaveData_t data, const ProfilingFB::SWTask* metadata, size_t inMemoryOffset)
            : RawProfilingRecord(metadata),
              DebugFormattableRecordMixin(inMemoryOffset),
              _data(data),
              _bufferId(metadata->bufferId()),
              _inClusterIndex(metadata->dataIndex()),
              _clusterId(metadata->clusterId()) {
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
        profInfoItem.stall_cycles = _data.stallCycles;
        return profInfoItem;
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_data.begin == 0 && _data.duration == 0, "Can't process ACT profiling data.");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::ACTSHAVE;
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

protected:
    ColDesc getColDesc() const override {
        return {{"Buffer ID", COL_WIDTH_32}, {"Cluster ID", COL_WIDTH_64}, {"Buffer offset", COL_WIDTH_64},
                {"Begin", COL_WIDTH_64},     {"Duration", COL_WIDTH_32},   {"Stall", COL_WIDTH_32}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto actShaveCol = getColDesc();
        const auto bufferOffsetBytes = _inClusterIndex * getDebugDataSize();

        outStream << std::setw(actShaveCol[0].second) << _bufferId << std::setw(actShaveCol[1].second) << _clusterId
                  << std::setw(actShaveCol[2].second) << bufferOffsetBytes << std::setw(actShaveCol[3].second)
                  << _data.begin << std::setw(actShaveCol[4].second) << _data.duration
                  << std::setw(actShaveCol[5].second) << _data.stallCycles;
    }

private:
    ActShaveData_t _data;
    uint32_t _bufferId;
    uint32_t _inClusterIndex;
    uint32_t _clusterId;
};

class ArrayRecord : public RawProfilingRecord {
public:
    ArrayRecord(const std::string name, const RawProfilingRecords& records)
            : RawProfilingRecord(name, records.front()->getLayerType(), records.front()->getWaitBarriers(),
                                 records.front()->getUpdateBarriers()),
              _records(records) {
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return std::accumulate(_records.cbegin(), _records.cend(), std::numeric_limits<TimeType>::max(),
                               [&](TimeType a, RawProfilingRecordPtr variant) -> TimeType {
                                   return std::min(a, variant->getStartTime(frequenciesSetup));
                               });
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return std::accumulate(_records.cbegin(), _records.cend(), std::numeric_limits<TimeType>::min(),
                               [&](TimeType a, RawProfilingRecordPtr variant) -> TimeType {
                                   return std::max(a, variant->getFinishTime(frequenciesSetup));
                               });
    }

    ExecutorType getExecutorType() const override {
        VPUX_THROW_WHEN(_records.size() == 0, "Empty ArrayRecord");
        return _records.front()->getExecutorType();
    }

protected:
    RawProfilingRecords _records;
};

using WorkpointRecords = std::vector<std::pair<WorkpointConfiguration_t, size_t>>;

// Container for conjucted storage of tasks of one format: RawProfilingRecordPtr/TaskInfo
struct RawProfilingData {
    RawProfilingRecords dmaTasks;
    RawProfilingRecords dpuTasks;
    RawProfilingRecords swTasks;
    // Pair of workpoint and offset
    WorkpointRecords workpoints;
    // Vector of [ExecutorType; offset in blob(bytes)]
    std::vector<std::pair<ExecutorType, size_t>> parseOrder;

    RawProfilingRecords getTaskOfType(ExecutorType type) const;
};

// Map of exec. type to section offset and size
using RawDataLayout = std::map<ExecutorType, std::pair<uint32_t, uint32_t>>;

struct RawData {
    RawDataLayout sections;
    RawProfilingData rawRecords;
    MVCNN::TargetDevice device;
    std::optional<double> maybe30XXNceFreq;
};

/**
 * @fn getTaskInfo
 * @brief Parse raw profiling output to get per-tasks info.
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param verbosity amount of DPU info to print, may be LOW|MEDIUM|HIGH
 * @param fpga whether buffer was obtained from FPGA
 * @param ignoreSanitizationErrors to ignore sanitization errors
 * @see TaskType
 * @see VerbosityLevel
 * @return std::vector of TaskInfo structures
 */
std::vector<TaskInfo> getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                  VerbosityLevel verbosity, bool fpga = false, bool ignoreSanitizationErrors = false);

/**
 * @fn getRawProfilingTasks
 * @brief Show raw counters for debug purpose. Intended for use in prof_parser only
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param ignoreSanitizationErrors to ignore sanitization errors
 * @return RawProfilingData
 */
RawData getRawProfilingTasks(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                             bool ignoreSanitizationErrors = false);

/**
 * @brief Helper function to parse profiling buffer name and extract buffer offsets and sizes
 *
 * @param profilingOutputName - profiling buffer name (e.g. 0_dpu_6432_actshave_6464_dma_15744_pll)
 * @param profilingOutputSize - total size of the profiling output in bytes
 * @return offsets and sizes per ExecutorType in bytes
 *
 */

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

/**
 * @brief Extract suffix from original task name
 *
 * @param name - task name
 * @return task name suffixes after original name separator (if present)
 *
 * Eg. for name = Subtract_1751?t_Add/cluster_0/variant_0
 * the function returns "t_Add/cluster_0/variant_0"
 * The original task name before ? is ignored.
 */
std::string getTaskNameSuffixes(const std::string& name);

/**
 * @brief Extract cluster id from task name suffixes
 *
 * @param name - task name
 * @return cluster id suffix
 *
 * Eg. for name = Subtract_1751?t_Add/cluster_0/variant_1
 * the function returns "0"
 * Original task name (i.e. string before ?) is ignored.
 */
std::string getClusterFromName(const std::string& name);

/**
 * @brief Extract variant id from task name suffixes
 *
 * @param name - task name
 * @return variant id suffix
 *
 * Eg. for name = Subtract_1751?t_Add/cluster_0/variant_0
 * the function returns "0"
 * Original task name (i.e. string before ?) is ignored.
 */
std::string getVariantFromName(const std::string& name);

/**
 * @brief Extract a value from a structured task name string
 *
 * @param name - structured task name string in format prefix1/prefix2/key1_val1/key2_val2
 * @param key - keyword to have value extracted eg: "key1"
 * @return std::string - extracted value starting a character after '_' and ending on either the end of the string
 * or a keyword delimiter '/'
 *
 * Eg.
 *
 * For "origTaskName?key1_val1/key2_val2" and key "key1", the function yields "val1",
 * for "origTaskName?key1_val1/key2_val2" and key "key2", the function yields "val2",
 * for "origTaskName?key1_val1/key2_val2" and key "key3", the function yields ""
 *
 * Original task name (i.e. string before ?) is ignored.
 */
std::string getValueFromStructuredTaskName(const std::string& name, std::string key);

}  // namespace profiling
}  // namespace vpux

#endif
