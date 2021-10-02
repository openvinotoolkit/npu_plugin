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

#pragma once

#include "vpux/compiler/core/disjoint_interval_set.hpp"
#include "vpux/compiler/core/feasible_scheduler_utils.hpp"

namespace vpux {

struct ControlEdge {
    ControlEdge(size_t source, size_t sink): _source(source), _sink(sink) {
    }

    bool operator<(const ControlEdge& o) const {
        return (_source != o._source) ? (_source < o._source) : (_sink < o._sink);
    }

    std::string sourceName() const {
        return std::to_string(static_cast<int>(_source));
    }
    std::string sinkName() const {
        return std::to_string(static_cast<int>(_sink));
    }

    size_t _source;
    size_t _sink;
};  // struct ControlEdge //

template <>
struct IntervalTraits<ScheduledOpOneResource> {
    using UnitType = size_t;
    using IntervalType = ScheduledOpOneResource;

    static UnitType intervalBegin(const IntervalType& interval) {
        return interval._addressStart;
    }

    static UnitType intervalEnd(const IntervalType& interval) {
        return interval._addressEnd;
    }
};  // struct IntervalTraits<ScheduledOpOneResource> //

class ControlEdgeSet {
public:
    ControlEdgeSet(): _controlEdgeSet() {
    }

    void operator()(const ScheduledOpOneResource& a, const ScheduledOpOneResource& b) {
        _controlEdgeSet.push_back(ControlEdge(a._op, b._op));
    }

    SmallVector<ControlEdge>::const_iterator begin() const {
        return _controlEdgeSet.begin();
    }
    SmallVector<ControlEdge>::const_iterator end() const {
        return _controlEdgeSet.end();
    }

private:
    SmallVector<ControlEdge> _controlEdgeSet;
};  //  class ControlEdgeSet //

// Given an iterator over sorted intervals the algorithm produces control
// edges for overlapping memory ranges in order of schedule
template <typename T>
class ControlEdgeGenerator {
public:
    using Traits = IntervalTraits<T>;
    using UnitType = typename Traits::UnitType;
    using IntervalType = typename Traits::IntervalType;
    using IntervalUtilsType = IntervalUtils<IntervalType>;
    using IntervalTreeType = DisjointIntervalSet<UnitType, IntervalType>;
    using IntervalQueryIteratorType = typename IntervalTreeType::IntervalIteratorType;
    struct NoopFunctorType {
        void operator()(const IntervalType&, const IntervalType&) {
        }
    };  // struct NoopFunctorType //

    ControlEdgeGenerator(): _intervalTree() {
    }

    template <typename IntervalIterator, typename OutputFunctor = NoopFunctorType>
    size_t generateControlEdges(IntervalIterator begin, IntervalIterator end, OutputFunctor& output = OutputFunctor()) {
        size_t edgeCount = 0UL;
        for (; begin != end; ++begin) {
            const auto currBeg = Traits::intervalBegin(*begin);
            const auto currEnd = Traits::intervalEnd(*begin);
            if (currBeg > currEnd) {
                continue;
            }
            edgeCount += processNextInterval(*begin, output);
        }
        return edgeCount;
    }

protected:
    bool overlaps(const IntervalQueryIteratorType& qitr, UnitType ibegin, UnitType iend) const {
        return IntervalUtilsType::intersects(qitr.intervalBegin(), qitr.intervalEnd(), ibegin, iend);
    }

    template <typename OutputFunctor = NoopFunctorType>
    size_t processNextInterval(const IntervalType& currInterval, OutputFunctor& output = OutputFunctor()) {
        size_t edgeCount = 0UL;
        const auto currBeg = Traits::intervalBegin(currInterval);
        const auto currEnd = Traits::intervalEnd(currInterval);
        assert(currBeg <= currEnd);

        auto qitr = _intervalTree.query(currBeg, currEnd);
        auto qitrEnd = _intervalTree.end(), qitrNext = qitrEnd;

        // Invariant: [currRemBeg, currRemEnd] is the reminder of the
        // current interval which does not overlap intervals until qitr //
        auto currRemBeg = currBeg, currRemEnd = currEnd;
        while ((qitr != qitrEnd) && overlaps(qitr, currRemBeg, currRemEnd)) {  // foreach overlap //
            auto qinterval = *qitr;

            assert((currRemBeg <= currRemEnd) && (currBeg <= currRemBeg) && (currRemEnd <= currEnd));

            // output the control edge //
            output(qinterval, currInterval);
            ++edgeCount;

            // we now have an overlap between [qbeg,qend] and
            // [currRemBeg,currRemEnd] //

            auto qbeg = qitr.intervalBegin();
            auto qend = qitr.intervalEnd();

            // erase the current interval //
            qitrNext = qitr;
            ++qitrNext;
            _intervalTree.erase(qbeg, qend);

            // compute the intersecting interval //
            auto interBeg = std::max(currRemBeg, qbeg);
            auto interEnd = std::min(currRemEnd, qend);
            assert(interBeg <= interEnd);

            UnitType resultBeg[2UL], resultEnd[2UL];
            IntervalType const* resultVal[2UL];
            size_t rcount;

            // now compute the xor interval(s) //
            rcount = IntervalUtilsType::intervalXor(qbeg, qend, currRemBeg, currRemEnd, resultBeg, resultEnd);

            auto nextRemBeg = std::numeric_limits<UnitType>::max();
            auto nextRemEnd = std::numeric_limits<UnitType>::min();
            assert(rcount <= 2UL);

            if (!rcount) {
                // no xor intervals //
                const bool status = _intervalTree.insert(interBeg, interEnd, currInterval);
                assert(status);
                std::ignore = status;
            } else {
                // NOTE: that first XOR interval comes before the second if there
                // are two XOR intervals. //

                // Determine the type of interval in the xor //
                //
                // qbeg     qend
                // [--------]
                //        [----------------]
                //      currRemBeg       currRemEnd
                //
                // the xor has atmost two parts:
                // [------)
                //          (--------------]
                for (size_t r = 0; r < rcount; r++) {
                    if (IntervalUtilsType::isSubset(resultBeg[r], resultEnd[r], currRemBeg, currRemEnd)) {
                        resultVal[r] = &currInterval;
                    } else {
                        assert(IntervalUtilsType::isSubset(resultBeg[r], resultEnd[r], qbeg, qend));
                        resultVal[r] = &qinterval;
                    }
                }

                size_t nextXorIntervalIndex = 0;

                if (resultEnd[nextXorIntervalIndex] < interBeg) {
                    // insert the interval part before [interBeg, interEnd] //
                    _intervalTree.insert(resultBeg[nextXorIntervalIndex], resultEnd[nextXorIntervalIndex],
                                         *(resultVal[nextXorIntervalIndex]));
                    ++nextXorIntervalIndex;
                }

                // insert the intersection part //
                _intervalTree.insert(interBeg, interEnd, currInterval);

                // process the interval above the intersection part //
                if (nextXorIntervalIndex < rcount) {
                    if (resultVal[nextXorIntervalIndex] != &currInterval) {
                        // need to insert this part back in the interval tree //
                        _intervalTree.insert(resultBeg[nextXorIntervalIndex], resultEnd[nextXorIntervalIndex],
                                             *(resultVal[nextXorIntervalIndex]));
                    } else {
                        // pass the remaining part of the currInterval to the next
                        // iteraton.
                        nextRemBeg = resultBeg[nextXorIntervalIndex];
                        nextRemEnd = resultEnd[nextXorIntervalIndex];
                    }
                }
            }

            // update the remaining part of the current interval //
            currRemBeg = nextRemBeg;
            currRemEnd = nextRemEnd;
            qitr = qitrNext;
        }  // foreach overlap //

        if (currRemBeg <= currRemEnd) {
            // process the trailing part this also covers the trailing case.//
            const auto status = _intervalTree.insert(currRemBeg, currRemEnd, currInterval);
            assert(status);
            std::ignore = status;
        }

        // TODO: do the merging within the update itself //
        mergeAbuttingIntervals(currInterval);
        return edgeCount;
    }

    void mergeAbuttingIntervals(const IntervalType& currInterval) {
        UnitType ibeg = Traits::intervalBegin(currInterval);
        UnitType iend = Traits::intervalEnd(currInterval);
        IntervalQueryIteratorType qitr = _intervalTree.query(ibeg, iend);
        IntervalQueryIteratorType qitrEnd = _intervalTree.end(), qitrNext, qitrStart;

        if ((qitr == qitrEnd) || !(*qitr == currInterval)) {
            return;
        }

        qitrStart = qitr;
        UnitType prevLeftEnd = qitr.intervalBegin();
        UnitType prevRightEnd = qitr.intervalEnd();

        ++qitr;
        while ((qitr != qitrEnd) && ((*qitr == currInterval) && ((prevRightEnd + 1) == qitr.intervalBegin()))) {
            prevRightEnd = qitr.intervalEnd();
            // TODO: implement an erase using iterators instead of
            // end point values so that this takes O(1) time.
            qitrNext = qitr;
            ++qitrNext;
            _intervalTree.erase(qitr.intervalBegin(), qitr.intervalEnd());
            qitr = qitrNext;
        }

        if (prevRightEnd > qitrStart.intervalEnd()) {
            _intervalTree.erase(qitrStart.intervalBegin(), qitrStart.intervalEnd());
            _intervalTree.insert(prevLeftEnd, prevRightEnd, currInterval);
        }
    }

    IntervalTreeType _intervalTree;
};  // class ControlEdgeGenerator //

}  // namespace vpux
