#include "include/mcm/target/kmb/pwl/pwl_dyn_fit.hpp"

#include <limits>
#include <cmath>
#include <iostream>
#include <algorithm>

void PWLFunction::Segment::checkFunction(const std::vector<int> & refFunction, int nBits) const {
    if (nBits > 32 || nBits == 0) {
        throw std::logic_error("PWLFit: invalid bitness: " + std::to_string(nBits));
    }
    // checking refFunction
    if ((1u << nBits) != refFunction.size()) {
        throw std::logic_error("PWLFunction::Segment reference function for " + std::to_string(nBits)
                               + " bits, contains only " + std::to_string(refFunction.size()) + " values");
    }

    int zeroPoint = (1 << (nBits - 1));

    // checking segment
    if (_data.begin() < -zeroPoint || _data.begin() >= zeroPoint) {
        throw std::logic_error("PWLFunction::Segment invalid segment starPoint " + std::to_string(_data.begin())
                               + " for given bitness: " + std::to_string(nBits));
    }

    if (_data.end() >  zeroPoint) {
        throw std::logic_error("PWLFunction::Segment invalid segment endPoint " + std::to_string(_data.end())
                               + " for given bitness: " + std::to_string(nBits));
    }
}
int PWLFunction::Segment::costInternal(const std::vector<int> & refFunction, int nBits) const {
    int zeroPoint = (1 << (nBits - 1));

    int diff = 0;
    for (int i = _data.begin(); i < _data.end(); i++) {
        auto value = evaluate(i);
        diff += abs(value - refFunction[zeroPoint + i]);
    }

    return diff;

}
int PWLFunction::Segment::cost(const std::vector<int> & refFunction, int nBits) const {
    checkFunction(refFunction, nBits);
    return costInternal(refFunction, nBits);
}

int PWLFunction::Segment::fitfast(const std::vector<int> & refFunction, int nBits) {
    checkFunction(refFunction, nBits);
    // check nothing to fit case
    if (_data.empty()) {
        throw std::logic_error("PWLFunction::Segment::fit empty interval for fitting: ["
                               + std::to_string(_data.begin()) + ", " + std::to_string(_data.end()) + ")");
    }

    int zeroPoint = (1 << (nBits - 1));

    // trivial fit
    if (_data.end() - _data.begin() == 1) {
        _shift = 0;
        auto y = refFunction[_data.begin() + zeroPoint];
        _bias = y - _data.begin();
        return costInternal(refFunction, nBits);
    }

    // least square error
    int sXY = 0;
    int sX  = 0;
    int sX2 = 0;
    int sY  = 0;


    for (auto x = _data.begin(); x < _data.end(); x++) {
        auto y = refFunction[x + zeroPoint];
        sX += x;
        sY += y;
        sXY += x*y;
        sX2 += x*x;
    }
    float N = (_data.end() - _data.begin());
    float a = (N * sXY - sX * sY) / (N * sX2 - sX * sX);

    // lets consider two cases floor(a) ceil(a)
    std::vector<std::vector<float>> slope_biases;
    auto fill_slope_bias = [&](float slope) {
        float b = (sY - pow(2, slope) * sX) / N;
        if (floor(b) == ceil(b)) {
            return std::vector<float>{slope, floorf(b)};
        }
        return std::vector<float>{slope, floorf(b), ceilf(b)};
    };

    if (a > 0) {
        auto s = log2(a) / log2(2);
        // check both slopes ceil(s) floor(s)

        slope_biases.push_back(fill_slope_bias(ceil(s)));
        if (floor(s) != ceil(s)) {
            slope_biases.push_back(fill_slope_bias(floor(s)));
        }
        // TODO: this case looks interesting since prediction using direct LSM works not good for slope prediction close to zero
        // and short number of points
        if (floor(s) < 0) {
            slope_biases.push_back(fill_slope_bias(floor(s) - 1));
        }
    } else {
        slope_biases.push_back({nBits * -1.f, 0.f});
    }

    auto applyShiftAndBias = [&] (const float shift_c, const float bias_c) {
        float b = bias_c;

        if (shift_c == -1.f * nBits) {
            _shift = -nBits;
            // if whole segment is negative
            if (_data.begin() < 0 && _data.end() <= 0) {
                 b = (sY + N) / N;
            }
            if (_data.begin() < 0 && _data.end() > 0) {
                b = (sY - _data.begin()) / N;
            }
            if (_data.begin() >= 0) {
                b = sY / N;
            }
        } else {
            _shift = roundf(shift_c);
        }
        _bias = roundf(b);
    };

    int minB = 0, minS = 0, minCost = costInfinity;
    for (auto && sb : slope_biases) {
        for (size_t i = 1; i != sb.size() && minCost != 0; i++) {
            applyShiftAndBias(sb.front(), sb[i]);
            auto cost = costInternal(refFunction, nBits);
            if (cost < minCost) {
                minCost = cost;
                minB = _bias;
                minS = _shift;
            }
        }
        if (minCost == 0) {
            break;
        }
    }

    _shift = minS;
    _bias = minB;

    return minCost;
}

int PWLFunction::Segment::fit(const std::vector<int> & refFunction, int nBits, const Range & biasRange, const Range & shiftRange) {
    checkFunction(refFunction, nBits);
    // check nothing to fit case
    if (_data.empty()) {
        throw std::logic_error("PWLFunction::Segment::fit empty interval for fitting: ["
                               + std::to_string(_data.begin()) + ", " + std::to_string(_data.end()) + ")");
    }

    if (biasRange.empty()) {
        throw std::logic_error("PWLFunction::Segment::fit empty bias range for fitting: ["
                               + std::to_string(biasRange.begin()) + ", " + std::to_string(biasRange.end()) + ")");
    }

    if (shiftRange.empty()) {
        throw std::logic_error("PWLFunction::Segment::fit empty shift range for fitting: ["
                               + std::to_string(shiftRange.begin()) + ", " + std::to_string(shiftRange.end()) + ")");
    }

    // bruteforce solution minimizing const function
    int minCost = std::numeric_limits<int>::max();
    int optBias = _bias;
    int optShift = _shift;

    for (auto b = biasRange.begin(); b != biasRange.end(); b++) {
        for (auto s = shiftRange.begin(); s != shiftRange.end(); s++) {
            _bias = b;
            _shift = s;

            auto currentCost = costInternal(refFunction, nBits);
            if (currentCost < minCost) {
                minCost = currentCost;
                optBias = b;
                optShift = s;
            }
        }
    }
    _bias = optBias;
    _shift = optShift;
    return minCost;
}

template <class Tx, class Ty>
PWLFit<Tx, Ty> & PWLFit<Tx, Ty>::setRange(Tx minX, Tx maxX) {
    bool invalidRangeError = false;
    if (std::is_floating_point<Tx>::value) {
        if (minX + std::numeric_limits<Tx>::epsilon() >= maxX) {
            invalidRangeError = true;
        }
    } else {
        if (minX >= maxX) {
            invalidRangeError = true;
        }
    }
    if (invalidRangeError) {
        throw std::logic_error("PWLFit: invalid range: " + std::to_string(minX) + ", " + std::to_string(maxX));
    }
    _range = {minX, maxX};
    return *this;
}

template <class Tx, class Ty>
PWLFit<Tx, Ty> & PWLFit<Tx, Ty>::setBitness(uint8_t nBits) {

    if (nBits > 32 || nBits == 0) {
        throw std::logic_error("PWLFit: invalid bitness: " + std::to_string(nBits));
    }

    this->_nBits = nBits;
    return *this;
}

template <class Tx, class Ty>
PWLFit<Tx, Ty> & PWLFit<Tx, Ty>::setMaxIntervals(uint8_t nIntervals) {
    if (nIntervals == 0) {
        throw std::logic_error("PWLFit: invalid number of intervals: " + std::to_string(nIntervals));
    }
    this->_nIntervals = nIntervals;
    return *this;
}

template <class Tx, class Ty>
PWLFunction  PWLFit<Tx, Ty>::solve() const {
    // check params
    if (!_aproximatedFunction) {
        throw std::logic_error("PWLFit: function not set");
    }
    if (_range == std::pair<Tx, Tx>()) {
        throw std::logic_error("PWLFit: range set");
    }
    if (_nBits == 0) {
        throw std::logic_error("PWLFit: bitness not set");
    }

    std::vector<PWLFunction::Segment> pwl_segments;

    auto  refFunction = genReferenceIntFunction();
    auto costFunction = genCostFunction(refFunction);

    auto make2D = [](int x, int y) {
        return std::vector<std::vector<int>>(x, std::vector<int>(y));
    };
    auto nPoints = _nIntervals + 1;
    auto solution = make2D(1 << _nBits, nPoints);
    auto path = make2D(1 << _nBits, nPoints);

    for (int j = 0; j != (1 << _nBits); j++) {
        for (int t = j; t < nPoints; t++) {
            solution[j][t] = PWLFunction::costInfinity;
            path[j][t] = -1;
        }
        if (j == 0) {
            solution[j][0] = costFunction[0][j + 1].second;
        } else {
            // there is no any approximation with 1 point
            solution[j][0] = PWLFunction::costInfinity;
        }

        path[j][0] = 0;

        for (int t = 1; t < std::min(static_cast<int>(nPoints), j + 1); t++) {
            solution[j][t] = PWLFunction::costInfinity;
            path[j][t] = 0;

            for (int i = t; i < j + 1; i++) {
                // cost function if triangle array with size of last row = 1
                auto addedCost = costFunction[i ][j - i + 1].second;
                auto prevSolutionCost = solution[i - 1][t - 1];

                if (addedCost == PWLFunction::costInfinity || prevSolutionCost == PWLFunction::costInfinity)  {
                    continue;
                }

                if (solution[j][t] > (prevSolutionCost + addedCost)) {
                    solution[j][t] = prevSolutionCost + addedCost;
                    path[j][t] = i;
                }
            }
        }
    }

    std::vector<int> segments;
    segments.push_back((1 << _nBits));
    for (int j = (1 << _nBits) - 1, t = nPoints - 1;t > 0; t--) {
        segments.push_back(path[j][t]);
        j = path[j][t];
    }
    // TODO: fix that
    segments[nPoints - 1] = 0;
    for (int j = _nIntervals; j > 0; j--) {
        auto startIdx = segments[j];
        auto endIndex = segments[j - 1] - segments[j];
        auto costSegment = costFunction[startIdx][endIndex];
        if (costSegment.first.getRange().empty()) {
            continue;
        }
        pwl_segments.push_back(costSegment.first);
    }

    return pwl_segments;
}

template <class Tx, class Ty>
std::vector<std::vector<PWLFunction::CostlySegment>>
PWLFit<Tx, Ty> :: genCostFunction(const std::vector<int> & refIntFunction) const {
    // compute E(i, j)
    int zeroPoint = (1 << (_nBits - 1));
    std::vector<std::vector<PWLFunction::CostlySegment>>  globalCost2D(1 << _nBits);

    // creating cost-function table for single interval between i and j point in segmentation
    for (int i = 0; i != (1 << _nBits); i++) {
        std::vector<std::pair<PWLFunction::Segment, int>>  globalCost((1<<_nBits) - i + 1);
        globalCost[0].second = PWLFunction::costInfinity;
        for (int j = i + 1; j <= (1 << _nBits); j++) {
            PWLFunction::Range startRange{i - zeroPoint, static_cast<unsigned int>(j - i)};
            PWLFunction::Segment s1{startRange};
            auto cost = s1.fitfast(refIntFunction, _nBits);//, {-512, 1024}, {-nBits, 2u * nBits});
            globalCost[j - i] = {s1, cost};
        }
        globalCost2D[i] = std::move(globalCost);
    }

    return globalCost2D;
}

template <class Tx, class Ty>
std::vector<int>  PWLFit<Tx, Ty> :: genReferenceIntFunction() const {
    std::vector<int> ref(1 << _nBits);
    int zeroPoint = 1 << (_nBits - 1);
    int levels = 1 << _nBits;
    for (int i = 0; i != (1 << _nBits); i++) {
        Tx xfp = i;
        xfp = xfp / (levels - 1) * (_range.second - _range.first) + _range.first;
        auto yfp = _aproximatedFunction(xfp);
        auto y = round((yfp - _range.first) / (_range.second - _range.first) * (levels - 1)) - zeroPoint;

        ref[i] = static_cast<int>(y);
    }

    return ref;
}
