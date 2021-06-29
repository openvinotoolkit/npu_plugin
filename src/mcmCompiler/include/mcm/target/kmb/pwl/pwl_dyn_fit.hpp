#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

enum class ApproximationSource {
    LeakyRelu,
    Mish
};

class PWLFunction {
public:
    /**
     * represents integer values in interval of [start, start + length]
    */
    class Range {
         int _start;
         unsigned int _length;
     public:
         Range() = default;
         Range(int start, unsigned int length) :
            _start(start), _length(length) {

         }
         int begin() const {
             return _start;
         }
         int end() const {
            return _start + _length;
         }
         bool empty() const {
             return begin() == end();
         }
         bool contains (int x) const {
             return x >= begin() && x < end();
         }
    };

    class Segment {
        Range _data;
        int _bias = 0;
        int _shift = 0;
    public:
        Segment() = default;
        explicit Segment(const Range & data) :
            _data(data) {
        }
        Segment(int startPoint, unsigned int length) :
            _data(startPoint, length) {
        }
        Segment(int startPoint, unsigned int length, int bias, int shift) :
            _data(startPoint, length), _bias(bias), _shift(shift) {
        }

        int cost(const std::vector<int> & refFunction, int bitness) const;
        int fit(const std::vector<int> & refFunction, int bitness, const Range & biasRange, const Range & shiftRange);

        /**
         * uses LeastSquares to get bias and shift
         */
        int fitfast(const std::vector<int> & refFunction, int bitness);

        int getBias() const {
            return _bias;
        }
        int getShift() const {
            return _shift;
        }
        // TODO: EISW-9403
        bool contains(int x) const {
            return _data.contains(x);
        }
        // TODO: EISW-9403
        int evaluate(int x) const {
            auto value = x;
            if (_shift > 0) {
                value <<= _shift;
            } else {
                value >>= -_shift;
            }
            value += _bias;
            return value;
        }
        Range getRange() const {
            return _data;
        }
    protected:
        void checkFunction(const std::vector<int> & refFunction, int bitness) const;
        int  costInternal(const std::vector<int> & refFunction, int bitness) const;
    };
    using CostlySegment = std::pair<PWLFunction::Segment, int>;
    static constexpr auto  costInfinity = std::numeric_limits<int>::max();
private:
    std::vector<Segment> pwl_segments;

public:
    PWLFunction (const std::vector<Segment> &s)  : pwl_segments(s) {
    }
    // TODO: EISW-9403
    int operator ()(int x) const {
        for (auto a : pwl_segments) {
            if (a.contains(x)) {
                return a.evaluate(x);
            }
        }
        throw std::runtime_error("PWL function not defined for: " + std::to_string(x));
    }
    std::vector<Segment> segments() const {
        return pwl_segments;
    }
};


template <class Tx, class Ty = Tx>
class PWLFit {
    std::pair<Tx, Tx> _range;
    uint8_t _nBits = 0;
    uint8_t _nIntervals = 8;
    std::function<Ty(Tx)> _aproximatedFunction;

public:
    template <class T>
    PWLFit & setFunction(const T& sourceFunction) {
        _aproximatedFunction = sourceFunction;
        return *this;
    }
    PWLFit & setRange(Tx minX, Tx maxX);
    PWLFit & setBitness(uint8_t nBits);
    PWLFit & setMaxIntervals(uint8_t nIntervals);
    PWLFunction solve() const;

protected:

    std::vector<int> genReferenceIntFunction() const;
    std::vector<std::vector<PWLFunction::CostlySegment>> genCostFunction(const std::vector<int> & refIntFunction) const;
};
// float->float functions supported
template class PWLFit<float>;
template class PWLFit<double>;

using PWLFloatFit = PWLFit<float>;
using PWLDoubleFit = PWLFit<double>;
