#include <cmath>
#include <algorithm>
#include "gtest/gtest.h"
#include "gtest/gtest-printers.h"
#include "include/mcm/target/kmb/pwl/pwl_dyn_fit.hpp"
#include "include/mcm/target/kmb/pwl/ref_functions.hpp"


double getAllowabledL2Error(const double maxQuant) {
    const std::map<int32_t, double> L2Error = {
            {99844, 3.87298},  {101641, 3.87298}, {101787, 3.87298}, {102020, 3.87298}, {102578, 3.74166},
            {103265, 3.74166}, {107429, 3.60555}, {112253, 2.44949}, {113965, 2.44949}, {116641, 2},
            {123109, 1},       {124766, 1},       {130938, 1},       {131719, 1},       {131797, 1},
            {132266, 1},       {133264, 1.41421}, {137817, 1.41421}, {160938, 0},       {161863, 0},
            {164375, 0},       {170423, 0},       {178438, 0},       {178438, 0},       {192601, 0},
            {198282, 0},       {205441, 0},       {206375, 0},       {207231, 0},       {216725, 0},
            {228631, 0},       {232989, 0},       {237500, 0},       {238438, 0},       {254856, 0},
            {261082, 0},       {269564, 0},       {285828, 0},       {307500, 0},       {355312, 0},
            {388125, 0},       {65000, 7.28011},  {81484, 5.19615},  {82656, 5.91608},  {82813, 5.09902},
            {86953, 4},        {88441, 4},        {89219, 4},        {91484, 4},        {91875, 4},
            {91865, 4},        {92344, 4},        {93516, 4},        {94666, 4},        {95781, 4.12311},
            {96172, 4.12311},  {96641, 4},        {97031, 3.87298},  {97935, 4},        {98281, 4},
            {98438, 4},        {98984, 4}};

    int32_t max_quant = std::round(maxQuant * 10000.f);
    const auto l2error = L2Error.find(max_quant);
    if (l2error == L2Error.end()) {
        throw std::runtime_error("Couldn't find L2 error bounds for " + std::to_string(maxQuant));
    }
    return l2error->second;
}

using ConfigTestParams = std::tuple<ApproximationSource, uint32_t, uint32_t, std::tuple<float, float>>;

struct TestsArgs {
    ApproximationSource acc_function;
    int32_t nBits;
    int32_t nIntervals;
    float range_min;
    float range_max;
};

class PWLAccuracy : public testing::TestWithParam<ConfigTestParams>, public TestsArgs {
protected:
    double (*refFunction)(double);
    int32_t minx;
    int32_t maxx;
public:
    static void PrintTo (const ConfigTestParams & params, std::ostream & os) {
        TestsArgs local;

        UnpackTo(params, local);

        switch (local.acc_function) {
            case ApproximationSource::LeakyRelu : os << "LeakyRelu, "; break;
            case ApproximationSource::Mish : os << "Mish, "; break;
            default:
                throw std::runtime_error ("Unexpected approximation functions: " + std::to_string((int)local.acc_function));
        }
        os << local.nIntervals << " segments, ";
        os << local.nBits <<" bit, ";
        os << "range: " << local.range_min << ", " << local.range_max;
    }

    void UnpackHere(const ConfigTestParams & params) {
        UnpackTo(params, *this);
    }

    static void UnpackTo ( const ConfigTestParams & params, TestsArgs & result) {
        auto tmp_tie = std::tie(result.range_min, result.range_max);
        std::tie(result.acc_function, result.nBits, result.nIntervals, tmp_tie) = params;
    }

    void SetUp() override {
        UnpackHere(GetParam());

        switch (acc_function)  {
            case ApproximationSource::Mish : refFunction = mish; break;
            case ApproximationSource::LeakyRelu : refFunction = leakyReLU; break;
        }
        // calc int range
        minx = -(1 << (nBits-1));
        maxx =  (1 << (nBits-1)) - 1;
    }
};

TEST_P(PWLAccuracy, test) {

    auto pwlfnc = PWLDoubleFit()
            .setRange(range_min, range_max)
            .setFunction(refFunction)
            .setBitness(nBits)
            .setMaxIntervals(nIntervals)
            .solve();

    float L2 = 0;
    for (int32_t i = minx; i <= maxx; i++) {
        // getting floating point X
        double x = ((double)i - (double)minx) / ((double) (1 << nBits) - 1) * (range_max - range_min) + range_min;

        // ref function result
        double y = refFunction(x);

        // project it into int range using same bitness
        double yscale = 1.0f;

        
        double  y_projected = round(((y - range_min) * yscale) * (double)((1 << nBits) - 1) / (range_max - range_min)) + minx;
        int y_projected_int = static_cast<int>(y_projected);
        int  pwl_approximated_y = pwlfnc(i);

        // std::cout << y_projected_int << " " << pwl_approximated_y << "\n";
        L2 += (y_projected_int - pwl_approximated_y) * (y_projected_int - pwl_approximated_y);
    }
    ASSERT_NEAR(sqrt(L2),getAllowabledL2Error(range_max), 0.00001);
    //std::cout << "range : " <<range_min<< ", " << range_max << ", L2 diff: " << sqrt(L2) << "\n";
    // for (size_t s = 0; s < pwlfnc.segments().size(); s++) {
    //     std::cout << "segment " << s << ": [" << pwlfnc.segments()[s].getRange().begin()
    //                                 << ", " <<pwlfnc.segments()[s].getRange().end() << "] "
    //                                 << "b = " <<pwlfnc.segments()[s].getBias() << ", "
    //                                 << "s = " <<pwlfnc.segments()[s].getShift()
    //                                 << std::endl;
    // }
}

auto Symmetrical = [](float low, float high, float step_range = 1.) {
    std::pair<float, float> ranges = {low, high};
    float min_range = 2.;

    std::vector<std::tuple<float, float>> pp;
    for(; low < high && abs(low) >= min_range && abs(high) >= min_range
        ; low += step_range, high -= step_range) {
        pp.push_back({low, high});
    }
    return pp;
};

auto Single = [] (float low, float high) {
    std::vector<std::tuple<float, float>> pp (1, {low, high});
    return pp;
};

auto  GenRanges = [](const std::initializer_list<std::vector<std::tuple<float, float>>> & ranges) {
    std::vector<std::tuple<float, float>> pp;
    for (auto && subRange : ranges) {
        pp.insert(pp.end(), subRange.begin(), subRange.end());
    }
    std::sort(pp.begin(), pp.end(), [] (const std::tuple<float, float> & lhs,
                                        const std::tuple<float, float> & rhs) {
        return std::get<1>(lhs) > std::get<1>(rhs);
    });
    return pp;
};

// TODO: following printer can be turned off in case of Gtest compilation issues
namespace testing {
    namespace internal {
        template<>
        class UniversalTersePrinter<ConfigTestParams> {
        public:
            static void Print(const ConfigTestParams &value, ::std::ostream *os) {
                PWLAccuracy::PrintTo(value, *os);
            }
        };
    }
}

INSTANTIATE_TEST_CASE_P(
        MatchesIntegerRef, PWLAccuracy,
        ::testing::Combine(
                ::testing::ValuesIn({ApproximationSource::Mish}),
                ::testing::ValuesIn({8u /*, 13u*/}), ::testing::ValuesIn({8u}),
                ::testing::ValuesIn(
                        GenRanges({Single(-10.0703, 9.98438), Single(-10.2422, 10.1641), Single(-10.2588, 10.1787),
                                   Single(-10.2824, 10.202),  Single(-10.3359, 10.2578), Single(-10.4078, 10.3265),
                                   Single(-10.8275, 10.7429), Single(-11.3137, 11.2253), Single(-11.4863, 11.3965),
                                   Single(-11.7578, 11.6641), Single(-12.4078, 12.3109), Single(-12.5703, 12.4766),
                                   Single(-13.1953, 13.0938), Single(-13.2734, 13.1719), Single(-13.2812, 13.1797),
                                   Single(-13.3281, 13.2266), Single(-13.4314, 13.3264), Single(-13.8902, 13.7817),
                                   Single(-16.2188, 16.0938), Single(-16.3137, 16.1863), Single(-16.5781, 16.4375),
                                   Single(-17.1765, 17.0423), Single(-17.9843, 17.8438), Single(-17.9843, 17.8438),
                                   Single(-19.4118, 19.2601), Single(-19.9843, 19.8282), Single(-20.7059, 20.5441),
                                   Single(-20.8, 20.6375),    Single(-20.8863, 20.7231), Single(-21.8431, 21.6725),
                                   Single(-23.0431, 22.8631), Single(-23.4824, 23.2989), Single(-23.9375, 23.75),
                                   Single(-24.0312, 23.8438), Single(-25.6863, 25.4856), Single(-26.3137, 26.1082),
                                   Single(-27.1686, 26.9564), Single(-28.8078, 28.5828), Single(-30.9844, 30.75),
                                   Single(-35.8125, 35.5312), Single(-39.0938, 38.8125), Single(-6.55078, 6.5),
                                   Single(-8.21094, 8.14844), Single(-8.32812, 8.26562), Single(-8.35156, 8.28125),
                                   Single(-8.76562, 8.69531), Single(-8.91373, 8.84409), Single(-8.99219, 8.92188),
                                   Single(-9.21875, 9.14844), Single(-9.25781, 9.1875),  Single(-9.25882, 9.18649),
                                   Single(-9.30469, 9.23438), Single(-9.42188, 9.35156), Single(-9.54118, 9.46664),
                                   Single(-9.65625, 9.57812), Single(-9.69531, 9.61719), Single(-9.73438, 9.66406),
                                   Single(-9.77344, 9.70312), Single(-9.87059, 9.79347),
                                   Single(-9.90625, 9.82812), Single(-9.92188, 9.84375), Single(-9.97656, 9.89844)}))));
