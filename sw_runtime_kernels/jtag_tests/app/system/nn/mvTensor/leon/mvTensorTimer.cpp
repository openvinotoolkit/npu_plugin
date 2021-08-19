// {% copyright %}
#include "mvTensorTimer.h"
#include <algorithm>

namespace mv
{
namespace tensor
{
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

Timer::Timer(double *store) :
    store_(store) {
    reset();
}

void Timer::reset()
{
    start_ = Clock::now();
}

double Timer::elapsed() const
{
    auto now = Clock::now();
    return std::chrono::duration_cast<ms>(now - start_).count();
}

Timer::~Timer()
{
    if (store_ != nullptr)
        *store_ = elapsed();
}
}
}
