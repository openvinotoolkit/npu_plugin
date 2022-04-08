/*
* {% copyright %}
*/
#ifndef NN_PERF_MEASUREMENT_H
#define NN_PERF_MEASUREMENT_H

namespace nn {
namespace shave_lib {
struct ShavePerfCounters {
  unsigned int cycles;
  unsigned int instrs;
  unsigned int stalls;
  unsigned int branches;
};
} // namespace shave_lib
} // namespace nn

#endif // NN_PERF_MEASUREMENT_H_
