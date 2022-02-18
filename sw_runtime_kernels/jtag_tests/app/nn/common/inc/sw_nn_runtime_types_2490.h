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

#ifndef SW_NN_RUNTIME_TYPES_H_2490__
#define SW_NN_RUNTIME_TYPES_H_2490__

#include <nn_perf_measurement.h>
#include <sw_layer.h>
#include <sw_shave_lib_common.h>
#include <sw_shave_res_manager.h>
#include <common_types.h>

// TODO: Better macro expansion allows these to live in a strongly typed enum
// now
/**
 *  Used to send action callbacks to LRT
 *  The basetype bit size must be <= the HW allowd "tag field"
 * Spec: "With the Myriad X (Myriad2 v3) architecture the tag field is widened
 * to 13-bits."
 */
#define SVU_NN_TAG_FIELD_SHUTDOWN 0x0000
#define SVU_NN_TAG_FIELD_PREAMBLE 0x0010
#define SVU_NN_TAG_FIELD_L2C_DATA_FLUSH 0x0011
#define SVU_NN_TAG_FIELD_L2C_INSTR_FLUSH 0x0012
#define SVU_NN_TAG_FIELD_PRINT_RT_TRACE 0x0013

namespace nn
{

  namespace util
  {
    struct TaskContext;
  }

  namespace shave_lib
  {
    constexpr uint8_t MAX_INPUT_TENSORS = MAX_KERNEL_INPUTS;
    constexpr uint8_t MAX_OUTPUT_TENSORS = MAX_KERNEL_OUTPUTS;

    static_assert(MAX_INPUT_TENSORS < (UINT8_MAX>> 2),
                  "IO tensor count should never come close to UINT8_MAX.");
    static_assert(MAX_OUTPUT_TENSORS < (UINT8_MAX >> 2),
                  "IO tensor count should never come close to UINT8_MAX.");

    struct AbsoluteAddresses
    {
      const unsigned char *inputs_[MAX_INPUT_TENSORS] = {nullptr};
      unsigned char *outputs_[MAX_OUTPUT_TENSORS] = {nullptr};
    };

    struct Layer;

    struct alignas(64) SoftLayerExec
    {
      AbsoluteAddresses abs_addr_;
      const Layer *layer_;
      shave_lib::ShavePerfCounters *counters_;

      // layer completion flag (always NNCMX resident)
      bool completed_;

      SoftLayerExec()
          : abs_addr_{}, layer_{nullptr}, counters_{nullptr},
            completed_{false} {}
    };

    enum struct RtWorkerState : uint8_t
    {
      IDLE = 0,
      IN_PROGRESS,
      FIFIO_ERROR,
      MEMORY_ERROR,
      UNKNOWN
    };

    enum struct RtDbgState : uint32_t
    {
      RuntimeInitStarting = 0,
      RuntimeInitComplete,
      RuntimeStarting,
      ReceivedSLE,
      DequeuedLayer,
      AbsoluteAddresses,
      ExecPreamble,
      KernelEntry,
      FinishedSLE,
      RuntimeComplete,
      RuntimeTerminate,
      SaveInputs,
      SaveOutputs,
      StackInstrumentation,
      StackEventExceeds,
      StackCrash,
      Marker01,
      Marker02,
      Marker03,
      Marker04,
      Marker05,
    };

// may add shareable state as features like multiple controllers are added
    struct alignas(64) svuNNRtCommonState
    {
#ifdef DEBUG_NN_SVU_RUNTIME
      RtDbgState dbgState;
      uint32_t dbgValue;
      uint32_t dbgValue2;
      uint32_t dbgShave;
      uint32_t irq_tx;
      uint32_t irq_rx;
#endif

      RtWorkerState workerRunState[TOTAL_NUM_SHAVES] = {
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
          RtWorkerState::UNKNOWN,
      };

      ShaveResourceManager *svuNNRtRef[TOTAL_NUM_SHAVES] = { 0 };
      ShaveResource totResources[TOTAL_NUM_SHAVES];
      ShaveResource preResources[TOTAL_NUM_SHAVES];
      ShaveResource newTotResources[TOTAL_NUM_SHAVES];
      int8_t preMapping[TOTAL_NUM_SHAVES] = {INVALID_SHAVE_ID};

      /** Total shave resource pool */
      uint32_t resCount{0};

      /** Preamble visible resource which is always a subset of totResources */
      uint32_t preResCount{0};

      uint32_t newResCount{0};

      volatile bool updateTotResources{false};
      volatile bool lrtInterruptServiced{false};

      ShavePerfCounters performanceCounters;
      char __attribute__((aligned(64))) execContext[SHAVE_LIB_EXEC_CONTEXT_SIZE];
    };

    // State used to init the SVU persistant runtimes
    struct alignas(64) svuNNRtInit
    {
      uint32_t svuMutexId;
      void *heapAddress;
      uint32_t heapSize;

      svuNNRtCommonState rtState;
    };
  } // namespace shave_lib
} // namespace nn

#endif //SW_NN_RUNTIME_TYPES_H_2490__
