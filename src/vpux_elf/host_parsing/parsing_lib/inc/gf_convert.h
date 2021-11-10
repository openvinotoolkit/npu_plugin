#pragma once
#include <data_types.h>
#include <graphfile_generated.h>
#include <nn_relocation.h>

namespace parsing_lib {
void convertNNDmaTask(const MVCNN::NNDMATask *gfTask, parsing_lib::DMATask &task);
bool convertRelativeAddress(const parsing_lib::TensorReference &tr, nn::common_runtime::RelativeAddress &ra);
void convertInvariant(const MVCNN::NCEInvariantFields *gfInv, parsing_lib::Invariant &inv);
void convertVariant(const MVCNN::NCEVariantFields *gfVar, parsing_lib::Variant &var);
void convertNCE2Task(const MVCNN::NCE2Task *gfTask, parsing_lib::NCE2Task &task);
}
