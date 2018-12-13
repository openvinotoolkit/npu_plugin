#include "include/mcm/pass/serializeKeenBay.hpp"
#include "include/mcm/utils/serializer/fakeGuest.hpp"

#include <chrono>

using namespace MVCNN;

// Simple trick to force the compiler to read any number of arguments passed to
// the function without using them so we can measure access time without printf.
#define print to_print_or_not_to_print
bool to_print = false;

template<typename... Args>
void to_print_or_not_to_print(volatile Args... args)
{
    // Comment out the printf to get the most accurate time
    // otherwise the compiler will emit some extra code.
    if (to_print)
        printf(args...);
    else
    //-----------------------------------------------------
        (void)(..., args);
}

void print_tensorref(auto& name, auto tref)
{
    if(tref == nullptr){
        printf("%-12s. NULL\n", name);
        return;
    }else{
        printf("not null %p\n", (void *)tref);
    }
    print("%-12s addr:0x%08x dtype:%d=%s dim:[", name, tref,
            tref->data_dtype(), dtypeLabels[tref->data_dtype()].c_str());
    for (auto e : *tref->dimensions())
        print("% 4u ", e);
    printf("]\n");
    print("%+17s", "str:[");
    for (auto e : *tref->strides())
        print("% 6u ", e);
    printf("]");
    print(" d_idx:%6u sp_idx:%6u l_off:%2u t_off:%2u",
            tref->data()->data_index(), tref->data()->sparsity_index(),
            tref->leading_offset(), tref->trailing_offset());
    print("\n");
}

void print_dpu_task(auto task)
{
    auto dpu = static_cast<const NCE2Task*>(task->task());
    auto dpui = dpu->invariant();

    print("    dpu common: op:%d=%s \n"
          "                kw:%i kh:%i ksw:%i ksh:%i \n",
            dpui->dpu_task_type(), dpuLayerTypeLabels[dpui->dpu_task_type()].c_str(),
            dpui->kernelW(), dpui->kernelH(),
            dpui->kernel_strideW(), dpui->kernel_strideH());

    print_tensorref("    input", dpui->input_data());
    print_tensorref("    output", dpui->output_data());
    print_tensorref("    weights", dpui->weights_data());
    print_tensorref("    bias", dpui->bias_data());

    print("    dpus: %lu\n", dpu->variant()->Length());
    for (auto dpuv : *dpu->variant())
    {
        print("      dpu%i cl_id:%lu mpe_m:%i"
             "   out xyz: %5i %5i %5i to xyz: %5i %5i %5i"
             "   pad(lrtb):%i-%i-%i-%i\n",
                dpuv->workloadID(),
                dpuv->clusterID(),
                dpuv->mpe_mode(),

                dpuv->workload_start_X(),
                dpuv->workload_start_Y(),
                dpuv->workload_start_Z(),
                dpuv->workload_end_X(),
                dpuv->workload_end_Y(),
                dpuv->workload_end_Z(),

                dpuv->padLeft(), dpuv->padRight(), dpuv->padTop(), dpuv->padBottom()
                );
    }
    if (dpui->ppe_task() != nullptr && dpui->ppe_task()->fixed_function() != nullptr )
    {

        for(auto x : *dpui->ppe_task()->fixed_function()){
            print("    ppe generic: ops:0x%08x clamp:%u\n",
                x, x->Clamp_High()
            );
            // print_tensorref("      bias", dpui->ppe_task()->bias_data());
        }
        print_tensorref("      scale", dpui->ppe_task()->scale_data());
    }
    else
    {
        print("    no ppe\n");
    }
}

void deserialize(const GraphFile* const g)
{
    print("Deserializing blob...\n");

    print("version: %lu.%lu.%lu\n",
              g->header()->version()->majorV(),
              g->header()->version()->minorV(),
              g->header()->version()->patchV());

    uint32_t cnt = g->binary_data()->Length();
    print("binary_data: %lu\n", cnt);
    for (auto bin_data: *g->binary_data())
    {
        print("  data%u addr:0x%08x u8:0x%08x fp16:0x%08x\n", --cnt, bin_data,
                bin_data->u8(), bin_data->fp16());
    }

    uint32_t tlCnt = g->task_lists()->Length();
    print("task_lists: %lu\n", tlCnt);
    for (auto task_list : *g->task_lists())
    {
        print("  task_list%u\n", tlCnt--);

        uint32_t tCnt = task_list->content()->Length();
        for (auto task : *task_list->content())
        {
            print("    task%u type:%d=%s addr:0x%08x\n", tCnt--,
                    task->task_type(),
                    specificTaskLabels[task->task_type()].c_str(),
                    task->task());

            switch (task->task_type())
            {
                case SpecificTask_NCE2Task:
                    {
                        print_dpu_task(task);
                    }
                break;
                default:
                    print("      Unsupported operation type: %d=%s\n",
                            task->task_type(),
                            specificTaskLabels[task->task_type()].c_str());
            }
        }
    }
}

void deserialize(const GraphFile* const graph, bool print)
{
    to_print = print;
    deserialize(graph);
}
