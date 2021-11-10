#include "host_parsed_inference.h"

#include <stdio.h>
#include <stddef.h>

#define SIZE(t) printf("%30s: %d\n", #t, sizeof(t))

using namespace host_parsing;

extern "C"
int main(int argc, char**)
{
    DmaDescriptor desc[3];

    printf("DESC 0: %p 1: %p 2: %p DELTA01: %lld DELTA12: %lld\n", &desc[0], &desc[1], &desc[2], ((uint64_t)&desc[1]) - ((uint64_t)&desc[0]), ((uint64_t)&desc[2]) - ((uint64_t)&desc[1]) );

    SIZE(void*);

    SIZE(HostParsedInference);
    SIZE(DmaWrapper);
    SIZE(DmaWrapper[2]);
    SIZE(DmaDescriptor);
    SIZE(DmaDescriptor[2]);
    SIZE(DPUInvariantWrapper[2]);
    SIZE(DPUVariantWrapper[2]);
    SIZE(DPUInvariantRegisters);
    SIZE(DPUVariantRegisters);
    SIZE(BarrierWrapper[2]);
    SIZE(TaskReference<DmaWrapper>[2]);
    SIZE(TaskReference<DPUInvariantWrapper>);
    SIZE(TaskReference<DPUVariantWrapper>);
    SIZE(MappedInference[2]);
    SIZE(DmaDescriptor[5]);
}
