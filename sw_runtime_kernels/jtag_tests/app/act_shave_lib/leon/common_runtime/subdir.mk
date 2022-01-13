# {% copyright %}

include-dirs-lrt-y += inc
include-dirs-shave_nn-y += inc

sys-nn-inference-runtime-common-components-srcs += $(wildcard src/*.c*)

srcs-lrt-y += $(sys-nn-inference-runtime-common-components-srcs)
srcs-shave_nn-y += $(sys-nn-inference-runtime-common-components-srcs)
