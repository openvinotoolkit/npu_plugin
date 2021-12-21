# {% copyright %}

#include-dirs-lnn-y += inc
include-dirs-lrt-y += inc
include-dirs-shave_nn-y += inc

sys-nn-inference-runtime-common-components-srcs += $(wildcard src/*.c*)

srcs-lrt-y += $(sys-nn-inference-runtime-common-components-srcs)
#srcs-lnn-$(CONFIG_HAS_LNN_SRCS) += $(sys-nn-inference-runtime-common-components-srcs)
#srcs-shave_nn-$(CONFIG_HAS_SHAVE_NN_SRCS) += $(sys-nn-inference-runtime-common-components-srcs)
srcs-shave_nn-y += $(sys-nn-inference-runtime-common-components-srcs)
