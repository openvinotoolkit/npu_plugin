#
# {% copyright %}
#

folders = .
folders += parsing_lib/inc

include-dirs-lrt-y += . parsing_lib/inc inc
include-dirs-lnn-y += .
include-dirs-shave_nn-y += .

parsing_lib_files = gf_convert.cpp utils.cpp dma.cpp dpu_common.cpp fp_utils.cpp cm_convolution.cpp convolution.cpp dw_convolution.cpp eltwise.cpp maxpool.cpp ppe_task.cpp dpu_soh_utils.cpp debug_utils.cpp

srcs-lrt-y += $(addprefix parsing_lib/src/, $(parsing_lib_files))
