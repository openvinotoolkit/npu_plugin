$(warning "entered subdir for act_shave_lib")

presilicon-dir = ../../../../../../../presilicon

include-dirs-lrt-y += inc inc/ShaveElfMetadata
include-dirs-shave_nn-y += inc inc/ShaveElfMetadata
include-dirs-shave_nn-y += $(presilicon-dir)/swCommon/shave_code/include
include-dirs-shave_nn-y += $(presilicon-dir)/drivers/shave/include

srcs-shave_nn-y += avgpooling.cpp
srcs-shave_nn-y += postops_3D_core.cpp
#srcs-shave_nn-y += custom_cpp.cpp
srcs-shave_nn-y += pre_postops.cpp
srcs-shave_nn-y += pre_custom_cpp.cpp
srcs-shave_nn-y += pre_softmax_single.cpp
srcs-shave_nn-y += shave_main.cpp
srcs-shave_nn-y += act_shave_res_mgr.cpp
srcs-shave_nn-y += dma_shave_nn.cpp
