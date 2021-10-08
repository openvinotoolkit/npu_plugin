$(info "entered subdir for act_shave_lib")

include-dirs-lrt-y += inc inc/ShaveElfMetadata
include-dirs-shave_nn-y += inc inc/ShaveElfMetadata

srcs-shave_nn-y += postops_3D_core.cpp
srcs-shave_nn-y += pre_postops.cpp
srcs-shave_nn-y += shave_main.cpp
srcs-shave_nn-y += act_shave_res_mgr.cpp
