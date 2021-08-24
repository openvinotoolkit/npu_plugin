mv-tensor-defines-y                                  += -DMVTENSOR_CMX_BUFFER=$(CONFIG_MVTENSOR_CMX_BUFFER)
mv-tensor-defines-$(CONFIG_MVTENSOR_FAST_SVU)        += -DMV_TENSOR_FAST__OS_DRV_SVU
mv-tensor-defines-$(CONFIG_USE_COMPONENT_MEMMANAGER) += -DMVTENSOR_USE_MEMORY_MANAGER
mv-tensor-defines-$(CONFIG_MVTENSOR_L2C_COPY_BACK)   += -DIS_LEON_L2C_MODE_COPY_BACK

subdirs-los-y   += shared modules leon
subdirs-lrt-y   += shared modules leon
#subdirs-los-y   += shared modules leon common shave_lib inference_runtime_common platform_abstraction inference_runtime_common
#subdirs-lrt-y   += shared modules leon common shave_lib inference_runtime_commonplatform_abstraction inference_runtime_common

ccopt-los-y   += $(mv-tensor-defines-y)
ccopt-lrt-y   += $(mv-tensor-defines-y)

ccopt-los-y   += -falign-functions=64 -falign-loops=64
ccopt-lrt-y   += -falign-functions=64 -falign-loops=64
#
#subdirs-lrt-$(CONFIG_NN_USE_APPCONFIG_LRT) += app_config
#subdirs-lnn-$(CONFIG_NN_USE_APPCONFIG_LNN) += app_config
#
#subdirs-lrt-$(CONFIG_USE_COMPONENT_NN) += common shave_lib 
##inference_runtime_common inference_manager
#subdirs-lnn-$(CONFIG_USE_COMPONENT_NN) += common
##inference_runtime_common inference_runtime
#subdirs-shave-$(CONFIG_USE_COMPONENT_NN) += common shave_lib
#subdirs-shave_nn-$(CONFIG_USE_COMPONENT_NN) += common
##act_runtime inference_runtime_common
#
#
#subdirs-shave-y += common shave_lib inference_runtime_common platform_abstraction inference_runtime_common
#subdirs-shave_nn-y += common inference_runtime_common

