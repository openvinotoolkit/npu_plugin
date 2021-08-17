mv-tensor-defines-y                                  += -DMVTENSOR_CMX_BUFFER=$(CONFIG_MVTENSOR_CMX_BUFFER)
mv-tensor-defines-$(CONFIG_MVTENSOR_FAST_SVU)        += -DMV_TENSOR_FAST__OS_DRV_SVU
mv-tensor-defines-$(CONFIG_USE_COMPONENT_MEMMANAGER) += -DMVTENSOR_USE_MEMORY_MANAGER
mv-tensor-defines-$(CONFIG_MVTENSOR_L2C_COPY_BACK)   += -DIS_LEON_L2C_MODE_COPY_BACK

subdirs-los-y   += shared modules leon
subdirs-lrt-y   += shared modules leon

ccopt-los-y   += $(mv-tensor-defines-y)
ccopt-lrt-y   += $(mv-tensor-defines-y)

ccopt-los-y   += -falign-functions=64 -falign-loops=64
ccopt-lrt-y   += -falign-functions=64 -falign-loops=64
