#
subdirs-lnn-$(CONFIG_SOC_LNN_MODE_BM) += rtemsStubs
subdirs-lrt-$(CONFIG_SOC_LRT_MODE_BM) += rtemsStubs
subdirs-lrt1-$(CONFIG_SOC_LRT1_MODE_BM) += rtemsStubs

subdirs-lnn-$(CONFIG_USE_COMPONENT_UNITTEST) += UnitTest
subdirs-lrt-$(CONFIG_USE_COMPONENT_UNITTEST) += UnitTest
subdirs-lrt1-$(CONFIG_USE_COMPONENT_UNITTEST) += UnitTest
subdirs-shave-$(CONFIG_USE_COMPONENT_UNITTEST) += UnitTest

subdirs-lnn-$(CONFIG_USE_COMPONENT_UNITTESTVCS) += UnitTestVcs
subdirs-lrt-$(CONFIG_USE_COMPONENT_UNITTESTVCS) += UnitTestVcs
subdirs-lrt1-$(CONFIG_USE_COMPONENT_UNITTESTVCS) += UnitTestVcs
subdirs-shave-$(CONFIG_USE_COMPONENT_UNITTESTVCS) += UnitTestVcs

subdirs-lnn-$(CONFIG_USE_COMPONENT_VCSHOOKS) += VcsHooks
subdirs-lrt-$(CONFIG_USE_COMPONENT_VCSHOOKS) += VcsHooks
subdirs-lrt1-$(CONFIG_USE_COMPONENT_VCSHOOKS) += VcsHooks
subdirs-shave-$(CONFIG_USE_COMPONENT_VCSHOOKS) += VcsHooks
subdirs-lnn-$(CONFIG_USE_KERNELS)   += validationApps/system/kernels
subdirs-lrt-$(CONFIG_USE_KERNELS)   += validationApps/system/kernels
subdirs-lrt1-$(CONFIG_USE_KERNELS)  += validationApps/system/kernels
subdirs-shave-$(CONFIG_USE_KERNELS) += validationApps/system/kernels

subdirs-lnn-y += validationApps/testImages
subdirs-lrt-y += validationApps/testImages
subdirs-lrt1-y += validationApps/testImages

ifeq "" "$(mdk-build-phase)"
ifeq "y" "$(CONFIG_VALIDATION_APP_ENABLED)"
    $(info  . [INFO]   =====  This is a validation application  =====)
endif
endif
