# {% copyright %}

ifndef VPUIP_2_Directory
$(info     )
$(info !!!!!!    'VPUIP_2_Directory' environment variable should be defined to build)
$(info ( export VPUIP_2_Directory=<path to directory with vpuip_2> ))
$(error )
endif
DirAppRoot=./
MDK_ROOT_PATH = $(abspath ${VPUIP_2_Directory})

#subdirs-los-y += ../../..
#subdirs-lrt-y += ../../..
#subdirs-shave-y += ../../..
#subdirs-shave-nn-y += ../../..
#include ../../../subdir.mk

include $(MDK_ROOT_PATH)/mdkApplication.mk

