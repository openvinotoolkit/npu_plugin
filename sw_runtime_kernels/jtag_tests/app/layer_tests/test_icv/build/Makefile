# {% copyright %}

ifndef FIRMWARE_VPU_DIR
$(info     )
$(info 'FIRMWARE_VPU_DIR' environment variable should be defined to build)
$(info ( export FIRMWARE_VPU_DIR=<path to directory with firmware.vpu.iot> ))
$(error )
endif
DirAppRoot=./
MDK_ROOT_PATH = $(abspath ${FIRMWARE_VPU_DIR})

include $(MDK_ROOT_PATH)/mdkApplication.mk

