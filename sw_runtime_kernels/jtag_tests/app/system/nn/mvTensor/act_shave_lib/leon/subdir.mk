presilicon-dir = ../../../../../../../presilicon

include-dirs-lrt-y += inc
include-dirs-lrt-y += $(presilicon-dir)/drivers/leon/drv/include
include-dirs-lrt-y += $(presilicon-dir)/swCommon/leon/include
include-dirs-lrt-y += $(presilicon-dir)/swCommon/shared/include


srcs-lrt-y += src/parser_postops.cpp
srcs-lrt-y += src/parser_custom_cpp.cpp
srcs-lrt-y += src/custom_common.cpp
srcs-lrt-y += src/act_shave_dispatcher.cpp
srcs-lrt-y += src/act_shave_runtime.cpp
srcs-lrt-y += src/ShaveElfMetadata/ShaveElfMetadataParser.cpp
srcs-lrt-y += src/common_functions.cpp
srcs-lrt-y += src/dma.c