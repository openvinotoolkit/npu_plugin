#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#
set(TARGET_NAME VPUXLoader)

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(WIN32)

set(Loader_Library_SRCS
  ${LOADER_DIR_INC}/vpux_headers/metadata.hpp
  ${LOADER_DIR_INC}/vpux_loader/vpux_loader.hpp
  ${LOADER_DIR_SRC}/vpux_loader.cpp
  ${ELF_DIR}/core/include/vpux_elf/reader.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer.hpp
  ${ELF_DIR}/core/include/vpux_elf/accessor.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/relocation_entry.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/symbol_entry.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/vpu_extensions.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/data_types.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/elf_header.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/elf_structs.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/program_header.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/relocation_entry.hpp
  ${ELF_DIR}/core/include/vpux_elf/types/section_header.hpp
  ${ELF_DIR}/core/include/vpux_elf/utils/utils.hpp
  ${ELF_DIR}/core/include/vpux_elf/utils/error.hpp
  ${ELF_DIR}/core/include/vpux_elf/utils/log.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/binary_data_section.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/empty_section.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/relocation.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/relocation_section.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/section.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/segment.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/string_section.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/symbol.hpp
  ${ELF_DIR}/core/include/vpux_elf/writer/symbol_section.hpp
)

add_library(${TARGET_NAME} STATIC ${Loader_Library_SRCS})

target_link_libraries(${TARGET_NAME}
  PUBLIC
        vpux_elf
        vpux_utils
)

target_include_directories(${TARGET_NAME}
  PUBLIC
        ${LOADER_DIR_INC}/
        ${LOADER_DIR_SRC}/
        ${LOADER_DIR}/
        ${ELF_DIR}/core/include/vpux_elf/
        ${ELF_DIR}/core/src/
)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
