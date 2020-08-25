set(TARGET_NAME "VPUXPlugin")
set(VPUX_DEVICE_NAME VPUX)

if((X86 OR X86_64) AND NOT ENABLE_HDDL2)
    return()
endif()

if(ENABLE_LTO)
    ie_enable_lto()
endif()

if (AARCH64 OR ARM)
    file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kmb_plugin/*.cpp)
    file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/kmb_plugin/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/kmb_plugin/*.h)
    set(ENGINE_SOURCE_FILE kmb_plugin/kmb_plugin.cpp)
    set(HEADERS_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/kmb_plugin
                     ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/vpu)
    set(LIBRARIES ${VPUAL_NN_LIBRARY} ${GAPI_SIPP_LIBRARY} ${GAPI_M2I_LIBRARY} ${VPUSMM_LIBRARY})
elseif(X86 OR X86_64)
    file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hddl2_plugin/src/*.cpp)
    file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/hddl2_plugin/include/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/hddl2_plugin/include/*.h)
    set(ENGINE_SOURCE_FILE hddl2_plugin/src/hddl2_plugin.cpp)
    set(HEADERS_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/hddl2_plugin/include
                   ${CMAKE_CURRENT_SOURCE_DIR}/hddl2_plugin/include/hddl_unite
                   ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/hddl2
                   ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/vpu)
    set(LIBRARIES HddlUnite)
else()
    message(FATAL_ERROR "Your platform is not supported for VPUX Plugin")
endif()

ie_add_plugin(NAME ${TARGET_NAME}
    DEVICE_NAME ${VPUX_DEVICE_NAME}
    SOURCES ${SOURCES} ${HEADERS}
    VERSION_DEFINES_FOR ${ENGINE_SOURCE_FILE})

target_compile_definitions(${TARGET_NAME} PRIVATE -DDEVICE_NAME="${VPUX_DEVICE_NAME}")

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
    PRIVATE
        ${FRONTEND_VPUX_INCLUDE_DIR}
        ${HEADERS_DIRS})

target_link_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine
        ${INTEL_ITT_LIBS}
        vpux
        ${LIBRARIES}
        kmb_utils
        IE::vpu_common_lib)

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

# install

ie_cpack_add_component(VPUX REQUIRED)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT vpu
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT vpu
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT vpu)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
