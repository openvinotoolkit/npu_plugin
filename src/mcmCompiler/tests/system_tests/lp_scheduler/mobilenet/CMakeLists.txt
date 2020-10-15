LINK_DIRECTORIES($ENV{MCM_HOME}/build/lib)
INCLUDE_DIRECTORIES($ENV{MCM_HOME})

add_executable(mobile_net mobile_net_aligned.cpp)
target_link_libraries(mobile_net cm)
set(TARGET_NAME mobile_net)


add_custom_target(generate_lp_scheduler_blob_${TARGET_NAME})
add_custom_command(TARGET generate_lp_scheduler_blob_${TARGET_NAME}
                   COMMAND ./${TARGET_NAME} -d
                    ${CMAKE_SOURCE_DIR}/comp_desc_lp_scheduler.json
                   COMMAND rm -rf ./output)
add_dependencies(generate_lp_scheduler_blob_${TARGET_NAME} ${TARGET_NAME})


add_custom_target(generate_default_blob_${TARGET_NAME})
add_custom_command(TARGET generate_default_blob_${TARGET_NAME}
                   COMMAND ./${TARGET_NAME} -d
                    ${CMAKE_SOURCE_DIR}/comp_desc_default.json
                   COMMAND rm -rf ./output)
add_dependencies(generate_default_blob_${TARGET_NAME} ${TARGET_NAME})
