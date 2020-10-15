LINK_DIRECTORIES($ENV{MCM_HOME}/build/lib)
INCLUDE_DIRECTORIES($ENV{MCM_HOME})

add_executable(yolo yolo_aligned_sc_pf1.cpp)
target_link_libraries(yolo cm)


add_custom_target(generate_lp_scheduler_blob_yolo)
add_custom_command(TARGET generate_lp_scheduler_blob_yolo
                   COMMAND ./yolo -d
                    ${CMAKE_SOURCE_DIR}/comp_desc_lp_scheduler.json
                   COMMAND rm -rf ./output)
add_dependencies(generate_lp_scheduler_blob_yolo yolo)


add_custom_target(generate_default_blob_yolo)
add_custom_command(TARGET generate_default_blob_yolo
                   COMMAND ./yolo -d
                    ${CMAKE_SOURCE_DIR}/comp_desc_default.json
                   COMMAND rm -rf ./output)
add_dependencies(generate_default_blob_yolo yolo)
