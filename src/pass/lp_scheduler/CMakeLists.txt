set(LP_SCHEDULER_PASS_HEADERS 
    "${CMAKE_CURRENT_SOURCE_DIR}/operation_precedence_dag.hpp")

set(LP_SCHEDULER_PASS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/lp_scheduler_pass.cpp")

#add_library(lp_scheduler_pass STATIC)
add_library(lp_scheduler_pass INTERFACE)
add_dependencies(lp_scheduler_pass lp_scheduler)

#target_sources(lp_scheduler_pass INTERFACE "${LP_SCHEDULER_PASS_HEADERS}"
#      PUBLIC "${LP_SCHEDULER_PASS_SOURCES}")

target_sources(lp_scheduler_pass INTERFACE "${LP_SCHEDULER_PASS_HEADERS}")

set_property(TARGET lp_scheduler_pass APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>)

#target_include_directories(lp_scheduler_pass
#    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
#    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>)
