set(LP_SCHEDULER_PASS_HEADERS 
    "${CMAKE_CURRENT_SOURCE_DIR}/operation_precedence_dag.hpp")

# Currently this is a generic programming library and involves only headers
# similar to STL or BOOST. Change this to STATIC/SHARED if any explicit
# specializations are necessary and include appropriate *.cpp files.
add_library(lp_scheduler_pass INTERFACE)
add_dependencies(lp_scheduler_pass lp_scheduler)

target_sources(lp_scheduler_pass INTERFACE "${LP_SCHEDULER_PASS_HEADERS}")
set_property(TARGET lp_scheduler_pass APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>)
