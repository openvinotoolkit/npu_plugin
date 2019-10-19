set(LP_SCHEDULER_HEADERS 
    "${CMAKE_CURRENT_SOURCE_DIR}/disjoint_interval_set.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/feasible_scheduler.hpp")

# Currently this is a generic programming library and involves only headers
# similar to STL or BOOST. Change this to STATIC/SHARED if any explicit
# specializations are necessary and include appropriate *.cpp files.
add_library(lp_scheduler INTERFACE)


target_sources(lp_scheduler INTERFACE "${LP_SCHEDULER_HEADERS}")
set_property(TARGET lp_scheduler APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>)
