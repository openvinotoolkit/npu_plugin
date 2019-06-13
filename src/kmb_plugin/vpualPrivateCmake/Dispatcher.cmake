set(VPUAL_DISPATCHER_BASE ${vpual_home}/host/source/Dispatcher)

include(${CMAKE_CURRENT_LIST_DIR}/XLink.cmake)

set(VPUAL_DISPATCHER_INCLUDE_DIRECTORIES
    ${VPUAL_DISPATCHER_BASE}/include
    ${XLINK_INCLUDE_DIRECTORIES}
)

set(VPUAL_DISPATCHER_SOURCES
    ${VPUAL_DISPATCHER_BASE}/source/VpualDispatcher.cpp
    ${VPUAL_DISPATCHER_BASE}/source/VpualMessage.cpp
)

set(VPUAL_DISPATCHER_PUBLIC_HEADERS
    ${VPUAL_DISPATCHER_BASE}/include/VpualDispatcher.h
    ${VPUAL_DISPATCHER_BASE}/include/VpualMessage.h
)

add_library(VpualDispatcher SHARED
    ${VPUAL_DISPATCHER_SOURCES}
)

target_link_libraries(VpualDispatcher XLink2)
