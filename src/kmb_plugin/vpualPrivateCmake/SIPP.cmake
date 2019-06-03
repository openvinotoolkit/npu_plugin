set(SIPP_DIRECTORY_BASE ${vpual_home}/host/source/SIPP)

set(SIPP_INCLUDE_DIRECTORIES
    ${SIPP_DIRECTORY_BASE}/include
)

set(SIPP_SOURCES
    ${SIPP_DIRECTORY_BASE}/source/sipp_api.cpp
)

set(PUBLIC_SIPP_HEADERS
    ${SIPP_DIRECTORY_BASE}/sipp_api.h
    ${SIPP_DIRECTORY_BASE}/sipp_messages.h
)

set(PUBLIC_HEADERS
    ${PUBLIC_SIPP_HEADERS}
)

add_library(sipp_custom SHARED
    ${SIPP_SOURCES}
)

target_link_libraries(sipp_custom)
