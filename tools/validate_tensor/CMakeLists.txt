cmake_minimum_required(VERSION 3.5.0)
project(validateTensor VERSION 0.0.1)

find_package(gflags REQUIRED)

include(ExternalProject)
set(NLOHMAN_SOURCE_DIR "${PROJECT_SOURCE_DIR}/contrib/nlohmann")
set(NLOHMAN_INSTALL_DIR "${PROJECT_SOURCE_DIR}/contrib/nlohmann/install")

if (NOT EXISTS NLOHMAN_SOURCE_DIR)
    ExternalProject_Add(
        project_nlohmann_json
        GIT_REPOSITORY    https://github.com/nlohmann/json
        GIT_TAG           master
        SOURCE_DIR        ${NLOHMAN_SOURCE_DIR}
        BUILD_COMMAND     ${MAKE}
        INSTALL_COMMAND   make DESTDIR=${NLOHMAN_INSTALL_DIR} install
        VERBATIM
    )
endif()

SET(NLOHMAN_INCLUDE_DIR "${NLOHMAN_INSTALL_DIR}/usr/local/include")
SET(NLOHMAN_LIBRARY ${NLOHMAN_INSTALL_DIR}/usr/local/lib/cmake/nlohmann_json)
SET(NLOHMAN_FOUND TRUE)
include_directories(${NLOHMAN_INCLUDE_DIR})

link_directories(${NLOHMAN_LIBRARY})

add_executable(validate 
    "validate.cpp"
)
# export(EXPORT SomeLib-export NAMESPACE SomeLib:: FILE SomeLibConfig.cmake) # Create SomeLibConfig.cmake
# export(PACKAGE SomeLib)                                                    # Store location of SomeLibConfig.cmake


find_package(nlohmann_json CONFIG HINTS ${NLOHMAN_LIBRARY})
MESSAGE(STATUS "FOUND: " ${NLOHMAN_JSON_FOUND})
# add_library(nlohmann_json SHARED IMPORTED)
# set_target_properties(nlohmann_json PROPERTIES IMPORTED_LOCATION ${NLOHMAN_LIBRARY})

# set(NLOHMAN_LIB = ${NLOHMAN_LIBRARY}::nlohmann_json)
# target_link_libraries(convert_tensor gflags ${NLOHMAN_LIB})
target_link_libraries(validate gflags nlohmann_json::nlohmann_json corebase)
# target_link_libraries(convert_tensor gflags nlohmann_json)
add_dependencies(validate project_nlohmann_json)

