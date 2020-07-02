cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

enable_testing()
include_directories(${GTEST_INCLUDE_DIRS})

#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
#add_executable(graph_container graph_container.cpp ${graph})
#include_directories(include/mcm/graph)

#file(GLOB base_src
#    "base/*.cpp"
#)
#
#file(GLOB op_src
#    "op/*.cpp"
#)
#
#file(GLOB pass_src
#    "pass/*.cpp"
#)
#
#file(GLOB compiler_src
#    "compiler/*.cpp"
#)
#
#file(GLOB model_src
#    "model/*.cpp"
#)
#
#file(GLOB graph_src
#    "graph/*.cpp"
#)
#
#file(GLOB resources
#    "resources/*.cpp"
#)
#
#file(GLOB tensor
#    "tensor/*.cpp"
#)
#
#file(GLOB nce1
#    "nce1/*.cpp"
#)
#
#file(GLOB hw
#    "hw/*.cpp"
#)
#
#file(GLOB kmb
#    "target/kmb/*.cpp"
#)
#
#file(GLOB utils
#    "utils/*.cpp"
#)

file(GLOB transitive_reduction_algorithm
    "${CMAKE_SOURCE_DIR}/tests/unit/graph/graph_transitive_reduction.cpp"
)

file(GLOB path_splitter_unit_tests 
    "${CMAKE_SOURCE_DIR}/tests/unit/graph/path_spiltter_test.cpp"
)

file(GLOB barrier_deps_test 
    "${CMAKE_SOURCE_DIR}/tests/unit/target/kmb/barriers_test.cpp"
)
file(GLOB scheduler_unit_tests
  "${CMAKE_SOURCE_DIR}/src/scheduler/*\_test.cpp"
  "${CMAKE_SOURCE_DIR}/src/pass/lp_scheduler/*\_test.cpp"
)


file(GLOB hde_unit_tests
  "${CMAKE_SOURCE_DIR}/tests/unit/target/kmb/hde/hde.cpp"
)

file(GLOB fake_quantize_processing_unit_tests
    "${CMAKE_SOURCE_DIR}/tests/unit/pass/fake_quantize_processing.cpp"
)

#file(GLOB runtime_model_unit_tests
# "${CMAKE_SOURCE_DIR}/tests/unit/runtime_model/runtime_model_test.cpp"
#)

add_executable(mcm_unit_tests
    mcm_unit_tests.cpp
    #${nce1}
    #${hw}
#    ${graph_src}
#    ${resources}
#    ${base_src}
#    ${tensor}
#    ${op_src}
#    ${pass_src}
#    ${compiler_src}
#    ${model_src}
#    ${kmb}
#    ${utils}
    ${scheduler_unit_tests}
    ${hde_unit_tests}
    ${runtime_model_unit_tests}
    ${barrier_deps_test}
    ${transitive_reduction_algorithm}
    ${path_splitter_unit_tests}
    ${fake_quantize_processing_unit_tests}
)

if(VALIDATION)
file(GLOB deployer_src
    "deployer/*.cpp"
)
add_executable(deployer_unit_tests
    mcm_unit_tests.cpp
    ${deployer_src}
)
target_link_libraries (deployer_unit_tests gtest m)
target_link_libraries(deployer_unit_tests cm)
target_link_libraries(deployer_unit_tests dep)
endif(VALIDATION)

target_link_libraries(mcm_unit_tests gtest m)
target_link_libraries(mcm_unit_tests cm)
target_include_directories(mcm_unit_tests PRIVATE utils
      INTERFACE lp_scheduler lp_scheduler_pass)
target_link_libraries(mcm_unit_tests lp_scheduler lp_scheduler_pass)
#target_link_libraries(mcm_unit_tests gcomparator)

add_subdirectory(layer)
add_subdirectory(system_tests/lp_scheduler/three_layer_conv_model)
#add_subdirectory(system_tests/lp_scheduler/resnet)
add_subdirectory(system_tests/lp_scheduler/tiny_yolo)
#add_subdirectory(system_tests/lp_scheduler/mobilenet)

set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES stdc++)
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "")
set_target_properties(mcm_unit_tests PROPERTIES LINKER_LANGUAGE C)

#file(GLOB layers
#    "layers/*.cpp"
#)
#
#add_executable(regression_tests
#    mcm_unit_tests.cpp
#    ${layers}
#)
#
#target_link_libraries (regression_tests gtest)
#target_link_libraries (regression_tests cm)
