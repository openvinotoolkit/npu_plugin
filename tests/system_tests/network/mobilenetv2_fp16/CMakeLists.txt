cmake_minimum_required(VERSION 3.14.5)

set(MOBILENETV2_FP16 net_test_mobilenetv2_fp16)

add_executable(
    ${MOBILENETV2_FP16}
    test.cpp
    mobilenetv2_fp16.cpp)

target_link_libraries(
    ${MOBILENETV2_FP16}
    cm)
