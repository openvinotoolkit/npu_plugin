cmake_minimum_required(VERSION 3.14.5)

set(RESNET50_FP16 net_test_resnet50_fp16)

add_executable(
    ${RESNET50_FP16}
    test.cpp
    resnet50_fp16.cpp)

target_link_libraries(
    ${RESNET50_FP16}
    cm)
