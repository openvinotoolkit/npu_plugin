cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

if (MSVC)

    # TODO - implement

else()

    add_subdirectory(mobilenetv2_fp16)
    add_subdirectory(resnet50_fp16)

endif(MSVC)
