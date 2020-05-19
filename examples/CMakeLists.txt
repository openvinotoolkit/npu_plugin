cmake_minimum_required(VERSION 3.5.0)
project(mcmComplier VERSION 0.0.1)

if (MSVC)

    # TODO - implement

else()

    add_executable(mobilenet_v2 mobilenet_v2.cpp)
    target_link_libraries(mobilenet_v2 cm)

endif(MSVC)
