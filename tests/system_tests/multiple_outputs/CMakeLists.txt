LINK_DIRECTORIES($ENV{MCM_HOME}/build/lib)
INCLUDE_DIRECTORIES($ENV{MCM_HOME})

add_executable(multiple_outputs multiple_outputs.cpp)
target_link_libraries(multiple_outputs cm)
