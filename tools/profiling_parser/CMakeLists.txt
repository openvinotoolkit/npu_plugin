cmake_minimum_required(VERSION 3.5.0)
project(validateTensor VERSION 0.0.1)

find_package(gflags QUIET)

add_executable(prof_parser 
    "prof_parser.cpp"
)

target_include_directories(prof_parser
    PRIVATE
        ${MCMROOT_DIR_PATH}
        ${META_DIR_PATH}/include
)

target_link_libraries(prof_parser flatbuffers gflags mcmCompiler)
