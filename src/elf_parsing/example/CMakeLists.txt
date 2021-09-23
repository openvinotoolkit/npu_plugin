# {% copyright %}

set (EXAMPLES simplereader simplewriter)

foreach(TARGET_NAME ${EXAMPLES})
    add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME}/main.cpp)

    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W4 /WX)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()

    target_link_libraries(${TARGET_NAME} elf)
endforeach()
