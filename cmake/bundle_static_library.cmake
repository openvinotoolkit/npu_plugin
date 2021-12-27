# MIT License
# 
# Copyright (c) 2019 Cristian Adam
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TODO: add a brief comment documenting the function
function(bundle_static_library TARGET_NAME BUNDLED_TARGET_NAME)
    list(APPEND STATIC_LIBS ${TARGET_NAME})

    set(STATIC_LIBS_TO_BUNDLE "")
    foreach(arg IN LISTS ARGN)
        if(NOT arg MATCHES "(${TARGET_NAME}|${BUNDLED_TARGET_NAME})")
            list(APPEND STATIC_LIBS_TO_BUNDLE ${arg})
        endif()
    endforeach()

    function(_recursively_collect_dependencies INPUT_TARGET DEPENDENCIES_TO_INCLUDE)
        set(PUBLIC_DEPENDENCIES "")
        if (DEPENDENCIES_TO_INCLUDE STREQUAL "")
            set(_INPUT_LINK_LIBRARIES LINK_LIBRARIES)
            get_target_property(_INPUT_TYPE ${INPUT_TARGET} TYPE)
            if (${_INPUT_TYPE} STREQUAL "INTERFACE_LIBRARY")
                set(_INPUT_LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            endif()
            get_target_property(PUBLIC_DEPENDENCIES ${INPUT_TARGET} ${_INPUT_LINK_LIBRARIES})
        else()
            set(PUBLIC_DEPENDENCIES ${DEPENDENCIES_TO_INCLUDE})
        endif()
        foreach(DEPENDENCY IN LISTS PUBLIC_DEPENDENCIES)
            if(TARGET ${DEPENDENCY})
                get_target_property(_ALIAS ${DEPENDENCY} ALIASED_TARGET)
                if (TARGET ${_ALIAS})
                    set(DEPENDENCY ${ALIAS})
                endif()
                get_target_property(_type ${DEPENDENCY} TYPE)
                if (${_type} STREQUAL "STATIC_LIBRARY")
                    list(APPEND STATIC_LIBS ${DEPENDENCY})
                endif()

                get_property(LIBRARY_ALREADY_ADDED
                    GLOBAL PROPERTY _${TARGET_NAME}_STATIC_BUNDLE_${DEPENDENCY})
                if(NOT LIBRARY_ALREADY_ADDED)
                    set_property(GLOBAL PROPERTY _${TARGET_NAME}_STATIC_BUNDLE_${DEPENDENCY} ON)
                    _recursively_collect_dependencies(${DEPENDENCY} "")
                endif()
            endif()
        endforeach()
        set(STATIC_LIBS ${STATIC_LIBS} PARENT_SCOPE)
    endfunction()

    _recursively_collect_dependencies(${TARGET_NAME} "${STATIC_LIBS_TO_BUNDLE}")

    list(REMOVE_DUPLICATES STATIC_LIBS)
    set(BUNDLED_TARGET_FULL_NAME
        ${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${BUNDLED_TARGET_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU)$")
        file(WRITE ${CMAKE_BINARY_DIR}/${BUNDLED_TARGET_NAME}.ar.in
            "CREATE ${BUNDLED_TARGET_FULL_NAME}\n"
        )
        
        foreach(TARGET IN LISTS STATIC_LIBS)
            file(APPEND ${CMAKE_BINARY_DIR}/${BUNDLED_TARGET_NAME}.ar.in
                "ADDLIB $<TARGET_FILE:${TARGET}>\n"
            )
        endforeach()
        
        file(APPEND ${CMAKE_BINARY_DIR}/${BUNDLED_TARGET_NAME}.ar.in "SAVE\n")
        file(APPEND ${CMAKE_BINARY_DIR}/${BUNDLED_TARGET_NAME}.ar.in "END\n")

        file(GENERATE
            OUTPUT ${CMAKE_BINARY_DIR}/${BUNDLED_TARGET_NAME}.ar
            INPUT ${CMAKE_BINARY_DIR}/${BUNDLED_TARGET_NAME}.ar.in)

        set(AR_TOOL ${CMAKE_AR})
        if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
            set(AR_TOOL ${CMAKE_CXX_COMPILER_AR})
        endif()

        add_custom_command(
            OUTPUT ${BUNDLED_TARGET_FULL_NAME}
            COMMAND ${AR_TOOL} -M < ${CMAKE_BINARY_DIR}/${BUNDLED_TARGET_NAME}.ar
            DEPENDS ${TARGET_NAME}
            COMMENT "Bundling ${BUNDLED_TARGET_NAME}"
            VERBATIM)
    elseif(MSVC)
        find_program(lib_tool lib)

        foreach(TARGET IN LISTS STATIC_LIBS)
            list(APPEND STATIC_LIBS_FULL_NAMES $<TARGET_FILE:${TARGET}>)
        endforeach()

        add_custom_command(
            OUTPUT ${BUNDLED_TARGET_FULL_NAME}
            COMMAND ${lib_tool} /NOLOGO /OUT:${BUNDLED_TARGET_FULL_NAME} ${STATIC_LIBS_FULL_NAMES}
            DEPENDS ${TARGET_NAME}
            COMMENT "Bundling ${BUNDLED_TARGET_NAME}"
            VERBATIM
        )
    else()
        message(FATAL_ERROR "Unknown bundle scenario!")
    endif()

    add_custom_target(bundling_target_${BUNDLED_TARGET_NAME} ALL DEPENDS ${BUNDLED_TARGET_FULL_NAME})

    add_library(${BUNDLED_TARGET_NAME} STATIC IMPORTED GLOBAL)
    set_target_properties(${BUNDLED_TARGET_NAME}
        PROPERTIES
            IMPORTED_LOCATION ${BUNDLED_TARGET_FULL_NAME}
            INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
            INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${TARGET_NAME},INCLUDE_DIRECTORIES>
    )

    add_dependencies(${BUNDLED_TARGET_NAME} bundling_target_${BUNDLED_TARGET_NAME})
endfunction()
