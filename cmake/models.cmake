# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_DOCKER)
    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
endif()

cmake_policy(SET CMP0054 NEW)

find_package(Git REQUIRED)

set(MODELS_LST "")
set(MODELS_LST_TO_FETCH "")

function (add_models_repo add_to_fetcher model_name)
    list(LENGTH ARGV add_models_args)
    if (add_models_args EQUAL 3)
        list(GET ARGV 2 branch_name)
    else()
        set(branch_name ${MODELS_BRANCH})
    endif()
    if (add_to_fetcher)
        set(model_name "${model_name}:${branch_name}")
        list(APPEND MODELS_LST_TO_FETCH ${model_name})
    endif()

    list(APPEND MODELS_LST ${model_name})

    set(MODELS_LST_TO_FETCH ${MODELS_LST_TO_FETCH} PARENT_SCOPE)
    set(MODELS_LST ${MODELS_LST} PARENT_SCOPE)
endfunction()

function(add_lfs_repo name prefix url tag)
    ExternalProject_Add(${name}
        PREFIX ${prefix}
        GIT_REPOSITORY ${url}
        GIT_TAG ${tag}
        GIT_CONFIG "http.sslverify=false"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        LOG_DOWNLOAD ON)

    execute_process(
        COMMAND ${GIT_EXECUTABLE} lfs install --local --skip-smudge
        WORKING_DIRECTORY ${prefix}/src/${name}
        OUTPUT_VARIABLE lfs_output
        RESULT_VARIABLE lfs_var)
    if(lfs_var)
        message(FATAL_ERROR "Git lfs must be installed in order to fetch models\nPlease install it from https://git-lfs.github.com/")
    endif()

    if(NOT PRUNE_LFS_MODELS)
        ExternalProject_Add_Step(${name} lfs
                COMMAND ${GIT_EXECUTABLE} lfs pull
                WORKING_DIRECTORY ${prefix}/src/${name}
                COMMENT "Pull LFS for ${name}"
                DEPENDEES download update
                ALWAYS TRUE
                LOG TRUE)
    else()
        ExternalProject_Add_Step(${name} lfs_fetch
                COMMAND ${GIT_EXECUTABLE} lfs fetch --prune
                WORKING_DIRECTORY ${prefix}/src/${name}
                COMMENT "Fetch LFS for ${name}"
                DEPENDEES download update
                ALWAYS TRUE
                LOG TRUE)

        ExternalProject_Add_Step(${name} lfs_checkout
                COMMAND ${GIT_EXECUTABLE} lfs checkout
                WORKING_DIRECTORY ${prefix}/src/${name}
                COMMENT "Checkout LFS for ${name}"
                DEPENDEES lfs_fetch
                ALWAYS TRUE
                LOG TRUE)
    endif()
endfunction()

function (fetch_models_and_validation_set)
    if (ENABLE_VALIDATION_SET)
        add_lfs_repo(
            "validation_set"
            "${TEMP}/validation_set"
            "git@gitlab-icv.inn.intel.com:inference-engine/validation-set.git"
            "${MODELS_BRANCH}")
    endif()
    set(VALIDATION_SET ${TEMP}/validation_set/src/validation_set PARENT_SCOPE)

    foreach(loop_var ${MODELS_LST_TO_FETCH})
        string(REPLACE ":" ";" MODEL_CONFIG_LST ${loop_var})

        list(GET MODEL_CONFIG_LST 0 folder_name)
        list(GET MODEL_CONFIG_LST 1 repo_name)
        list(GET MODEL_CONFIG_LST 2 branch_name)

        add_lfs_repo(
            "${folder_name}"
            "${TEMP}/models"
            "git@gitlab-icv.inn.intel.com:${repo_name}"
            "${branch_name}")
    endforeach(loop_var)
endfunction()
