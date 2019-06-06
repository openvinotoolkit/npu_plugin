# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

function (fetch_mcmCompiler)

        list(LENGTH ARGV args)
        if (args EQUAL 1)
            list(GET ARGV 0 branch_name)
        else()
            set(branch_name "master")
        endif()
        set(MCM_BASE_DIR ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/movidius/mcmCompiler)
        if (NOT DEFINED ENV{MCM_HOME})
            message( "\nMCM_HOME environment variable must be defined")
            message( "most likely by:")
            message( "export MCM_HOME=${MCM_BASE_DIR}/src/mcmCompiler" )
            message( FATAL_ERROR "\n")
        endif()
        set(BIN_DIR "${MCM_BASE_DIR}/src/mcmCompiler-build")
        set(STAMP_DIR "${MCM_BASE_DIR}/src/mcmCompiler-stamp")
        ExternalProject_Add(
            mcmCompiler
            PREFIX ${MCM_BASE_DIR}
            GIT_REPOSITORY "git@github.com:movidius/mcmCompiler.git"
            GIT_TAG ${branch_name}
            INSTALL_COMMAND ""
            LOG_CONFIGURE 1
        )
endfunction(fetch_mcmCompiler)
