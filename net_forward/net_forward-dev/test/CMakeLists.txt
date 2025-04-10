# CMakeList.txt: opticMasterBeta �?CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑�?#
cmake_minimum_required (VERSION 3.8)

project(net_test C CXX)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
# set(PATH_TEST ${PROJECT_SOURCE_DIR})
# set(PATH_LIB ${PROJECT_SOURCE_DIR}/lib)
# set(PATH_NET ${PROJECT_SOURCE_DIR}/../net)
# set(PATH_PARA ${PATH_NET}/net_param)
set(PATH_TEST .)
set(PATH_LIB ./lib)
set(PATH_NET ../net)
#set(PATH_PARA  ../net/net_param)
set(PATH_PARA  ../net/${chip})
MESSAGE(STATUS PATH_TEST_ORI = ${PROJECT_SOURCE_DIR})
MESSAGE(STATUS PATH_TEST = ${PATH_TEST})
MESSAGE(STATUS PATH_LIB  = ${PATH_LIB})
MESSAGE(STATUS PATH_NET  = ${PATH_NET})
MESSAGE(STATUS PATH_PARA = ${PATH_PARA})


ADD_DEFINITIONS(-DRUN_TST=1)
add_definitions(-DPROJECT_57_ZX)
add_definitions(-ftrapv)

find_package(PkgConfig)
pkg_search_module(SLXML REQUIRED libxml-2.0)
IF (WIN32)
        MESSAGE(STATUS "Now is windows")
        add_definitions(-DSL_WINDOWS)
        add_definitions(-DNEON_OFF)
        set(PKG_CONFIG_EXECUTABLE "D:/soft/pkg-config/bin/pkg-config.exe")
ELSEIF (APPLE)
        MESSAGE(STATUS "Now is Apple systens.")
        add_definitions(-DSL_MAC)
        add_definitions(-DNEON_OFF)
        set(ENV{PKG_CONFIG_PATH} "D:/lib/pangolin/share/pkgconfig")
ELSEIF (UNIX)
        MESSAGE(STATUS "Now is UNIX-like OS's.")
        add_definitions(-DSL_LINUX)
        add_definitions(-DNEON_OFF)
        set(ENV{PKG_CONFIG_PATH} /usr/local/lib/pkgconfig)
ENDIF ()


set(LIST_SRC1
        ./main_test.c
        ./bmp.c
        ../net/${chip}/alog.c
        ../net/android.c
        ../net/net_cnn_common.c
        ../net/SL_Math.c
        ../net/block_function.c

)


include_directories(${PATH_PARA})
include_directories(${PATH_NET})
include_directories(${PATH_TEST})


IF (${chip} STREQUAL "6135")
    ADD_DEFINITIONS(-DCHIP6135)
    set(LIST_SRC2
        ../net/${chip}/net_mistouch.c
        ../net/${chip}/net_spoof.c
    )

ELSEIF (${chip} STREQUAL "6159")
    ADD_DEFINITIONS(-DCHIP6159)
    set(LIST_SRC2
        ../net/${chip}/net_mistouch.c
        ../net/${chip}/net_spoof.c
    )
ELSEIF (${chip} STREQUAL "6191")
    ADD_DEFINITIONS(-DCHIP6191)
    set(LIST_SRC2
        ../net/${chip}/net_mistouch.c
        ../net/${chip}/net_spoof.c
    )
ELSEIF (${chip} STREQUAL "6192")
    ADD_DEFINITIONS(-DCHIP6192)
    set(LIST_SRC2

        ../net/${chip}/net_mistouch.c
        ../net/${chip}/net_spoof.c
        ../net/${chip}/net_ori_enh.c
        ../net/${chip}/net_patch.c
        ../net/${chip}/net_spd.c
        ../net/${chip}/net_quality.c
    )
ELSEIF  (${chip} STREQUAL "6193")
    ADD_DEFINITIONS(-DCHIP6193)
    set(LIST_SRC2

    ../net/${chip}/net_mistouch.c
    ../net/${chip}/net_spoof.c
    ../net/${chip}/net_spd.c
    ../net/${chip}/net_mask.c
    ../net/${chip}/net_enhance.c
    ../net/${chip}/net_patch.c
    ../net/${chip}/net_exp.c
    ../net/${chip}/net_quality.c
    )
ENDIF ()
MESSAGE(STATUS chip= ${chip})

IF (${mod} STREQUAL "net")
    ADD_DEFINITIONS(-DEXP=1)
    ADD_DEFINITIONS(-DENHANCE=1)
    ADD_DEFINITIONS(-DMASK=1)
    ADD_DEFINITIONS(-DSPD=1)
    ADD_DEFINITIONS(-DSPOOF=1)
    ADD_DEFINITIONS(-DQUALITY=1)
    ADD_DEFINITIONS(-DPATCH=1)
    ADD_DEFINITIONS(-DMISTOUCH=1)
ELSEIF (${mod} STREQUAL "exp")
    ADD_DEFINITIONS(-DEXP=1)
ELSEIF (${mod} STREQUAL "enhance")
    ADD_DEFINITIONS(-DENHANCE=1)
ELSEIF (${mod} STREQUAL "mask")
    ADD_DEFINITIONS(-DMASK=1)
ELSEIF (${mod} STREQUAL "spd")
    ADD_DEFINITIONS(-DSPD=1)
ELSEIF (${mod} STREQUAL "spoof")
    ADD_DEFINITIONS(-DSPOOF=1)
ELSEIF (${mod} STREQUAL "quality")
    ADD_DEFINITIONS(-DQUALITY=1)
ELSEIF (${mod} STREQUAL "mistouch")
    ADD_DEFINITIONS(-DMISTOUCH=1)
ELSEIF (${mod} STREQUAL "patch")
    ADD_DEFINITIONS(-DPATCH=1)
ENDIF ()

enable_testing()

add_definitions(-DUSE_MEMORYPOOL=0)

SET(MAIN_TEST "main_test.c")
add_executable (main_test ${MAIN_TEST} ${LIST_SRC1} ${LIST_SRC2})
target_include_directories(main_test PUBLIC  ${PATH_PARA} ${PATH_NET}  ${PATH_TEST} )#

