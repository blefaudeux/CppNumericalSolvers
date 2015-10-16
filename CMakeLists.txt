<<<<<<< HEAD
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set(PROJECT_NAME_STR CppSolver)

project( ${PROJECT_NAME_STR} )

include(CppNumericalSolvers.config)

find_package(Threads REQUIRED)

set(CMAKE_PREFIX_PATH "/usr/include/eigen3/")

# Set debug or release configurations
if( CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]" )
    message( "INFO: DEBUG BUILD" )
    # for debug type builds, turn on verbose makefiles
    SET(CMAKE_VERBOSE_MAKEFILE ON)

    # Tell other CMake files that we're doing a debug build
    SET( DEBUG_BUILD 1 )

    # Tell C/C++ that we're doing a debug build
    ADD_DEFINITIONS( -DDEBUG )
endif()

if( CMAKE_BUILD_TYPE MATCHES "[Rr][Ee][Ll][Ee][Aa][Ss][Ee]" )
    message( "INFO: RELEASE BUILD" )
endif()

# check existence of c++11 compiler
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Find Eigen3 on this machine
set( EIGEN3_INCLUDE_DIR "/usr/include/eigen3/")
include_directories( "${EIGEN3_INCLUDE_DIR}" )

# Recurse into the subdirectories. This does not actually
add_subdirectory (lib)
add_subdirectory (demo)

# ----------------------------------------
# Handle Google framework for unit testing
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/lib/headers )

link_directories (${PROJECT_BINARY_DIR}/lib)

add_subdirectory(${EXT_PROJECTS_DIR}/gtest)

enable_testing()
set(PROJECT_TEST_NAME ${PROJECT_NAME_STR}_test)
include_directories(${GTEST_INCLUDE_DIRS} ${COMMON_INCLUDES})

file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/unittest/*.cpp)
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
add_dependencies(${PROJECT_TEST_NAME} googletest)

if(NOT WIN32)
    target_link_libraries(${PROJECT_TEST_NAME}
        ${GTEST_LIBS_DIR}/libgtest.a
        ${GTEST_LIBS_DIR}/libgtest_main.a
        )
else()
    target_link_libraries(${PROJECT_TEST_NAME}
        debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
        optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
        )
    target_link_libraries(${PROJECT_TEST_NAME}
        debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
        optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
        )
endif()

set(LIBSOLVER_BINARY_DIR ${PROJECT_BINARY_DIR}/lib)
link_directories (${LIBSOLVER_BINARY_DIR})
MESSAGE("LIBSOLVER Path :\n ${LIBSOLVER_BINARY_DIR}\n")

target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT} ${PROJECT_NAME_STR}_dynamic )

add_test(test1 ${PROJECT_TEST_NAME})
