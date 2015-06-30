# Define the project's name
project(Demo)

cmake_minimum_required(VERSION 2.6)

# Make sure the compiler can find include files from our UKF library. 
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/../lib)
include_directories(${PROJECT_SOURCE_DIR}/../lib/include)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Set debug/release flags
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

# Add all the files we're interested in (mostly for QTCreator)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/*.cpp)

# Make sure the linker can find the libSolver library once it is built. 
set(LIBSOLVER_BINARY_DIR ${PROJECT_SOURCE_DIR}/../lib/bin)

link_directories (${LIBSOLVER_BINARY_DIR}) 
MESSAGE("LIBSOLVER Path :\n ${LIBSOLVER_BINARY_DIR}\n")

# Add executable "main"
add_executable(main ${SRC} ${HEADERS})

# Link the executable to the UKF library.
target_link_libraries (main CppSolver)
