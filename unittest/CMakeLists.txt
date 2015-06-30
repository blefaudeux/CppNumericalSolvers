# Define the project's name
project(UnitTest)

# Make sure the compiler can find include files from our UKF library.
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/../lib)
include_directories(${PROJECT_SOURCE_DIR}/../lib/include)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Add all the files we're interested in (mostly for QTCreator)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/*.cpp)

# Make sure the linker can find the libSolver library once it is built.
set(LIBSOLVER_BINARY_DIR ${PROJECT_BINARY_DIR}/../lib)

link_directories (${LIBSOLVER_BINARY_DIR})
MESSAGE("LIBSOLVER Path :\n ${LIBSOLVER_BINARY_DIR}\n")

# Add executable "main"
add_executable(UnitTest ${SRC} ${HEADERS})

set_property(TARGET UnitTest PROPERTY CXX_STANDARD 11)
set_property(TARGET UnitTest PROPERTY CXX_STANDARD_REQUIRED ON)

# Link the executable to the UKF library.
target_link_libraries (UnitTest CppSolver)
