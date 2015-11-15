# Define the project's name
project(CppSolver)

# Add all the files we're interested in (mostly for QTCreator)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/headers/*.h ${PROJECT_SOURCE_DIR}/headers/linesearch/*.h)

# Define the included directories
include_directories(${PROJECT_SOURCE_DIR}/headers)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/linesearch)

# Create static and dynamic libraries
add_library (CppSolver_static STATIC ${SRC} ${HEADERS})
add_library (CppSolver_dynamic SHARED ${SRC} ${HEADERS})

set_property(TARGET CppSolver_dynamic PROPERTY CXX_STANDARD 11)
set_property(TARGET CppSolver_dynamic PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET CppSolver_static PROPERTY CXX_STANDARD 11)
set_property(TARGET CppSolver_static PROPERTY CXX_STANDARD_REQUIRED ON)

### Define the installation procedure
install(TARGETS CppSolver_dynamic CppSolver_static DESTINATION ${PROJECT_SOURCE_DIR}/bin)
install(FILES ${HEADERS} DESTINATION ${PROJECT_SOURCE_DIR}/include)
