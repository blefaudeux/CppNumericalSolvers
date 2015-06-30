cmake_minimum_required (VERSION 2.6) 
project (CppNumericalSolvers) 

set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} "/usr/include/eigen3/")

# Recurse into the subdirectories. This does not actually 
# cause another cmake executable to run. The same process will walk through 
# the project's entire directory structure.
add_subdirectory (lib) 
add_subdirectory (demo) 
add_subdirectory (unittest) 

