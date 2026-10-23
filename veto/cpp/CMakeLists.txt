cmake_minimum_required(VERSION 3.16)
project(g4dis CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Geant4 REQUIRED)

pybind11_add_module(g4dis g4dis.cc)
target_include_directories(g4dis PRIVATE ${Geant4_INCLUDE_DIRS})
target_link_libraries(g4dis PRIVATE ${Geant4_LIBRARIES})
