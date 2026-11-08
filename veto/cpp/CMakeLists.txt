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

# transport of shield-iron interaction final states to the shield exit (bin/shield_dis.py)
pybind11_add_module(g4shield g4shield.cc)
target_include_directories(g4shield PRIVATE ${Geant4_INCLUDE_DIRS})
target_link_libraries(g4shield PRIVATE ${Geant4_LIBRARIES})

# transport of vessel / helium interaction final states to the straw stations and the timing detector (bin/g4_vessel_transport.py)
pybind11_add_module(g4vessel g4vessel.cc)
target_include_directories(g4vessel PRIVATE ${Geant4_INCLUDE_DIRS})
target_link_libraries(g4vessel PRIVATE ${Geant4_LIBRARIES})

pybind11_add_module(g4slab g4slab.cc)                       # EM debris behind slabs (bin/em_debris_study.py)
target_include_directories(g4slab PRIVATE ${Geant4_INCLUDE_DIRS})
target_link_libraries(g4slab PRIVATE ${Geant4_LIBRARIES})

pybind11_add_module(g4rock g4rock.cc)                       # cavern-wall DIS: what leaves the rock (bin/rock_dis.py)
target_include_directories(g4rock PRIVATE ${Geant4_INCLUDE_DIRS})
target_link_libraries(g4rock PRIVATE ${Geant4_LIBRARIES})
