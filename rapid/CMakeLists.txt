cmake_minimum_required(VERSION 3.16)
project(wilson_rapid CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

add_library(rapid_core STATIC
  src/fuels.cpp
  src/fetch.cpp
  src/landscape.cpp
  src/spread.cpp
  src/ensemble.cpp
  src/calibrate.cpp
  src/tactics.cpp
  src/planner.cpp
  src/render.cpp
  src/incident.cpp
  src/stb_impl.cpp
)
set_source_files_properties(src/stb_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_include_directories(rapid_core PUBLIC src third_party)
# Must be visible to every translation unit that includes httplib.h.
target_compile_definitions(rapid_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
target_link_libraries(rapid_core PUBLIC Threads::Threads OpenSSL::SSL OpenSSL::Crypto)
if(NOT MSVC)
  target_compile_options(rapid_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

add_executable(wilson_rapid src/main.cpp)
target_link_libraries(wilson_rapid PRIVATE rapid_core)

add_executable(rapid_tests tests/tests.cpp)
target_link_libraries(rapid_tests PRIVATE rapid_core)

enable_testing()
add_test(NAME rapid_tests COMMAND rapid_tests)
