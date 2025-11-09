cmake_minimum_required(VERSION 3.10)
project(multipath_rng CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

include_directories(src)

add_executable(generator
    src/generator.cpp
    src/util.cpp
    src/logging.cpp
)

add_executable(dispatcher
    src/dispatcher.cpp
    src/util.cpp
    src/logging.cpp
    src/rng.cpp
)

add_executable(backend
    src/backend.cpp
    src/util.cpp
    src/logging.cpp
)

add_executable(predictor
    src/predictor.cpp
    src/util.cpp
    src/logging.cpp
)

target_link_libraries(generator pthread)
target_link_libraries(dispatcher pthread)
target_link_libraries(backend pthread)
target_link_libraries(predictor pthread)

