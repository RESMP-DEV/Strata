cmake_minimum_required(VERSION 3.24)
project(strata_deepseek_contracts LANGUAGES CXX)

# This is intentionally separate from the engine build. The first DeepSeek
# V4.1 slices are dependency-free CPU contracts, so configuring them must not
# fetch or compile any backend runtime.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

function(add_deepseek_test target_name source_name)
  add_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR}/${source_name})
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  target_compile_features(${target_name} PRIVATE cxx_std_20)
  target_compile_options(${target_name} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wextra>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wpedantic>)
  add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

add_deepseek_test(deepseek_geometry_test geometry_test.cpp)
add_deepseek_test(deepseek_candidate_select_test candidate_select_test.cpp)
