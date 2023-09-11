add_library(
    ops STATIC
    activation.cpp
    attention.cpp
    cache.cpp
    layernorm.cpp
    pos_encoding.cpp
)

target_link_libraries(ops PUBLIC kernel ${TORCH_LIBRARIES})
target_include_directories(ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(ops PROPERTIES POSITION_INDEPENDENT_CODE ON)