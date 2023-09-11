add_library(
    kernel STATIC
    activation.hip
    attention.hip
    cache.hip
    layernorm.hip
    pos_encoding.hip
)

target_link_libraries(kernel PUBLIC util)
set_property(TARGET kernel PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET kernel PROPERTY HIP_ARCHITECTURES gfx906 gfx908 gfx90a gfx1100) # Tested on MI50, MI100, MI210, W7900
set_property(TARGET kernel PROPERTY HIP_STANDARD 17)


