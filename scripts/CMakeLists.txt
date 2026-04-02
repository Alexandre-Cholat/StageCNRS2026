# Define the MFCC executable
add_executable(mfcc_calc mfcc_calc.cpp)

# Link math library for Linux (TDM-GCC on Windows usually handles this automatically)
if(UNIX AND NOT APPLE)
    target_link_libraries(mfcc_calc m)
endif()