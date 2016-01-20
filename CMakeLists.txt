set(SOURCES
    "src/camera_calibration.cpp"
    "src/interface.cpp"
)

set(HEADERS
    "include/camera_calibration.h"
)

include_directories(include)
add_library(camera_calibration MODULE ${SOURCES} ${HEADERS})
target_link_libraries(camera_calibration PRIVATE lmscore)
