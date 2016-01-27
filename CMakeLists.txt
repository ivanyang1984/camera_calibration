set(SOURCES
    "src/camera_calibration.cpp"
    "src/interface.cpp"
)

set(HEADERS
    "include/camera_calibration.h"
    "include/calibration.h"
)

find_package( OpenCV REQUIRED )

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)
add_library(camera_calibration MODULE ${SOURCES} ${HEADERS})
target_link_libraries(camera_calibration PRIVATE lmscore imaging ${OpenCV_LIBS})
