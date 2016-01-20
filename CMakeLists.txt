set(SOURCES
    "src/camera_calibration.cpp"
    "src/interface.cpp"
)

set(HEADERS
    "include/camera_calibration.h"
)

find_package( OpenCV REQUIRED )

message(STATUS ${OpenCV_INCLUDE_DIRS})

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)
add_library(camera_calibration MODULE ${SOURCES} ${HEADERS})
target_link_libraries(camera_calibration PRIVATE lmscore ${OpenCV_LIBS})
