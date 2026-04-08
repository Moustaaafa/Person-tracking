TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

SOURCES += \
    config.cpp \
    detector.cpp \
    main.cpp \
    pose.cpp \
    reid.cpp \
    tracker.cpp \
    visualization.cpp

HEADERS += \
    config.h \
    detector.h \
    pose.h \
    reid.h \
    tracker.h \
    visualization.h

INCLUDEPATH += /home/mosta/Downloads/onnxruntime-linux-x64-1.24.2/include

LIBS += -L/home/mosta/Downloads/onnxruntime-linux-x64-1.24.2/lib -lonnxruntime

QMAKE_LFLAGS += -Wl,-rpath,/home/mosta/Downloads/onnxruntime-linux-x64-1.24.2/lib

# Only if you actually use jsoncpp in this project:
INCLUDEPATH += /usr/include/jsoncpp
LIBS += -ljsoncpp

# Optional if you get linker errors:
LIBS += -lpthread -ldl
