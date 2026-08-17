cmake_minimum_required(VERSION 3.21)

project(TextAnnotator
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

qt_add_executable(TextAnnotator
    WIN32

    src/main.cpp

    src/MainWindow.h
    src/MainWindow.cpp

    src/Annotation.h

    src/AnnotationModel.h
    src/AnnotationModel.cpp

    src/AnnotationEditor.h
    src/AnnotationEditor.cpp

    src/JsonStorage.h
    src/JsonStorage.cpp
)

target_link_libraries(TextAnnotator
    PRIVATE
        Qt6::Widgets
)