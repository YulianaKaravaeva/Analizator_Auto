# Analizator_Auto

Андроид-приложение для анализа повреждения автомобиля.

Цель данного приложения: определить и выделить повреждения автомобиля на фотографии, а также соотнести его к одному из следующих классов: разбитая фара, разбитое стекло, вмятина и царапина.

Особенности приложения: возможность загружать фотографии и получать ответ в виде исходного фото, но с выделенными и подписанными повреждениями. На одном фото может быть обнаруженно более одного повреждения.

Инструменты: данное приложение было написанно в Android Studio с использованием языков Kotlin и C++. Также для обработки изображений была обученна нейронная сеть модели Yolov8.

Реализация использования Yolov8 в Android Studio была выполнена с помощью данного репозитория: https://github.com/lamegaton/YOLOv8-Custom-Object-Detection-Android/tree/main
