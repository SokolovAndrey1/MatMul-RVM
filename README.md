# RISC-V THEAD Matrix Multiplication practice

Репозиторий для практической работы по оптимизации умножения матриц для архитектуры RISC-V с помощью RISC-V THEAD Matrix Extention

## Задача

Разработать реализацию блочного умножения матриц с float элементами используюя инструкции матричного расширения THEAD.\
Размер блока должен быть 4х4. Референсная реализация для квадратных матриц [gemm_block4x4_ref](lib/src/gemm_blocked_ref.c).

* [Описание инструкций матричного расширения](https://github.com/T-head-Semi/riscv-matrix-extension-spec/blob/master/spec/matrix_instructions.adoc)
* [RVM intrinsics](https://github.com/T-head-Semi/riscv-matrix-extension-spec/blob/master/doc/intrinsic/rvm-intrinsic-api.adoc)
* [Пример использования матричного расширения](https://github.com/T-head-Semi/riscv-matrix-extension-spec/blob/master/demos/intrinsic_matmul/matmul.c) (поэлементное умножение матриц с целыми числами)

## Алгоритм работы

1) Создать fork репозитория.
2) Настроить окружение с помощью скрипта [env.sh](env.sh).
2) Собрать проект с помощью скрипта [build.sh](build.sh).
3) Запустить тесты на RISC-V Qemu для квадратных и прямоугольных матриц.
5) Реализовать [gemm_block4x4_rvm](lib/src/gemm_blocked_rvm.c), используя инструкции из матричного расширения:
    * Для квадратных матриц с размерами, кратные 4. Должны проходить тесты `_build/test/test_rvm_square`.
    * _Опционально:_ Для произвольных размеров матриц должны проходить тесты `_build/test/test_rvm_nonsquare`.
6) Открыть Pull Request c оптимизациями.
    * Соотвествующий произведенным оптимизациям CI должен быть зеленым.

## Структура репозитория

* [lib](lib) - библиотека с реализацией умножения матриц
* [test](test) - функциональные тесты для проверки корректности алгоритмов

## Настройка окружения

После запуска скрипта настройки окружения [env.sh](env.sh) в корневой папке репозитория должна быть папка `tools` с `gcc` (тулчейн для кросс-компиляции) и `qemu` (RISC-V симулятор).

## Сборка проекта

Для сборки проекта можно использовать скрипт [build.sh](build.sh). В нем выставлены пути для кросс-компиляторов относительно путей в шаге настройке окружения. Можно изменить параметры сборки (например, `BUILD_TYPE` для отладки)

Опции конфигурации проекта:
* `ENABLE_TEST` - Сборка тестов (`ON`\\`OFF`)
* `BUILD_TYPE` - Режим сборки (`Release`\\`Debug`)
* `BUILD_FOLDER` - Папка для артефактов сборки
* `BUILD_STATIC` - Включить статическую линковку (`ON`\\`OFF`)

## Пример запуска с помощью QEMU

``
./tools/qemu/bin/qemu-riscv64 -cpu c907fdvm-rv64 ./_build/test/test_rvm_square
``\
``
./tools/qemu/bin/qemu-riscv64 -cpu c907fdvm-rv64 ./_build/test/test_rvm_nonsquare
``\
где `c907fdvm-rv64` нужен для поддержки RISC-V Matrix Extention

В случае ошибки:\
``
qemu-riscv64: Could not open '/lib/ld-linux-riscv64-lp64d.so.1': No such file or directory
``\
необходимо сконфигурировать проект с `-DBUILD_STATIC=ON` или добавить ключ запуска `-L ./tools/gcc/sysroot/`

## Отладка кода на RISC-V

[Краткая инструкция](docs/How2Debug.md)
