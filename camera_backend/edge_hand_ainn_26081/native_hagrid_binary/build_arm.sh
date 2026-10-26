#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
TASK_COMPILER=/home/zhangwnag/nyx310-hand-ainn-20261016/toolchain/gcc-arm-13.0-2025.09-x86_64-aarch64-none-linux-gnu/bin/aarch64-buildroot-linux-gnu-g++
TASK_SYSROOT="$("$TASK_COMPILER" -print-sysroot)"
cd "$TASK_ROOT"
cmake -S . -B build-arm -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_CXX_COMPILER="$TASK_COMPILER"
cmake --build build-arm -j4
"$TASK_COMPILER" -std=c++14 -Iinclude tests/abi_probe.cpp -o build-arm/abi_probe -ldl
qemu/usr/bin/qemu-aarch64-static "$TASK_SYSROOT/usr/lib/ld-linux-aarch64.so.1" --library-path "$TASK_SYSROOT/usr/lib" build-arm/hagrid_probe hagrid_fixtures.bin
qemu/usr/bin/qemu-aarch64-static "$TASK_SYSROOT/usr/lib/ld-linux-aarch64.so.1" --library-path "$TASK_SYSROOT/usr/lib" build-arm/core_probe ../native_lowmem/arm_fixtures.bin
qemu/usr/bin/qemu-aarch64-static "$TASK_SYSROOT/usr/lib/ld-linux-aarch64.so.1" --library-path "$TASK_SYSROOT/usr/lib" build-arm/raw_probe ../native_lowmem/raw_fixtures.bin
qemu/usr/bin/qemu-aarch64-static "$TASK_SYSROOT/usr/lib/ld-linux-aarch64.so.1" --library-path "sdk/3rdparty/librknn_api/aarch64:$TASK_SYSROOT/usr/lib" build-arm/abi_probe build-arm/libalg.so
