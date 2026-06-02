#!/bin/bash
# Build and run all benchmarks on Jetson Nano
set -e

echo "=== Jetson Nano CUDA SGM Benchmark ==="
echo ""
echo "System info:"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "  OpenCV: $(pkg-config --modversion opencv4 2>/dev/null || echo 'check manually')"
echo "  GPU: $(tegrastats 2>/dev/null | head -1 || echo 'jetson_clocks not available')"
echo ""

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

echo ""
echo "=== Test 1: Quick smoke test (left.jpg/right.jpg) ==="
cd ../..
if [ -f x64/Debug/left.jpg ]; then
    cp x64/Debug/left.jpg . 2>/dev/null || true
    cp x64/Debug/right.jpg . 2>/dev/null || true
fi
./jetson/build/sgm_adaptive

echo ""
echo "=== Test 2: KITTI evaluation (first 10 frames) ==="
# For a quick test, create a subset of KITTI data
if [ -d "data/KITTI/training/image_2" ]; then
    echo "Running KITTI eval (this will take a while for 200 frames)..."
    ./jetson/build/kitti_eval
else
    echo "KITTI data not found. Copy data/KITTI/ from PC first."
fi

echo ""
echo "=== Power monitoring ==="
echo "Run this in another terminal during benchmarks:"
echo "  tegrastats --interval 500"
