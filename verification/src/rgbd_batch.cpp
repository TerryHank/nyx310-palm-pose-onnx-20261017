#include <cmath>
#include "Scepter_api.h"

// Exact manufacturer projection, one Python/native crossing per frame.
extern "C" __declspec(dllexport) int project_depth_batch(
    ScDeviceHandle device, const uint16_t* mm, const ScVector3f* cloud,
    int width, int height, int rgbWidth, int rgbHeight,
    float* uv, float* xyz, int* sourceIndices) {
    int count = 0;
    const ScVector2u16 size{static_cast<uint16_t>(rgbWidth), static_cast<uint16_t>(rgbHeight)};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const int index = y * width + x;
        const auto p = cloud[index];
        if (!mm[index] || mm[index] == 65535 || p.z <= 0 ||
            !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        ScVector2u16 pixel{};
        if (scTransformDepthPointToColorPoint(device, {x,y,mm[index]}, size, &pixel) != 0 ||
            pixel.x == 0 || pixel.y == 0 || pixel.x >= rgbWidth || pixel.y >= rgbHeight) continue;
        uv[count*2] = pixel.x; uv[count*2+1] = pixel.y;
        xyz[count*3] = p.x/1000.f; xyz[count*3+1] = p.y/1000.f; xyz[count*3+2] = p.z/1000.f;
        sourceIndices[count] = index;
        ++count;
    }
    return count;
}
