#include "initializer.h"

#include <math.h>

static uint32_t next_random(uint32_t *state)
{
    uint32_t value = *state;

    if (value == 0) {
        value = 0x9E3779B9u;
    }

    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

TensorStatus tensor_fill_xavier(
    Tensor *tensor,
    size_t fan_in,
    size_t fan_out,
    uint32_t seed)
{
    float *data;
    float limit;
    uint32_t state = seed == 0 ? 0x12345678u : seed;
    size_t i;

    if (tensor == NULL || fan_in == 0 || fan_out == 0) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    data = tensor_data_mut(tensor);
    for (i = 0; i < tensor_numel(tensor); ++i) {
        const float unit = (float)next_random(&state) / 4294967295.0f;
        data[i] = (2.0f * unit - 1.0f) * limit;
    }

    return TENSOR_OK;
}
