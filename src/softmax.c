#include "softmax.h"

#include <math.h>

Tensor *softmax_forward(const Tensor *input)
{
    const size_t *shape;
    const float *input_data;
    float *output_data;
    Tensor *output;
    size_t row;
    size_t column;

    if (input == NULL || tensor_ndim(input) != 2) {
        return NULL;
    }

    shape = tensor_shape(input);
    output = tensor_create(shape, 2);
    if (output == NULL) {
        return NULL;
    }

    input_data = tensor_data(input);
    output_data = tensor_data_mut(output);

    for (row = 0; row < shape[0]; ++row) {
        const size_t row_offset = row * shape[1];
        float maximum = input_data[row_offset];
        float sum = 0.0f;

        for (column = 1; column < shape[1]; ++column) {
            if (input_data[row_offset + column] > maximum) {
                maximum = input_data[row_offset + column];
            }
        }

        for (column = 0; column < shape[1]; ++column) {
            output_data[row_offset + column] =
                expf(input_data[row_offset + column] - maximum);
            sum += output_data[row_offset + column];
        }

        for (column = 0; column < shape[1]; ++column) {
            output_data[row_offset + column] /= sum;
        }
    }

    return output;
}
