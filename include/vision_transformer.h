#ifndef VISION_TRANSFORMER_H
#define VISION_TRANSFORMER_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VisionTransformer VisionTransformer;

VisionTransformer *vision_transformer_create(
    size_t image_height,
    size_t image_width,
    size_t patch_size,
    size_t channels,
    size_t embedding_dimension,
    size_t number_of_heads,
    size_t mlp_hidden_dimension,
    size_t number_of_encoder_blocks,
    size_t number_of_classes);

void vision_transformer_free(VisionTransformer *model);

/* Return logits with shape [1, number_of_classes]. */
Tensor *vision_transformer_forward(
    const VisionTransformer *model,
    const Tensor *image);

/* Return the encoded CLS feature with shape [1, embedding_dimension]. */
Tensor *vision_transformer_forward_features(
    const VisionTransformer *model,
    const Tensor *image);

/* Update only the classification head using one labeled example. */
TensorStatus vision_transformer_train_classifier_step(
    VisionTransformer *model,
    const Tensor *features,
    size_t target_class,
    float learning_rate,
    float *loss);

size_t vision_transformer_number_of_classes(const VisionTransformer *model);
size_t vision_transformer_number_of_blocks(const VisionTransformer *model);

#ifdef __cplusplus
}
#endif

#endif /* VISION_TRANSFORMER_H */
