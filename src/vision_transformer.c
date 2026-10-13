#include "vision_transformer.h"

#include <stdlib.h>

#include "cls_token.h"
#include "patch_embedding.h"
#include "position_embedding.h"
#include "transformer_encoder.h"

struct VisionTransformer {
    size_t embedding_dimension;
    size_t number_of_classes;
    size_t number_of_encoder_blocks;
    PatchEmbedding *patch_embedding;
    ClsToken *cls_token;
    PositionEmbedding *position_embedding;
    TransformerEncoder **encoder_blocks;
    Tensor *classifier_weight;
    Tensor *classifier_bias;
};

static Tensor *create_classifier_weight(size_t input_dimension, size_t output_dimension)
{
    size_t shape[2] = {input_dimension, output_dimension};
    Tensor *weight = tensor_zeros(shape, 2);
    float *data;
    size_t diagonal;
    size_t i;

    if (weight == NULL) {
        return NULL;
    }

    data = tensor_data_mut(weight);
    diagonal = input_dimension < output_dimension ? input_dimension : output_dimension;
    for (i = 0; i < diagonal; ++i) {
        data[i * output_dimension + i] = 1.0f;
    }
    return weight;
}

static Tensor *create_classifier_bias(size_t number_of_classes)
{
    size_t shape[1] = {number_of_classes};
    return tensor_zeros(shape, 1);
}

VisionTransformer *vision_transformer_create(
    size_t image_height,
    size_t image_width,
    size_t patch_size,
    size_t channels,
    size_t embedding_dimension,
    size_t number_of_heads,
    size_t mlp_hidden_dimension,
    size_t number_of_encoder_blocks,
    size_t number_of_classes)
{
    VisionTransformer *model;
    size_t patch_count;
    size_t sequence_length;
    size_t i;

    if (number_of_encoder_blocks == 0 || number_of_classes == 0) {
        return NULL;
    }

    model = (VisionTransformer *)calloc(1, sizeof(*model));
    if (model == NULL) {
        return NULL;
    }

    model->embedding_dimension = embedding_dimension;
    model->number_of_classes = number_of_classes;
    model->number_of_encoder_blocks = number_of_encoder_blocks;
    model->patch_embedding = patch_embedding_create(
        image_height, image_width, patch_size, channels, embedding_dimension);

    if (model->patch_embedding == NULL) {
        vision_transformer_free(model);
        return NULL;
    }

    patch_count = patch_embedding_num_patches(model->patch_embedding);
    if (patch_count == (size_t)-1) {
        vision_transformer_free(model);
        return NULL;
    }
    sequence_length = patch_count + 1;
    model->cls_token = cls_token_create(embedding_dimension);
    model->position_embedding = position_embedding_create(
        sequence_length, embedding_dimension);
    model->encoder_blocks = (TransformerEncoder **)calloc(
        number_of_encoder_blocks, sizeof(*model->encoder_blocks));

    if (model->cls_token == NULL || model->position_embedding == NULL
        || model->encoder_blocks == NULL) {
        vision_transformer_free(model);
        return NULL;
    }

    for (i = 0; i < number_of_encoder_blocks; ++i) {
        model->encoder_blocks[i] = transformer_encoder_create(
            embedding_dimension, number_of_heads, mlp_hidden_dimension);
        if (model->encoder_blocks[i] == NULL) {
            vision_transformer_free(model);
            return NULL;
        }
    }

    model->classifier_weight = create_classifier_weight(
        embedding_dimension, number_of_classes);
    model->classifier_bias = create_classifier_bias(number_of_classes);
    if (model->classifier_weight == NULL || model->classifier_bias == NULL) {
        vision_transformer_free(model);
        return NULL;
    }

    return model;
}

void vision_transformer_free(VisionTransformer *model)
{
    size_t i;

    if (model == NULL) {
        return;
    }

    if (model->encoder_blocks != NULL) {
        for (i = 0; i < model->number_of_encoder_blocks; ++i) {
            transformer_encoder_free(model->encoder_blocks[i]);
        }
        free(model->encoder_blocks);
    }

    patch_embedding_free(model->patch_embedding);
    cls_token_free(model->cls_token);
    position_embedding_free(model->position_embedding);
    tensor_free(model->classifier_weight);
    tensor_free(model->classifier_bias);
    free(model);
}

Tensor *vision_transformer_forward(
    const VisionTransformer *model,
    const Tensor *image)
{
    Tensor *patches;
    Tensor *tokens;
    Tensor *positioned_tokens;
    Tensor *encoded_tokens;
    Tensor *next_tokens;
    Tensor *logits;
    const float *encoded_data;
    const float *classifier_weight;
    const float *classifier_bias;
    float *logit_data;
    size_t shape[2];
    size_t class_index;
    size_t embedding_index;
    size_t i;

    if (model == NULL || image == NULL) {
        return NULL;
    }

    patches = patch_embedding_forward(model->patch_embedding, image);
    tokens = cls_token_forward(model->cls_token, patches);
    positioned_tokens = position_embedding_forward(model->position_embedding, tokens);
    if (patches == NULL || tokens == NULL || positioned_tokens == NULL) {
        tensor_free(patches);
        tensor_free(tokens);
        tensor_free(positioned_tokens);
        return NULL;
    }

    encoded_tokens = positioned_tokens;
    for (i = 0; i < model->number_of_encoder_blocks; ++i) {
        next_tokens = transformer_encoder_forward(
            model->encoder_blocks[i], encoded_tokens);
        if (next_tokens == NULL) {
            tensor_free(patches);
            tensor_free(tokens);
            tensor_free(positioned_tokens);
            if (encoded_tokens != positioned_tokens) {
                tensor_free(encoded_tokens);
            }
            return NULL;
        }

        if (encoded_tokens != positioned_tokens) {
            tensor_free(encoded_tokens);
        }
        encoded_tokens = next_tokens;
    }

    shape[0] = 1;
    shape[1] = model->number_of_classes;
    logits = tensor_zeros(shape, 2);
    if (logits == NULL) {
        tensor_free(patches);
        tensor_free(tokens);
        tensor_free(positioned_tokens);
        if (encoded_tokens != positioned_tokens) {
            tensor_free(encoded_tokens);
        }
        return NULL;
    }

    encoded_data = tensor_data(encoded_tokens);
    classifier_weight = tensor_data(model->classifier_weight);
    classifier_bias = tensor_data(model->classifier_bias);
    logit_data = tensor_data_mut(logits);

    for (class_index = 0; class_index < model->number_of_classes; ++class_index) {
        float value = classifier_bias[class_index];
        for (embedding_index = 0;
             embedding_index < model->embedding_dimension;
             ++embedding_index) {
            value += encoded_data[embedding_index]
                * classifier_weight[embedding_index * model->number_of_classes + class_index];
        }
        logit_data[class_index] = value;
    }

    tensor_free(patches);
    tensor_free(tokens);
    tensor_free(positioned_tokens);
    if (encoded_tokens != positioned_tokens) {
        tensor_free(encoded_tokens);
    }
    return logits;
}

size_t vision_transformer_number_of_classes(const VisionTransformer *model)
{
    return model == NULL ? 0 : model->number_of_classes;
}

size_t vision_transformer_number_of_blocks(const VisionTransformer *model)
{
    return model == NULL ? 0 : model->number_of_encoder_blocks;
}
