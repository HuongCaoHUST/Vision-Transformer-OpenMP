#include "vision_transformer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cls_token.h"
#include "cross_entropy.h"
#include "initializer.h"
#include "layer_norm.h"
#include "mlp.h"
#include "multi_head_attention.h"
#include "patch_embedding.h"
#include "position_embedding.h"
#include "transformer_encoder.h"

struct VisionTransformer {
    size_t image_height;
    size_t image_width;
    size_t patch_size;
    size_t channels;
    size_t embedding_dimension;
    size_t number_of_heads;
    size_t mlp_hidden_dimension;
    size_t number_of_classes;
    size_t number_of_encoder_blocks;
    PatchEmbedding *patch_embedding;
    ClsToken *cls_token;
    PositionEmbedding *position_embedding;
    TransformerEncoder **encoder_blocks;
    Tensor *classifier_weight;
    Tensor *classifier_bias;
};

static const unsigned char weights_magic[8] = {
    'V', 'I', 'T', 'W', '0', '0', '1', '\0'
};

static int write_float_array(FILE *file, const float *data, size_t count)
{
    return data != NULL && fwrite(data, sizeof(*data), count, file) == count;
}

static int read_float_array(FILE *file, float *data, size_t count)
{
    return data != NULL && fread(data, sizeof(*data), count, file) == count;
}

static int write_block_weights(FILE *file, TransformerEncoder *block)
{
    LayerNorm *attention_norm = transformer_encoder_attention_norm_mut(block);
    MultiHeadAttention *attention = transformer_encoder_attention_mut(block);
    LayerNorm *mlp_norm = transformer_encoder_mlp_norm_mut(block);
    MlpBlock *mlp = transformer_encoder_mlp_mut(block);
    const size_t dimension = layer_norm_dimension(attention_norm);
    const size_t hidden_dimension = mlp_hidden_dimension(mlp);
    const size_t matrix_size = dimension * dimension;

    return write_float_array(
               file, layer_norm_gamma_mut(attention_norm), dimension)
        && write_float_array(
               file, layer_norm_beta_mut(attention_norm), dimension)
        && write_float_array(
               file, multi_head_attention_query_weights_mut(attention), matrix_size)
        && write_float_array(
               file, multi_head_attention_key_weights_mut(attention), matrix_size)
        && write_float_array(
               file, multi_head_attention_value_weights_mut(attention), matrix_size)
        && write_float_array(
               file, multi_head_attention_output_weights_mut(attention), matrix_size)
        && write_float_array(
               file, multi_head_attention_query_bias_mut(attention), dimension)
        && write_float_array(
               file, multi_head_attention_key_bias_mut(attention), dimension)
        && write_float_array(
               file, multi_head_attention_value_bias_mut(attention), dimension)
        && write_float_array(
               file, multi_head_attention_output_bias_mut(attention), dimension)
        && write_float_array(
               file, layer_norm_gamma_mut(mlp_norm), dimension)
        && write_float_array(
               file, layer_norm_beta_mut(mlp_norm), dimension)
        && write_float_array(
               file, mlp_input_weights_mut(mlp), dimension * hidden_dimension)
        && write_float_array(
               file, mlp_input_bias_mut(mlp), hidden_dimension)
        && write_float_array(
               file, mlp_output_weights_mut(mlp), hidden_dimension * dimension)
        && write_float_array(
               file, mlp_output_bias_mut(mlp), dimension);
}

static int read_block_weights(FILE *file, TransformerEncoder *block)
{
    LayerNorm *attention_norm = transformer_encoder_attention_norm_mut(block);
    MultiHeadAttention *attention = transformer_encoder_attention_mut(block);
    LayerNorm *mlp_norm = transformer_encoder_mlp_norm_mut(block);
    MlpBlock *mlp = transformer_encoder_mlp_mut(block);
    const size_t dimension = layer_norm_dimension(attention_norm);
    const size_t hidden_dimension = mlp_hidden_dimension(mlp);
    const size_t matrix_size = dimension * dimension;

    return read_float_array(
               file, layer_norm_gamma_mut(attention_norm), dimension)
        && read_float_array(
               file, layer_norm_beta_mut(attention_norm), dimension)
        && read_float_array(
               file, multi_head_attention_query_weights_mut(attention), matrix_size)
        && read_float_array(
               file, multi_head_attention_key_weights_mut(attention), matrix_size)
        && read_float_array(
               file, multi_head_attention_value_weights_mut(attention), matrix_size)
        && read_float_array(
               file, multi_head_attention_output_weights_mut(attention), matrix_size)
        && read_float_array(
               file, multi_head_attention_query_bias_mut(attention), dimension)
        && read_float_array(
               file, multi_head_attention_key_bias_mut(attention), dimension)
        && read_float_array(
               file, multi_head_attention_value_bias_mut(attention), dimension)
        && read_float_array(
               file, multi_head_attention_output_bias_mut(attention), dimension)
        && read_float_array(
               file, layer_norm_gamma_mut(mlp_norm), dimension)
        && read_float_array(
               file, layer_norm_beta_mut(mlp_norm), dimension)
        && read_float_array(
               file, mlp_input_weights_mut(mlp), dimension * hidden_dimension)
        && read_float_array(
               file, mlp_input_bias_mut(mlp), hidden_dimension)
        && read_float_array(
               file, mlp_output_weights_mut(mlp), hidden_dimension * dimension)
        && read_float_array(
               file, mlp_output_bias_mut(mlp), dimension);
}

static int write_model_weights(FILE *file, VisionTransformer *model)
{
    const size_t embedding_dimension = model->embedding_dimension;
    const size_t number_of_classes = model->number_of_classes;
    const size_t patch_dimension =
        patch_embedding_patch_dimension(model->patch_embedding);
    const size_t sequence_length =
        patch_embedding_num_patches(model->patch_embedding) + 1;
    size_t i;

    if (!write_float_array(
            file,
            patch_embedding_weights_mut(model->patch_embedding),
            patch_dimension * embedding_dimension)
        || !write_float_array(
            file,
            patch_embedding_bias_mut(model->patch_embedding),
            embedding_dimension)
        || !write_float_array(
            file, cls_token_data_mut(model->cls_token), embedding_dimension)
        || !write_float_array(
            file,
            position_embedding_data_mut(model->position_embedding),
            sequence_length * embedding_dimension)) {
        return 0;
    }

    for (i = 0; i < model->number_of_encoder_blocks; ++i) {
        if (!write_block_weights(file, model->encoder_blocks[i])) {
            return 0;
        }
    }

    return write_float_array(
               file,
               tensor_data(model->classifier_weight),
               embedding_dimension * number_of_classes)
        && write_float_array(
               file, tensor_data(model->classifier_bias), number_of_classes);
}

static int read_model_weights(FILE *file, VisionTransformer *model)
{
    const size_t embedding_dimension = model->embedding_dimension;
    const size_t number_of_classes = model->number_of_classes;
    const size_t patch_dimension =
        patch_embedding_patch_dimension(model->patch_embedding);
    const size_t sequence_length =
        patch_embedding_num_patches(model->patch_embedding) + 1;
    size_t i;

    if (!read_float_array(
            file,
            patch_embedding_weights_mut(model->patch_embedding),
            patch_dimension * embedding_dimension)
        || !read_float_array(
            file,
            patch_embedding_bias_mut(model->patch_embedding),
            embedding_dimension)
        || !read_float_array(
            file, cls_token_data_mut(model->cls_token), embedding_dimension)
        || !read_float_array(
            file,
            position_embedding_data_mut(model->position_embedding),
            sequence_length * embedding_dimension)) {
        return 0;
    }

    for (i = 0; i < model->number_of_encoder_blocks; ++i) {
        if (!read_block_weights(file, model->encoder_blocks[i])) {
            return 0;
        }
    }

    return read_float_array(
               file,
               tensor_data_mut(model->classifier_weight),
               embedding_dimension * number_of_classes)
        && read_float_array(
               file, tensor_data_mut(model->classifier_bias), number_of_classes);
}

static int write_weights_header(FILE *file, const VisionTransformer *model)
{
    const uint64_t config[] = {
        (uint64_t)model->image_height,
        (uint64_t)model->image_width,
        (uint64_t)model->patch_size,
        (uint64_t)model->channels,
        (uint64_t)model->embedding_dimension,
        (uint64_t)model->number_of_heads,
        (uint64_t)model->mlp_hidden_dimension,
        (uint64_t)model->number_of_encoder_blocks,
        (uint64_t)model->number_of_classes
    };

    return fwrite(weights_magic, sizeof(weights_magic), 1, file) == 1
        && fwrite(config, sizeof(config[0]), sizeof(config) / sizeof(config[0]), file)
            == sizeof(config) / sizeof(config[0]);
}

static int read_weights_header(FILE *file, size_t *config)
{
    unsigned char magic[sizeof(weights_magic)];
    uint64_t raw_config[9];
    size_t i;

    if (fread(magic, sizeof(magic), 1, file) != 1
        || fread(raw_config, sizeof(raw_config[0]),
                 sizeof(raw_config) / sizeof(raw_config[0]), file)
            != sizeof(raw_config) / sizeof(raw_config[0])) {
        return 0;
    }

    for (i = 0; i < sizeof(weights_magic); ++i) {
        if (magic[i] != weights_magic[i]) {
            return 0;
        }
    }

    for (i = 0; i < sizeof(raw_config) / sizeof(raw_config[0]); ++i) {
        if (raw_config[i] > (uint64_t)SIZE_MAX) {
            return 0;
        }
        config[i] = (size_t)raw_config[i];
    }

    return 1;
}

static Tensor *create_classifier_weight(size_t input_dimension, size_t output_dimension)
{
    size_t shape[2] = {input_dimension, output_dimension};
    Tensor *weight = tensor_zeros(shape, 2);

    if (weight == NULL) {
        return NULL;
    }

    (void)tensor_fill_xavier(weight, input_dimension, output_dimension, 0x4001u);
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
    model->image_height = image_height;
    model->image_width = image_width;
    model->patch_size = patch_size;
    model->channels = channels;
    model->number_of_heads = number_of_heads;
    model->mlp_hidden_dimension = mlp_hidden_dimension;
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
        model->encoder_blocks[i] = transformer_encoder_create_seeded(
            embedding_dimension,
            number_of_heads,
            mlp_hidden_dimension,
            0x5000u + (uint32_t)i * 0x100u);
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

TensorStatus vision_transformer_save_weights(
    const VisionTransformer *model,
    const char *path)
{
    FILE *file;
    int success;

    if (model == NULL || path == NULL) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    file = fopen(path, "wb");
    if (file == NULL) {
        return TENSOR_ERROR_IO;
    }

    /* The save helper only reads the model; its mutating accessors are used
       internally because the layer APIs expose their parameter buffers that way. */
    success = write_weights_header(file, model)
        && write_model_weights(file, (VisionTransformer *)model);
    if (fclose(file) != 0) {
        success = 0;
    }

    return success ? TENSOR_OK : TENSOR_ERROR_IO;
}

VisionTransformer *vision_transformer_load_weights(const char *path)
{
    FILE *file;
    size_t config[9];
    VisionTransformer *model;
    int extra_byte;

    if (path == NULL) {
        return NULL;
    }

    file = fopen(path, "rb");
    if (file == NULL || !read_weights_header(file, config)) {
        if (file != NULL) {
            fclose(file);
        }
        return NULL;
    }

    model = vision_transformer_create(
        config[0], config[1], config[2], config[3], config[4],
        config[5], config[6], config[7], config[8]);
    if (model == NULL || !read_model_weights(file, model)) {
        fclose(file);
        vision_transformer_free(model);
        return NULL;
    }

    extra_byte = fgetc(file);
    fclose(file);
    if (extra_byte != EOF) {
        vision_transformer_free(model);
        return NULL;
    }

    return model;
}

Tensor *vision_transformer_forward_features(
    const VisionTransformer *model,
    const Tensor *image)
{
    Tensor *patches;
    Tensor *tokens;
    Tensor *positioned_tokens;
    Tensor *encoded_tokens;
    Tensor *next_tokens;
    Tensor *features;
    const float *encoded_data;
    float *feature_data;
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

    {
        size_t feature_shape[2] = {1, model->embedding_dimension};
        features = tensor_zeros(feature_shape, 2);
    }
    if (features == NULL) {
        tensor_free(patches);
        tensor_free(tokens);
        tensor_free(positioned_tokens);
        if (encoded_tokens != positioned_tokens) {
            tensor_free(encoded_tokens);
        }
        return NULL;
    }

    encoded_data = tensor_data(encoded_tokens);
    feature_data = tensor_data_mut(features);
    for (i = 0; i < model->embedding_dimension; ++i) {
        feature_data[i] = encoded_data[i];
    }

    tensor_free(patches);
    tensor_free(tokens);
    tensor_free(positioned_tokens);
    if (encoded_tokens != positioned_tokens) {
        tensor_free(encoded_tokens);
    }
    return features;
}

static Tensor *classify_features(
    const VisionTransformer *model,
    const Tensor *features)
{
    const float *feature_data;
    const float *weight_data;
    const float *bias_data;
    float *logit_data;
    size_t shape[2] = {1, model->number_of_classes};
    Tensor *logits = tensor_zeros(shape, 2);
    size_t class_index;
    size_t embedding_index;

    if (logits == NULL) {
        return NULL;
    }

    feature_data = tensor_data(features);
    weight_data = tensor_data(model->classifier_weight);
    bias_data = tensor_data(model->classifier_bias);
    logit_data = tensor_data_mut(logits);

    for (class_index = 0; class_index < model->number_of_classes; ++class_index) {
        float value = bias_data[class_index];
        for (embedding_index = 0;
             embedding_index < model->embedding_dimension;
             ++embedding_index) {
            value += feature_data[embedding_index]
                * weight_data[embedding_index * model->number_of_classes + class_index];
        }
        logit_data[class_index] = value;
    }

    return logits;
}

TensorStatus vision_transformer_train_classifier_step(
    VisionTransformer *model,
    const Tensor *features,
    size_t target_class,
    float learning_rate,
    float *loss)
{
    Tensor *logits;
    Tensor *gradient;
    const float *feature_data;
    const float *gradient_data;
    float *weight_data;
    float *bias_data;
    size_t class_index;
    size_t embedding_index;
    TensorStatus status;

    if (model == NULL || features == NULL || loss == NULL || learning_rate <= 0.0f
        || tensor_ndim(features) != 2
        || tensor_shape(features)[0] != 1
        || tensor_shape(features)[1] != model->embedding_dimension) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    logits = classify_features(model, features);
    if (logits == NULL) {
        return TENSOR_ERROR_ALLOCATION;
    }

    gradient = NULL;
    status = cross_entropy_loss(logits, target_class, loss, &gradient);
    tensor_free(logits);
    if (status != TENSOR_OK) {
        return status;
    }

    feature_data = tensor_data(features);
    gradient_data = tensor_data(gradient);
    weight_data = tensor_data_mut(model->classifier_weight);
    bias_data = tensor_data_mut(model->classifier_bias);

    for (class_index = 0; class_index < model->number_of_classes; ++class_index) {
        for (embedding_index = 0;
             embedding_index < model->embedding_dimension;
             ++embedding_index) {
            weight_data[embedding_index * model->number_of_classes + class_index] -=
                learning_rate * feature_data[embedding_index] * gradient_data[class_index];
        }
        bias_data[class_index] -= learning_rate * gradient_data[class_index];
    }

    tensor_free(gradient);
    return TENSOR_OK;
}

TensorStatus vision_transformer_train_step(
    VisionTransformer *model,
    const Tensor *image,
    size_t target_class,
    float learning_rate,
    float *loss)
{
    Tensor *patches = NULL;
    Tensor *tokens = NULL;
    Tensor *positioned_tokens = NULL;
    Tensor *features = NULL;
    Tensor *logits = NULL;
    Tensor *gradient = NULL;
    Tensor *feature_gradient = NULL;
    Tensor *current_gradient = NULL;
    Tensor *next_gradient = NULL;
    Tensor *patch_gradient = NULL;
    Tensor **block_inputs = NULL;
    Tensor **block_outputs = NULL;
    TensorStatus status = TENSOR_ERROR_ALLOCATION;
    size_t feature_shape[2];
    size_t token_shape[2];
    size_t i;

    if (model == NULL || image == NULL || loss == NULL || learning_rate <= 0.0f) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    block_inputs = (Tensor **)calloc(
        model->number_of_encoder_blocks, sizeof(*block_inputs));
    block_outputs = (Tensor **)calloc(
        model->number_of_encoder_blocks, sizeof(*block_outputs));
    if (block_inputs == NULL || block_outputs == NULL) {
        goto cleanup;
    }

    patches = patch_embedding_forward(model->patch_embedding, image);
    tokens = cls_token_forward(model->cls_token, patches);
    positioned_tokens = position_embedding_forward(model->position_embedding, tokens);
    if (patches == NULL || tokens == NULL || positioned_tokens == NULL) {
        goto cleanup;
    }

    for (i = 0; i < model->number_of_encoder_blocks; ++i) {
        block_inputs[i] = i == 0 ? positioned_tokens : block_outputs[i - 1];
        block_outputs[i] = transformer_encoder_forward(
            model->encoder_blocks[i], block_inputs[i]);
        if (block_outputs[i] == NULL) {
            goto cleanup;
        }
    }

    feature_shape[0] = 1;
    feature_shape[1] = model->embedding_dimension;
    features = tensor_zeros(feature_shape, 2);
    if (features == NULL) {
        goto cleanup;
    }
    for (i = 0; i < model->embedding_dimension; ++i) {
        tensor_data_mut(features)[i] =
            tensor_data(block_outputs[model->number_of_encoder_blocks - 1])[i];
    }

    logits = classify_features(model, features);
    if (logits == NULL) {
        goto cleanup;
    }

    status = cross_entropy_loss(logits, target_class, loss, &gradient);
    if (status != TENSOR_OK) {
        goto cleanup;
    }

    /* Compute d(feature) with the old classifier weights before updating them. */
    feature_gradient = tensor_zeros(feature_shape, 2);
    if (feature_gradient == NULL) {
        status = TENSOR_ERROR_ALLOCATION;
        goto cleanup;
    }
    for (i = 0; i < model->embedding_dimension; ++i) {
        size_t class_index;
        float value = 0.0f;
        for (class_index = 0; class_index < model->number_of_classes; ++class_index) {
            value += tensor_data(gradient)[class_index]
                * tensor_data(model->classifier_weight)[
                    i * model->number_of_classes + class_index];
        }
        tensor_data_mut(feature_gradient)[i] = value;
    }

    for (i = 0; i < model->embedding_dimension; ++i) {
        size_t class_index;
        for (class_index = 0; class_index < model->number_of_classes; ++class_index) {
            tensor_data_mut(model->classifier_weight)[
                i * model->number_of_classes + class_index] -=
                learning_rate * tensor_data(features)[i] * tensor_data(gradient)[class_index];
        }
    }
    for (i = 0; i < model->number_of_classes; ++i) {
        tensor_data_mut(model->classifier_bias)[i] -=
            learning_rate * tensor_data(gradient)[i];
    }

    token_shape[0] = tensor_shape(block_outputs[
        model->number_of_encoder_blocks - 1])[0];
    token_shape[1] = model->embedding_dimension;
    current_gradient = tensor_zeros(token_shape, 2);
    if (current_gradient == NULL) {
        status = TENSOR_ERROR_ALLOCATION;
        goto cleanup;
    }
    for (i = 0; i < model->embedding_dimension; ++i) {
        tensor_data_mut(current_gradient)[i] = tensor_data(feature_gradient)[i];
    }

    for (i = model->number_of_encoder_blocks; i > 0; --i) {
        next_gradient = transformer_encoder_backward(
            model->encoder_blocks[i - 1],
            block_inputs[i - 1],
            current_gradient,
            learning_rate);
        tensor_free(current_gradient);
        current_gradient = next_gradient;
        next_gradient = NULL;
        if (current_gradient == NULL) {
            status = TENSOR_ERROR_ALLOCATION;
            goto cleanup;
        }
    }

    next_gradient = position_embedding_backward(
        model->position_embedding, current_gradient, learning_rate);
    tensor_free(current_gradient);
    current_gradient = next_gradient;
    next_gradient = NULL;
    if (current_gradient == NULL) {
        status = TENSOR_ERROR_ALLOCATION;
        goto cleanup;
    }

    patch_gradient = cls_token_backward(model->cls_token, current_gradient, learning_rate);
    tensor_free(current_gradient);
    current_gradient = NULL;
    if (patch_gradient == NULL) {
        status = TENSOR_ERROR_ALLOCATION;
        goto cleanup;
    }

    status = patch_embedding_backward(
        model->patch_embedding, image, patch_gradient, learning_rate);

cleanup:
    tensor_free(patches);
    tensor_free(tokens);
    tensor_free(positioned_tokens);
    tensor_free(features);
    tensor_free(logits);
    tensor_free(gradient);
    tensor_free(feature_gradient);
    tensor_free(current_gradient);
    tensor_free(next_gradient);
    tensor_free(patch_gradient);
    if (block_outputs != NULL) {
        for (i = 0; i < model->number_of_encoder_blocks; ++i) {
            tensor_free(block_outputs[i]);
        }
    }
    free(block_inputs);
    free(block_outputs);
    return status;
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
