#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <omp.h>

#include "cifar10_dataset.hpp"
#include "cross_entropy.h"
#include "tensor.h"
#include "vision_transformer.h"

using namespace std;

struct EvaluationResult {
    float total_loss = 0.0f;
    size_t successful_samples = 0;
    size_t failed_samples = 0;
    size_t correct_samples = 0;
};

static void evaluate_dataset(
    const Cifar10Dataset &dataset,
    const VisionTransformer *model,
    EvaluationResult *result)
{
    if (result == nullptr) {
        return;
    }

    *result = EvaluationResult{};
    for (size_t index = 0; index < dataset.size(); ++index) {
        Tensor *image = nullptr;
        Tensor *logits = nullptr;
        Tensor *gradient = nullptr;
        size_t label = 0;
        float loss = 0.0f;

        if (!dataset.get(index, &image, &label)) {
            ++result->failed_samples;
            continue;
        }

        logits = vision_transformer_forward(model, image);
        if (logits == nullptr
            || cross_entropy_loss(logits, label, &loss, &gradient) != TENSOR_OK) {
            ++result->failed_samples;
            tensor_free(image);
            tensor_free(logits);
            tensor_free(gradient);
            continue;
        }

        size_t predicted_class = 0;
        const float *logit_data = tensor_data(logits);
        for (size_t class_index = 1;
             class_index < tensor_shape(logits)[1];
             ++class_index) {
            if (logit_data[class_index] > logit_data[predicted_class]) {
                predicted_class = class_index;
            }
        }

        result->total_loss += loss;
        ++result->successful_samples;
        if (predicted_class == label) {
            ++result->correct_samples;
        }

        tensor_free(image);
        tensor_free(logits);
        tensor_free(gradient);
    }
}

static float average_loss(const EvaluationResult &result)
{
    return result.successful_samples == 0
        ? 0.0f
        : result.total_loss / static_cast<float>(result.successful_samples);
}

static float accuracy(const EvaluationResult &result)
{
    return result.successful_samples == 0
        ? 0.0f
        : 100.0f * static_cast<float>(result.correct_samples)
            / static_cast<float>(result.successful_samples);
}

int main(int argc, char **argv)
{
    const size_t epochs = argc > 1 ? strtoul(argv[1], nullptr, 10) : 1;
    const size_t max_samples = argc > 2 ? strtoul(argv[2], nullptr, 10) : 100;
    const float learning_rate = 0.001f;
    Cifar10Dataset train_dataset("datasets/cifar10", "train", max_samples);
    Cifar10Dataset test_dataset("datasets/cifar10", "test", max_samples);
    VisionTransformer *model;

    if (epochs == 0) {
        cerr << "The number of epochs must be greater than zero\n";
        return EXIT_FAILURE;
    }

    if (train_dataset.size() == 0) {
        cerr << "The CIFAR-10 training dataset is empty\n";
        return EXIT_FAILURE;
    }

    model = vision_transformer_create(32, 32, 8, 3, 64, 4, 256, 2, 10);
    if (model == nullptr) {
        cerr << "Unable to create Vision Transformer\n";
        return EXIT_FAILURE;
    }

    cout << "Training samples: " << train_dataset.size() << '\n';
    cout << "Evaluation samples: " << test_dataset.size() << '\n';
    cout << "Epochs: " << epochs << '\n';
    cout << "OpenMP max threads: " << omp_get_max_threads() << '\n';
    cout << "Trainable parameters: full ViT (online SGD)\n";
    cout << "Learning rate: " << learning_rate << '\n';

    const double training_start = omp_get_wtime();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        const double epoch_start = omp_get_wtime();
        size_t successful_updates = 0;
        size_t failed_updates = 0;

        train_dataset.shuffle(static_cast<uint32_t>(epoch + 1));
        for (size_t index = 0; index < train_dataset.size(); ++index) {
            Tensor *image = nullptr;
            size_t label = 0;
            float loss = 0.0f;

            if (!train_dataset.get(index, &image, &label)
                || vision_transformer_train_step(
                       model, image, label, learning_rate, &loss) != TENSOR_OK) {
                ++failed_updates;
                tensor_free(image);
                continue;
            }

            ++successful_updates;
            tensor_free(image);
        }
        const double update_seconds = omp_get_wtime() - epoch_start;

        EvaluationResult train_result;
        EvaluationResult test_result;
        evaluate_dataset(train_dataset, model, &train_result);
        evaluate_dataset(test_dataset, model, &test_result);

        const double epoch_seconds = omp_get_wtime() - epoch_start;
        cout << "Epoch " << (epoch + 1)
             << "/" << epochs
             << " - train loss: " << fixed << setprecision(6)
             << average_loss(train_result)
             << " - train accuracy: " << setprecision(2)
             << accuracy(train_result) << "%"
             << " - test loss: " << setprecision(6)
             << average_loss(test_result)
             << " - test accuracy: " << setprecision(2)
             << accuracy(test_result) << "%"
             << " - updates: " << successful_updates
             << " - failed: " << failed_updates
             << " - train eval failed: " << train_result.failed_samples
             << " - test eval failed: " << test_result.failed_samples
             << " - time: " << setprecision(6) << epoch_seconds << " s"
             << " - train updates/s: "
             << (update_seconds > 0.0
                     ? static_cast<double>(successful_updates) / update_seconds
                     : 0.0)
             << '\n';
    }

    const double training_seconds = omp_get_wtime() - training_start;
    cout << "Total training time: " << fixed << setprecision(6)
         << training_seconds << " s\n";
    vision_transformer_free(model);
    return EXIT_SUCCESS;
}
