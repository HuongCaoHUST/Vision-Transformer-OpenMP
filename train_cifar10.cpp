#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
<<<<<<< HEAD

#include <omp.h>
=======
>>>>>>> 4f94e60 (add confusion matrix)

#include "cifar10_dataset.hpp"
#include "metrics_utils.hpp"
#include "tensor.h"
#include "vision_transformer.h"

using namespace std;

static size_t predicted_class(const Tensor *logits, size_t number_of_classes)
{
    const float *logit_data = tensor_data(logits);
    size_t prediction = 0;

    for (size_t class_index = 1; class_index < number_of_classes; ++class_index) {
        if (logit_data[class_index] > logit_data[prediction]) {
            prediction = class_index;
        }
    }
    return prediction;
}

static float macro_precision(
    const vector<size_t> &true_positives,
    const vector<size_t> &predicted_positives,
    const vector<size_t> &actual_positives)
{
    float sum = 0.0f;
    size_t observed_classes = 0;

    for (size_t class_index = 0; class_index < true_positives.size(); ++class_index) {
        if (actual_positives[class_index] == 0
            && predicted_positives[class_index] == 0) {
            continue;
        }
        const size_t denominator = predicted_positives[class_index];
        sum += denominator == 0
            ? 0.0f
            : static_cast<float>(true_positives[class_index])
                / static_cast<float>(denominator);
        ++observed_classes;
    }

    return observed_classes == 0 ? 0.0f : sum / static_cast<float>(observed_classes);
}

static float macro_recall(
    const vector<size_t> &true_positives,
    const vector<size_t> &actual_positives)
{
    float sum = 0.0f;
    size_t observed_classes = 0;

    for (size_t class_index = 0; class_index < true_positives.size(); ++class_index) {
        if (actual_positives[class_index] == 0) {
            continue;
        }
        sum += static_cast<float>(true_positives[class_index])
            / static_cast<float>(actual_positives[class_index]);
        ++observed_classes;
    }

    return observed_classes == 0 ? 0.0f : sum / static_cast<float>(observed_classes);
}

int main(int argc, char **argv)
{
    const size_t epochs = argc > 1 ? strtoul(argv[1], nullptr, 10) : 1;
    const size_t max_samples = argc > 2 ? strtoul(argv[2], nullptr, 10) : 100;
    Cifar10Dataset dataset("datasets/cifar10", "train", max_samples);
    VisionTransformer *model;
    float learning_rate = 0.01f;

    if (dataset.size() == 0) {
        cerr << "The CIFAR-10 training dataset is empty\n";
        return EXIT_FAILURE;
    }

    model = vision_transformer_create(32, 32, 8, 3, 64, 4, 256, 2, 10);
    if (model == nullptr) {
        cerr << "Unable to create Vision Transformer\n";
        return EXIT_FAILURE;
    }

    cout << "Training samples: " << dataset.size() << '\n';
    cout << "Epochs: " << epochs << '\n';
    cout << "OpenMP threads: " << omp_get_max_threads() << '\n';
    cout << "Trainable parameters: classification head only\n";

    const double training_start = omp_get_wtime();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        const double epoch_start = omp_get_wtime();
        float total_loss = 0.0f;
        size_t successful_samples = 0;
<<<<<<< HEAD
        const size_t sample_count = dataset.size();
        vector<Tensor *> features(sample_count, nullptr);
        vector<size_t> labels(sample_count, 0);
        vector<unsigned char> valid_samples(sample_count, 0);
=======
        size_t correct_predictions = 0;
        const size_t number_of_classes = vision_transformer_number_of_classes(model);
        vector<size_t> true_positives(number_of_classes, 0);
        vector<size_t> predicted_positives(number_of_classes, 0);
        vector<size_t> actual_positives(number_of_classes, 0);
        vector<vector<size_t>> confusion_matrix(
            number_of_classes, vector<size_t>(number_of_classes, 0));
>>>>>>> 4f94e60 (add confusion matrix)

        dataset.shuffle(static_cast<uint32_t>(epoch + 1));
        /* Feature extraction is read-only and can safely run in parallel. */
        #pragma omp parallel for schedule(dynamic)
        for (long long index = 0; index < static_cast<long long>(sample_count); ++index) {
            Tensor *image = nullptr;
<<<<<<< HEAD
=======
            Tensor *features = nullptr;
            Tensor *logits = nullptr;
>>>>>>> 4f94e60 (add confusion matrix)
            size_t label = 0;

            if (!dataset.get(static_cast<size_t>(index), &image, &label)) {
                continue;
            }

<<<<<<< HEAD
            features[static_cast<size_t>(index)] =
                vision_transformer_forward_features(model, image);
            labels[static_cast<size_t>(index)] = label;
            valid_samples[static_cast<size_t>(index)] = static_cast<unsigned char>(
                features[static_cast<size_t>(index)] != nullptr);
            tensor_free(image);
        }

        /* Classifier updates remain sequential because they modify shared weights. */
        for (size_t index = 0; index < sample_count; ++index) {
            if (!valid_samples[index]) {
                continue;
            }

            float loss = 0.0f;
            if (vision_transformer_train_classifier_step(
                    model, features[index], labels[index], learning_rate, &loss) != TENSOR_OK) {
                tensor_free(features[index]);
                features[index] = nullptr;
=======
            logits = vision_transformer_forward(model, image);
            features = vision_transformer_forward_features(model, image);
            if (logits == nullptr || features == nullptr
                || vision_transformer_train_classifier_step(
                       model, features, label, learning_rate, &loss) != TENSOR_OK) {
                cerr << "Skipping failed training sample " << index << '\n';
                tensor_free(image);
                tensor_free(logits);
                tensor_free(features);
>>>>>>> 4f94e60 (add confusion matrix)
                continue;
            }

            const size_t prediction = predicted_class(logits, number_of_classes);
            ++actual_positives[label];
            ++predicted_positives[prediction];
            ++confusion_matrix[label][prediction];
            if (prediction == label) {
                ++correct_predictions;
                ++true_positives[label];
            }

            total_loss += loss;
            ++successful_samples;
<<<<<<< HEAD
            tensor_free(features[index]);
            features[index] = nullptr;
=======
            tensor_free(image);
            tensor_free(logits);
            tensor_free(features);
>>>>>>> 4f94e60 (add confusion matrix)
        }

        const float average_loss = successful_samples == 0
            ? 0.0f : total_loss / static_cast<float>(successful_samples);
<<<<<<< HEAD
        const double epoch_seconds = omp_get_wtime() - epoch_start;
=======
        const float accuracy = successful_samples == 0
            ? 0.0f
            : static_cast<float>(correct_predictions)
                / static_cast<float>(successful_samples);
        const float precision = macro_precision(
            true_positives, predicted_positives, actual_positives);
        const float recall = macro_recall(true_positives, actual_positives);
        const string confusion_matrix_path = "runs/confusion_matrix_epoch_"
            + to_string(epoch + 1) + ".png";
        if (!save_confusion_matrix(confusion_matrix, confusion_matrix_path)) {
            cerr << "Unable to save confusion matrix to "
                 << confusion_matrix_path << '\n';
        }
>>>>>>> 4f94e60 (add confusion matrix)
        cout << "Epoch " << (epoch + 1)
             << "/" << epochs
             << " - average loss: "
             << fixed << setprecision(6) << average_loss
<<<<<<< HEAD
             << " - time: " << epoch_seconds << " s"
             << " - samples/s: "
             << (epoch_seconds > 0.0
                     ? static_cast<double>(successful_samples) / epoch_seconds
                     : 0.0)
             << '\n';
=======
             << " - P: " << precision
             << " - R: " << recall
             << " - Accuracy: " << accuracy << '\n';
>>>>>>> 4f94e60 (add confusion matrix)
    }

    const double training_seconds = omp_get_wtime() - training_start;
    cout << "Total training time: " << fixed << setprecision(6)
         << training_seconds << " s\n";
    vision_transformer_free(model);
    return EXIT_SUCCESS;
}
