#ifndef METRICS_UTILS_HPP
#define METRICS_UTILS_HPP

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

/*
 * Save a confusion matrix as a color-coded image.
 * Rows are actual classes and columns are predicted classes.
 */
inline bool save_confusion_matrix(
    const std::vector<std::vector<size_t>> &confusion_matrix,
    const std::string &output_path)
{
    const size_t number_of_classes = confusion_matrix.size();
    if (number_of_classes == 0) {
        return false;
    }
    for (const auto &row : confusion_matrix) {
        if (row.size() != number_of_classes) {
            return false;
        }
    }

    const int cell_size = 64;
    const int margin_left = 100;
    const int margin_top = 80;
    const int margin_right = 20;
    const int margin_bottom = 80;
    const int matrix_size = static_cast<int>(number_of_classes) * cell_size;
    cv::Mat image(
        margin_top + matrix_size + margin_bottom,
        margin_left + matrix_size + margin_right,
        CV_8UC3,
        cv::Scalar(255, 255, 255));

    size_t maximum_count = 0;
    for (const auto &row : confusion_matrix) {
        for (const size_t count : row) {
            maximum_count = std::max(maximum_count, count);
        }
    }

    cv::Mat heatmap(matrix_size, matrix_size, CV_8UC1, cv::Scalar(0));
    for (size_t actual = 0; actual < number_of_classes; ++actual) {
        for (size_t predicted = 0; predicted < number_of_classes; ++predicted) {
            const double normalized = maximum_count == 0
                ? 0.0
                : static_cast<double>(confusion_matrix[actual][predicted])
                    / static_cast<double>(maximum_count);
            const uchar intensity = static_cast<uchar>(normalized * 255.0);
            cv::rectangle(
                heatmap,
                cv::Rect(
                    static_cast<int>(predicted) * cell_size,
                    static_cast<int>(actual) * cell_size,
                    cell_size,
                    cell_size),
                cv::Scalar(intensity),
                cv::FILLED);
        }
    }

    cv::Mat colored_heatmap;
    cv::applyColorMap(heatmap, colored_heatmap, cv::COLORMAP_JET);
    colored_heatmap.copyTo(
        image(cv::Rect(margin_left, margin_top, matrix_size, matrix_size)));

    const double font_scale = number_of_classes > 12 ? 0.35 : 0.45;
    for (size_t actual = 0; actual < number_of_classes; ++actual) {
        for (size_t predicted = 0; predicted < number_of_classes; ++predicted) {
            const std::string text = std::to_string(
                confusion_matrix[actual][predicted]);
            const cv::Scalar text_color =
                confusion_matrix[actual][predicted] * 2 > maximum_count
                    ? cv::Scalar(255, 255, 255)
                    : cv::Scalar(0, 0, 0);
            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(
                text, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
            const int x = margin_left + static_cast<int>(predicted) * cell_size
                + (cell_size - text_size.width) / 2;
            const int y = margin_top + static_cast<int>(actual) * cell_size
                + (cell_size + text_size.height) / 2;
            cv::putText(
                image,
                text,
                cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv::LINE_AA);
        }
    }

    cv::putText(
        image,
        "Confusion Matrix",
        cv::Point(margin_left, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 0, 0),
        2,
        cv::LINE_AA);
    cv::putText(
        image,
        "Predicted class",
        cv::Point(margin_left + matrix_size / 2 - 80, image.rows - 25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 0, 0),
        1,
        cv::LINE_AA);
    cv::putText(
        image,
        "Actual",
        cv::Point(15, margin_top + matrix_size / 2),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 0, 0),
        1,
        cv::LINE_AA);

    for (size_t class_index = 0; class_index < number_of_classes; ++class_index) {
        const std::string label = std::to_string(class_index);
        const int x = margin_left + static_cast<int>(class_index) * cell_size
            + cell_size / 2 - 5;
        const int y = margin_top - 12;
        cv::putText(image, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX,
                    0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(image, label, cv::Point(75, margin_top
                    + static_cast<int>(class_index) * cell_size + cell_size / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1,
                    cv::LINE_AA);
    }

    const std::filesystem::path path(output_path);
    if (!path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            return false;
        }
    }
    return cv::imwrite(output_path, image);
}

#endif /* METRICS_UTILS_HPP */
