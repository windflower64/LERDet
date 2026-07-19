#include "stagdet_detector.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace {

struct Geometry {
    float scale_x = 1.0F;
    float scale_y = 1.0F;
    float pad_left = 0.0F;
    float pad_top = 0.0F;
    int source_width = 320;
    int source_height = 320;
};

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (!field.empty() && field.back() == '\r') {
            field.pop_back();
        }
        fields.push_back(field);
    }
    return fields;
}

Geometry load_geometry(const std::string& manifest_path, const std::string& name) {
    std::ifstream input(manifest_path);
    if (!input) {
        throw std::runtime_error("Failed to open manifest: " + manifest_path);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Manifest is empty: " + manifest_path);
    }
    const std::vector<std::string> header = split_csv(line);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index) {
        columns[header[index]] = index;
    }
    const std::vector<std::string> required = {
        "name", "visible_source_width", "visible_source_height", "scale_x", "scale_y", "pad_left", "pad_top"};
    for (const std::string& column : required) {
        if (columns.count(column) == 0) {
            throw std::runtime_error("Missing manifest column: " + column);
        }
    }
    while (std::getline(input, line)) {
        const std::vector<std::string> fields = split_csv(line);
        if (fields.size() != header.size() || fields[columns["name"]] != name) {
            continue;
        }
        return {
            std::stof(fields[columns["scale_x"]]),
            std::stof(fields[columns["scale_y"]]),
            std::stof(fields[columns["pad_left"]]),
            std::stof(fields[columns["pad_top"]]),
            std::stoi(fields[columns["visible_source_width"]]),
            std::stoi(fields[columns["visible_source_height"]]),
        };
    }
    throw std::runtime_error("No manifest row for pair: " + name);
}

StagDetDetection restore(const StagDetDetection& detection, const Geometry& geometry) {
    StagDetDetection restored = detection;
    const float max_x = static_cast<float>(geometry.source_width - 1);
    const float max_y = static_cast<float>(geometry.source_height - 1);
    restored.x1 = std::clamp((detection.x1 - geometry.pad_left) / geometry.scale_x, 0.0F, max_x);
    restored.x2 = std::clamp((detection.x2 - geometry.pad_left) / geometry.scale_x, 0.0F, max_x);
    restored.y1 = std::clamp((detection.y1 - geometry.pad_top) / geometry.scale_y, 0.0F, max_y);
    restored.y2 = std::clamp((detection.y2 - geometry.pad_top) / geometry.scale_y, 0.0F, max_y);
    return restored;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.rknn> <visible.jpg> <infrared.jpg> <manifest.csv>\n";
        return 1;
    }
    try {
        const cv::Mat visible = cv::imread(argv[2], cv::IMREAD_COLOR);
        const cv::Mat infrared = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
        if (visible.empty() || infrared.empty()) {
            throw std::runtime_error("Failed to read the visible/infrared pair");
        }
        const std::string name = std::filesystem::path(argv[2]).stem().string();
        const Geometry geometry = load_geometry(argv[4], name);

        StagDetDetector detector(argv[1]);
        StagDetTimings timings;
        const std::vector<StagDetDetection> detections = detector.detect(visible, infrared, &timings);
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Pair: " << name << "\n";
        std::cout << "Detections: " << detections.size() << "\n";
        for (std::size_t index = 0; index < detections.size(); ++index) {
            const StagDetDetection restored = restore(detections[index], geometry);
            std::cout << "[" << index << "] class=uav score=" << restored.score
                      << " xyxy=(" << restored.x1 << ", " << restored.y1
                      << ", " << restored.x2 << ", " << restored.y2 << ")\n";
        }
        std::cout << "Preprocess: " << timings.preprocess_ms << " ms\n";
        std::cout << "Inference: " << timings.inference_ms << " ms\n";
        std::cout << "Postprocess: " << timings.postprocess_ms << " ms\n";
        std::cout << "Total compute: " << timings.total_ms() << " ms\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "STAGDet image inference failed: " << error.what() << "\n";
        return 2;
    }
}
