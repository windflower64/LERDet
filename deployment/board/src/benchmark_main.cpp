#include "stagdet_detector.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.rknn> <visible.jpg> <infrared.jpg> <warmup> <runs>\n";
        return 1;
    }
    try {
        const int warmup = std::stoi(argv[4]);
        const int runs = std::stoi(argv[5]);
        if (warmup < 0 || runs < 1) {
            throw std::invalid_argument("warmup must be >= 0 and runs must be >= 1");
        }
        const cv::Mat visible = cv::imread(argv[2], cv::IMREAD_COLOR);
        const cv::Mat infrared = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
        if (visible.empty() || infrared.empty()) {
            throw std::runtime_error("Failed to read the visible/infrared pair");
        }

        StagDetDetector detector(argv[1]);
        for (int index = 0; index < warmup; ++index) {
            detector.detect(visible, infrared);
        }

        StagDetTimings totals;
        std::size_t last_detection_count = 0;
        for (int index = 0; index < runs; ++index) {
            StagDetTimings current;
            last_detection_count = detector.detect(visible, infrared, &current).size();
            totals.preprocess_ms += current.preprocess_ms;
            totals.inference_ms += current.inference_ms;
            totals.postprocess_ms += current.postprocess_ms;
        }
        totals.preprocess_ms /= runs;
        totals.inference_ms /= runs;
        totals.postprocess_ms /= runs;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Warmup runs: " << warmup << "\n";
        std::cout << "Measured runs: " << runs << "\n";
        std::cout << "Last detection count: " << last_detection_count << "\n";
        std::cout << "Average preprocess: " << totals.preprocess_ms << " ms\n";
        std::cout << "Average RKNN inference: " << totals.inference_ms << " ms\n";
        std::cout << "Average postprocess: " << totals.postprocess_ms << " ms\n";
        std::cout << "Average compute: " << totals.total_ms() << " ms\n";
        std::cout << "Compute FPS: " << 1000.0 / totals.total_ms() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "STAGDet benchmark failed: " << error.what() << "\n";
        return 2;
    }
}
