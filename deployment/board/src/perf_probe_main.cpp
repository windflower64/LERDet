#include "stagdet_detector.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

namespace {
constexpr int kWarmupRuns = 10;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.rknn> <visible.jpg> <infrared.jpg> <report.txt>\n";
        return 1;
    }
    try {
        const cv::Mat visible = cv::imread(argv[2], cv::IMREAD_COLOR);
        const cv::Mat infrared = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
        if (visible.empty() || infrared.empty()) {
            throw std::runtime_error("Failed to read the visible/infrared pair");
        }
        StagDetDetector detector(argv[1], 0.25F, 0.50F, true);
        for (int index = 0; index < kWarmupRuns; ++index) {
            detector.detect(visible, infrared);
        }
        const StagDetPerfReport report = detector.collect_perf_report(visible, infrared);

        std::ofstream output(argv[4], std::ios::binary);
        if (!output) {
            throw std::runtime_error("Failed to open report path");
        }
        output << "Model: " << argv[1] << "\n";
        output << "Visible: " << argv[2] << "\n";
        output << "Infrared: " << argv[3] << "\n";
        output << "RKNN API version: " << report.sdk_api_version << "\n";
        output << "RKNN driver version: " << report.sdk_driver_version << "\n";
        output << "Warmup runs: " << kWarmupRuns << "\n";
        output << "RKNN run duration: " << report.run_duration_us << " us\n";
        output << "\n=== RKNN PERF DETAIL ===\n" << report.perf_detail;
        std::cout << "Saved RKNN performance report: " << argv[4] << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "STAGDet performance probe failed: " << error.what() << "\n";
        return 2;
    }
}
