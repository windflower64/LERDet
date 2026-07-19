#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <rknn_api.h>

struct StagDetDetection {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float score = 0.0F;
    int class_id = 0;
};

struct StagDetTimings {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;

    double total_ms() const {
        return preprocess_ms + inference_ms + postprocess_ms;
    }
};

struct StagDetPerfReport {
    std::string sdk_api_version;
    std::string sdk_driver_version;
    std::int64_t run_duration_us = 0;
    std::string perf_detail;
};

class StagDetDetector {
public:
    explicit StagDetDetector(
        const std::string& model_path,
        float confidence_threshold = 0.25F,
        float iou_threshold = 0.50F,
        bool collect_perf = false);
    ~StagDetDetector();

    StagDetDetector(const StagDetDetector&) = delete;
    StagDetDetector& operator=(const StagDetDetector&) = delete;

    std::vector<StagDetDetection> detect(
        const cv::Mat& visible_bgr,
        const cv::Mat& infrared_gray,
        StagDetTimings* timings = nullptr);
    StagDetPerfReport collect_perf_report(
        const cv::Mat& visible_bgr,
        const cv::Mat& infrared_gray);

private:
    struct OutputLevel {
        std::uint32_t box_index;
        std::uint32_t score_index;
        int stride;
        int prediction_count;
    };

    void load_model(const std::string& model_path);
    void query_tensor_contract();
    std::vector<StagDetDetection> detect_impl(
        const cv::Mat& visible_bgr,
        const cv::Mat& infrared_gray,
        StagDetTimings* timings,
        StagDetPerfReport* perf_report);
    void release() noexcept;

    rknn_context context_ = 0;
    rknn_input_output_num io_count_{};
    std::vector<rknn_tensor_attr> input_attributes_;
    std::vector<rknn_tensor_attr> output_attributes_;
    std::vector<OutputLevel> output_levels_;
    std::uint32_t visible_input_index_ = 0;
    std::uint32_t infrared_input_index_ = 1;
    std::uint32_t box_output_index_ = 0;
    std::uint32_t scores_output_index_ = 1;
    int input_width_ = 0;
    int input_height_ = 0;
    int prediction_count_ = 0;
    int box_output_channels_ = 0;
    bool dual_input_ = false;
    bool raw_dfl_output_ = false;
    bool per_scale_output_ = false;
    float confidence_threshold_ = 0.25F;
    float iou_threshold_ = 0.50F;
    bool collect_perf_ = false;
};
