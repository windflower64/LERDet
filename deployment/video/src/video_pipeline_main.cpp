#include "stagdet_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

constexpr int kInputWidth = 320;
constexpr int kContentHeight = 180;
constexpr int kInputHeight = 192;
constexpr int kPadTop = 6;
constexpr int kPadBottom = 6;
constexpr int kPadValue = 114;

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct StageTotals {
    double decode_ms = 0.0;
    double video_preprocess_ms = 0.0;
    double merge_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double output_ms = 0.0;
};

struct SourceGeometry {
    int width;
    int height;
    float scale_x;
    float scale_y;
};

std::string json_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' || character == '"') {
            result.push_back('\\');
        }
        result.push_back(character);
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

cv::Mat infrared_to_gray(const cv::Mat& frame) {
    if (frame.channels() == 1) {
        return frame;
    }
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported infrared channel count: " + std::to_string(frame.channels()));
    }
    return gray;
}

void prepare_pair(
    const cv::Mat& visible_source,
    const cv::Mat& infrared_source,
    cv::Mat& visible_prepared,
    cv::Mat& infrared_prepared) {
    if (visible_source.empty() || infrared_source.empty()) {
        throw std::runtime_error("Decoded an empty video frame");
    }
    if (visible_source.channels() != 3) {
        throw std::runtime_error("Visible decoder must return BGR frames");
    }

    const cv::Mat infrared_gray = infrared_to_gray(infrared_source);
    cv::Mat infrared_aligned;
    cv::resize(
        infrared_gray,
        infrared_aligned,
        visible_source.size(),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    cv::Mat visible_resized;
    cv::Mat infrared_resized;
    cv::resize(
        visible_source,
        visible_resized,
        cv::Size(kInputWidth, kContentHeight),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    cv::resize(
        infrared_aligned,
        infrared_resized,
        cv::Size(kInputWidth, kContentHeight),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    cv::copyMakeBorder(
        visible_resized,
        visible_prepared,
        kPadTop,
        kPadBottom,
        0,
        0,
        cv::BORDER_CONSTANT,
        cv::Scalar(kPadValue, kPadValue, kPadValue));
    cv::copyMakeBorder(
        infrared_resized,
        infrared_prepared,
        kPadTop,
        kPadBottom,
        0,
        0,
        cv::BORDER_CONSTANT,
        cv::Scalar(kPadValue));
}

StagDetDetection restore(const StagDetDetection& detection, const SourceGeometry& geometry) {
    StagDetDetection restored = detection;
    const float max_x = static_cast<float>(geometry.width - 1);
    const float max_y = static_cast<float>(geometry.height - 1);
    restored.x1 = std::clamp(detection.x1 / geometry.scale_x, 0.0F, max_x);
    restored.x2 = std::clamp(detection.x2 / geometry.scale_x, 0.0F, max_x);
    restored.y1 = std::clamp((detection.y1 - kPadTop) / geometry.scale_y, 0.0F, max_y);
    restored.y2 = std::clamp((detection.y2 - kPadTop) / geometry.scale_y, 0.0F, max_y);
    return restored;
}

void write_summary(
    const std::filesystem::path& path,
    const std::string& sequence_id,
    const std::string& visible_path,
    const std::string& infrared_path,
    std::size_t processed_pairs,
    std::size_t detections,
    int visible_width,
    int visible_height,
    int infrared_width,
    int infrared_height,
    double visible_fps,
    double infrared_fps,
    double visible_frame_count,
    double infrared_frame_count,
    double confidence,
    double nms_iou,
    double wall_ms,
    const StageTotals& totals,
    const std::vector<double>& pts_differences,
    bool visible_read_failed,
    bool infrared_read_failed) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to create summary: " + path.string());
    }
    const double count = static_cast<double>(processed_pairs);
    const auto average = [count](double value) { return count > 0.0 ? value / count : 0.0; };
    const double pts_average = pts_differences.empty()
        ? 0.0
        : std::accumulate(pts_differences.begin(), pts_differences.end(), 0.0)
            / static_cast<double>(pts_differences.size());
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"sequence_id\": \"" << json_escape(sequence_id) << "\",\n"
           << "  \"visible_video\": \"" << json_escape(visible_path) << "\",\n"
           << "  \"infrared_video\": \"" << json_escape(infrared_path) << "\",\n"
           << "  \"decode_backend\": \"opencv_video_capture\",\n"
           << "  \"alignment_mode\": \"stretch_ir_to_visible_then_resize\",\n"
           << "  \"input_shape\": [1, 192, 320, 4],\n"
           << "  \"channel_order\": \"B,G,R,T\",\n"
           << "  \"confidence\": " << confidence << ",\n"
           << "  \"nms_iou\": " << nms_iou << ",\n"
           << "  \"processed_pairs\": " << processed_pairs << ",\n"
           << "  \"detections\": " << detections << ",\n"
           << "  \"visible_metadata\": {\"width\": " << visible_width
           << ", \"height\": " << visible_height << ", \"fps\": " << visible_fps
           << ", \"frame_count\": " << visible_frame_count << "},\n"
           << "  \"infrared_metadata\": {\"width\": " << infrared_width
           << ", \"height\": " << infrared_height << ", \"fps\": " << infrared_fps
           << ", \"frame_count\": " << infrared_frame_count << "},\n"
           << "  \"visible_read_failed_at_end\": " << (visible_read_failed ? "true" : "false") << ",\n"
           << "  \"infrared_read_failed_at_end\": " << (infrared_read_failed ? "true" : "false") << ",\n"
           << "  \"wall_ms\": " << wall_ms << ",\n"
           << "  \"end_to_end_fps\": " << (wall_ms > 0.0 ? count * 1000.0 / wall_ms : 0.0) << ",\n"
           << "  \"average_ms\": {\n"
           << "    \"decode_pair\": " << average(totals.decode_ms) << ",\n"
           << "    \"video_preprocess\": " << average(totals.video_preprocess_ms) << ",\n"
           << "    \"four_channel_merge\": " << average(totals.merge_ms) << ",\n"
           << "    \"rknn_inference\": " << average(totals.inference_ms) << ",\n"
           << "    \"postprocess\": " << average(totals.postprocess_ms) << ",\n"
           << "    \"csv_output\": " << average(totals.output_ms) << "\n"
           << "  },\n"
           << "  \"pts_difference_ms\": {\"mean\": " << pts_average
           << ", \"p50\": " << percentile(pts_differences, 0.50)
           << ", \"p95\": " << percentile(pts_differences, 0.95)
           << ", \"max\": "
           << (pts_differences.empty() ? 0.0 : *std::max_element(pts_differences.begin(), pts_differences.end()))
           << "}\n"
           << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.rknn> <visible.mp4> <infrared.mp4> <output-dir>"
                  << " [confidence=0.25] [nms-iou=0.70] [max-frames=0]\n";
        return 1;
    }
    try {
        const std::string model_path = argv[1];
        const std::string visible_path = argv[2];
        const std::string infrared_path = argv[3];
        const std::filesystem::path output_dir = argv[4];
        const float confidence = argc >= 6 ? std::stof(argv[5]) : 0.25F;
        const float nms_iou = argc >= 7 ? std::stof(argv[6]) : 0.70F;
        const int max_frames = argc >= 8 ? std::stoi(argv[7]) : 0;
        if (!(confidence > 0.0F && confidence < 1.0F)
            || !(nms_iou > 0.0F && nms_iou < 1.0F)
            || max_frames < 0) {
            throw std::invalid_argument("Invalid confidence, NMS IoU, or max frame count");
        }

        std::filesystem::create_directories(output_dir);
        const std::string sequence_id = std::filesystem::path(visible_path).parent_path().filename().string();
        StagDetDetector detector(model_path, confidence, nms_iou);

        const Clock::time_point wall_start = Clock::now();
        cv::VideoCapture visible_capture(visible_path);
        cv::VideoCapture infrared_capture(infrared_path);
        if (!visible_capture.isOpened() || !infrared_capture.isOpened()) {
            throw std::runtime_error("OpenCV failed to open one or both videos");
        }

        const int visible_width = static_cast<int>(visible_capture.get(cv::CAP_PROP_FRAME_WIDTH));
        const int visible_height = static_cast<int>(visible_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        const int infrared_width = static_cast<int>(infrared_capture.get(cv::CAP_PROP_FRAME_WIDTH));
        const int infrared_height = static_cast<int>(infrared_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double visible_fps = visible_capture.get(cv::CAP_PROP_FPS);
        const double infrared_fps = infrared_capture.get(cv::CAP_PROP_FPS);
        const double visible_frame_count = visible_capture.get(cv::CAP_PROP_FRAME_COUNT);
        const double infrared_frame_count = infrared_capture.get(cv::CAP_PROP_FRAME_COUNT);
        if (visible_width <= 0 || visible_height <= 0 || infrared_width <= 0 || infrared_height <= 0) {
            throw std::runtime_error("Video metadata contains invalid dimensions");
        }
        if (std::abs(visible_fps - infrared_fps) > 1e-3) {
            std::cerr << "Warning: visible/infrared FPS differ: "
                      << visible_fps << " vs " << infrared_fps << "\n";
        }
        if (std::abs(visible_frame_count - infrared_frame_count) > 0.5) {
            std::cerr << "Warning: visible/infrared frame counts differ: "
                      << visible_frame_count << " vs " << infrared_frame_count << "\n";
        }

        std::ofstream detections_csv(output_dir / "detections.csv");
        std::ofstream timings_csv(output_dir / "frame_timings.csv");
        if (!detections_csv || !timings_csv) {
            throw std::runtime_error("Failed to create output CSV files");
        }
        detections_csv << "sequence_id,frame_index,visible_pts_ms,infrared_pts_ms,score,x1,y1,x2,y2\n";
        timings_csv << "sequence_id,frame_index,decode_ms,pts_difference_ms,video_preprocess_ms,"
                       "merge_ms,inference_ms,postprocess_ms,csv_output_ms,detections\n";

        const SourceGeometry geometry{
            visible_width,
            visible_height,
            static_cast<float>(kInputWidth) / static_cast<float>(visible_width),
            static_cast<float>(kContentHeight) / static_cast<float>(visible_height),
        };
        StageTotals totals;
        std::vector<double> pts_differences;
        std::size_t processed_pairs = 0;
        std::size_t total_detections = 0;
        bool visible_read_failed = false;
        bool infrared_read_failed = false;
        cv::Mat first_visible_prepared;
        cv::Mat first_infrared_prepared;

        while (max_frames == 0 || static_cast<int>(processed_pairs) < max_frames) {
            cv::Mat visible_frame;
            cv::Mat infrared_frame;
            const Clock::time_point decode_start = Clock::now();
            const bool visible_ok = visible_capture.read(visible_frame);
            const bool infrared_ok = infrared_capture.read(infrared_frame);
            const Clock::time_point decode_end = Clock::now();
            if (!visible_ok || !infrared_ok) {
                visible_read_failed = !visible_ok;
                infrared_read_failed = !infrared_ok;
                break;
            }
            const double visible_pts_ms = visible_capture.get(cv::CAP_PROP_POS_MSEC);
            const double infrared_pts_ms = infrared_capture.get(cv::CAP_PROP_POS_MSEC);
            const double pts_difference_ms = std::abs(visible_pts_ms - infrared_pts_ms);

            cv::Mat visible_prepared;
            cv::Mat infrared_prepared;
            const Clock::time_point video_preprocess_start = Clock::now();
            prepare_pair(visible_frame, infrared_frame, visible_prepared, infrared_prepared);
            const Clock::time_point video_preprocess_end = Clock::now();
            if (processed_pairs == 0) {
                first_visible_prepared = visible_prepared.clone();
                first_infrared_prepared = infrared_prepared.clone();
            }

            StagDetTimings detector_timings;
            const std::vector<StagDetDetection> detections =
                detector.detect(visible_prepared, infrared_prepared, &detector_timings);
            const Clock::time_point output_start = Clock::now();
            for (const StagDetDetection& detection : detections) {
                const StagDetDetection restored = restore(detection, geometry);
                detections_csv << sequence_id << ',' << processed_pairs << ','
                               << std::setprecision(9) << visible_pts_ms << ',' << infrared_pts_ms << ','
                               << restored.score << ',' << restored.x1 << ',' << restored.y1 << ','
                               << restored.x2 << ',' << restored.y2 << '\n';
            }
            const Clock::time_point output_end = Clock::now();

            const double decode_ms = elapsed_ms(decode_start, decode_end);
            const double video_preprocess_ms = elapsed_ms(video_preprocess_start, video_preprocess_end);
            const double output_ms = elapsed_ms(output_start, output_end);
            timings_csv << sequence_id << ',' << processed_pairs << ','
                        << std::fixed << std::setprecision(6)
                        << decode_ms << ',' << pts_difference_ms << ',' << video_preprocess_ms << ','
                        << detector_timings.preprocess_ms << ',' << detector_timings.inference_ms << ','
                        << detector_timings.postprocess_ms << ',' << output_ms << ','
                        << detections.size() << '\n';
            totals.decode_ms += decode_ms;
            totals.video_preprocess_ms += video_preprocess_ms;
            totals.merge_ms += detector_timings.preprocess_ms;
            totals.inference_ms += detector_timings.inference_ms;
            totals.postprocess_ms += detector_timings.postprocess_ms;
            totals.output_ms += output_ms;
            pts_differences.push_back(pts_difference_ms);
            total_detections += detections.size();
            ++processed_pairs;
            if (processed_pairs % 100 == 0) {
                std::cout << sequence_id << ": " << processed_pairs << " frame pairs\n";
            }
        }
        detections_csv.close();
        timings_csv.close();
        const Clock::time_point wall_end = Clock::now();
        const double wall_ms = elapsed_ms(wall_start, wall_end);

        if (!first_visible_prepared.empty()) {
            cv::imwrite((output_dir / "debug_visible_320x192.png").string(), first_visible_prepared);
            cv::imwrite((output_dir / "debug_infrared_320x192.png").string(), first_infrared_prepared);
        }
        write_summary(
            output_dir / "summary.json",
            sequence_id,
            visible_path,
            infrared_path,
            processed_pairs,
            total_detections,
            visible_width,
            visible_height,
            infrared_width,
            infrared_height,
            visible_fps,
            infrared_fps,
            visible_frame_count,
            infrared_frame_count,
            confidence,
            nms_iou,
            wall_ms,
            totals,
            pts_differences,
            visible_read_failed,
            infrared_read_failed);

        const double count = static_cast<double>(processed_pairs);
        std::cout << std::fixed << std::setprecision(4)
                  << "\nSTAGDet original-video detection E2E\n"
                  << "Sequence: " << sequence_id << "\n"
                  << "Processed frame pairs: " << processed_pairs << "\n"
                  << "Detections: " << total_detections << "\n"
                  << "Average pair decode: " << (count > 0.0 ? totals.decode_ms / count : 0.0) << " ms\n"
                  << "Average video preprocess: "
                  << (count > 0.0 ? totals.video_preprocess_ms / count : 0.0) << " ms\n"
                  << "Average four-channel merge: " << (count > 0.0 ? totals.merge_ms / count : 0.0) << " ms\n"
                  << "Average RKNN inference: " << (count > 0.0 ? totals.inference_ms / count : 0.0) << " ms\n"
                  << "Average postprocess: " << (count > 0.0 ? totals.postprocess_ms / count : 0.0) << " ms\n"
                  << "Wall time: " << wall_ms << " ms\n"
                  << "End-to-end FPS: " << (wall_ms > 0.0 ? count * 1000.0 / wall_ms : 0.0) << "\n"
                  << "PTS difference P50/P95: " << percentile(pts_differences, 0.50)
                  << "/" << percentile(pts_differences, 0.95) << " ms\n"
                  << "Outputs: " << output_dir << "\n";
        return processed_pairs > 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "STAGDet video pipeline failed: " << error.what() << "\n";
        return 2;
    }
}
