#include "stagdet_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace {

constexpr int kInputChannels = 4;
constexpr int kDflBins = 16;
constexpr int kRegressionChannels = 4 * kDflBins;
constexpr int kDecodedBoxChannels = 4;
constexpr int kScoreChannels = 1;

void check_rknn(int code, const std::string& operation) {
    if (code != RKNN_SUCC) {
        throw std::runtime_error(operation + " failed with RKNN code " + std::to_string(code));
    }
}

std::vector<std::uint8_t> read_binary_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Failed to open RKNN model: " + path);
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        throw std::runtime_error("RKNN model is empty: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (!input.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Failed to read RKNN model: " + path);
    }
    return data;
}

std::string dimensions(const rknn_tensor_attr& attribute) {
    std::string result = "[";
    for (std::uint32_t index = 0; index < attribute.n_dims; ++index) {
        if (index > 0) {
            result += ",";
        }
        result += std::to_string(attribute.dims[index]);
    }
    return result + "]";
}

struct TensorGeometry {
    int width;
    int height;
    int channels;
};

TensorGeometry input_geometry(const rknn_tensor_attr& attribute) {
    if (attribute.n_dims != 4) {
        throw std::runtime_error("Expected a rank-4 input tensor: " + dimensions(attribute));
    }
    if (attribute.fmt == RKNN_TENSOR_NCHW) {
        return {
            static_cast<int>(attribute.dims[3]),
            static_cast<int>(attribute.dims[2]),
            static_cast<int>(attribute.dims[1]),
        };
    }
    if (attribute.fmt == RKNN_TENSOR_NHWC) {
        return {
            static_cast<int>(attribute.dims[2]),
            static_cast<int>(attribute.dims[1]),
            static_cast<int>(attribute.dims[3]),
        };
    }
    throw std::runtime_error("Unsupported input layout: " + dimensions(attribute));
}

int expected_prediction_count(int width, int height) {
    if (width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0) {
        throw std::runtime_error("STAGDet input width and height must be positive multiples of 32");
    }
    int result = 0;
    for (const int stride : {4, 8, 16, 32}) {
        result += (width / stride) * (height / stride);
    }
    return result;
}

bool matches_output_tensor(const rknn_tensor_attr& attribute, int channels, int predictions) {
    if (attribute.n_elems != static_cast<std::uint32_t>(channels * predictions)) {
        return false;
    }
    if (attribute.n_dims == 3) {
        return attribute.dims[1] == static_cast<std::uint32_t>(channels)
            || attribute.dims[2] == static_cast<std::uint32_t>(channels);
    }
    if (attribute.n_dims == 4) {
        return attribute.dims[1] == static_cast<std::uint32_t>(channels)
            || attribute.dims[3] == static_cast<std::uint32_t>(channels);
    }
    return false;
}

bool is_channel_major(const rknn_tensor_attr& attribute, int channels, int predictions) {
    if (!matches_output_tensor(attribute, channels, predictions)) {
        throw std::runtime_error("Output dimensions do not match the expected tensor: " + dimensions(attribute));
    }
    if (attribute.dims[1] == static_cast<std::uint32_t>(channels)) {
        return true;
    }
    if (attribute.dims[attribute.n_dims - 1] == static_cast<std::uint32_t>(channels)) {
        return false;
    }
    if (attribute.fmt == RKNN_TENSOR_NCHW) {
        return true;
    }
    if (attribute.fmt == RKNN_TENSOR_NHWC) {
        return false;
    }
    throw std::runtime_error("Unsupported output layout: " + dimensions(attribute));
}

float tensor_value(
    const rknn_output& output,
    const rknn_tensor_attr& attribute,
    std::size_t index) {
    if (index >= attribute.n_elems) {
        throw std::out_of_range("RKNN output index is outside tensor bounds");
    }
    switch (attribute.type) {
        case RKNN_TENSOR_INT8: {
            const auto value = static_cast<const std::int8_t*>(output.buf)[index];
            return (static_cast<int>(value) - attribute.zp) * attribute.scale;
        }
        case RKNN_TENSOR_UINT8: {
            const auto value = static_cast<const std::uint8_t*>(output.buf)[index];
            return (static_cast<int>(value) - attribute.zp) * attribute.scale;
        }
        case RKNN_TENSOR_FLOAT32:
            return static_cast<const float*>(output.buf)[index];
        default:
            throw std::runtime_error("Unsupported RKNN output tensor type: " + std::to_string(attribute.type));
    }
}

std::size_t output_index(
    bool channel_major,
    int channel,
    int prediction,
    int channels,
    int predictions) {
    return channel_major
        ? static_cast<std::size_t>(channel * predictions + prediction)
        : static_cast<std::size_t>(prediction * channels + channel);
}

float dfl_expectation(
    const rknn_output& output,
    const rknn_tensor_attr& attribute,
    bool channel_major,
    int side,
    int prediction,
    int predictions) {
    float logits[kDflBins];
    float maximum = -std::numeric_limits<float>::infinity();
    for (int bin = 0; bin < kDflBins; ++bin) {
        const int channel = side * kDflBins + bin;
        logits[bin] = tensor_value(
            output,
            attribute,
            output_index(channel_major, channel, prediction, kRegressionChannels, predictions));
        maximum = std::max(maximum, logits[bin]);
    }
    float denominator = 0.0F;
    float numerator = 0.0F;
    for (int bin = 0; bin < kDflBins; ++bin) {
        const float probability = std::exp(logits[bin] - maximum);
        denominator += probability;
        numerator += probability * static_cast<float>(bin);
    }
    return numerator / denominator;
}

struct Anchor {
    float x;
    float y;
    float stride;
};

Anchor anchor_for_prediction(int prediction, int input_width, int input_height) {
    constexpr int strides[] = {4, 8, 16, 32};
    int offset = 0;
    for (int level = 0; level < 4; ++level) {
        const int columns = input_width / strides[level];
        const int rows = input_height / strides[level];
        const int count = rows * columns;
        if (prediction < offset + count) {
            const int local = prediction - offset;
            return {
                (static_cast<float>(local % columns) + 0.5F) * strides[level],
                (static_cast<float>(local / columns) + 0.5F) * strides[level],
                static_cast<float>(strides[level]),
            };
        }
        offset += count;
    }
    throw std::out_of_range("Prediction index is outside all feature levels");
}

Anchor anchor_for_level(int prediction, int input_width, int stride) {
    const int columns = input_width / stride;
    return {
        (static_cast<float>(prediction % columns) + 0.5F) * stride,
        (static_cast<float>(prediction / columns) + 0.5F) * stride,
        static_cast<float>(stride),
    };
}

float intersection_over_union(const StagDetDetection& first, const StagDetDetection& second) {
    const float x1 = std::max(first.x1, second.x1);
    const float y1 = std::max(first.y1, second.y1);
    const float x2 = std::min(first.x2, second.x2);
    const float y2 = std::min(first.y2, second.y2);
    const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    const float first_area = std::max(0.0F, first.x2 - first.x1) * std::max(0.0F, first.y2 - first.y1);
    const float second_area = std::max(0.0F, second.x2 - second.x1) * std::max(0.0F, second.y2 - second.y1);
    const float union_area = first_area + second_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<StagDetDetection> nms(
    const std::vector<StagDetDetection>& candidates,
    float iou_threshold) {
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return candidates[left].score > candidates[right].score;
    });

    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<StagDetDetection> kept;
    for (std::size_t position = 0; position < order.size(); ++position) {
        const std::size_t current = order[position];
        if (suppressed[current]) {
            continue;
        }
        kept.push_back(candidates[current]);
        for (std::size_t next = position + 1; next < order.size(); ++next) {
            const std::size_t compared = order[next];
            if (!suppressed[compared]
                && intersection_over_union(candidates[current], candidates[compared]) > iou_threshold) {
                suppressed[compared] = true;
            }
        }
    }
    return kept;
}

void validate_input_pair(
    const cv::Mat& visible_bgr,
    const cv::Mat& infrared_gray,
    int expected_width,
    int expected_height) {
    if (visible_bgr.type() != CV_8UC3
        || visible_bgr.rows != expected_height
        || visible_bgr.cols != expected_width) {
        throw std::runtime_error("Visible input dimensions or BGR uint8 type do not match the RKNN model");
    }
    if (infrared_gray.type() != CV_8UC1
        || infrared_gray.rows != expected_height
        || infrared_gray.cols != expected_width) {
        throw std::runtime_error("Infrared input dimensions or grayscale uint8 type do not match the RKNN model");
    }
}

cv::Mat merge_rgbt(
    const cv::Mat& visible_bgr,
    const cv::Mat& infrared_gray,
    int expected_width,
    int expected_height) {
    validate_input_pair(visible_bgr, infrared_gray, expected_width, expected_height);
    std::vector<cv::Mat> channels;
    cv::split(visible_bgr, channels);
    channels.push_back(infrared_gray);
    cv::Mat rgbt;
    cv::merge(channels, rgbt);
    if (!rgbt.isContinuous()) {
        rgbt = rgbt.clone();
    }
    return rgbt;
}

}  // namespace

StagDetDetector::StagDetDetector(
    const std::string& model_path,
    float confidence_threshold,
    float iou_threshold,
    bool collect_perf)
    : confidence_threshold_(confidence_threshold),
      iou_threshold_(iou_threshold),
      collect_perf_(collect_perf) {
    if (!(confidence_threshold > 0.0F && confidence_threshold < 1.0F)) {
        throw std::invalid_argument("Confidence threshold must be in (0, 1)");
    }
    if (!(iou_threshold > 0.0F && iou_threshold < 1.0F)) {
        throw std::invalid_argument("IoU threshold must be in (0, 1)");
    }
    try {
        load_model(model_path);
        query_tensor_contract();
    } catch (...) {
        release();
        throw;
    }
}

StagDetDetector::~StagDetDetector() {
    release();
}

void StagDetDetector::load_model(const std::string& model_path) {
    const std::vector<std::uint8_t> model = read_binary_file(model_path);
    const std::uint32_t flags = collect_perf_ ? RKNN_FLAG_COLLECT_PERF_MASK : 0U;
    check_rknn(
        rknn_init(
            &context_,
            const_cast<std::uint8_t*>(model.data()),
            static_cast<std::uint32_t>(model.size()),
            flags,
            nullptr),
        "rknn_init");
}

void StagDetDetector::query_tensor_contract() {
    check_rknn(
        rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_count_, sizeof(io_count_)),
        "RKNN_QUERY_IN_OUT_NUM");
    if ((io_count_.n_input != 1 && io_count_.n_input != 2)
        || (io_count_.n_output != 2 && io_count_.n_output != 8)) {
        throw std::runtime_error("STAGDet RKNN must have one or two inputs and two or eight outputs");
    }

    input_attributes_.resize(io_count_.n_input);
    bool found_visible = false;
    bool found_infrared = false;
    for (std::uint32_t index = 0; index < io_count_.n_input; ++index) {
        rknn_tensor_attr& attribute = input_attributes_[index];
        std::memset(&attribute, 0, sizeof(attribute));
        attribute.index = index;
        check_rknn(
            rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &attribute, sizeof(attribute)),
            "RKNN_QUERY_INPUT_ATTR");
        std::cout << "input[" << index << "] " << attribute.name
                  << " dims=" << dimensions(attribute)
                  << " elements=" << attribute.n_elems
                  << " type=" << attribute.type
                  << " fmt=" << attribute.fmt << "\n";
        const TensorGeometry geometry = input_geometry(attribute);
        if (index == 0) {
            input_width_ = geometry.width;
            input_height_ = geometry.height;
        } else if (geometry.width != input_width_ || geometry.height != input_height_) {
            throw std::runtime_error("All STAGDet inputs must have matching spatial dimensions");
        }
        if (io_count_.n_input == 2 && geometry.channels == 3) {
            visible_input_index_ = index;
            found_visible = true;
        } else if (io_count_.n_input == 2 && geometry.channels == 1) {
            infrared_input_index_ = index;
            found_infrared = true;
        }
    }
    dual_input_ = io_count_.n_input == 2;
    if (dual_input_ && (!found_visible || !found_infrared)) {
        throw std::runtime_error("Expected visible three-channel and infrared one-channel STAGDet inputs");
    }
    if (!dual_input_ && input_geometry(input_attributes_[0]).channels != kInputChannels) {
        throw std::runtime_error("Expected a four-channel STAGDet input");
    }
    prediction_count_ = expected_prediction_count(input_width_, input_height_);

    output_attributes_.resize(io_count_.n_output);
    bool found_box_output = false;
    bool found_scores = false;
    for (std::uint32_t index = 0; index < io_count_.n_output; ++index) {
        rknn_tensor_attr& attribute = output_attributes_[index];
        std::memset(&attribute, 0, sizeof(attribute));
        attribute.index = index;
        check_rknn(
            rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attribute, sizeof(attribute)),
            "RKNN_QUERY_OUTPUT_ATTR");
        std::cout << "output[" << index << "] " << attribute.name
                  << " dims=" << dimensions(attribute)
                  << " elements=" << attribute.n_elems
                  << " type=" << attribute.type
                  << " zp=" << attribute.zp
                  << " scale=" << attribute.scale << "\n";
        if (attribute.type != RKNN_TENSOR_INT8
            || attribute.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
            || attribute.scale <= 0.0F) {
            throw std::runtime_error("STAGDet outputs must be affine-asymmetric INT8 tensors");
        }
        if (io_count_.n_output == 2
            && matches_output_tensor(attribute, kRegressionChannels, prediction_count_)) {
            box_output_index_ = index;
            box_output_channels_ = kRegressionChannels;
            raw_dfl_output_ = true;
            found_box_output = true;
        } else if (io_count_.n_output == 2
                   && matches_output_tensor(attribute, kDecodedBoxChannels, prediction_count_)) {
            box_output_index_ = index;
            box_output_channels_ = kDecodedBoxChannels;
            raw_dfl_output_ = false;
            found_box_output = true;
        } else if (io_count_.n_output == 2
                   && matches_output_tensor(attribute, kScoreChannels, prediction_count_)) {
            scores_output_index_ = index;
            found_scores = true;
        }
    }
    per_scale_output_ = io_count_.n_output == 8;
    if (per_scale_output_) {
        raw_dfl_output_ = true;
        for (const int stride : {4, 8, 16, 32}) {
            const int level_predictions = (input_width_ / stride) * (input_height_ / stride);
            int box_index = -1;
            int score_index = -1;
            for (std::uint32_t index = 0; index < io_count_.n_output; ++index) {
                if (matches_output_tensor(output_attributes_[index], kRegressionChannels, level_predictions)) {
                    if (box_index >= 0) {
                        throw std::runtime_error("Multiple regression outputs match one feature level");
                    }
                    box_index = static_cast<int>(index);
                }
                if (matches_output_tensor(output_attributes_[index], kScoreChannels, level_predictions)) {
                    if (score_index >= 0) {
                        throw std::runtime_error("Multiple score outputs match one feature level");
                    }
                    score_index = static_cast<int>(index);
                }
            }
            if (box_index < 0 || score_index < 0) {
                throw std::runtime_error("Missing per-scale regression or score output");
            }
            output_levels_.push_back({
                static_cast<std::uint32_t>(box_index),
                static_cast<std::uint32_t>(score_index),
                stride,
                level_predictions,
            });
        }
    } else if (!found_box_output || !found_scores) {
        throw std::runtime_error("Unexpected STAGDet output element counts");
    }
}

std::vector<StagDetDetection> StagDetDetector::detect(
    const cv::Mat& visible_bgr,
    const cv::Mat& infrared_gray,
    StagDetTimings* timings) {
    return detect_impl(visible_bgr, infrared_gray, timings, nullptr);
}

StagDetPerfReport StagDetDetector::collect_perf_report(
    const cv::Mat& visible_bgr,
    const cv::Mat& infrared_gray) {
    if (!collect_perf_) {
        throw std::runtime_error("Performance collection was not enabled");
    }
    StagDetPerfReport report;
    detect_impl(visible_bgr, infrared_gray, nullptr, &report);
    return report;
}

std::vector<StagDetDetection> StagDetDetector::detect_impl(
    const cv::Mat& visible_bgr,
    const cv::Mat& infrared_gray,
    StagDetTimings* timings,
    StagDetPerfReport* perf_report) {
    const auto preprocess_start = std::chrono::steady_clock::now();
    cv::Mat input_rgbt;
    cv::Mat visible_input;
    cv::Mat infrared_input;
    if (dual_input_) {
        validate_input_pair(visible_bgr, infrared_gray, input_width_, input_height_);
        visible_input = visible_bgr.isContinuous() ? visible_bgr : visible_bgr.clone();
        infrared_input = infrared_gray.isContinuous() ? infrared_gray : infrared_gray.clone();
    } else {
        input_rgbt = merge_rgbt(visible_bgr, infrared_gray, input_width_, input_height_);
    }
    const auto preprocess_end = std::chrono::steady_clock::now();

    std::vector<rknn_input> inputs(io_count_.n_input);
    const auto set_input = [&](std::uint32_t index, const cv::Mat& value) {
        rknn_input& input = inputs[index];
        input.index = index;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = static_cast<std::uint32_t>(value.total() * value.elemSize());
        input.buf = value.data;
        input.pass_through = 0;
    };
    if (dual_input_) {
        set_input(visible_input_index_, visible_input);
        set_input(infrared_input_index_, infrared_input);
    } else {
        set_input(0, input_rgbt);
    }
    check_rknn(
        rknn_inputs_set(context_, io_count_.n_input, inputs.data()),
        "rknn_inputs_set");

    const auto inference_start = std::chrono::steady_clock::now();
    check_rknn(rknn_run(context_, nullptr), "rknn_run");
    std::vector<rknn_output> outputs(io_count_.n_output);
    for (std::uint32_t index = 0; index < io_count_.n_output; ++index) {
        outputs[index].index = index;
        outputs[index].want_float = 0;
        outputs[index].is_prealloc = 0;
    }
    check_rknn(rknn_outputs_get(context_, io_count_.n_output, outputs.data(), nullptr), "rknn_outputs_get");
    const auto inference_end = std::chrono::steady_clock::now();

    std::vector<StagDetDetection> detections;
    const auto postprocess_start = std::chrono::steady_clock::now();
    try {
        std::vector<StagDetDetection> candidates;
        candidates.reserve(256);
        if (per_scale_output_) {
            for (const OutputLevel& level : output_levels_) {
                const rknn_tensor_attr& box_attribute = output_attributes_[level.box_index];
                const rknn_tensor_attr& score_attribute = output_attributes_[level.score_index];
                const bool box_channel_major = is_channel_major(
                    box_attribute, kRegressionChannels, level.prediction_count);
                const bool score_channel_major = is_channel_major(
                    score_attribute, kScoreChannels, level.prediction_count);
                const rknn_output& box_output = outputs[level.box_index];
                const rknn_output& score_output = outputs[level.score_index];
                const int quantized_threshold = std::clamp(
                    static_cast<int>(std::ceil(
                        confidence_threshold_ / score_attribute.scale + score_attribute.zp)),
                    -128,
                    127);
                const auto* quantized_scores = static_cast<const std::int8_t*>(score_output.buf);
                for (int prediction = 0; prediction < level.prediction_count; ++prediction) {
                    const std::size_t score_index = output_index(
                        score_channel_major, 0, prediction, kScoreChannels, level.prediction_count);
                    if (static_cast<int>(quantized_scores[score_index]) < quantized_threshold) {
                        continue;
                    }
                    const float score = tensor_value(score_output, score_attribute, score_index);
                    const Anchor anchor = anchor_for_level(prediction, input_width_, level.stride);
                    const float left = dfl_expectation(
                        box_output, box_attribute, box_channel_major, 0, prediction, level.prediction_count);
                    const float top = dfl_expectation(
                        box_output, box_attribute, box_channel_major, 1, prediction, level.prediction_count);
                    const float right = dfl_expectation(
                        box_output, box_attribute, box_channel_major, 2, prediction, level.prediction_count);
                    const float bottom = dfl_expectation(
                        box_output, box_attribute, box_channel_major, 3, prediction, level.prediction_count);
                    candidates.push_back({
                        anchor.x - left * anchor.stride,
                        anchor.y - top * anchor.stride,
                        anchor.x + right * anchor.stride,
                        anchor.y + bottom * anchor.stride,
                        score,
                        0,
                    });
                }
            }
        } else {
            const rknn_tensor_attr& box_attribute = output_attributes_[box_output_index_];
            const rknn_tensor_attr& score_attribute = output_attributes_[scores_output_index_];
            const bool box_channel_major = is_channel_major(
                box_attribute, box_output_channels_, prediction_count_);
            const bool scores_channel_major = is_channel_major(
                score_attribute, kScoreChannels, prediction_count_);
            const rknn_output& box_output = outputs[box_output_index_];
            const rknn_output& score_output = outputs[scores_output_index_];
            const int quantized_threshold = std::clamp(
                static_cast<int>(std::ceil(
                    confidence_threshold_ / score_attribute.scale + score_attribute.zp)),
                -128,
                127);
            const auto* quantized_scores = static_cast<const std::int8_t*>(score_output.buf);
            for (int prediction = 0; prediction < prediction_count_; ++prediction) {
                const std::size_t score_index = output_index(
                    scores_channel_major, 0, prediction, kScoreChannels, prediction_count_);
                if (static_cast<int>(quantized_scores[score_index]) < quantized_threshold) {
                    continue;
                }
                const float score = tensor_value(score_output, score_attribute, score_index);
                if (raw_dfl_output_) {
                    const Anchor anchor = anchor_for_prediction(
                        prediction, input_width_, input_height_);
                    const float left = dfl_expectation(
                        box_output, box_attribute, box_channel_major, 0, prediction, prediction_count_);
                    const float top = dfl_expectation(
                        box_output, box_attribute, box_channel_major, 1, prediction, prediction_count_);
                    const float right = dfl_expectation(
                        box_output, box_attribute, box_channel_major, 2, prediction, prediction_count_);
                    const float bottom = dfl_expectation(
                        box_output, box_attribute, box_channel_major, 3, prediction, prediction_count_);
                    candidates.push_back({
                        anchor.x - left * anchor.stride,
                        anchor.y - top * anchor.stride,
                        anchor.x + right * anchor.stride,
                        anchor.y + bottom * anchor.stride,
                        score,
                        0,
                    });
                    continue;
                }
                const float center_x = tensor_value(
                    box_output,
                    box_attribute,
                    output_index(
                        box_channel_major, 0, prediction, kDecodedBoxChannels, prediction_count_));
                const float center_y = tensor_value(
                    box_output,
                    box_attribute,
                    output_index(
                        box_channel_major, 1, prediction, kDecodedBoxChannels, prediction_count_));
                const float width = tensor_value(
                    box_output,
                    box_attribute,
                    output_index(
                        box_channel_major, 2, prediction, kDecodedBoxChannels, prediction_count_));
                const float height = tensor_value(
                    box_output,
                    box_attribute,
                    output_index(
                        box_channel_major, 3, prediction, kDecodedBoxChannels, prediction_count_));
                candidates.push_back({
                    center_x - width * 0.5F,
                    center_y - height * 0.5F,
                    center_x + width * 0.5F,
                    center_y + height * 0.5F,
                    score,
                    0,
                });
            }
        }
        detections = nms(candidates, iou_threshold_);

        if (perf_report != nullptr) {
            rknn_sdk_version sdk_version{};
            check_rknn(
                rknn_query(context_, RKNN_QUERY_SDK_VERSION, &sdk_version, sizeof(sdk_version)),
                "RKNN_QUERY_SDK_VERSION");
            perf_report->sdk_api_version = sdk_version.api_version;
            perf_report->sdk_driver_version = sdk_version.drv_version;
            rknn_perf_run perf_run{};
            check_rknn(
                rknn_query(context_, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run)),
                "RKNN_QUERY_PERF_RUN");
            perf_report->run_duration_us = perf_run.run_duration;
            rknn_perf_detail perf_detail{};
            check_rknn(
                rknn_query(context_, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail)),
                "RKNN_QUERY_PERF_DETAIL");
            if (perf_detail.perf_data != nullptr && perf_detail.data_len > 0) {
                perf_report->perf_detail.assign(perf_detail.perf_data, perf_detail.data_len);
            }
        }
    } catch (...) {
        rknn_outputs_release(context_, io_count_.n_output, outputs.data());
        throw;
    }
    const auto postprocess_end = std::chrono::steady_clock::now();
    check_rknn(
        rknn_outputs_release(context_, io_count_.n_output, outputs.data()),
        "rknn_outputs_release");

    if (timings != nullptr) {
        timings->preprocess_ms = std::chrono::duration<double, std::milli>(
            preprocess_end - preprocess_start).count();
        timings->inference_ms = std::chrono::duration<double, std::milli>(
            inference_end - inference_start).count();
        timings->postprocess_ms = std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start).count();
    }
    return detections;
}

void StagDetDetector::release() noexcept {
    if (context_ != 0) {
        rknn_destroy(context_);
        context_ = 0;
    }
}
