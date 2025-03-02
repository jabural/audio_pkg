// Copyright 2025 <Your Name>. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define MINIAUDIO_IMPLEMENTATION
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "miniaudio.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

// AudioPublisher is a ROS 2 node that records audio using miniaudio and publishes
// it as Float32MultiArray messages. It buffers samples to control the publish rate.
class AudioPublisher : public rclcpp::Node {
 public:
  // Constructs the AudioPublisher node, initializing miniaudio and starting recording.
  AudioPublisher() : Node("audio_publisher") {
    // Create a publisher for audio data on the "audio_data" topic with a queue size of 10.
    publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("audio_data", 10);

    // Set the buffer size to 4000 samples (~90 ms at 44.1 kHz) to batch audio data
    // before publishing, reducing message frequency.
    buffer_size_ = 4000;

    // Initialize the miniaudio context for audio device access.
    if (ma_context_init(NULL, 0, NULL, &context_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize miniaudio context.");
      throw std::runtime_error("Miniaudio context init failed");
    }

    // Configure the recording device for mono, 44.1 kHz, float32 format.
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format = ma_format_f32;
    deviceConfig.capture.channels = 1;
    deviceConfig.sampleRate = 44100;
    deviceConfig.dataCallback = AudioPublisher::data_callback;
    deviceConfig.pUserData = this;  // Pass this instance to the callback.

    // Initialize and validate the recording device.
    if (ma_device_init(&context_, &deviceConfig, &device_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize recording device.");
      ma_context_uninit(&context_);
      throw std::runtime_error("Device init failed");
    }

    // Start capturing audio immediately upon node creation.
    RCLCPP_INFO(this->get_logger(), "Starting audio recording...");
    ma_device_start(&device_);
  }

  // Destructor ensures proper cleanup of miniaudio resources.
  ~AudioPublisher() {
    ma_device_stop(&device_);
    ma_device_uninit(&device_);
    ma_context_uninit(&context_);
  }

 private:
  // Publishes the accumulated audio samples as a ROS 2 message and clears the buffer.
  void publish_buffer() {
    if (recorded_samples_.empty()) return;

    auto msg = std_msgs::msg::Float32MultiArray();
    msg.data = recorded_samples_;
    publisher_->publish(msg);

    // Log the first publish event to confirm operation, avoiding spam.
    RCLCPP_INFO_ONCE(this->get_logger(), "Published %zu samples", recorded_samples_.size());

    // Clear the buffer to reset for the next batch, keeping memory usage bounded.
    recorded_samples_.clear();
  }

  // Callback invoked by miniaudio when new audio data is captured.
  // Args:
  //   pDevice: Pointer to the miniaudio device.
  //   pOutput: Unused in capture mode (marked maybe_unused).
  //   pInput: Raw audio input data from the microphone.
  //   frameCount: Number of frames (samples per channel) in this callback.
  static void data_callback(ma_device* pDevice, [[maybe_unused]] void* pOutput,
                            const void* pInput, ma_uint32 frameCount) {
    AudioPublisher* node = static_cast<AudioPublisher*>(pDevice->pUserData);
    if (!node || !pInput) return;

    const float* input = static_cast<const float*>(pInput);

    // Append new samples to the buffer for batching.
    std::vector<float> samples(input, input + frameCount * pDevice->capture.channels);
    node->recorded_samples_.insert(node->recorded_samples_.end(), samples.begin(), samples.end());

    // Publish when the buffer reaches or exceeds the target size.
    if (node->recorded_samples_.size() >= node->buffer_size_) {
      node->publish_buffer();
    }
  }

  // ROS 2 publisher for sending audio data.
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher_;

  // Miniaudio context and device for audio capture.
  ma_context context_;
  ma_device device_;

  // Buffer to accumulate audio samples before publishing.
  std::vector<float> recorded_samples_;

  // Target number of samples to collect before publishing a message.
  size_t buffer_size_;
};

// Main entry point: initializes ROS 2, spins the AudioPublisher node, and shuts down.
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AudioPublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}