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
#include "miniaudio.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vector>
#include <mutex>
#include <chrono>
#include <thread>

// AudioPlayer is a ROS 2 node that subscribes to audio data and plays it using miniaudio.
class AudioPlayer : public rclcpp::Node {
 public:
  // Constructs the AudioPlayer node, setting up miniaudio playback and ROS 2 subscription.
  AudioPlayer() : Node("audio_player") {
    // Initialize the miniaudio context for audio device management.
    if (ma_context_init(NULL, 0, NULL, &context_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize miniaudio context.");
      throw std::runtime_error("Miniaudio context init failed");
    }

    // Configure the playback device for mono, 44.1 kHz, float32 format.
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format = ma_format_f32;
    deviceConfig.playback.channels = 1;
    deviceConfig.sampleRate = 44100;
    deviceConfig.dataCallback = AudioPlayer::data_callback;
    deviceConfig.pUserData = this;  // Pass this instance to the callback.

    // Initialize and validate the playback device.
    if (ma_device_init(&context_, &deviceConfig, &device_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize playback device.");
      ma_context_uninit(&context_);
      throw std::runtime_error("Playback device init failed");
    }

    // Subscribe to the "audio_data" topic with a queue size of 10 to receive audio samples.
    subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "audio_data", 10, std::bind(&AudioPlayer::audio_callback, this, std::placeholders::_1));

    // Start playback immediately to process incoming audio data.
    RCLCPP_INFO(this->get_logger(), "Starting audio playback...");
    ma_device_start(&device_);
  }

  // Destructor ensures proper cleanup of miniaudio resources.
  ~AudioPlayer() {
    ma_device_stop(&device_);
    ma_device_uninit(&device_);
    ma_context_uninit(&context_);
  }

 private:
  // Callback invoked by miniaudio to request audio data for playback.
  // Args:
  //   pDevice: Pointer to the miniaudio device.
  //   pOutput: Buffer to fill with audio samples for playback.
  //   pInput: Unused in playback mode.
  //   frameCount: Number of frames (samples per channel) requested.
  static void data_callback(ma_device* pDevice, void* pOutput, [[maybe_unused]] const void* pInput, ma_uint32 frameCount) {
    AudioPlayer* node = static_cast<AudioPlayer*>(pDevice->pUserData);
    float* output = static_cast<float*>(pOutput);

    // Lock the buffer to safely access audio samples from the subscriber thread.
    std::lock_guard<std::mutex> lock(node->buffer_mutex_);
    ma_uint32 framesToCopy = std::min(frameCount, static_cast<ma_uint32>(node->audio_buffer_.size()));

    if (framesToCopy > 0) {
      // Copy available samples to the output buffer and remove them from the queue.
      std::copy(node->audio_buffer_.begin(), node->audio_buffer_.begin() + framesToCopy, output);
      node->audio_buffer_.erase(node->audio_buffer_.begin(), node->audio_buffer_.begin() + framesToCopy);
    } else {
      // Handle buffer underrun by filling with silence to avoid audio glitches.
      std::fill(output, output + frameCount, 0.0f);
    }
  }

  // Callback invoked when a new ROS 2 Float32MultiArray message is received.
  // Args:
  //   msg: Shared pointer to the received audio data message.
  void audio_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    // Lock the buffer to append new samples safely from the subscriber thread.
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    audio_buffer_.insert(audio_buffer_.end(), msg->data.begin(), msg->data.end());

    // Log receipt of samples for debugging and monitoring buffer growth.
    RCLCPP_INFO(this->get_logger(), "Received %zu samples, buffer size now %zu",
                msg->data.size(), audio_buffer_.size());
  }

  // Miniaudio context and device for audio playback.
  ma_context context_;
  ma_device device_;

  // Buffer to store audio samples received from the ROS 2 topic.
  std::vector<float> audio_buffer_;

  // Mutex to synchronize access to audio_buffer_ between subscriber and playback threads.
  std::mutex buffer_mutex_;

  // ROS 2 subscription to receive audio data messages.
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr subscription_;
};

// Main entry point: initializes ROS 2, spins the AudioPlayer node, and shuts down.
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AudioPlayer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}