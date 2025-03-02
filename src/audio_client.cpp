// Copyright 2025 Javier. All rights reserved.
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
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "audio_pkg/action/audio_playback_action.hpp"
#include <vector>
#include <chrono>
#include <thread>
#include <memory>

// AudioRecorderClient records 3 seconds of audio and sends it as a goal to the AudioPlayer action server.
class AudioRecorderClient : public rclcpp::Node {
 public:
  using AudioPlaybackAction = audio_pkg::action::AudioPlaybackAction;
  using GoalHandleAudioPlaybackAction = rclcpp_action::ClientGoalHandle<AudioPlaybackAction>;

  AudioRecorderClient() : Node("audio_recorder_client") {
    // Initialize miniaudio context.
    if (ma_context_init(NULL, 0, NULL, &context_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize miniaudio context.");
      throw std::runtime_error("Miniaudio context init failed");
    }

    // Configure recording device: mono, 44.1 kHz, float32.
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format = ma_format_f32;
    deviceConfig.capture.channels = 1;
    deviceConfig.sampleRate = 44100;
    deviceConfig.dataCallback = AudioRecorderClient::data_callback;
    deviceConfig.pUserData = this;

    // Initialize the recording device.
    if (ma_device_init(&context_, &deviceConfig, &device_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize recording device.");
      ma_context_uninit(&context_);
      throw std::runtime_error("Device init failed");
    }

    // Create action client for the AudioPlayer server.
    action_client_ = rclcpp_action::create_client<AudioPlaybackAction>(this, "play_audio");

    // Record and send audio after node initialization.
    record_and_send_audio();
  }

  ~AudioRecorderClient() {
    ma_device_stop(&device_);
    ma_device_uninit(&device_);
    ma_context_uninit(&context_);
  }

 private:
  // Records 3 seconds of audio and sends it as an action goal.
  void record_and_send_audio() {
    RCLCPP_INFO(this->get_logger(), "Recording 3 seconds of audio...");
    ma_device_start(&device_);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    ma_device_stop(&device_);

    RCLCPP_INFO(this->get_logger(), "Captured %zu samples.", recorded_samples_.size());

    // Prepare the action goal with recorded audio.
    auto goal_msg = AudioPlaybackAction::Goal();
    goal_msg.audio_data.data = recorded_samples_;

    // Wait for the action server to be available.
    if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available after 5 seconds.");
      return;
    }

    // Send the goal asynchronously with feedback and result callbacks.
    auto send_goal_options = rclcpp_action::Client<AudioPlaybackAction>::SendGoalOptions();
    send_goal_options.feedback_callback =
        std::bind(&AudioRecorderClient::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    send_goal_options.result_callback =
        std::bind(&AudioRecorderClient::result_callback, this, std::placeholders::_1);

    RCLCPP_INFO(this->get_logger(), "Sending audio goal with %zu samples.", recorded_samples_.size());
    action_client_->async_send_goal(goal_msg, send_goal_options);

    // Keep node alive to process callbacks (temporary solution; ideally use a timer or executor).
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }

  // Callback for recording audio data from miniaudio.
  static void data_callback(ma_device* pDevice, [[maybe_unused]] void* pOutput, const void* pInput,
                            ma_uint32 frameCount) {
    AudioRecorderClient* node = static_cast<AudioRecorderClient*>(pDevice->pUserData);
    if (!node || !pInput) return;

    const float* input = static_cast<const float*>(pInput);
    node->recorded_samples_.insert(node->recorded_samples_.end(), input, input + frameCount);
  }

  // Callback for receiving feedback from the action server.
  void feedback_callback(
      GoalHandleAudioPlaybackAction::SharedPtr,
      const std::shared_ptr<const AudioPlaybackAction::Feedback> feedback) {
    RCLCPP_INFO(this->get_logger(), "Feedback: Elapsed time %.2f s", feedback->elapsed_time);
  }

  // Callback for receiving the result from the action server.
  void result_callback(const GoalHandleAudioPlaybackAction::WrappedResult& result) {
    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        RCLCPP_INFO(this->get_logger(), "Result: Audio played for %.2f s", result.result->total_time);
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(this->get_logger(), "Goal was aborted.");
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_WARN(this->get_logger(), "Goal was canceled.");
        break;
      default:
        RCLCPP_ERROR(this->get_logger(), "Unknown result code.");
        break;
    }
    rclcpp::shutdown();  // Shutdown after receiving the result (optional).
  }

  ma_context context_;
  ma_device device_;
  std::vector<float> recorded_samples_;  // Buffer to store 3 seconds of audio.
  rclcpp_action::Client<AudioPlaybackAction>::SharedPtr action_client_;
};

// Main entry point: initializes ROS 2 and runs the AudioRecorderClient node.
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AudioRecorderClient>();
  rclcpp::spin(node);  // Spin to handle callbacks.
  rclcpp::shutdown();
  return 0;
}