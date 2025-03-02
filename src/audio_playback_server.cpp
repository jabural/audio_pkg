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
#include <mutex>
#include <chrono>
#include <thread>
#include <memory>
#include <condition_variable>

// AudioPlayer is a ROS 2 action server that plays audio received as goals and provides playback time feedback.
// The playback device remains open continuously to minimize latency.
class AudioPlayer : public rclcpp::Node {
 public:
  using AudioPlaybackAction = audio_pkg::action::AudioPlaybackAction;
  using GoalHandleAudioPlaybackAction = rclcpp_action::ServerGoalHandle<AudioPlaybackAction>;

  // Constructs the AudioPlayer node and initializes the playback device.
  AudioPlayer() : Node("audio_player") {
    // Initialize the miniaudio context for audio device management.
    if (ma_context_init(NULL, 0, NULL, &context_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize miniaudio context.");
      throw std::runtime_error("Miniaudio context init failed");
    }

    // Configure the playback device to use mono audio at 16 kHz with 32-bit float samples.
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format = ma_format_f32;
    deviceConfig.playback.channels = 1;
    deviceConfig.sampleRate = 16000;
    deviceConfig.dataCallback = AudioPlayer::data_callback;
    deviceConfig.pUserData = this;

    // Initialize the playback device and start it immediately to keep it always open.
    if (ma_device_init(&context_, &deviceConfig, &device_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize playback device.");
      ma_context_uninit(&context_);
      throw std::runtime_error("Playback device init failed");
    }
    if (ma_device_start(&device_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to start playback device.");
      ma_device_uninit(&device_);
      ma_context_uninit(&context_);
      throw std::runtime_error("Playback start failed");
    }
    RCLCPP_INFO(this->get_logger(), "Playback device started and will remain open.");

    // Set up the action server to receive audio playback goals on the "play_audio" topic.
    action_server_ = rclcpp_action::create_server<AudioPlaybackAction>(
        this, "play_audio",
        std::bind(&AudioPlayer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&AudioPlayer::handle_cancel, this, std::placeholders::_1),
        std::bind(&AudioPlayer::handle_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "AudioPlayer action server started.");
  }

  // Cleans up miniaudio resources when the node is destroyed.
  ~AudioPlayer() {
    ma_device_stop(&device_);
    ma_device_uninit(&device_);
    ma_context_uninit(&context_);
  }

 private:
  // Handles incoming action goals, rejecting empty audio data.
  // Args:
  //   uuid: Unique identifier for the goal (unused).
  //   goal: The audio data to play.
  rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID& uuid,
      std::shared_ptr<const AudioPlaybackAction::Goal> goal) {
    (void)uuid;
    if (goal->audio_data.data.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Received empty audio data goal.");
      return rclcpp_action::GoalResponse::REJECT;
    }
    RCLCPP_INFO(this->get_logger(), "Received goal with %zu samples.", goal->audio_data.data.size());
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  // Handles goal cancellation by clearing the buffer and notifying completion.
  // Args:
  //   goal_handle: Handle to the goal being canceled.
  rclcpp_action::CancelResponse handle_cancel(
      const std::shared_ptr<GoalHandleAudioPlaybackAction> goal_handle) {
    RCLCPP_INFO(this->get_logger(), "Received cancel request.");
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    audio_buffer_.clear();
    playback_finished_ = true;
    finish_cv_.notify_one();
    goal_handle->canceled(make_result(0.0f));
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  // Executes an accepted goal by loading audio into the buffer and monitoring playback.
  // Args:
  //   goal_handle: Handle to the accepted goal.
  void handle_accepted(const std::shared_ptr<GoalHandleAudioPlaybackAction> goal_handle) {
    std::thread([this, goal_handle]() {
      const auto goal = goal_handle->get_goal();
      float total_samples, sample_rate, expected_duration;

      // Load the audio data into the buffer and calculate expected playback duration.
      {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        audio_buffer_ = goal->audio_data.data;
        playback_finished_ = false;
        RCLCPP_INFO(this->get_logger(), "Loaded %zu samples into buffer.", audio_buffer_.size());
        total_samples = static_cast<float>(audio_buffer_.size());
        sample_rate = 16000.0f;
        expected_duration = total_samples / sample_rate;
      }

      auto start_time = std::chrono::steady_clock::now();
      while (goal_handle->is_active()) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();

        // Send periodic feedback with the elapsed playback time.
        auto feedback = std::make_shared<AudioPlaybackAction::Feedback>();
        feedback->elapsed_time = elapsed;
        goal_handle->publish_feedback(feedback);

        // Wait for playback to complete or timeout, checking every 100 ms.
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        if (finish_cv_.wait_for(lock, std::chrono::milliseconds(100),
                                [this] { return playback_finished_ || audio_buffer_.empty(); })) {
          RCLCPP_INFO(this->get_logger(), "Playback finished: buffer empty=%d", audio_buffer_.empty());
          break;
        }

        // Exit if playback exceeds expected duration plus a 1-second buffer.
        if (elapsed > expected_duration + 1.0f) {
          RCLCPP_WARN(this->get_logger(), "Timeout after %.2f s, stopping playback.", elapsed);
          break;
        }
      }

      // Calculate total playback time and send the result if the goal is still active.
      float total_time = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
      if (goal_handle->is_active()) {
        goal_handle->succeed(make_result(total_time));
      } else {
        RCLCPP_INFO(this->get_logger(), "Goal not active, no result sent.");
      }
    }).detach();
  }

  // Creates a result message with the total playback time.
  // Args:
  //   total_time: Duration of playback in seconds.
  std::shared_ptr<AudioPlaybackAction::Result> make_result(float total_time) {
    auto result = std::make_shared<AudioPlaybackAction::Result>();
    result->total_time = total_time;
    RCLCPP_INFO(this->get_logger(), "Playback completed, total time: %.2f s", total_time);
    return result;
  }

  // Callback invoked by miniaudio to provide audio data for playback.
  // Args:
  //   pDevice: Pointer to the miniaudio device.
  //   pOutput: Buffer to fill with audio samples.
  //   pInput: Unused input data (nullptr for playback).
  //   frameCount: Number of frames requested by the audio hardware.
  static void data_callback(ma_device* pDevice, void* pOutput, [[maybe_unused]] const void* pInput,
                            ma_uint32 frameCount) {
    AudioPlayer* node = static_cast<AudioPlayer*>(pDevice->pUserData);
    float* output = static_cast<float*>(pOutput);

    // Copy available samples to the output buffer or fill with silence if none are available.
    std::lock_guard<std::mutex> lock(node->buffer_mutex_);
    ma_uint32 framesToCopy = std::min(frameCount, static_cast<ma_uint32>(node->audio_buffer_.size()));

    if (framesToCopy > 0) {
      std::copy(node->audio_buffer_.begin(), node->audio_buffer_.begin() + framesToCopy, output);
      node->audio_buffer_.erase(node->audio_buffer_.begin(), node->audio_buffer_.begin() + framesToCopy);
      RCLCPP_INFO(node->get_logger(), "Played %u samples, %zu remaining.", framesToCopy, node->audio_buffer_.size());
      if (node->audio_buffer_.empty()) {
        node->playback_finished_ = true;
        node->finish_cv_.notify_one();
      }
    } else {
      std::fill(output, output + frameCount, 0.0f);
      RCLCPP_DEBUG(node->get_logger(), "No samples, filling %u with silence.", frameCount);
    }
  }

  // Miniaudio context and device for continuous audio playback.
  ma_context context_;
  ma_device device_;

  // Buffer holding audio samples to be played, protected by a mutex.
  std::vector<float> audio_buffer_;
  std::mutex buffer_mutex_;

  // Flag indicating playback completion, synchronized with a condition variable.
  bool playback_finished_ = false;
  std::condition_variable finish_cv_;

  // Action server for receiving audio playback goals.
  rclcpp_action::Server<AudioPlaybackAction>::SharedPtr action_server_;
};

// Main entry point: initializes ROS 2, spins the AudioPlayer node, and shuts down.
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AudioPlayer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}