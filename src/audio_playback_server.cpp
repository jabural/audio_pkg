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

class AudioPlayer : public rclcpp::Node {
 public:
  using AudioPlaybackAction = audio_pkg::action::AudioPlaybackAction;
  using GoalHandleAudioPlaybackAction = rclcpp_action::ServerGoalHandle<AudioPlaybackAction>;

  AudioPlayer() : Node("audio_player") {
    if (ma_context_init(NULL, 0, NULL, &context_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize miniaudio context.");
      throw std::runtime_error("Miniaudio context init failed");
    }

    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format = ma_format_f32;
    deviceConfig.playback.channels = 1;
    deviceConfig.sampleRate = 44100;
    deviceConfig.dataCallback = AudioPlayer::data_callback;
    deviceConfig.pUserData = this;

    if (ma_device_init(&context_, &deviceConfig, &device_) != MA_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize playback device.");
      ma_context_uninit(&context_);
      throw std::runtime_error("Playback device init failed");
    }

    action_server_ = rclcpp_action::create_server<AudioPlaybackAction>(
        this, "play_audio",
        std::bind(&AudioPlayer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&AudioPlayer::handle_cancel, this, std::placeholders::_1),
        std::bind(&AudioPlayer::handle_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "AudioPlayer action server started.");
  }

  ~AudioPlayer() {
    ma_device_stop(&device_);
    ma_device_uninit(&device_);
    ma_context_uninit(&context_);
  }

 private:
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

  rclcpp_action::CancelResponse handle_cancel(
      const std::shared_ptr<GoalHandleAudioPlaybackAction> goal_handle) {
    RCLCPP_INFO(this->get_logger(), "Received cancel request.");
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    audio_buffer_.clear();
    playback_finished_ = true;
    ma_device_stop(&device_);
    goal_handle->canceled(make_result(0.0f));
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleAudioPlaybackAction> goal_handle) {
    std::thread([this, goal_handle]() {
      const auto goal = goal_handle->get_goal();
      float total_samples, sample_rate, expected_duration;

      {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        audio_buffer_ = goal->audio_data.data;
        playback_finished_ = false;
        RCLCPP_INFO(this->get_logger(), "Loaded %zu samples into buffer.", audio_buffer_.size());
        total_samples = static_cast<float>(audio_buffer_.size());
        sample_rate = 44100.0f;
        expected_duration = total_samples / sample_rate;
      }

      if (!ma_device_is_started(&device_)) {
        if (ma_device_start(&device_) != MA_SUCCESS) {
          RCLCPP_ERROR(this->get_logger(), "Failed to start playback device.");
          goal_handle->abort(make_result(0.0f));
          return;
        }
        RCLCPP_INFO(this->get_logger(), "Playback device started.");
      }

      auto start_time = std::chrono::steady_clock::now();
      while (goal_handle->is_active()) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();

        {
          std::lock_guard<std::mutex> lock(buffer_mutex_);
          if (audio_buffer_.empty() || playback_finished_) {
            RCLCPP_INFO(this->get_logger(), "Playback finished: buffer empty=%d", audio_buffer_.empty());
            break;
          }
        }

        if (elapsed > expected_duration + 1.0f) {
          RCLCPP_WARN(this->get_logger(), "Timeout after %.2f s, stopping playback.", elapsed);
          break;
        }

        auto feedback = std::make_shared<AudioPlaybackAction::Feedback>();
        feedback->elapsed_time = elapsed;
        goal_handle->publish_feedback(feedback);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      ma_device_stop(&device_);
      RCLCPP_INFO(this->get_logger(), "Playback device stopped.");

      float total_time = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
      if (goal_handle->is_active()) {
        goal_handle->succeed(make_result(total_time));
      } else {
        RCLCPP_INFO(this->get_logger(), "Goal not active, no result sent.");
      }
    }).detach();
  }

  std::shared_ptr<AudioPlaybackAction::Result> make_result(float total_time) {
    auto result = std::make_shared<AudioPlaybackAction::Result>();
    result->total_time = total_time;
    RCLCPP_INFO(this->get_logger(), "Playback completed, total time: %.2f s", total_time);
    return result;
  }

  static void data_callback(ma_device* pDevice, void* pOutput, [[maybe_unused]] const void* pInput,
                            ma_uint32 frameCount) {
    AudioPlayer* node = static_cast<AudioPlayer*>(pDevice->pUserData);
    float* output = static_cast<float*>(pOutput);

    std::lock_guard<std::mutex> lock(node->buffer_mutex_);
    ma_uint32 framesToCopy = std::min(frameCount, static_cast<ma_uint32>(node->audio_buffer_.size()));

    if (framesToCopy > 0) {
      std::copy(node->audio_buffer_.begin(), node->audio_buffer_.begin() + framesToCopy, output);
      node->audio_buffer_.erase(node->audio_buffer_.begin(), node->audio_buffer_.begin() + framesToCopy);
      RCLCPP_INFO(node->get_logger(), "Played %u samples, %zu remaining.", framesToCopy, node->audio_buffer_.size());
      if (node->audio_buffer_.empty()) {
        node->playback_finished_ = true;
      }
    } else {
      std::fill(output, output + frameCount, 0.0f);
      RCLCPP_WARN(node->get_logger(), "Underrun: No samples available, filling %u with silence.", frameCount);
    }
  }

  ma_context context_;
  ma_device device_;
  std::vector<float> audio_buffer_;
  std::mutex buffer_mutex_;
  bool playback_finished_ = false;  // Flag to signal playback completion.
  rclcpp_action::Server<AudioPlaybackAction>::SharedPtr action_server_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AudioPlayer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}