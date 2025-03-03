#!/usr/bin/env python3

# Copyright 2025 Javier. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy  # ROS 2 Python client library for node creation and message handling
from rclpy.node import Node  # Base class for creating ROS 2 nodes
from std_msgs.msg import Float32MultiArray, Bool  # ROS 2 message types for audio data and VAD results
import torch  # PyTorch library for tensor operations and model loading
import torchaudio  # PyTorch audio processing library (not used here but imported)
import numpy as np  # NumPy for numerical operations (commented out in some places)
import time  # For timing operations like speech padding
from collections import deque  # Double-ended queue for managing audio buffer

# SileroVADNode detects voice activity in audio streams using Silero VAD or a fallback energy-based method.
class SileroVADNode(Node):
    """ROS 2 node for voice activity detection (VAD) on audio data."""
    def __init__(self):
        # Initialize the ROS 2 node with the name 'silero_vad_node'
        super().__init__('silero_vad_node')

        # Declare configurable parameters with default values for flexibility
        self.declare_parameter('sample_rate', 16000)  # Audio sample rate in Hz
        self.declare_parameter('threshold', 0.3)  # VAD probability threshold for speech detection
        self.declare_parameter('window_size_ms', 96)  # Size of audio window in milliseconds
        self.declare_parameter('speech_pad_ms', 500)  # Padding duration in ms to extend speech detection

        # Retrieve the parameter values from ROS 2 parameter server
        self.sample_rate = self.get_parameter('sample_rate').value
        self.threshold = self.get_parameter('threshold').value
        self.window_size_ms = self.get_parameter('window_size_ms').value
        self.speech_pad_ms = self.get_parameter('speech_pad_ms').value

        # Convert window size from milliseconds to number of samples based on sample rate
        self.window_size_samples = int(self.window_size_ms * self.sample_rate / 1000)

        # Initialize a fixed-size buffer (deque) to store audio chunks, with maxlen limiting its size
        self.audio_buffer = deque(maxlen=self.window_size_samples)

        # Set up the VAD model (Silero or fallback) during initialization
        self.setup_vad_model()

        # Variables to track speech state and timing for padding logic
        self.is_speech = False  # Flag indicating if speech is currently detected
        self.last_speech_time = 0  # Timestamp of the last detected speech
        self.last_msg = False  # Track the last published VAD state to avoid redundant messages

        # Create a subscription to receive audio data from the 'audio_data' topic
        self.audio_subscription = self.create_subscription(
            Float32MultiArray,  # Message type for incoming audio data
            'audio_data',  # Topic name to subscribe to
            self.audio_callback,  # Callback function to process audio messages
            10  # QoS (Quality of Service) depth, buffer size for incoming messages
        )

        # Create a publisher to send VAD results to the 'voice_activity' topic
        self.vad_publisher = self.create_publisher(Bool, 'voice_activity', 10)

        # Log a message to indicate successful initialization
        self.get_logger().info('Silero VAD Node initialized')


    def setup_vad_model(self):
        """Loads the Silero VAD model from snakers4/silero-vad or falls back to energy-based VAD."""
        # Log the start of the VAD model loading process
        self.get_logger().info('Loading Silero VAD model...')

        # Choose the device (CUDA GPU or CPU) based on CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        try:
            # Attempt to load the Silero VAD model from the GitHub repository
            self.get_logger().info('Loading Silero VAD from snakers4/silero-vad...')
            model, utils = torch.hub.load(
                'snakers4/silero-vad',  # Repository name for Silero VAD
                'silero_vad',  # Model name to load
                source='github',  # Source of the model (GitHub)
                force_reload=False  # Use cached model if available, no forced redownload
            )
            # Move the loaded model to the selected device (GPU or CPU)
            self.model = model.to(self.device)

            # Log successful loading of the Silero VAD model
            self.get_logger().info('Successfully loaded Silero VAD')

        except Exception as e:
            # Log any error that occurs during model loading
            self.get_logger().error(f'Failed to load Silero VAD: {str(e)}')
            self.get_logger().error('Falling back to simple energy-based VAD...')
            # Log the switch to the fallback method (though no explicit assignment is made here)
            self.get_logger().info('Using simple energy-based VAD')

    def energy_vad_iterator(self, audio_tensor):
        """Implements a simple energy-based VAD as a fallback when Silero fails."""
        # Calculate the root mean square (RMS) energy of the audio chunk
        energy = torch.sqrt(torch.mean(audio_tensor**2))
        # Return True if energy exceeds a threshold (0.1), indicating speech
        return energy > 0.1  # Threshold may need tuning based on audio characteristics

    def audio_callback(self, msg):
        """Handles incoming audio messages and publishes VAD results."""
        # Convert the incoming ROS message data (Float32MultiArray) to a PyTorch tensor
        audio_tensor = torch.tensor(msg.data, dtype=torch.float32, device=self.device)

        # Define the chunk size (512 samples) for processing audio in segments
        chunk_size = 512
        num_samples = audio_tensor.shape[0]  # Total number of samples in the audio tensor
        speech_detected = False  # Flag to indicate if speech is detected in any chunk

        # Process the audio tensor in chunks of 512 samples
        for i in range(0, num_samples, chunk_size):
            chunk = audio_tensor[i:i + chunk_size]  # Extract a chunk of audio
            if chunk.shape[0] == chunk_size:  # Only process chunks that are full-sized
                speech_detected = self.process_audio(chunk)  # Check for speech in the chunk
                if speech_detected:  # If speech is detected, stop processing further chunks
                    break

        # Publish the VAD result only if it differs from the last published state
        if speech_detected != self.last_msg:
            vad_msg = Bool()  # Create a Bool message for the VAD result
            vad_msg.data = speech_detected  # Set the message data to True or False
            self.vad_publisher.publish(vad_msg)  # Publish the result to the 'voice_activity' topic
            self.last_msg = speech_detected  # Update the last published state

    def process_audio(self, audio_tensor):
        """Detects speech in an audio chunk using the configured VAD method."""
        try:
            # Process the audio tensor with the Silero VAD model without computing gradients
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()  # Get speech probability

            # Implement speech padding logic to extend detection beyond brief silences
            current_time = time.time()  # Get the current timestamp
            if speech_prob > self.threshold:  # If speech probability exceeds threshold
                if not self.is_speech:  # If this is a new speech segment
                    self.get_logger().info('Speech detected')
                self.is_speech = True  # Mark as speech active
                self.last_speech_time = current_time  # Update the last speech timestamp
            elif self.is_speech and (current_time - self.last_speech_time) * 1000 > self.speech_pad_ms:
                # If speech was active but padding duration has elapsed, mark it as ended
                self.is_speech = False
                self.get_logger().info('Speech ended')

            # Return True if speech probability exceeds 0.5
            return self.is_speech

        except Exception as e:
            # Log any processing errors and return False (no speech detected)
            self.get_logger().error(f'Error in audio processing: {str(e)}')
            return False

def main(args=None):
    """Initializes and runs the Silero VAD node."""
    rclpy.init(args=args)  # Initialize the ROS 2 Python client library
    node = SileroVADNode()  # Create an instance of the SileroVADNode

    try:
        rclpy.spin(node)  # Keep the node running to process messages
    except KeyboardInterrupt:
        # Handle manual shutdown (e.g., Ctrl+C)
        node.get_logger().info('Shutting down Silero VAD node...')
    finally:
        # Ensure proper cleanup on shutdown
        if rclpy.ok():  # Check if ROS 2 context is still valid
            node.destroy_node()  # Destroy the node
            rclpy.shutdown()  # Shutdown the ROS 2 client library

if __name__ == '__main__':
    # Entry point: run the main function if this script is executed directly
    main()