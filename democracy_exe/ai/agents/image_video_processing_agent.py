# democracy_exe/ai/agents/image_video_processing_agent.py
from __future__ import annotations

import os

from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from PIL import Image

from democracy_exe.ai.base import AgentState, BaseAgent


class ImageVideoProcessingAgent(BaseAgent):
    """Agent for processing image and video files.

    Handles operations like cropping, resizing, filtering, and encoding of media files.
    Supports common image formats (PNG, JPG, GIF) and video formats (MP4, AVI, MOV).
    """

    def crop(self, media: str | np.ndarray | Image.Image, params: dict[str, Any] | None = None) -> str | np.ndarray | Image.Image:
        """Crop the media according to specified parameters.

        Args:
            media: Input media file path or loaded media object
            params: Dictionary of cropping parameters

        Returns:
            Cropped media object or path to processed file
        """
        return self.process_image(media) if isinstance(media, str) else media

    def resize(self, media: str | np.ndarray | Image.Image, params: dict[str, Any] | None = None) -> str | np.ndarray | Image.Image:
        """Resize the media according to specified parameters.

        Args:
            media: Input media file path or loaded media object
            params: Dictionary of resizing parameters

        Returns:
            Resized media object or path to processed file
        """
        return media

    def apply_filters(self, media: str | np.ndarray | Image.Image, params: dict[str, Any] | None = None) -> str | np.ndarray | Image.Image:
        """Apply filters to the media according to specified parameters.

        Args:
            media: Input media file path or loaded media object
            params: Dictionary of filter parameters

        Returns:
            Filtered media object or path to processed file
        """
        return media

    def encode(self, media: str | np.ndarray | Image.Image, params: dict[str, Any] | None = None) -> str | np.ndarray | Image.Image:
        """Encode the media according to specified parameters.

        Args:
            media: Input media file path or loaded media object
            params: Dictionary of encoding parameters

        Returns:
            Encoded media object or path to processed file
        """
        return media

    def process_image(self, image_path: str, target_size: tuple[int, int] = (1080, 1350)) -> str:
        """Process an image file by resizing and cropping.

        Args:
            image_path: Path to the input image file
            target_size: Desired output dimensions (width, height)

        Returns:
            Path to the processed image file
        """
        img = Image.open(image_path)
        img_resized = self.resize_and_crop(img, target_size)
        output_path = f"processed_{os.path.basename(image_path)}"
        img_resized.save(output_path)
        return output_path

    def process_video(self, video_path: str, target_size: tuple[int, int] = (1080, 1350)) -> str:
        """Process a video file by resizing and cropping each frame.

        Args:
            video_path: Path to the input video file
            target_size: Desired output dimensions (width, height)

        Returns:
            Path to the processed video file
        """
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f"processed_{os.path.basename(video_path)}"
        out = cv2.VideoWriter(output_path, fourcc, 30.0, target_size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = self.resize_and_crop_cv2(frame, target_size)
            out.write(frame_resized)

        cap.release()
        out.release()
        return output_path

    def resize_and_crop(self, img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        """Resize and crop a PIL Image to target dimensions while maintaining aspect ratio.

        Args:
            img: Input PIL Image object
            target_size: Desired output dimensions (width, height)

        Returns:
            Processed PIL Image object
        """
        img_ratio = img.width / img.height
        target_ratio = target_size[0] / target_size[1]

        if img_ratio > target_ratio:
            # Image is wider than target, crop width
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img_cropped = img.crop((left, 0, left + new_width, img.height))
        else:
            # Image is taller than target, crop height
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img_cropped = img.crop((0, top, img.width, top + new_height))

        return img_cropped.resize(target_size, Image.Resampling.LANCZOS)

    def resize_and_crop_cv2(self, img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize and crop an OpenCV image to target dimensions while maintaining aspect ratio.

        Args:
            img: Input OpenCV image array
            target_size: Desired output dimensions (width, height)

        Returns:
            Processed OpenCV image array
        """
        img_ratio = img.shape[1] / img.shape[0]
        target_ratio = target_size[0] / target_size[1]

        if img_ratio > target_ratio:
            # Image is wider than target, crop width
            new_width = int(img.shape[0] * target_ratio)
            left = (img.shape[1] - new_width) // 2
            img_cropped = img[:, left:left + new_width]
        else:
            # Image is taller than target, crop height
            new_height = int(img.shape[1] / target_ratio)
            top = (img.shape[0] - new_height) // 2
            img_cropped = img[top:top + new_height, :]

        return cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_LANCZOS4)

    async def process(self, state: AgentState) -> AgentState:
        """Process a media file based on the agent state.

        Args:
            state: Current agent state containing file path and processing parameters

        Returns:
            Updated agent state with processing results or error message
        """
        try:
            file_path = state["file_path"]
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                processed_path = self.process_image(file_path)
                state["response"] = f"Image processed and saved to: {processed_path}"
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                processed_path = self.process_video(file_path)
                state["response"] = f"Video processed and saved to: {processed_path}"
            else:
                state["response"] = "Unsupported file type"
        except Exception as e:
            state["response"] = f"An error occurred while processing the file: {e!s}"
        return state

image_video_processing_agent = ImageVideoProcessingAgent()
