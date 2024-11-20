# democracy_exe/ai/agents/image_analysis_agent.py
from __future__ import annotations

from typing import List, Tuple

import torch

from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from democracy_exe.ai.base import AgentState, BaseAgent


class ImageAnalysisAgent(BaseAgent):
    """Agent for analyzing images using ResNet50 model.

    This agent performs image classification and object detection using a pre-trained
    ResNet50 model. It can process images and generate natural language descriptions
    of their contents.

    Attributes:
        device: The torch device (CPU/GPU) to run inference on
        model: Pre-trained ResNet50 model
        transform: Composition of image transformations
        classes: List of ImageNet class labels
    """
    def __init__(self):
        """Initialize the ImageAnalysisAgent with ResNet50 model and transforms."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.model.eval()
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with open('imagenet_classes.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]


    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess an image for model inference.

            Args:
                image_path: Path to the image file

            Returns:
                Preprocessed image tensor ready for model input
        """
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def classify_image(self, image_tensor: torch.Tensor) -> list[tuple[str, float]]:
        """Classify an image using the ResNet50 model.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            List of tuples containing (class_name, probability) for top 5 predictions
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        return [(self.classes[idx], prob.item()) for idx, prob in zip(top5_catid, top5_prob, strict=False)]

    def detect_objects(self, image_tensor: torch.Tensor) -> list[tuple[str, float]]:
        """Detect objects in the image using the model.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            List of tuples containing (object_class, confidence) for detected objects
        """
        return self.classify_image(image_tensor)  # Using classification as object detection for now

    def generate_description(self, classifications: list[tuple[str, float]]) -> str:
        """Generate a natural language description of image classifications.

        Args:
            classifications: List of (class_name, probability) tuples

        Returns:
            Formatted string describing the image contents
        """
        description = "This image appears to contain:\n"
        for cls, prob in classifications:
            description += f"- {cls} (confidence: {prob:.2%})\n"
        return description

    async def process(self, state: AgentState) -> AgentState:
        """Process an image and update the agent state with analysis results.

        Args:
            state: Current agent state containing image_path

        Returns:
            Updated agent state with analysis response

        Raises:
            Exception: If image processing fails
        """
        try:
            image_path = state["image_path"]
            image_tensor = self.preprocess_image(image_path)
            classifications = self.classify_image(image_tensor)
            description = self.generate_description(classifications)
            state["response"] = description
        except Exception as e:
            state["response"] = f"An error occurred while analyzing the image: {e!s}"
        return state

image_analysis_agent = ImageAnalysisAgent()
