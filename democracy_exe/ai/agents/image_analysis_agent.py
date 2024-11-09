# democracy_exe/ai/agents/image_analysis_agent.py
from __future__ import annotations

from typing import List, Tuple

import torch

from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from democracy_exe.ai.base import AgentState, BaseAgent


class ImageAnalysisAgent(BaseAgent):
    def __init__(self):
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
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def classify_image(self, image_tensor: torch.Tensor) -> list[tuple[str, float]]:
        with torch.no_grad():
            outputs = self.model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        return [(self.classes[idx], prob.item()) for idx, prob in zip(top5_catid, top5_prob, strict=False)]

    def generate_description(self, classifications: list[tuple[str, float]]) -> str:
        description = "This image appears to contain:\n"
        for cls, prob in classifications:
            description += f"- {cls} (confidence: {prob:.2%})\n"
        return description

    async def process(self, state: AgentState) -> AgentState:
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
