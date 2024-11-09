from __future__ import annotations

from langgraph.graph import Graph

from democracy_exe.ai.agents.image_video_processing_agent import ImageVideoProcessingAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class ImageVideoProcessingGraph(BaseGraph):
    def __init__(self):
        super().__init__()
        self.processing_agent = ImageVideoProcessingAgent()

    def build(self) -> Graph:
        # Add processing agent node
        self.graph.add_node("process_media", AgentNode(self.processing_agent))

        # Add nodes for different processing tasks
        self.graph.add_node("crop_media", self.crop_media)
        self.graph.add_node("resize_media", self.resize_media)
        self.graph.add_node("apply_filters", self.apply_filters)
        self.graph.add_node("encode_media", self.encode_media)

        # Add edges to create the processing flow
        self.graph.add_edge("process_media", "crop_media")
        self.graph.add_edge("crop_media", "resize_media")
        self.graph.add_edge("resize_media", "apply_filters")
        self.graph.add_edge("apply_filters", "encode_media")
        self.graph.add_edge("encode_media", "process_media")

        # Set the entry point
        self.graph.set_entry_point("process_media")

        return self.graph

    def process(self, state: AgentState) -> AgentState:
        compiled_graph = self.compile()
        return compiled_graph(state)

    def crop_media(self, state: AgentState) -> AgentState:
        cropped = self.processing_agent.crop(state["media"], state.get("crop_params"))
        state["cropped_media"] = cropped
        return state

    def resize_media(self, state: AgentState) -> AgentState:
        resized = self.processing_agent.resize(state["cropped_media"], state.get("resize_params"))
        state["resized_media"] = resized
        return state

    def apply_filters(self, state: AgentState) -> AgentState:
        filtered = self.processing_agent.apply_filters(state["resized_media"], state.get("filter_params"))
        state["filtered_media"] = filtered
        return state

    def encode_media(self, state: AgentState) -> AgentState:
        encoded = self.processing_agent.encode(state["filtered_media"], state.get("encode_params"))
        state["response"] = encoded
        return state

image_video_processing_graph = ImageVideoProcessingGraph()


"""
This image_video_processing_graph.py file defines the ImageVideoProcessingGraph class, which encapsulates the functionality for processing images and videos:
The ImageVideoProcessingGraph class inherits from BaseGraph and implements the build method.
It has an ImageVideoProcessingAgent instance to handle the actual media processing operations.
The build method constructs the graph:
It adds the main processing agent node.
It adds nodes for different media processing tasks: cropping, resizing, applying filters, and encoding.
It adds edges to create a complete media processing flow.
It sets the entry point of the graph to the main processing node.
The process method compiles the graph and processes the given state.
Each method (crop_media, resize_media, etc.) corresponds to a different media processing task. They update the state with the results of each task.
An instance of ImageVideoProcessingGraph is created at the module level for easy access.
This implementation provides a structured approach to media processing within the AI system. The processing tasks are broken down into distinct stages, each handled by a separate node in the graph. This modular design makes it easy to modify or extend individual parts of the processing pipeline as needed.
The cyclic nature of the graph allows for iterative refinement of media processing if such functionality is desired in future applications.
"""
