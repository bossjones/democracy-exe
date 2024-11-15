from __future__ import annotations

from typing import Any

from langgraph.graph import Graph

from democracy_exe.ai.agents.image_video_processing_agent import ImageVideoProcessingAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class ImageVideoProcessingGraph(BaseGraph):
    """Graph for orchestrating image and video processing workflows.

    This graph manages the flow of media processing operations through multiple stages:
    cropping, resizing, filter application, and encoding. It coordinates these operations
    using a directed graph structure where each node represents a specific processing stage.

    Attributes:
        processing_agent: Instance of ImageVideoProcessingAgent that performs the actual
            processing operations
    """

    def __init__(self) -> None:
        """Initialize the image/video processing graph with its agent."""
        super().__init__()
        self.processing_agent = ImageVideoProcessingAgent()

    def build(self) -> Graph:
        """Construct the media processing workflow graph.

        Creates a directed graph with nodes for each processing stage and edges
        defining the flow between stages.

        Returns:
            Configured LangGraph Graph instance ready for execution
        """
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
        """Process media through the processing workflow.

        Args:
            state: Current agent state containing the media to process

        Returns:
            Updated agent state with processing results
        """
        compiled_graph = self.compile()
        return compiled_graph(state)

    def crop_media(self, state: AgentState) -> AgentState:
        """Crop the input media according to specified parameters.

        Args:
            state: Agent state containing the media and optional crop_params

        Returns:
            Updated state with cropped media
        """
        cropped = self.processing_agent.crop(state["media"], state.get("crop_params"))
        state["cropped_media"] = cropped
        return state

    def resize_media(self, state: AgentState) -> AgentState:
        """Resize the cropped media according to specified parameters.

        Args:
            state: Agent state containing the cropped media and optional resize_params

        Returns:
            Updated state with resized media
        """
        resized = self.processing_agent.resize(state["cropped_media"], state.get("resize_params"))
        state["resized_media"] = resized
        return state

    def apply_filters(self, state: AgentState) -> AgentState:
        """Apply specified filters to the resized media.

        Args:
            state: Agent state containing the resized media and optional filter_params

        Returns:
            Updated state with filtered media
        """
        filtered = self.processing_agent.apply_filters(state["resized_media"], state.get("filter_params"))
        state["filtered_media"] = filtered
        return state

    def encode_media(self, state: AgentState) -> AgentState:
        """Encode the processed media according to specified parameters.

        Args:
            state: Agent state containing the filtered media and optional encode_params

        Returns:
            Updated state with encoded media in the response field
        """
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
