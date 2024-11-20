from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from langgraph.graph import Graph

from democracy_exe.ai.agents.social_media_agent import SocialMediaAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class SocialMediaGraph(BaseGraph):
    """Graph for orchestrating social media content processing operations.

    This graph manages the flow of social media operations through multiple stages:
    content fetching, media processing (screenshots/videos), and formatting.
    It coordinates these operations using a directed graph structure where each
    node represents a specific processing stage.

    Attributes:
        social_media_agent: Instance of SocialMediaAgent that performs the actual
            social media operations
    """

    def __init__(self) -> None:
        """Initialize the social media graph with its agent."""
        super().__init__()
        self.social_media_agent = SocialMediaAgent()

    def build(self) -> Graph:
        """Construct the social media processing workflow graph.

        Creates a directed graph with nodes for each processing stage and edges
        defining the flow between stages. Includes conditional edges for
        different types of media content.

        Returns:
            Configured LangGraph Graph instance ready for execution
        """
        # Add social media agent node
        self.graph.add_node("process_media", AgentNode(self.social_media_agent))

        # Add nodes for different processing stages
        self.graph.add_node("fetch_content", self.fetch_content)
        self.graph.add_node("process_video", self.process_video)
        self.graph.add_node("process_image", self.process_image)
        self.graph.add_node("format_output", self.format_output)

        # Add conditional edges based on content type
        self.graph.add_conditional_edges(
            "fetch_content",
            self.determine_content_type,
            {
                "video": "process_video",
                "image": "process_image"
            }
        )

        # Add remaining edges
        self.graph.add_edge("process_media", "fetch_content")
        self.graph.add_edge("process_video", "format_output")
        self.graph.add_edge("process_image", "format_output")
        self.graph.add_edge("format_output", "process_media")

        # Set the entry point
        self.graph.set_entry_point("process_media")

        return self.graph

    def process(self, state: AgentState) -> AgentState:
        """Process a social media content request through the workflow.

        Args:
            state: Current agent state containing the content URL or data

        Returns:
            Updated agent state with processing results
        """
        compiled_graph = self.compile()
        return compiled_graph(state)

    def fetch_content(self, state: AgentState) -> AgentState:
        """Fetch content data from the provided URL.

        Args:
            state: Agent state containing the content URL

        Returns:
            Updated state with fetched content data
        """
        content_data = self.social_media_agent.fetch_tweet(state["url"])
        state["content_data"] = content_data
        return state

    def determine_content_type(self, state: AgentState) -> Literal["video", "image"]:
        """Determine the type of content to be processed.

        Args:
            state: Agent state containing the content data

        Returns:
            String indicating content type ("video" or "image")
        """
        is_video = self.social_media_agent.is_video_tweet(state["content_data"])
        return "video" if is_video else "image"

    async def process_video(self, state: AgentState) -> AgentState:
        """Process video content from the social media post.

        Args:
            state: Agent state containing the video content data

        Returns:
            Updated state with processed video data
        """
        video_path = await self.social_media_agent.download_video(state["url"])
        state["processed_content"] = video_path
        return state

    async def process_image(self, state: AgentState) -> AgentState:
        """Process image content from the social media post.

        Args:
            state: Agent state containing the image content data

        Returns:
            Updated state with processed image data
        """
        screenshot = await self.social_media_agent.take_screenshot(state["url"])
        state["processed_content"] = screenshot
        return state

    def format_output(self, state: AgentState) -> AgentState:
        """Format the processed content for final output.

        Args:
            state: Agent state containing the processed content

        Returns:
            Updated state with formatted content in the response field
        """
        content_type = self.determine_content_type(state)
        if content_type == "video":
            state["response"] = f"Video processed and saved to: {state['processed_content']}"
        else:
            state["response"] = f"Image processed and saved with size: {len(state['processed_content'])} bytes"
        return state


social_media_graph = SocialMediaGraph()


"""
This social_media_graph.py file defines the SocialMediaGraph class, which encapsulates the functionality for handling social media tasks, particularly focused on Twitter:
The SocialMediaGraph class inherits from BaseGraph and implements the build method.
It has a SocialMediaAgent instance to handle the actual social media operations.
The build method constructs the graph:
It adds the main social media agent node.
It adds nodes for different social media tasks: fetching tweets, taking screenshots, identifying important regions, cropping images, and downloading videos.
It adds edges to create the task flow, allowing for processing of both image and video tweets.
It includes a conditional edge that directs the flow to video download if the tweet contains a video.
It sets the entry point of the graph to the main social media node.
The process method compiles the graph and processes the given state.
Each method (fetch_tweet, take_screenshot, etc.) corresponds to a different task in processing social media content. They update the state with the results of each task.
The is_video_tweet method acts as a decision point, determining whether to process the tweet as an image or video.
An instance of SocialMediaGraph is created at the module level for easy access.
This implementation provides a structured approach to handling social media tasks within the AI system. It can process tweets, take screenshots, identify important regions in images, crop images to specific dimensions, and download videos. The conditional edge allows for different processing paths depending on whether the tweet contains an image or a video.
This design allows for flexible handling of different types of social media content, with the ability to easily extend or modify specific tasks as needed.
"""
