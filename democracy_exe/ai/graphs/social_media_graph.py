from __future__ import annotations

from langgraph.graph import Graph

from democracy_exe.ai.agents.social_media_agent import SocialMediaAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class SocialMediaGraph(BaseGraph):
    def __init__(self):
        super().__init__()
        self.social_media_agent = SocialMediaAgent()

    def build(self) -> Graph:
        # Add social media agent node
        self.graph.add_node("social_media", AgentNode(self.social_media_agent))

        # Add nodes for different social media tasks
        self.graph.add_node("fetch_tweet", self.fetch_tweet)
        self.graph.add_node("take_screenshot", self.take_screenshot)
        self.graph.add_node("identify_regions", self.identify_regions)
        self.graph.add_node("crop_image", self.crop_image)
        self.graph.add_node("download_video", self.download_video)

        # Add edges to create the social media task flow
        self.graph.add_edge("social_media", "fetch_tweet")
        self.graph.add_edge("fetch_tweet", "take_screenshot")
        self.graph.add_edge("take_screenshot", "identify_regions")
        self.graph.add_edge("identify_regions", "crop_image")
        self.graph.add_edge("crop_image", "social_media")

        # Add conditional edge for video download
        self.graph.add_conditional_edge("fetch_tweet", self.is_video_tweet,
                                        {True: "download_video", False: "take_screenshot"})
        self.graph.add_edge("download_video", "social_media")

        # Set the entry point
        self.graph.set_entry_point("social_media")

        return self.graph

    def process(self, state: AgentState) -> AgentState:
        compiled_graph = self.compile()
        return compiled_graph(state)

    def fetch_tweet(self, state: AgentState) -> AgentState:
        tweet_data = self.social_media_agent.fetch_tweet(state["tweet_url"])
        state["tweet_data"] = tweet_data
        return state

    def is_video_tweet(self, state: AgentState) -> bool:
        return self.social_media_agent.is_video_tweet(state["tweet_data"])

    def take_screenshot(self, state: AgentState) -> AgentState:
        screenshot = self.social_media_agent.take_screenshot(state["tweet_data"])
        state["screenshot"] = screenshot
        return state

    def identify_regions(self, state: AgentState) -> AgentState:
        regions = self.social_media_agent.identify_regions(state["screenshot"])
        state["important_regions"] = regions
        return state

    def crop_image(self, state: AgentState) -> AgentState:
        cropped_image = self.social_media_agent.crop_image(
            state["screenshot"],
            state["important_regions"],
            aspect_ratio=(1, 1),
            target_size=(1080, 1350)
        )
        state["response"] = cropped_image
        return state

    def download_video(self, state: AgentState) -> AgentState:
        video_path = self.social_media_agent.download_video(state["tweet_data"])
        state["response"] = video_path
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
