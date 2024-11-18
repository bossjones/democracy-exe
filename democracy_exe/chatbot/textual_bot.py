# from textual.app import App
# from textual.widgets import Input
# from textual.scroll_view import ScrollView

# class ChatApp(App):
#     async def on_mount(self):
#         self.chat_history = ScrollView()
#         self.input = Input()
#         await self.view.dock(self.chat_history, edge="top", size=20)
#         await self.view.dock(self.input, edge="bottom")

#     async def on_input_submitted(self, message: str):
#         # Add user message to chat history
#         self.chat_history.write(f"You: {message}\n")

#         # Send message to langgraph model and get response
#         response = await self.langgraph_model.send(message)

#         # Add model response to chat history
#         self.chat_history.write(f"Model: {response}\n")

#         # Clear input field
#         self.input.value = ""

# if __name__ == "__main__":
#     app = ChatApp()
#     app.run()
