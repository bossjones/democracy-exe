Hi I have a Langgraph Studio + Langsmith python module question  about this change to include multiprocessing in _background_thread.py https://github.com/langchain-ai/langsmith-sdk/blob/0243f793c7a5f7d61c3d2c6bd8fcf85c85933f96/python/langsmith/_internal/_background_thread.py

I was recently playing with a demo repo to onboard it to langgraph studio w/ a proper langgraph.json w/ correct dockerfile_lines config etc, I noticed that when I tried asking my graph questions, the traces started segfaulting, eg:

```
langgraph-api-1       | warning | /api/langgraph_api/api/assistants.py:36: PydanticDeprecatedSince20: The `schema` method is deprecated; use `model_json_schema` instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/

langgraph-api-1       | warning | /api/langgraph_api/api/assistants.py:219: PydanticDeprecatedSince20: The `__fields__` attribute is deprecated, use `model_fields` instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/

langgraph-api-1       | warning | /api/langgraph_api/api/assistants.py:218: PydanticDeprecatedSince20: The `__fields__` attribute is deprecated, use `model_fields` instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/

langgraph-api-1       | warning | /api/langgraph_api/api/assistants.py:218: PydanticDeprecatedSince20: The `schema` method is deprecated; use `model_json_schema` instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/

langgraph-api-1       | info | GET /assistants/166c0d22-706c-5a5b-9027-ea37f7308a85/schemas 200 96ms
langgraph-api-1       | info | GET /assistants/166c0d22-706c-5a5b-9027-ea37f7308a85/subgraphs 200 96ms
langgraph-api-1       | info | GET /assistants/166c0d22-706c-5a5b-9027-ea37f7308a85/graph 200 10ms
langgraph-api-1       | info | HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
langgraph-api-1       | info | HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
langgraph-api-1       | info | HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
langgraph-api-1       | [1:writes] Finished step 1 with writes to 1 channel:
langgraph-api-1       | - messages -> [AIMessage(content="Hello! I'm here and ready to help you with your ToDo list. How can I assist you today?", additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782'}, id='run-cfd118f1-3f24-4cec-bf51-d3536e3324db')]
langgraph-api-1       | [1:checkpoint] State at the end of step 1:
langgraph-api-1       | {'messages': [HumanMessage(content='hi how are you doing today?\n', additional_kwargs={}, response_metadata={}, id='0ec25255-39e9-4180-a644-14fe80e0515b'),
langgraph-api-1       |               AIMessage(content="Hello! I'm here and ready to help you with your ToDo list. How can I assist you today?", additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782'}, id='run-cfd118f1-3f24-4cec-bf51-d3536e3324db')]}
langgraph-api-1       | info | Background run succeeded
langgraph-api-1       | [1:writes] Finished step 1 with writes to 1 channel:
langgraph-api-1       | - messages -> [AIMessage(content="Hello! I'm here and ready to help you with your ToDo list. How can I assist you today?", additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782'}, id='run-cfd118f1-3f24-4cec-bf51-d3536e3324db')]
langgraph-api-1       | [1:checkpoint] State at the end of step 1:
langgraph-api-1       | {'messages': [HumanMessage(content='hi how are you doing today?\n', additional_kwargs={}, response_metadata={}, id='0ec25255-39e9-4180-a644-14fe80e0515b'),
langgraph-api-1       |               AIMessage(content="Hello! I'm here and ready to help you with your ToDo list. How can I assist you today?", additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782'}, id='run-cfd118f1-3f24-4cec-bf51-d3536e3324db')]}
langgraph-api-1       | info | Background run succeeded
langgraph-api-1       | [1:writes] Finished step 1 with writes to 1 channel:
langgraph-api-1       | - messages -> [AIMessage(content="Hello! I'm here and ready to help you with your ToDo list. How can I assist you today?", additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782'}, id='run-cfd118f1-3f24-4cec-bf51-d3536e3324db')]
langgraph-api-1       | [1:checkpoint] State at the end of step 1:
langgraph-api-1       | {'messages': [HumanMessage(content='hi how are you doing today?\n', additional_kwargs={}, response_metadata={}, id='0ec25255-39e9-4180-a644-14fe80e0515b'),
langgraph-api-1       |               AIMessage(content="Hello! I'm here and ready to help you with your ToDo list. How can I assist you today?", additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_d28bcae782'}, id='run-cfd118f1-3f24-4cec-bf51-d3536e3324db')]}
langgraph-api-1       | info | Background run succeeded
langgraph-api-1       | Fatal Python error: Segmentation fault
langgraph-api-1       |
langgraph-api-1       | Thread 0x0000ffff5740f180 (most recent call first):
langgraph-api-1       |   File "/usr/local/lib/python3.12/ssl.py", line 1103 in read
langgraph-api-1       |   File "/usr/local/lib/python3.12/ssl.py", line 1251 in recv_into
langgraph-api-1       |   File "/usr/local/lib/python3.12/socket.py", line 720 in readinto
langgraph-api-1       |   File "/usr/local/lib/python3.12/http/client.py", line 292 in _read_status
langgraph-api-1       |   File "/usr/local/lib/python3.12/http/client.py", line 331 in begin
langgraph-api-1       |   File "/usr/local/lib/python3.12/http/client.py", line 1428 in getresponse
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/urllib3/connection.py", line 516 in getresponse
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/urllib3/connectionpool.py", line 534 in _make_request
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/urllib3/connectionpool.py", line 787 in urlopen
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/requests/adapters.py", line 667 in send
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/requests/sessions.py", line 703 in send
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/requests/sessions.py", line 589 in request
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/langsmith/client.py", line 749 in request_with_retries
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/langsmith/client.py", line 1783 in _send_multipart_req
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/langsmith/client.py", line 1609 in _multipart_ingest_ops
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/langsmith/_internal/_background_thread.py", line 145 in _tracing_thread_handle_batch
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/langsmith/_internal/_background_thread.py", line 256 in tracing_control_thread_func
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1012 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
langgraph-api-1       |
langgraph-api-1       | Thread 0x0000ffff5d20f180 (most recent call first):
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 355 in wait
langgraph-api-1       |   File "/usr/local/lib/python3.12/queue.py", line 171 in get
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 994 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
langgraph-api-1       |
langgraph-api-1       | Thread 0x0000ffff57e0f180 (most recent call first):
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 355 in wait
langgraph-api-1       |   File "/usr/local/lib/python3.12/queue.py", line 171 in get
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 994 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
langgraph-api-1       |
langgraph-api-1       | Thread 0x0000ffff8340f180 (most recent call first):
langgraph-api-1       |   File "/usr/local/lib/python3.12/concurrent/futures/thread.py", line 90 in _worker
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1012 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
langgraph-api-1       |
langgraph-api-1       | Thread 0x0000ffff83e0f180 (most recent call first):
langgraph-api-1       |   File "/usr/local/lib/python3.12/concurrent/futures/thread.py", line 90 in _worker
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1012 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
langgraph-api-1       |   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
langgraph-api-1       |
langgraph-api-1       | Current thread 0x0000ffff92f6d020 (most recent call first):
langgraph-api-1       |   File "/usr/local/lib/python3.12/traceback.py", line 336 in walk_stack
langgraph-api-1       |   File "/usr/local/lib/python3.12/traceback.py", line 392 in extended_frame_gen
langgraph-api-1       |   File "/usr/local/lib/python3.12/traceback.py", line 418 in _extract_from_extended_frame_gen
langgraph-api-1       |   File "/usr/local/lib/python3.12/traceback.py", line 395 in extract
langgraph-api-1       |   File "<shim>", line ??? in <interpreter trampoline>
langgraph-api-1       |   File "/api/langgraph_api/sse.py", line 59 in stream_response
langgraph-api-1       |   File "/api/langgraph_api/sse.py", line 31 in wrap
langgraph-api-1       |   File "/usr/local/lib/python3.12/asyncio/runners.py", line 118 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/asyncio/runners.py", line 194 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/uvicorn/server.py", line 66 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/site-packages/uvicorn/_subprocess.py", line 80 in subprocess_started
langgraph-api-1       |   File "/usr/local/lib/python3.12/multiprocessing/process.py", line 108 in run
langgraph-api-1       |   File "/usr/local/lib/python3.12/multiprocessing/process.py", line 314 in _bootstrap
langgraph-api-1       |   File "/usr/local/lib/python3.12/multiprocessing/spawn.py", line 135 in _main
langgraph-api-1       |   File "/usr/local/lib/python3.12/multiprocessing/spawn.py", line 122 in spawn_main
langgraph-api-1       |   File "<string>", line 1 in <module>
```

I started digging into it a bit more and it seems as though it's an issue my loguru and the multiprocessing module when using the equeue=True option to log things from async/threaded things in python. https://github.com/Delgan/loguru/issues/916 and https://github.com/Delgan/loguru/issues/912

Help me come with a solution that will allow everything to work properly with loguru.

More context you can use:

https://loguru.readthedocs.io/en/stable/resources/recipes.html#compatibility-with-multiprocessing-using-enqueue-argument

https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

Think about your answer then walk me through your thought process step by step.

you'll need to look at democracy_exe/bot_logger/__init__.py to see how the InterceptHandler is configured and how global_log_config is defined.
