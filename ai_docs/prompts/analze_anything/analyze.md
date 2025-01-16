# Analyze anything

@structlog_expert.md  Find quotes from the the structlog docs that are relevant to writing quality tests using pytest for python modules that use structlog. Place these in <quotes> tags. Then, based on these quotes, list all information that would help the a programer adhere to these standards. Place your diagnostic information in <info> tags. Once you are done, we will make modifications to @test_logsetup.py @logsetup.py  to include these details. @Web

# fix testing using source of truth


@structlog_expert.md  Find quotes from the the structlog docs that are relevant to fixing my test errors that use structlog. Place these in <quotes> tags. Then, based on these quotes, list all information that would help the a programer fix these issues. Place your diagnostic information in <info> tags. Once you are done, we will make modifications to @test_logsetup.py @logsetup.py to include these details. @Web



@democracy_exe_clients.xml

```
# error message


2025-01-10 09:07:11 Fatal Python error: Segmentation fault
2025-01-10 09:07:11
2025-01-10 09:07:11 Thread 0x0000ffff5f40f180 (most recent call first):
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/concurrent/futures/thread.py", line 90 in _worker
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1012 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
2025-01-10 09:07:11
2025-01-10 09:07:11 Thread 0x0000ffff5fe0f180 (most recent call first):
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/ssl.py", line 1103 in read
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/ssl.py", line 1251 in recv_into
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/socket.py", line 720 in readinto
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/http/client.py", line 292 in _read_status
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/http/client.py", line 331 in begin
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/http/client.py", line 1428 in getresponse
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/urllib3/connectionpool.py", line 463 in _make_request
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/urllib3/connectionpool.py", line 716 in urlopen
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/requests/adapters.py", line 667 in send
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/requests/sessions.py", line 703 in send
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/requests/sessions.py", line 589 in request
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/langsmith/client.py", line 753 in request_with_retries
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/langsmith/client.py", line 1622 in _send_multipart_req
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/langsmith/client.py", line 1512 in _multipart_ingest_ops
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/langsmith/_internal/_background_thread.py", line 100 in _tracing_thread_handle_batch
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/langsmith/_internal/_background_thread.py", line 193 in tracing_control_thread_func
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1012 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
2025-01-10 09:07:11
2025-01-10 09:07:11 Thread 0x0000ffff6540f180 (most recent call first):
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 355 in wait
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/queue.py", line 171 in get
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 951 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
2025-01-10 09:07:11
2025-01-10 09:07:11 Thread 0x0000ffff64a0f180 (most recent call first):
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 355 in wait
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/queue.py", line 171 in get
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 951 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
2025-01-10 09:07:11
2025-01-10 09:07:11 Thread 0x0000ffff8be0f180 (most recent call first):
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/concurrent/futures/thread.py", line 90 in _worker
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1012 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
2025-01-10 09:07:11
2025-01-10 09:07:11 Thread 0x0000ffff90a0f180 (most recent call first):
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/concurrent/futures/thread.py", line 90 in _worker
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1012 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1075 in _bootstrap_inner
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/threading.py", line 1032 in _bootstrap
2025-01-10 09:07:11
2025-01-10 09:07:11 Current thread 0x0000ffff9b1c4020 (most recent call first):
2025-01-10 09:07:11   File "/api/langgraph_api/sse.py", line 59 in stream_response
2025-01-10 09:07:11   File "/api/langgraph_api/sse.py", line 31 in wrap
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/asyncio/runners.py", line 118 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/asyncio/runners.py", line 194 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/uvicorn/server.py", line 66 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/site-packages/uvicorn/_subprocess.py", line 80 in subprocess_started
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/multiprocessing/process.py", line 108 in run
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/multiprocessing/process.py", line 314 in _bootstrap
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/multiprocessing/spawn.py", line 135 in _main
2025-01-10 09:07:11   File "/usr/local/lib/python3.12/multiprocessing/spawn.py", line 122 in spawn_main
2025-01-10 09:07:11   File "<string>", line 1 in <module>
2025-01-10 09:07:11
2025-01-10 09:07:11 Extension modules: greenlet._greenlet, _brotli, tornado.speedups, msgpack._cmsgpack, charset_normalizer.md, requests.packages.charset_normalizer.md, requests.packages.chardet.md, _cffi_backend, uvloop.loop, httptools.parser.parser, httptools.parser.url_parser, websockets.speedups, psutil._psutil_linux, psutil._psutil_posix, psycopg_binary.pq, psycopg_binary._psycopg, regex._regex, yaml._yaml, multidict._multidict, yarl._quoting_c, propcache._helpers_c, aiohttp._http_writer, aiohttp._http_parser, aiohttp._websocket.mask, aiohttp._websocket.reader_c, frozenlist._frozenlist, numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.fft._pocketfft_internal, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator, PIL._imaging (total: 40)
```

Find quotes from the democracy_exe python docs and files that are relevant to fixing this Fatal Python error: Segmentation fault error. Place these in <quotes> tags. Then, based on these quotes, list all information that would help the a programer fix these issues. Place your diagnostic information in <info> tags. Once you are done, we will write these guidelines into a new file and called segfault_fixer_expert.xml. @Web

<!-- uv run pytest -s --record-mode=none --pdb --pdbcls bpdb:BPdb --verbose -vvvvvv --showlocals --tb=short tests/test_logsetup.py -k test_logger_initialization -->


Find quotes from the the cpython test_asyncio filesthat are relevant to writing high quality asyncio tests. Even though a lot of these tests involve the use of unittest instead of peference of pytest, you should be able to extract strategies that can be used with both testing frameworks. Place these in <quotes> tags. Then, based on these quotes, list all information that would help the a programer adhere to these standards. Place your diagnostic information in <info> tags. Once you are done, we will make modifications to @test_logsetup.py @logsetup.py  to include these details.



Find quotes from the langgraph_api/langraph_cli python files that are relevant to fixing this  Fatal Python error: Segmentation fault error. Place these in <quotes> tags. Then, based on these quotes, list all information that would help the a programer fix these issues. Place your diagnostic information in <info> tags.

---

# use xml tags to structure claude
@home_assistant_code.xml @segfault_fixer_expert.xml

Find quotes from the home_assistant python docs and files that are relevant to fixing this Fatal Python error: Segmentation fault error. Place these in <quotes> tags. Then, based on these quotes, list all information that would help the a programer fix these issues. Place your diagnostic information in <info> tags. We will use this information to enrich segfault_fixer_expert.xml. @Web

# let claude think
Think before you write the checklist in <thinking> tags. First, think through what patterns will provide the most benifit for THIS code base. Then, think through what aspects of the pattern need to written to make our segfault_fixer_expert.xml more capable of solving our segmentation fault issue. Finally, write the personalized checklist item <checklist> tags, using your analysis.


----

Find quotes from the chat history I just sent you that is relevant to this task of creating a self improving analysis prompt. Place these in <quotes> tags. Then, based on these quotes, list all information that would help structrue a good follow up prompt. Place your diagnostic information in <info> tags. We will use this information to enrich my follow up prompt. Think before you write anything in <thinking> tags.
