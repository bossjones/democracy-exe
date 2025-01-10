@structlog_expert.md  Find quotes from the the structlog docs that are relevant to writing quality tests using pytest for python modules that use structlog. Place these in <quotes> tags. Then, based on these quotes, list all information that would help the a programer adhere to these standards. Place your diagnostic information in <info> tags. Once you are done, we will make modifications to @test_logsetup.py @logsetup.py  to include these details. @Web


uv run pytest -s --record-mode=none --pdb --pdbcls bpdb:BPdb --verbose -vvvvvv --showlocals --tb=short tests/test_logsetup.py -k test_logger_initialization
