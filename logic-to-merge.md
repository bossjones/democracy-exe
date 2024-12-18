Getting Started with dpytest
===========================

Welcome to ``dpytest``, a python library for testing discord bots written using ``discord.py``. This tutorial
will explain how to install ``dpytest`` and set it up in your project, and write a simple test. If you already
know how to install libraries with pip, you probably want to skip to `Using Pytest`_.

Installing Dpytest
------------------

To start with, you should install dpytest with ``pip``. This will look a bit different, depending if you're
on Windows or Mac/Linux:

- Windows: ``py -m pip install dpytest``
- Linux: ``python3 -m pip install dpytest``

Using Dpytest
-------------

Once installed, you will need to import ``dpytest`` before you can use it. As it is an extension to ``discord.py``,
it goes into the ``discord.py`` extensions module. So, the most basic usage of dpytest would look like this:

.. code:: python

    import asyncio
    import discord.ext.test as dpytest


    async def test_ping():
        bot = ...  # However you create your bot.
        dpytest.configure(bot)
        await dpytest.message("!ping")
        assert dpytest.verify().message().contains().content("Ping:")


    async def test_foo():
        bot = ... # Same setup as above
        dpytest.configure(bot)
        await dpytest.message("!hello")
        assert dpytest.verify().message().content("Hello World!")


    asyncio.run(test_ping())
    asyncio.run(test_foo())

One problem that could happen is that the ``sent_queue`` is shared between the tests. So in order not to mess between
your tests (``verify()`` pops **one** message from the queue, so in general, you won't need to do anything) you can
explicitly call ``empty_queue()``, as shown in the next example (and later, in the ``conftests.py``).

If that looks like a lot of code just to run tests, don't worry, there's a better way! We can use pytest,
a popular Python testing library.

--------------------

**Next Tutorial**: `Using Pytest`_

.. _Using Pytest: ./using_pytest.html

# Using Pytest

So, you already have `dpytest` installed, and can import it. However, setting up a client for every test is a pain. The library is designed to work well with `pytest` (Thus the name), and it can make writing tests much easier. In the following tutorial we'll show how to set it up.

## Starting with Pytest

`pytest` can be installed through pip the same way `dpytest` is. Once that's done, using it is as easy as:

- Windows: `py -m pytest`
- Linux: `python3 -m pytest`

`pytest` will detect any functions starting with 'test' in directories it searches, and run them. It also supports a feature we will use heavily, called 'fixtures'. Fixtures are functions that do some common test setup, and then can be used in tests to always perform that setup, they can also return an object that will be passed to the test. Finally, they allow you to perform test teardown, cleaning message queue for example. See https://docs.pytest.org/en/latest/how-to/fixtures.html#teardown-cleanup-aka-fixture-finalization

The final piece of this is `pytest-asyncio`, a library for allowing `pytest` to run async tests. It is automatically installed when you get `dpytest` from pip, so you don't need to worry about installing it.

Putting all this together, we can rewrite our previous tests to look like this:

```
import discord
import discord.ext.commands as commands
from discord.ext.commands import Cog, command
import pytest
import pytest_asyncio
import discord.ext.test as dpytest


class Misc(Cog):
    @command()
    async def ping(self, ctx):
        await ctx.send("Pong !")

    @command()
    async def echo(self, ctx, text: str):
        await ctx.send(text)


@pytest_asyncio.fixture
async def bot():
    # Setup
    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True
    b = commands.Bot(command_prefix="!",
                     intents=intents)
    await b._async_setup_hook()  # setup the loop
    await b.add_cog(Misc())

    dpytest.configure(b)

    yield b

    # Teardown
    await dpytest.empty_queue() # empty the global message queue as test teardown


@pytest.mark.asyncio
async def test_ping(bot):
    await dpytest.message("!ping")
    assert dpytest.verify().message().content("Pong !")


@pytest.mark.asyncio
async def test_echo(bot):
    await dpytest.message("!echo Hello world")
    assert dpytest.verify().message().contains().content("Hello")
```

Much less writing the same code over and over again, and tests will be automatically run by pytest, then the results output in a nice pretty format once it's done.

## What is conftest.py?

As you write tests, you may want to split them into multiple files. One file for testing this cog, another for ensuring reactions work right. As it stands, you'll still need to copy your bot fixture into every file. To fix this, you need to create a file named `conftest.py` at the root of where you're putting your tests. If you haven't already, you should probably put them all in their own directory. Then you can put any fixtures you want in `conftest.py`, and pytest will let you use them in any other test file. `pytest` also recognizes certain function names with special meanings, for example `pytest_sessionfinish` will be run after all tests are done if defined in your conftest.

An example `conftest.py` might look like this:

```
import glob
import os
import pytest_asyncio
import discord
import discord.ext.commands as commands
import discord.ext.test as dpytest


@pytest_asyncio.fixture
async def bot():
    # Setup
    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True
    b = commands.Bot(command_prefix="!",
                    intents=intents)
    await b._async_setup_hook()
    dpytest.configure(b)

    yield b

    # Teardown
    await dpytest.empty_queue() # empty the global message queue as test teardown


def pytest_sessionfinish(session, exitstatus):
    """ Code to execute after all tests. """

    # dat files are created when using attachements
    print("\n-------------------------\nClean dpytest_*.dat files")
    fileList = glob.glob('./dpytest_*.dat')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except Exception:
            print("Error while deleting file : ", filePath)
```

With that, you should be ready to use `dpytest` with your bot.

## Troubleshooting

- I wrote a fixture, but I can't use the bot

Make sure your tests take a parameter with the exact same name as the fixture, pytest runs them based on name, including capitalization.

- I use dpytest.verify().message() and it fails but it shouldn't
- dpytest.get_message() returns a message from another test

Make sure you properly emptied the queue in the previous test, otherwise you could have remaining messages from previous tests messing up.

------

This is currently the end of the tutorials. Take a look at the [Runner Documentation](https://dpytest.readthedocs.io/en/latest/modules/runner.html) to see all the things you can do with `dpytest`.
