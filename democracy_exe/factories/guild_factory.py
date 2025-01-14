"""guild_factory.py"""

from __future__ import annotations

from democracy_exe.aio_settings import aiosettings


# SOURCE: https://stackoverflow.com/a/63483209/814221
class Singleton(type):
    # Inherit from "type" in order to gain access to method __call__
    def __init__(self, *args, **kwargs):
        self.__instance = None  # Create a variable to store the object reference
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            # if the object has not already been created
            self.__instance = super().__call__(
                *args, **kwargs
            )  # Call the __init__ method of the subclass (Spam) and save the reference
        return self.__instance


# SOURCE: https://stackoverflow.com/a/63483209/814221
class Guild(metaclass=Singleton):
    def __init__(self, id=aiosettings.discord_server_id, prefix=aiosettings.prefix):
        # print('Creating Guild')
        self.id = id
        self.prefix = prefix


# smoke tests
if __name__ == "__main__":
    test_guild_metadata = Guild(id=int(aiosettings.discord_server_id), prefix=aiosettings.prefix)
    print(test_guild_metadata)
    print(test_guild_metadata.id)
    print(test_guild_metadata.prefix)

    test_guild_metadata2 = Guild()
    print(test_guild_metadata2)
    print(test_guild_metadata2.id)
    print(test_guild_metadata2.prefix)
