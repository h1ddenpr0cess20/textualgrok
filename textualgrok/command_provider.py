"""System command provider for the chat app."""

from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.app import SystemCommand


class OrderedSystemCommandsProvider(Provider):
    """Yield system commands in the order defined by the app."""

    async def discover(self) -> Hits:
        for title, help_text, callback, discover in self.app.get_system_commands(self.screen):
            if discover:
                yield DiscoveryHit(title, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for title, help_text, callback, *_ in self.app.get_system_commands(self.screen):
            if (match := matcher.match(title)) > 0:
                yield Hit(match, matcher.highlight(title), callback, help=help_text)

