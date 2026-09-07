from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class BrowserProvider(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...
    def open(self, url: str) -> dict[str, Any]: ...


class BookingProvider(Protocol):
    def search(self, service: str, location: str, date: str, party_size: int = 1) -> list[dict[str, Any]]: ...
    def reserve(self, offer_id: str, *, confirmed: bool = False) -> dict[str, Any]: ...


class MessagingProvider(Protocol):
    def compose(self, recipient: str, subject: str, body: str) -> dict[str, Any]: ...
    def send(self, recipient: str, subject: str, body: str, *, confirmed: bool = False) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TravelPlan:
    origin: str
    destination: str
    departure_time: str | None = None
    avoid_traffic: bool = True


class ExternalActionGateway:
    """Boundary for bookings, company contact, traffic and other external actions.

    Reading/searching can be automated by the host. Reservations and messages
    remain explicit confirmation points so the agent cannot spend money or send
    communications unexpectedly.
    """

    def __init__(self, browser: BrowserProvider | None = None,
                 booking: BookingProvider | None = None,
                 messaging: MessagingProvider | None = None) -> None:
        self.browser = browser
        self.booking = booking
        self.messaging = messaging

    def research(self, query: str) -> list[dict[str, Any]]:
        if self.browser is None:
            raise RuntimeError("browser provider not configured")
        return self.browser.search(query)

    def find_booking(self, service: str, location: str, date: str, party_size: int = 1) -> list[dict[str, Any]]:
        if self.booking is None:
            raise RuntimeError("booking provider not configured")
        return self.booking.search(service, location, date, party_size)

    def reserve(self, offer_id: str, confirmed: bool = False) -> dict[str, Any]:
        if not confirmed:
            raise PermissionError("reservation requires explicit confirmation")
        if self.booking is None:
            raise RuntimeError("booking provider not configured")
        return self.booking.reserve(offer_id, confirmed=True)

    def contact_company(self, recipient: str, subject: str, body: str, confirmed: bool = False) -> dict[str, Any]:
        if not confirmed:
            raise PermissionError("sending a message requires explicit confirmation")
        if self.messaging is None:
            raise RuntimeError("messaging provider not configured")
        return self.messaging.send(recipient, subject, body, confirmed=True)
