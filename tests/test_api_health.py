from api.server import create_app, health


def test_health_payload():
    h = health()
    assert h["status"] == "ok"
    assert h["provider_independent"] is True


def test_handler_routes():
    Handler = create_app()
    assert "/health" in Handler.routes
    assert Handler.routes["/health"]()["status"] == "ok"
