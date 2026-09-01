from api.server import create_app, health


def test_generate_stub():
    Handler = create_app()
    assert "/v1/generate" in Handler.post_routes
    out = Handler.post_routes["/v1/generate"]({"prompt": "hi", "max_tokens": 8})
    assert out["backend"] == "local-stub"
    assert out["external_ai_api_required"] is None if "external_ai_api_required" not in out else True


def test_custom_generate_fn():
    def gen(payload):
        return {"text": f"echo:{payload.get('prompt')}", "backend": "test"}

    Handler = create_app(generate_fn=gen)
    out = Handler.post_routes["/v1/chat"]({"prompt": "abc"})
    assert out["text"] == "echo:abc"


def test_health_still_ok():
    assert health()["provider_independent"] is True
