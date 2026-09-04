from api.server import create_app, health, _unbound_generate


def test_generate_unbound_is_error_not_stub():
    Handler = create_app()
    assert "/v1/generate" in Handler.post_routes
    out = Handler.post_routes["/v1/generate"]({"prompt": "hi", "max_tokens": 8})
    assert out.get("error") == "model_not_bound"
    assert out.get("code") == "model_not_bound"


def test_unbound_helper():
    out = _unbound_generate({"prompt": "x"})
    assert out["error"] == "model_not_bound"


def test_custom_generate_fn():
    def gen(payload):
        return {"text": f"echo:{payload.get('prompt')}", "backend": "test"}

    Handler = create_app(generate_fn=gen)
    out = Handler.post_routes["/v1/chat"]({"prompt": "abc"})
    assert out["text"] == "echo:abc"


def test_health_still_ok():
    assert health()["provider_independent"] is True
