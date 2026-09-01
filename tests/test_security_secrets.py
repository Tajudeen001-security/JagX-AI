from security.secrets import contains_secrets, scan_text


def test_detects_aws_key():
    text = "key = AKIAIOSFODNN7EXAMPLE"
    hits = scan_text(text)
    assert any(h.rule == "aws_access_key" for h in hits)


def test_detects_private_key_header():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
    assert contains_secrets(text)


def test_clean_text():
    assert not contains_secrets("def add(a, b): return a + b")
