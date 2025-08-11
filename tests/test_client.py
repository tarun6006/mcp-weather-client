import pytest
from app import app

@pytest.fixture
def client_flask():
    app.config['TESTING'] = True
    return app.test_client()

def test_url_verification(client_flask):
    resp = client_flask.post("/slack/events", json={"type":"url_verification","challenge":"abc"})
    assert resp.status_code == 200 and resp.json["challenge"] == "abc"
