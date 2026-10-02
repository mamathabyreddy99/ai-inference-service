def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_positive_and_negative(client):
    pos = client.post("/predict", json={"text": "the service was excellent"}).json()
    neg = client.post("/predict", json={"text": "the app was terrible"}).json()
    assert pos["label"] == "positive"
    assert neg["label"] == "negative"
    assert 0.0 <= pos["confidence"] <= 1.0


def test_blank_text_rejected(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_missing_field_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


def test_too_long_text_rejected(client):
    assert client.post("/predict", json={"text": "a" * 2001}).status_code == 422


def test_batch_returns_one_prediction_per_text(client):
    texts = ["this tool was amazing", "the delivery was awful", "the interface felt superb"]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    assert [p["label"] for p in r.json()["predictions"]] == ["positive", "negative", "positive"]


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_batch_with_blank_item_rejected(client):
    assert client.post("/predict/batch", json={"texts": ["fine", " "]}).status_code == 422


def test_naive_and_optimized_agree(client):
    body = {"text": "customer support was frustrating"}
    a = client.post("/predict/naive", json=body).json()
    b = client.post("/predict", json=body).json()
    assert a["label"] == b["label"]
    assert a["confidence"] == b["confidence"]


def test_repeated_input_hits_cache(client):
    body = {"text": "the setup process was wonderful, cache check"}
    before = client.get("/health").json()["cache_hits"]
    client.post("/predict", json=body)
    client.post("/predict", json=body)
    after = client.get("/health").json()["cache_hits"]
    assert after >= before + 1


def test_history_records_predictions(client):
    client.post("/predict", json={"text": "history check was great"})
    rows = client.get("/history?limit=5").json()
    assert len(rows) >= 1
    assert {"id", "text", "label", "confidence", "endpoint", "created_at"} <= rows[0].keys()


def test_history_limit_validated(client):
    assert client.get("/history?limit=0").status_code == 422
