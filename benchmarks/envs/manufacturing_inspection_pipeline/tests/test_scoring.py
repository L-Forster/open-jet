from src.scoring import classify


def test_classify():
    assert classify(0.9) == "reject"
    assert classify(0.5) == "accept"
