from src.uplink_policy import can_transmit


def test_can_transmit():
    assert can_transmit(True, False) is True
