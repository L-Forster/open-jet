from src.control_loop import brake_distance


def test_brake_distance():
    assert brake_distance(8, 2) == 4
