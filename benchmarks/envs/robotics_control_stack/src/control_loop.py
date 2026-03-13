def brake_distance(speed: float, decel: float) -> float:
    if decel <= 0:
        raise ValueError("decel must be positive")
    return speed * decel


def should_brake(distance_remaining: float, speed: float, decel: float) -> bool:
    return brake_distance(speed, decel) >= distance_remaining
