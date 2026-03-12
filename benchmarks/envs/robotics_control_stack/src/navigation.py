from .control_loop import should_brake


def next_action(distance_remaining: float, speed: float, decel: float) -> str:
    return "brake" if should_brake(distance_remaining, speed, decel) else "advance"
