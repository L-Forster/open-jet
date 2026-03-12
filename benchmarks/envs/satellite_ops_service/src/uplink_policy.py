INTERMITTENT_UPLINK = True


def can_transmit(window_open: bool, manual_hold: bool) -> bool:
    return window_open and not manual_hold
