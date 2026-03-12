def summarize_alerts(alerts: list[str]) -> str:
    if not alerts:
        return "no active alerts"
    return "; ".join(alerts[:3])
