"""Read lightweight CPU, power, battery, and thermal metrics from Linux."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

_MICROWATTS_PER_WATT = 1_000_000.0
_MICROVOLT_MICROAMP_PER_WATT = 1_000_000_000_000.0


class SystemMetricsReader:
    """Read point-in-time system metrics from procfs and sysfs."""

    proc_stat_path = Path("/proc/stat")
    power_supply_root = Path("/sys/class/power_supply")
    powercap_root = Path("/sys/class/powercap")
    hwmon_root = Path("/sys/class/hwmon")
    thermal_root = Path("/sys/class/thermal")

    def __init__(self) -> None:
        """Store the previous CPU and RAPL samples needed for delta metrics."""
        self._prev_cpu_sample: tuple[int, int] | None = None
        self._prev_powercap_sample: dict[str, tuple[float, float, float | None]] = {}

    def read_cpu_percent(self) -> float | None:
        """Compare /proc/stat against the previous sample and return busy percent."""
        try:
            first = self.proc_stat_path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            return None

        parts = first.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        try:
            counters = [int(value) for value in parts[1:]]
        except ValueError:
            return None

        idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
        sample = (sum(counters), idle)
        previous = self._prev_cpu_sample
        self._prev_cpu_sample = sample
        if previous is None:
            return None

        total_delta = sample[0] - previous[0]
        idle_delta = sample[1] - previous[1]
        if total_delta <= 0:
            return None
        return round(((total_delta - idle_delta) / total_delta) * 100.0, 1)

    def read_power_metrics(self) -> tuple[float | None, float | None]:
        """Prefer power-supply data and fall back to powercap or hwmon sensors."""
        watts_total = 0.0
        max_watts_total = 0.0
        found_watts = False

        for dev in self._iter_dirs(self.power_supply_root):
            watts = self.read_device_watts(dev)
            max_watts = self.read_device_max_watts(dev)
            if watts is not None:
                watts_total += watts
                found_watts = True
            if max_watts is not None:
                max_watts_total += max_watts

        if found_watts:
            pct = None if max_watts_total <= 0 else round(min(100.0, (watts_total / max_watts_total) * 100.0), 1)
            return round(watts_total, 2), pct

        watts = self.read_powercap_watts()
        if watts is None:
            watts = self.read_hwmon_watts()
        return watts, None

    def read_powercap_watts(self) -> float | None:
        """Estimate package watts from RAPL energy counters."""
        now = time.monotonic()
        total_watts = 0.0
        found_delta = False

        for zone in self._iter_dirs(self.powercap_root, "intel-rapl:*"):
            # Ignore nested subdomains such as intel-rapl:0:0 to avoid double counting.
            if zone.name.count(":") != 1:
                continue

            energy_uj = self.read_number_file(zone / "energy_uj")
            if energy_uj is None:
                continue

            max_energy_uj = self.read_number_file(zone / "max_energy_range_uj")
            key = str(zone)
            previous = self._prev_powercap_sample.get(key)
            self._prev_powercap_sample[key] = (energy_uj, now, max_energy_uj)
            if previous is None:
                continue

            prev_energy, prev_time, prev_max = previous
            elapsed = now - prev_time
            if elapsed <= 0:
                continue

            delta_uj = energy_uj - prev_energy
            wrap_uj = max_energy_uj if max_energy_uj is not None else prev_max
            if delta_uj < 0 and wrap_uj and wrap_uj > 0:
                delta_uj += wrap_uj
            if delta_uj < 0:
                continue

            watts = (delta_uj / _MICROWATTS_PER_WATT) / elapsed
            if watts >= 0:
                total_watts += watts
                found_delta = True

        return round(total_watts, 2) if found_delta else None

    def read_hwmon_watts(self) -> float | None:
        """Read watts from hwmon sensors, preferring direct power inputs."""
        direct_watts = 0.0
        found_direct = False

        for dev in self._iter_dirs(self.hwmon_root, "hwmon*"):
            for path in dev.glob("power*_input"):
                value = self.read_number_file(path)
                if value is None:
                    continue
                direct_watts += value / _MICROWATTS_PER_WATT
                found_direct = True

        if found_direct:
            return round(direct_watts, 2)

        rail_watts_total = 0.0
        found_rail = False
        for dev in self._iter_dirs(self.hwmon_root, "hwmon*"):
            if (self.read_text_file(dev / "name") or "").lower() != "ina3221":
                continue

            rails: dict[int, float] = {}
            for in_path in dev.glob("in*_input"):
                suffix = in_path.name.removeprefix("in").removesuffix("_input")
                if not suffix.isdigit():
                    continue

                index = int(suffix)
                current_path = dev / f"curr{index}_input"
                if not current_path.is_file():
                    continue

                millivolts = self.read_number_file(in_path)
                milliamps = self.read_number_file(current_path)
                if millivolts is None or milliamps is None:
                    continue
                rails[index] = (millivolts * milliamps) / _MICROWATTS_PER_WATT

            if not rails:
                continue

            for index, watts in rails.items():
                label = (self.read_text_file(dev / f"in{index}_label") or "").strip()
                if "VDD_IN" in label.upper():
                    return round(max(0.0, watts), 2)

            for index, watts in rails.items():
                label = (self.read_text_file(dev / f"in{index}_label") or "").lower()
                if "sum of shunt" in label:
                    continue
                rail_watts_total += max(0.0, watts)
                found_rail = True

        return round(rail_watts_total, 2) if found_rail else None

    def read_battery_metrics(self) -> dict[str, float | str | None] | None:
        """Return status, charge, ETA, and draw for the first battery device."""
        for dev in self._iter_dirs(self.power_supply_root):
            dev_type = (self.read_text_file(dev / "type") or "").lower()
            if dev_type != "battery" and not dev.name.upper().startswith("BAT"):
                continue

            status = self.read_text_file(dev / "status")
            capacity_pct = self.read_number_file(dev / "capacity")
            if capacity_pct is None:
                for now_name, full_name in (("energy_now", "energy_full"), ("charge_now", "charge_full")):
                    capacity_pct = self._ratio_percent(
                        self.read_number_file(dev / now_name),
                        self.read_number_file(dev / full_name),
                    )
                    if capacity_pct is not None:
                        break

            watts = self.read_device_watts(dev)
            return {
                "status": status,
                "capacity_pct": round(capacity_pct, 1) if capacity_pct is not None else None,
                "remaining_hours": self.estimate_battery_remaining_hours(dev, status),
                "watts": round(watts, 2) if watts is not None else None,
            }

        return None

    def read_thermal_metrics(self) -> dict[str, float | str | None] | None:
        """Return the hottest thermal zone and how many zones reported data."""
        hottest_temp_c: float | None = None
        hottest_zone: str | None = None
        sample_count = 0

        for zone in self._iter_dirs(self.thermal_root, "thermal_zone*"):
            temp_c = self.read_temperature_c(zone / "temp")
            if temp_c is None:
                continue

            sample_count += 1
            zone_name = self.read_text_file(zone / "type") or zone.name
            if hottest_temp_c is None or temp_c > hottest_temp_c:
                hottest_temp_c = temp_c
                hottest_zone = zone_name

        if hottest_temp_c is None:
            return None
        return {
            "hottest_temp_c": round(hottest_temp_c, 1),
            "hottest_zone": hottest_zone,
            "sample_count": sample_count,
        }

    def estimate_battery_remaining_hours(self, dev: Path, status: str | None) -> float | None:
        """Estimate time to empty or full using kernel timers or live draw."""
        status_raw = (status or "").strip().lower()
        if status_raw == "discharging":
            seconds = self.read_number_file(dev / "time_to_empty_now")
            if seconds is not None:
                return max(0.0, seconds / 3600.0)
        elif status_raw == "charging":
            seconds = self.read_number_file(dev / "time_to_full_now")
            if seconds is not None:
                return max(0.0, seconds / 3600.0)
        else:
            return None

        for now_name, full_name, rate_name in (
            ("energy_now", "energy_full", "power_now"),
            ("charge_now", "charge_full", "current_now"),
        ):
            now_value = self.read_number_file(dev / now_name)
            rate_value = self.read_number_file(dev / rate_name)
            if now_value is None or rate_value is None or rate_value <= 0:
                continue
            if status_raw == "discharging":
                return max(0.0, now_value / rate_value)

            full_value = self.read_number_file(dev / full_name)
            if full_value is not None:
                return max(0.0, (full_value - now_value) / rate_value)

        return None

    def read_device_watts(self, dev: Path) -> float | None:
        """Read live device draw from direct power or current/voltage values."""
        return self._read_power_value(dev, "power_now", "current_now", "voltage_now")

    def read_device_max_watts(self, dev: Path) -> float | None:
        """Read device design capacity from direct power or current/voltage values."""
        return self._read_power_value(dev, "power_max_design", "current_max", "voltage_max")

    def read_number_file(self, path: Path) -> float | None:
        """Parse a numeric procfs/sysfs file and ignore unreadable values."""
        try:
            return float(path.read_text(encoding="utf-8").strip())
        except (OSError, TypeError, UnicodeDecodeError, ValueError):
            return None

    def read_text_file(self, path: Path) -> str | None:
        """Read a text procfs/sysfs file and return None when it is unavailable."""
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, TypeError, UnicodeDecodeError):
            return None

    def read_temperature_c(self, path: Path) -> float | None:
        """Normalise thermal readings to Celsius and reject implausible values."""
        value = self.read_number_file(path)
        if value is None:
            return None
        if value > 1000.0:
            value /= 1000.0
        if not -100.0 <= value <= 300.0:
            return None
        return value

    def _read_power_value(self, dev: Path, power_name: str, current_name: str, voltage_name: str) -> float | None:
        """Read watts from a direct power file or from current times voltage."""
        direct_power = self.read_number_file(dev / power_name)
        if direct_power is not None:
            return direct_power / _MICROWATTS_PER_WATT

        current = self.read_number_file(dev / current_name)
        voltage = self.read_number_file(dev / voltage_name)
        if current is None or voltage is None:
            return None
        return (current * voltage) / _MICROVOLT_MICROAMP_PER_WATT

    def _iter_dirs(self, root: Path, pattern: str | None = None) -> Iterator[Path]:
        """Yield only directories so callers do not repeat root checks."""
        if not root.exists() or not root.is_dir():
            return
        entries = root.glob(pattern) if pattern is not None else root.iterdir()
        for entry in entries:
            if entry.is_dir():
                yield entry

    @staticmethod
    def _ratio_percent(current: float | None, total: float | None) -> float | None:
        """Convert current and total values into a percentage when possible."""
        if current is None or total is None or total <= 0:
            return None
        return (current / total) * 100.0


def format_hours(hours: float) -> str:
    """Format an hour estimate as a compact ``1h23m`` or ``45m`` string."""
    if hours <= 0:
        return "0m"
    total_minutes = int(round(hours * 60.0))
    h, m = divmod(total_minutes, 60)
    return f"{m}m" if h <= 0 else f"{h}h{m:02d}m"
