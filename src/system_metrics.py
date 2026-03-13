from __future__ import annotations

import time
from pathlib import Path


class SystemMetricsReader:
    def __init__(self) -> None:
        self._prev_cpu_sample: tuple[int, int] | None = None
        self._prev_powercap_sample: dict[str, tuple[float, float, float | None]] = {}

    def read_cpu_percent(self) -> float | None:
        try:
            first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            return None

        parts = first.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        try:
            nums = [int(v) for v in parts[1:]]
        except ValueError:
            return None

        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        prev = self._prev_cpu_sample
        self._prev_cpu_sample = (total, idle)
        if prev is None:
            return None

        total_delta = total - prev[0]
        idle_delta = idle - prev[1]
        if total_delta <= 0:
            return None
        busy = total_delta - idle_delta
        return round((busy / total_delta) * 100.0, 1)

    def read_power_metrics(self) -> tuple[float | None, float | None]:
        root = Path("/sys/class/power_supply")
        if not root.exists() or not root.is_dir():
            rapl_watts = self.read_powercap_watts()
            if rapl_watts is None:
                rapl_watts = self.read_hwmon_watts()
            return (rapl_watts, None)

        watts_total = 0.0
        watts_max_total = 0.0
        found_watts = False

        for dev in root.iterdir():
            if not dev.is_dir():
                continue
            watts = self.read_device_watts(dev)
            if watts is not None:
                watts_total += watts
                found_watts = True
            max_watts = self.read_device_max_watts(dev)
            if max_watts is not None:
                watts_max_total += max_watts

        if not found_watts:
            rapl_watts = self.read_powercap_watts()
            if rapl_watts is None:
                rapl_watts = self.read_hwmon_watts()
            return (rapl_watts, None)

        pct: float | None = None
        if watts_max_total > 0:
            pct = round(min(100.0, (watts_total / watts_max_total) * 100.0), 1)
        return (round(watts_total, 2), pct)

    def read_powercap_watts(self) -> float | None:
        root = Path("/sys/class/powercap")
        if not root.exists() or not root.is_dir():
            return None

        now = time.monotonic()
        total_watts = 0.0
        have_delta = False

        for zone in root.glob("intel-rapl:*"):
            # Ignore nested subdomains (e.g. intel-rapl:0:0) to avoid double counting.
            if zone.name.count(":") != 1:
                continue
            energy_uj = self.read_number_file(zone / "energy_uj")
            if energy_uj is None:
                continue
            max_energy_uj = self.read_number_file(zone / "max_energy_range_uj")
            key = str(zone)
            prev = self._prev_powercap_sample.get(key)
            self._prev_powercap_sample[key] = (energy_uj, now, max_energy_uj)
            if prev is None:
                continue

            prev_energy, prev_time, prev_max = prev
            dt = now - prev_time
            if dt <= 0:
                continue
            delta = energy_uj - prev_energy
            wrap = max_energy_uj if max_energy_uj is not None else prev_max
            if delta < 0 and wrap and wrap > 0:
                delta += wrap
            if delta < 0:
                continue

            watts = (delta / 1_000_000.0) / dt
            if watts >= 0:
                total_watts += watts
                have_delta = True

        if not have_delta:
            return None
        return round(total_watts, 2)

    def read_hwmon_watts(self) -> float | None:
        root = Path("/sys/class/hwmon")
        if not root.exists() or not root.is_dir():
            return None

        # First choice: direct power sensors, typically in microwatts.
        direct_watts = 0.0
        found_direct = False
        for dev in root.glob("hwmon*"):
            if not dev.is_dir():
                continue
            for path in dev.glob("power*_input"):
                value = self.read_number_file(path)
                if value is None:
                    continue
                direct_watts += value / 1_000_000.0
                found_direct = True
        if found_direct:
            return round(direct_watts, 2)

        # Fallback for INA3221-style rails: W = (mV * mA) / 1_000_000.
        # Prefer the VDD_IN rail if available; it is usually total input draw.
        total_pairs_watts = 0.0
        found_pairs = False
        for dev in root.glob("hwmon*"):
            if not dev.is_dir():
                continue
            if (self.read_text_file(dev / "name") or "").strip().lower() != "ina3221":
                continue

            rail_watts: dict[int, float] = {}
            for in_path in dev.glob("in*_input"):
                suffix = in_path.name[len("in") : -len("_input")]
                if not suffix.isdigit():
                    continue
                idx = int(suffix)
                curr_path = dev / f"curr{idx}_input"
                if not curr_path.is_file():
                    continue
                mv = self.read_number_file(in_path)
                ma = self.read_number_file(curr_path)
                if mv is None or ma is None:
                    continue
                rail_watts[idx] = (mv * ma) / 1_000_000.0

            if not rail_watts:
                continue

            for idx, watts in rail_watts.items():
                label = (self.read_text_file(dev / f"in{idx}_label") or "").strip().upper()
                if "VDD_IN" in label:
                    return round(max(0.0, watts), 2)

            for idx, watts in rail_watts.items():
                label = (self.read_text_file(dev / f"in{idx}_label") or "").strip().lower()
                if "sum of shunt" in label:
                    continue
                total_pairs_watts += max(0.0, watts)
                found_pairs = True

        if found_pairs:
            return round(total_pairs_watts, 2)
        return None

    def read_battery_metrics(self) -> dict[str, float | str | None] | None:
        root = Path("/sys/class/power_supply")
        if not root.exists() or not root.is_dir():
            return None

        for dev in root.iterdir():
            if not dev.is_dir():
                continue
            dev_type = (self.read_text_file(dev / "type") or "").strip().lower()
            is_battery = dev_type == "battery" or dev.name.upper().startswith("BAT")
            if not is_battery:
                continue

            status = self.read_text_file(dev / "status")
            capacity = self.read_number_file(dev / "capacity")
            if capacity is None:
                energy_now = self.read_number_file(dev / "energy_now")
                energy_full = self.read_number_file(dev / "energy_full")
                charge_now = self.read_number_file(dev / "charge_now")
                charge_full = self.read_number_file(dev / "charge_full")
                if energy_now is not None and energy_full and energy_full > 0:
                    capacity = (energy_now / energy_full) * 100.0
                elif charge_now is not None and charge_full and charge_full > 0:
                    capacity = (charge_now / charge_full) * 100.0

            watts = self.read_device_watts(dev)
            remaining_hours = self.estimate_battery_remaining_hours(dev, status)
            return {
                "status": status,
                "capacity_pct": round(capacity, 1) if capacity is not None else None,
                "remaining_hours": remaining_hours,
                "watts": round(watts, 2) if watts is not None else None,
            }

        return None

    def read_thermal_metrics(self) -> dict[str, float | str | None] | None:
        root = Path("/sys/class/thermal")
        if not root.exists() or not root.is_dir():
            return None

        hottest_temp_c: float | None = None
        hottest_zone: str | None = None
        sample_count = 0

        for zone in root.glob("thermal_zone*"):
            if not zone.is_dir():
                continue
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
        status_raw = (status or "").strip().lower()

        # Prefer kernel-provided seconds when available.
        time_to_empty = self.read_number_file(dev / "time_to_empty_now")
        if status_raw == "discharging" and time_to_empty is not None:
            return max(0.0, time_to_empty / 3600.0)
        time_to_full = self.read_number_file(dev / "time_to_full_now")
        if status_raw == "charging" and time_to_full is not None:
            return max(0.0, time_to_full / 3600.0)

        if status_raw not in {"discharging", "charging"}:
            return None

        energy_now = self.read_number_file(dev / "energy_now")
        energy_full = self.read_number_file(dev / "energy_full")
        power_now = self.read_number_file(dev / "power_now")
        if power_now and power_now > 0:
            if status_raw == "discharging" and energy_now is not None:
                return max(0.0, energy_now / power_now)
            if status_raw == "charging" and energy_now is not None and energy_full is not None:
                return max(0.0, (energy_full - energy_now) / power_now)

        charge_now = self.read_number_file(dev / "charge_now")
        charge_full = self.read_number_file(dev / "charge_full")
        current_now = self.read_number_file(dev / "current_now")
        if current_now and current_now > 0:
            if status_raw == "discharging" and charge_now is not None:
                return max(0.0, charge_now / current_now)
            if status_raw == "charging" and charge_now is not None and charge_full is not None:
                return max(0.0, (charge_full - charge_now) / current_now)

        return None

    def read_device_watts(self, dev: Path) -> float | None:
        power_now = self.read_number_file(dev / "power_now")
        if power_now is not None:
            return power_now / 1_000_000.0

        current_now = self.read_number_file(dev / "current_now")
        voltage_now = self.read_number_file(dev / "voltage_now")
        if current_now is not None and voltage_now is not None:
            return (current_now * voltage_now) / 1_000_000_000_000.0
        return None

    def read_device_max_watts(self, dev: Path) -> float | None:
        power_max = self.read_number_file(dev / "power_max_design")
        if power_max is not None:
            return power_max / 1_000_000.0

        current_max = self.read_number_file(dev / "current_max")
        voltage_max = self.read_number_file(dev / "voltage_max")
        if current_max is not None and voltage_max is not None:
            return (current_max * voltage_max) / 1_000_000_000_000.0
        return None

    def read_number_file(self, path: Path) -> float | None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except (OSError, TypeError, UnicodeDecodeError):
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def read_text_file(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, TypeError, UnicodeDecodeError):
            return None

    def read_temperature_c(self, path: Path) -> float | None:
        value = self.read_number_file(path)
        if value is None:
            return None
        if value > 1000.0:
            value /= 1000.0
        if value < -100.0 or value > 300.0:
            return None
        return value


def format_hours(hours: float) -> str:
    if hours <= 0:
        return "0m"
    total_minutes = int(round(hours * 60.0))
    h, m = divmod(total_minutes, 60)
    if h <= 0:
        return f"{m}m"
    return f"{h}h{m:02d}m"
