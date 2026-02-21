#!/usr/bin/env python3
"""
Satellite Tracker
Downloads TLEs from Celestrak and calculates satellite passes.

Usage:
    python sat_tracker.py --lat 19.4326 --lon -99.1332 --start 20:00 --end 23:00 --days 3 --interval 10
"""

import argparse
import configparser
import csv
import math
import sys
from pathlib import Path

import arrow
import requests
from skyfield.api import EarthSatellite, load, wgs84


CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"
SATS_FILE = "sats-search.txt"
TLE_DIR = "tle"
CONFIG_FILE = "sat_tracker.cfg"


def load_config(config_path: str) -> dict:
    """Load configuration from INI file."""
    config = configparser.ConfigParser()
    defaults = {
        "latitude": None,
        "longitude": None,
        "elevation": 0,
        "interval": 10,
        "min_altitude": 0,
        "sats_file": SATS_FILE,
        "tle_dir": TLE_DIR,
    }

    path = Path(config_path)
    if not path.exists():
        return defaults

    try:
        config.read(config_path)

        if config.has_section("observer"):
            if config.has_option("observer", "latitude"):
                defaults["latitude"] = config.getfloat("observer", "latitude")
            if config.has_option("observer", "longitude"):
                defaults["longitude"] = config.getfloat("observer", "longitude")
            if config.has_option("observer", "elevation"):
                defaults["elevation"] = config.getfloat("observer", "elevation")

        if config.has_section("tracking"):
            if config.has_option("tracking", "interval"):
                defaults["interval"] = config.getint("tracking", "interval")
            if config.has_option("tracking", "min_altitude"):
                defaults["min_altitude"] = config.getfloat("tracking", "min_altitude")

        if config.has_section("files"):
            if config.has_option("files", "sats_file"):
                defaults["sats_file"] = config.get("files", "sats_file")
            if config.has_option("files", "tle_dir"):
                defaults["tle_dir"] = config.get("files", "tle_dir")

    except Exception as e:
        print(f"Warning: Error reading {config_path}: {e}")

    return defaults


def load_satellite_names(filepath: str) -> list[str]:
    """Load satellite names from search file."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found {filepath}")
        print(f"Creating example file: {filepath}")
        create_example_sats_file(filepath)
        sys.exit(1)

    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def create_example_sats_file(filepath: str) -> None:
    """Create an example satellite search file."""
    example_content = """# Satellite search file
# One satellite per line or wildcard
# Examples:
ISS (ZARYA)
STARLINK-*
MOLNIYA 1-91
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(example_content)


def download_tle(sat_name: str) -> list[tuple[str, str, str]]:
    """
    Download TLE from Celestrak for a given satellite.
    Returns list of tuples (name, line1, line2).
    """
    params = {
        "NAME": sat_name,
        "FORMAT": "TLE"
    }

    try:
        response = requests.get(CELESTRAK_URL, params=params, timeout=30)
        response.raise_for_status()

        if not response.text.strip():
            print(f"  No results for: {sat_name}")
            return []

        lines = response.text.strip().split("\n")
        satellites = []

        # TLE has 3 lines: name, line 1, line 2
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()

                if line1.startswith("1 ") and line2.startswith("2 "):
                    satellites.append((name, line1, line2))

        return satellites

    except requests.RequestException as e:
        print(f"  Error downloading TLE for {sat_name}: {e}")
        return []


def download_all_tles(sat_names: list[str]) -> list[tuple[str, str, str]]:
    """Download TLEs for all satellites in the list."""
    all_satellites = []
    seen = set()

    print("\nDownloading TLEs from Celestrak...")
    for name in sat_names:
        print(f"  Searching: {name}")
        satellites = download_tle(name)

        for sat in satellites:
            sat_id = (sat[1], sat[2])  # Use TLE lines as unique ID
            if sat_id not in seen:
                seen.add(sat_id)
                all_satellites.append(sat)
                print(f"    + {sat[0]}")

    print(f"\nTotal satellites found: {len(all_satellites)}")
    return all_satellites


def round_to_interval(dt: arrow.Arrow, interval_minutes: int) -> arrow.Arrow:
    """Round time to nearest interval."""
    minutes = dt.minute
    remainder = minutes % interval_minutes
    if remainder < interval_minutes / 2:
        # Round down
        rounded_minutes = minutes - remainder
    else:
        # Round up
        rounded_minutes = minutes + (interval_minutes - remainder)

    result = dt.replace(minute=0, second=0, microsecond=0)
    result = result.shift(minutes=rounded_minutes)
    return result


def get_default_time_range(interval_minutes: int) -> tuple[str, str]:
    """
    Get default start and end times.
    Start: current time rounded to nearest interval
    End: 1 hour after start
    """
    now = arrow.now()
    start = round_to_interval(now, interval_minutes)
    end = start.shift(hours=1)

    return start.format("HH:mm"), end.format("HH:mm")


def parse_time_range(start_str: str, end_str: str, base_date: arrow.Arrow) -> tuple[arrow.Arrow, arrow.Arrow]:
    """
    Parse time range and return arrow dates.
    If end is less than start, assumes end is next day.
    """
    start_parts = start_str.split(":")
    end_parts = end_str.split(":")

    start_hour = int(start_parts[0])
    start_min = int(start_parts[1]) if len(start_parts) > 1 else 0

    end_hour = int(end_parts[0])
    end_min = int(end_parts[1]) if len(end_parts) > 1 else 0

    start_time = base_date.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
    end_time = base_date.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)

    # If end is less than start, it's next day
    if end_time <= start_time:
        end_time = end_time.shift(days=1)

    return start_time, end_time


def calculate_satellite_position(
    satellite: EarthSatellite,
    observer_location,
    time_point,
    ts
) -> dict:
    """
    Calculate satellite position relative to observer.
    Returns dictionary with alt, az, ra, dec, magnitude and visibility.
    """
    t = ts.from_datetime(time_point.datetime)

    # Satellite position
    geocentric = satellite.at(t)

    # Position relative to observer
    difference = satellite - observer_location
    topocentric = difference.at(t)

    # Alt/Az
    alt, az, distance = topocentric.altaz()

    # RA/Dec
    ra, dec, _ = topocentric.radec()

    # Determine visibility (above horizon)
    is_visible = alt.degrees > 0

    # Calculate visual magnitude
    std_mag = get_standard_magnitude(satellite.name)
    magnitude = calculate_visual_magnitude(distance.km, std_mag, alt.degrees)

    return {
        "time": time_point,
        "visible": is_visible,
        "altitude": alt.degrees,
        "azimuth": az.degrees,
        "ra": ra.hours,
        "dec": dec.degrees,
        "distance_km": distance.km,
        "magnitude": magnitude
    }


# Standard magnitudes at 1000km for common satellite types
STANDARD_MAGNITUDES = {
    "ISS": -1.0,
    "ZARYA": -1.0,
    "TIANGONG": 0.0,
    "STARLINK": 5.5,
    "ONEWEB": 6.0,
    "IRIDIUM": 5.0,
    "MOLNIYA": 4.5,
    "MERIDIAN": 5.0,
    "COSMOS": 5.5,
    "DEFAULT": 5.0,  # Default for unknown satellites
}


def get_standard_magnitude(sat_name: str) -> float:
    """Get standard magnitude for a satellite based on its name."""
    sat_upper = sat_name.upper()
    for key, mag in STANDARD_MAGNITUDES.items():
        if key in sat_upper:
            return mag
    return STANDARD_MAGNITUDES["DEFAULT"]


def calculate_visual_magnitude(distance_km: float, std_mag: float, altitude_deg: float) -> float | None:
    """
    Calculate approximate visual magnitude of a satellite.

    Args:
        distance_km: Distance to satellite in km
        std_mag: Standard magnitude at 1000 km
        altitude_deg: Altitude above horizon in degrees

    Returns:
        Approximate visual magnitude, or None if below horizon
    """
    if altitude_deg <= 0:
        return None

    # Basic magnitude calculation based on distance
    # mag = std_mag + 5 * log10(distance / 1000)
    mag = std_mag + 5 * math.log10(distance_km / 1000)

    # Atmospheric extinction correction (dimmer near horizon)
    # Approximate airmass using Pickering (2002) formula
    if altitude_deg > 0:
        airmass = 1 / math.sin(math.radians(altitude_deg + 244 / (165 + 47 * altitude_deg**1.1)))
        extinction = 0.2 * (airmass - 1)  # ~0.2 mag per airmass
        mag += extinction

    return mag


def hours_to_hms(hours: float) -> str:
    """Convert decimal hours to HH:MM:SS format."""
    h = int(hours)
    m = int((hours - h) * 60)
    s = ((hours - h) * 60 - m) * 60
    return f"{h:02d}h{m:02d}m{s:05.2f}s"


def degrees_to_dms(degrees: float) -> str:
    """Convert decimal degrees to DD°MM'SS\" format."""
    sign = "+" if degrees >= 0 else "-"
    degrees = abs(degrees)
    d = int(degrees)
    m = int((degrees - d) * 60)
    s = ((degrees - d) * 60 - m) * 60
    return f"{sign}{d:02d}°{m:02d}'{s:05.2f}\""


def azimuth_to_cardinal(azimuth: float) -> str:
    """
    Convert azimuth in degrees to cardinal direction.

    N = 0°, E = 90°, S = 180°, W = 270°
    """
    # Normalize azimuth to 0-360
    az = azimuth % 360

    # 16-point compass rose
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]

    # Each direction covers 22.5 degrees (360/16)
    # Offset by 11.25 so N is centered at 0
    index = int((az + 11.25) / 22.5) % 16
    return directions[index]


def format_position(pos: dict, sat_name: str) -> str:
    """Format satellite position for display."""
    time_str = pos["time"].format("YYYY-MM-DD HH:mm:ss")

    if not pos["visible"]:
        return f"{time_str} | {sat_name:25s} | NOT-VISIBLE (alt: {pos['altitude']:+7.2f}°)"

    ra_hms = hours_to_hms(pos['ra'])
    dec_dms = degrees_to_dms(pos['dec'])
    mag_str = f"{pos['magnitude']:+5.1f}" if pos['magnitude'] is not None else "  N/A"
    cardinal = azimuth_to_cardinal(pos['azimuth'])

    return (
        f"{time_str} | {sat_name:20s} | {mag_str} | {cardinal:3s} | "
        f"{pos['altitude']:+7.2f}° | {pos['azimuth']:7.2f}° | "
        f"RA: {ra_hms} | Dec: {dec_dms} | "
        f"{pos['distance_km']:,.0f} km"
    )


def sanitize_filename(name: str) -> str:
    """Sanitize satellite name for use as filename."""
    # Replace invalid filename characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    result = name
    for char in invalid_chars:
        result = result.replace(char, '-')
    # Remove multiple dashes
    while '--' in result:
        result = result.replace('--', '-')
    return result.strip('-')


def save_tles_to_dir(satellites: list[tuple[str, str, str]], tle_dir: str) -> None:
    """Save each TLE to a separate file in the specified directory."""
    # Create directory if it doesn't exist
    tle_path = Path(tle_dir)
    tle_path.mkdir(parents=True, exist_ok=True)

    # Get current date and time
    now = arrow.now()
    timestamp = now.format("YYYY-MM-DD_HH-mm")

    print(f"\nSaving TLEs to directory: {tle_dir}/")

    for name, line1, line2 in satellites:
        # Create sanitized filename
        safe_name = sanitize_filename(name)
        filename = f"{safe_name}_{timestamp}.txt"
        filepath = tle_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{name}\n{line1}\n{line2}\n")

        print(f"  + {filename}")

    print(f"Total: {len(satellites)} TLE files saved")


def export_to_csv(results: list[dict], filepath: str) -> None:
    """Export results to CSV file."""
    if not results:
        return

    fieldnames = [
        "datetime", "satellite", "visible", "magnitude", "direction",
        "altitude", "azimuth", "ra_hms", "dec_dms", "distance_km"
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            mag = r.get('magnitude')
            mag_str = f"{mag:.1f}" if mag is not None else ""

            writer.writerow({
                "datetime": r["time"].format("YYYY-MM-DD HH:mm:ss"),
                "satellite": r["satellite"],
                "visible": "YES" if r["visible"] else "NO",
                "magnitude": mag_str,
                "direction": azimuth_to_cardinal(r['azimuth']),
                "altitude": f"{r['altitude']:.4f}",
                "azimuth": f"{r['azimuth']:.4f}",
                "ra_hms": hours_to_hms(r['ra']),
                "dec_dms": degrees_to_dms(r['dec']),
                "distance_km": f"{r['distance_km']:.2f}"
            })

    print(f"\nResults exported to: {filepath}")


def track_satellites(
    satellites: list[tuple[str, str, str]],
    lat: float,
    lon: float,
    start_time: str,
    end_time: str,
    days: int,
    interval_minutes: int,
    elevation: float = 0,
    visible_only: bool = False,
    min_altitude: float = 0,
    csv_output: str | None = None
) -> None:
    """
    Track satellites in the specified time range.
    """
    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation)

    # Create EarthSatellite objects
    earth_sats = []
    for name, line1, line2 in satellites:
        try:
            sat = EarthSatellite(line1, line2, name, ts)
            earth_sats.append((name, sat))
        except Exception as e:
            print(f"Error creating satellite {name}: {e}")

    if not earth_sats:
        print("No valid satellites to track.")
        return

    print(f"\n{'='*100}")
    print(f"Observer: Lat {lat:.4f}°, Lon {lon:.4f}°, Elevation {elevation}m")
    print(f"Calculation interval: every {interval_minutes} minutes")
    print(f"Time range: {start_time} - {end_time}")
    print(f"Days to calculate: {days}")
    if visible_only:
        print(f"Filter: Visible only (altitude > {min_altitude}°)")
    else:
        print("Filter: Showing visible and not visible (-v)")
    print(f"{'='*100}\n")

    all_results = []

    # Calculate for each day
    base_date = arrow.now().floor("day")

    for day_offset in range(days):
        current_date = base_date.shift(days=day_offset)
        start_dt, end_dt = parse_time_range(start_time, end_time, current_date)

        print(f"\n{'─'*100}")
        print(f"Date: {current_date.format('YYYY-MM-DD')} ({start_dt.format('HH:mm')} - {end_dt.format('HH:mm')})")
        print(f"{'─'*100}")

        # Generate time points
        current_time = start_dt
        while current_time <= end_dt:
            time_results = []

            for sat_name, sat in earth_sats:
                try:
                    pos = calculate_satellite_position(sat, observer, current_time, ts)
                    pos["satellite"] = sat_name

                    # Apply filters
                    if visible_only and (not pos["visible"] or pos["altitude"] < min_altitude):
                        continue

                    time_results.append(pos)
                    all_results.append(pos)

                except Exception as e:
                    if not visible_only:
                        print(f"  {current_time.format('HH:mm:ss')} | {sat_name:25s} | ERROR: {e}")

            # Show results for this interval
            if time_results:
                print(f"\n[{current_time.format('HH:mm:ss')}]")
                for pos in time_results:
                    output = format_position(pos, pos["satellite"])
                    print(f"  {output}")

            current_time = current_time.shift(minutes=interval_minutes)

    # Export to CSV if specified
    if csv_output:
        export_to_csv(all_results, csv_output)

    # Summary
    print(f"\n{'='*100}")
    print(f"Summary: {len(all_results)} positions calculated")
    if visible_only:
        visible_count = sum(1 for r in all_results if r["visible"])
        print(f"Visible positions: {visible_count}")
    print(f"{'='*100}")


def main():
    # Load configuration first to use as defaults
    cfg = load_config(CONFIG_FILE)

    parser = argparse.ArgumentParser(
        description="Track satellites using TLEs from Celestrak",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s --start 20:00 --end 23:00
  %(prog)s --lat 40.4168 --lon -3.7038 --start 22:00 --end 2:00 --days 5
  %(prog)s --start 21:00 --end 1:00 --interval 5 --days 3

Configuration:
  The {CONFIG_FILE} file contains default values for lat/lon and other parameters.
        """
    )

    parser.add_argument(
        "--lat", type=float, default=cfg["latitude"],
        help=f"Observer latitude (config: {cfg['latitude']})"
    )
    parser.add_argument(
        "--lon", type=float, default=cfg["longitude"],
        help=f"Observer longitude (config: {cfg['longitude']})"
    )
    parser.add_argument(
        "--elevation", type=float, default=cfg["elevation"],
        help=f"Observer elevation in meters (config: {cfg['elevation']})"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start time (format HH:MM, e.g.: 20:00). Default: current time rounded to interval"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End time (format HH:MM, e.g.: 23:00 or 1:00 for next day). Default: 1 hour after start"
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of days to calculate (default: 1)"
    )
    parser.add_argument(
        "--interval", type=int, default=cfg["interval"],
        help=f"Calculation interval in minutes (config: {cfg['interval']})"
    )
    parser.add_argument(
        "--file", type=str, default=cfg["sats_file"],
        help=f"File with satellite names to search (default from config: {cfg['sats_file']})"
    )
    parser.add_argument(
        "-v", "--show-hidden", action="store_true",
        help="Also show satellites not visible (below horizon)"
    )
    parser.add_argument(
        "--min-alt", type=float, default=cfg["min_altitude"],
        help=f"Minimum altitude to consider visible (config: {cfg['min_altitude']})"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--tle-dir", type=str, default=cfg["tle_dir"],
        help=f"Directory to save TLEs (config: {cfg['tle_dir']})"
    )
    parser.add_argument(
        "--no-save-tle", action="store_true",
        help="Don't save downloaded TLEs"
    )
    parser.add_argument(
        "--config", type=str, default=CONFIG_FILE,
        help=f"Configuration file (default: {CONFIG_FILE})"
    )

    args = parser.parse_args()

    # If a different config file was specified, reload
    if args.config != CONFIG_FILE:
        cfg = load_config(args.config)
        # Apply defaults from new config if not passed via CLI
        if args.lat is None:
            args.lat = cfg["latitude"]
        if args.lon is None:
            args.lon = cfg["longitude"]

    # Set default start/end times if not specified
    if args.start is None or args.end is None:
        default_start, default_end = get_default_time_range(args.interval)
        if args.start is None:
            args.start = default_start
        if args.end is None:
            # If start was specified but not end, add 1 hour to start
            if args.start != default_start:
                start_parts = args.start.split(":")
                start_hour = int(start_parts[0])
                start_min = int(start_parts[1]) if len(start_parts) > 1 else 0
                end_time = arrow.now().replace(hour=start_hour, minute=start_min).shift(hours=1)
                args.end = end_time.format("HH:mm")
            else:
                args.end = default_end
        print(f"Using time range: {args.start} - {args.end}")

    # Validate that we have lat/lon
    if args.lat is None:
        print(f"Error: Latitude required. Use --lat or configure in {CONFIG_FILE}")
        sys.exit(1)

    if args.lon is None:
        print(f"Error: Longitude required. Use --lon or configure in {CONFIG_FILE}")
        sys.exit(1)

    # Validate ranges
    if not (-90 <= args.lat <= 90):
        print("Error: Latitude must be between -90 and 90")
        sys.exit(1)

    if not (-180 <= args.lon <= 180):
        print("Error: Longitude must be between -180 and 180")
        sys.exit(1)

    if args.days < 1:
        print("Error: Days must be at least 1")
        sys.exit(1)

    if args.interval < 1:
        print("Error: Interval must be at least 1 minute")
        sys.exit(1)

    # Load satellite names
    sat_names = load_satellite_names(args.file)
    if not sat_names:
        print(f"Error: No satellites in {args.file}")
        sys.exit(1)

    print(f"Satellites to search: {len(sat_names)}")
    for name in sat_names:
        print(f"  - {name}")

    # Download TLEs
    satellites = download_all_tles(sat_names)
    if not satellites:
        print("Error: Could not download TLEs")
        sys.exit(1)

    # Save TLEs by default (unless --no-save-tle is used)
    if not args.no_save_tle:
        save_tles_to_dir(satellites, args.tle_dir)

    # Track satellites
    # By default only shows visible, with -v shows also not visible
    track_satellites(
        satellites=satellites,
        lat=args.lat,
        lon=args.lon,
        start_time=args.start,
        end_time=args.end,
        days=args.days,
        interval_minutes=args.interval,
        elevation=args.elevation,
        visible_only=not args.show_hidden,
        min_altitude=args.min_alt,
        csv_output=args.csv
    )


if __name__ == "__main__":
    main()
