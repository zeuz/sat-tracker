# Satellite Tracker

Command-line tool for predicting the visibility of highly elliptical orbit (HEO) and other specialized satellites from a configurable geographic location.

## Features

- Automatic download and caching of TLE data from Celestrak
- Calculates altitude, azimuth, RA/Dec, and apparent visual magnitude
- Filters satellites visible above the horizon
- Supports time ranges that cross midnight (e.g. 22:00–01:00)
- Multi-day consecutive calculations
- CSV export
- INI-based configuration file

## Installation

### 1. Clone / download the project

```bash
cd /path/to/project
```

### 2. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Edit `sat_tracker.cfg` to set the observer location and default parameters:

```ini
[observer]
latitude  = 22.16       # Latitude (decimal degrees, N positive)
longitude = -100.993    # Longitude (decimal degrees, E positive)
elevation = 1820        # Elevation above sea level (meters)

[tracking]
interval     = 10       # Time between calculations (minutes)
min_altitude = 0        # Minimum altitude to consider a satellite visible (degrees)

[files]
sats_file = sats-to-search.txt
tle_dir   = tle
```

## Satellites to track

The file `sats-to-search.txt` contains one satellite name per line. Names must match the Celestrak catalog:

```
MOLNIYA-1-91
MERIDIAN-10
ARKTIKA-M-1
QZS-1
```

## Usage

### Basic syntax

```bash
python sat_tracker.py --start HH:MM --end HH:MM [options]
```

### Examples

```bash
# Visibility window from 20:00 to 23:00 (location from config)
python sat_tracker.py --start 20:00 --end 23:00

# Custom location (Madrid)
python sat_tracker.py --lat 40.4168 --lon -3.7038 --start 20:00 --end 23:00

# Midnight-crossing range over 3 consecutive days
python sat_tracker.py --start 22:00 --end 01:00 --days 3

# 5-minute interval, show non-visible satellites too
python sat_tracker.py --start 20:00 --end 23:00 --interval 5 -v

# Export results to CSV
python sat_tracker.py --start 20:00 --end 23:00 --csv results.csv

# Skip saving downloaded TLE files
python sat_tracker.py --start 20:00 --end 23:00 --no-save-tle

# Use an alternative config file
python sat_tracker.py --config other.cfg --start 20:00 --end 23:00
```

### Available arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--start` | Start time `HH:MM` | Current time rounded to interval |
| `--end` | End time `HH:MM` | 1 hour after start |
| `--days` | Number of consecutive days | `1` |
| `--interval` | Time between calculations (minutes) | From config (`10`) |
| `--lat` | Observer latitude | From config |
| `--lon` | Observer longitude | From config |
| `--elevation` | Elevation in meters | From config |
| `--min-alt` | Minimum altitude for "visible" (degrees) | From config (`0`) |
| `--file` | File with satellite names | From config |
| `--tle-dir` | Directory to save TLE files | From config (`tle/`) |
| `--no-save-tle` | Do not save downloaded TLEs | (save by default) |
| `--csv` | Export results to this CSV file | (no export) |
| `--config` | Path to configuration file | `sat_tracker.cfg` |
| `-v`, `--show-hidden` | Show non-visible satellites as well | (visible only) |

## Output format

```
[HH:MM:SS]
  YYYY-MM-DD HH:mm:ss | MOLNIYA-1-91  | +4.2 | NNE |  +42.15° |   45.32° | RA: 12h34m56.78s | Dec: +23°45'12.34" | 38,234 km
  YYYY-MM-DD HH:mm:ss | MERIDIAN-10   | NOT-VISIBLE (alt:  -5.45°)
```

| Field | Description |
|-------|-------------|
| Date/time | UTC of the calculation |
| Name | Satellite name |
| Magnitude | Apparent visual magnitude (`+` = dimmer) |
| Azimuth | Direction on a 16-point compass rose (N, NNE, NE…) |
| Altitude | Elevation above the horizon (degrees) |
| Azimuth° | Numeric azimuth (degrees) |
| RA | Right ascension (equatorial coordinates) |
| Dec | Declination (equatorial coordinates) |
| Distance | Distance to the satellite (km) |

## Project structure

```
sat-tracker/
├── sat_tracker.py        # Main application
├── sat_tracker.cfg       # Observer configuration
├── requirements.txt      # Python dependencies
├── sats-to-search.txt    # List of satellites to track
└── tle/                  # Downloaded TLE cache

```

## Dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| `skyfield` | 1.48 | Orbital position calculations |
| `arrow` | 1.3.0 | Date and time handling |
| `requests` | 2.31.0 | TLE downloads from Celestrak |
