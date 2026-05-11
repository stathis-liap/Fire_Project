import os
import requests

# ---------------------------------------------------------------------------
# 1. Telemetry parser
# ---------------------------------------------------------------------------
def read_drone_telemetry(filepath):
    """Reads Latitude and Longitude from a `Lat: ... \n Lon: ...` text file."""
    telemetry = {}
    print(f"[Fetcher] Reading telemetry from: {filepath}...")

    with open(filepath, 'r', encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) == 2:
                telemetry[parts[0].strip()] = float(parts[1].strip())

    return telemetry['Lat'], telemetry['Lon']


# ---------------------------------------------------------------------------
# 2. Copernicus DEM
# ---------------------------------------------------------------------------
def fetch_terrain_from_api(lat, lon, api_key, output_dir=".", buffer=0.01):
    """ Fetches DEM from OpenTopography. Has Offline Fallback. """
    west, east = lon - buffer, lon + buffer
    south, north = lat - buffer, lat + buffer
    print(f"\n[Fetcher] Target Location: Latitude {lat}, Longitude {lon}")
    print(f"[Fetcher] Bounding Box: S:{south:.4f}, N:{north:.4f}, W:{west:.4f}, E:{east:.4f}")
    
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        'demtype': 'COP30',
        'south': south, 'north': north, 'west': west, 'east': east,
        'outputFormat': 'GTiff',
        'API_Key': api_key
    }
    
    filename = os.path.join(output_dir, "auto_downloaded_terrain.tif")
    print("[Fetcher] Sending request to OpenTopography API...")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"[Fetcher] >>> SUCCESS! Terrain saved automatically as '{filename}' <<<")
        return filename
    except Exception as e:
        print(f"[Fetcher] >>> WARNING: Failed to download DEM (Internet/Server Error).")
        if os.path.exists(filename):
            print(f"[Fetcher] >>> OFFLINE MODE: Using existing local file '{filename}'.")
            return filename
        else:
            print("[Fetcher] >>> CRITICAL: No local file found to fallback on. Aborting.")
            return None


# ---------------------------------------------------------------------------
# 3. ESRI World Imagery satellite texture
# ---------------------------------------------------------------------------
def fetch_satellite_image_by_bounds(west, south, east, north, width, height, output_dir="."):
    """Fetches an ESRI World Imagery JPG perfectly preserving the DEM's aspect ratio."""
    print(f"\n[Fetcher] Fetching satellite texture perfectly scaled to DEM bounds...")

    # Calculate a scale factor to get a high-res image (~2048 max) without distorting it!
    scale_factor = max(1, 2048 // max(width, height))
    tex_w = width * scale_factor
    tex_h = height * scale_factor

    url = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
           "World_Imagery/MapServer/export")
    params = {
        'bbox':    f"{west},{south},{east},{north}",
        'bboxSR':  '4326',
        'size':    f'{tex_w},{tex_h}', # Preserves the exact rectangular aspect ratio
        'imageSR': '4326',
        'format':  'jpg',
        'f':       'image'
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "texture.jpg")
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"[Fetcher] >>> SUCCESS: '{filename}' ({tex_w}x{tex_h}) <<<")
        return filename
    else:
        print(f"[Fetcher] >>> ERROR: status {response.status_code}")
        return None


# ---------------------------------------------------------------------------
# 4. EEA CORINE Land Cover
# ---------------------------------------------------------------------------
def fetch_corine_land_cover(west, south, east, north, width, height, output_dir="."):
    """ Fetches CORINE land cover. Has Offline Fallback. """
    print(f"\n[Fetcher] Fetching CORINE Land Cover (CLC 2018)...")

    url = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/export"
    params = {
        'bbox': f"{west},{south},{east},{north}",
        'bboxSR': '4326',
        'size': f"{width},{height}",
        'imageSR': '4326',
        'format': 'png',
        'transparent': 'false',
        'f': 'image'
    }

    filename = os.path.join(output_dir, "corine_cover.png")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"[Fetcher] >>> SUCCESS! CORINE saved as '{filename}' <<<")
        return filename
    except Exception as e:
        print(f"[Fetcher] >>> WARNING: Failed to download CORINE (Internet/Server Error).")
        if os.path.exists(filename):
            print(f"[Fetcher] >>> OFFLINE MODE: Using existing local file '{filename}'.")
            return filename
        else:
            raise Exception("No internet and no local fallback file available.")