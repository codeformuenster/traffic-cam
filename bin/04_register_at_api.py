"""
Register device with API.
See https://counting-backend.codeformuenster.org/docs#/default/create_device_devices__post
"""

import json
import requests

from traffic_cam import paths

# read devices registered with API
devices = json.loads(
    requests.get("https://counting-backend.codeformuenster.org/devices").text
)
device_ids = [device["id"] for device in devices]

# read image classes in device format
with open(str(paths.CLASS_LOCATION), "r") as f:
    locations = json.loads(f.read())

for location in locations:
    # if location already registered with API: remove it
    if location["id"] in device_ids:
        response = requests.delete(
            f"https://counting-backend.codeformuenster.org/devices/{location['id']}",
        )
        assert response.status_code == 200, "Status code on deletion is not 200."
    # register location
    response = requests.post(
        "https://counting-backend.codeformuenster.org/devices/",
        json=location,
    )
    assert response.status_code == 201, "Response status code on insert not 201."
