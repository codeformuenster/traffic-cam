"""
Register device with API.
See https://counting-backend.codeformuenster.org/docs#/default/create_device_devices__post
"""

import json

from traffic_cam import api, paths

# read devices registered with API
devices = api.get_devices()
device_ids = [device["id"] for device in devices]

# read image classes in device format
with open(str(paths.CLASS_LOCATION), "r") as f:
    locations = json.loads(f.read())

for location in locations:
    # if location already registered with API: remove it
    if location["id"] in device_ids:
        api.delete_device(id=location["id"])
    # register location
    response = api.create_device(device=location)
