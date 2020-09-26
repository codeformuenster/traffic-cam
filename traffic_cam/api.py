""" Functions to interact with API. """

import json
import requests
from typing import List

from traffic_cam import paths


def get_devices() -> List[dict]:
    url = f"{paths.API_URL}/devices"
    response = requests.get(url)
    devices = json.loads(response.text)
    return devices


def delete_device(id: str):
    url = f"{paths.API_URL}/devices/{id}"
    response = requests.delete(url)
    assert response.status_code == 200, "Status code on deletion is not 200."


def create_device(device: dict):
    url = f"{paths.API_URL}/devices/"
    response = requests.post(
        url=url,
        json=device,
    )
    assert response.status_code == 201, "Response status code on insert not 201."
