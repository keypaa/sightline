"""Read a COCO instances file, keeping each image's own licence.

COCO's dataset-level licence covers the annotations, not the photographs: the
consortium states it does not own image copyright and that use of the images
follows the Flickr terms. Each image therefore carries a licence id, and that id
is what a gate can act on. Dropping it here would make every later stage guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_annotations(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return per-image records and the licence table keyed by licence id.

    No filtering happens here. A missing or unknown licence id yields
    ``license_name=None`` and the record is still returned, so the caller can
    count how many images are affected rather than silently losing them.
    """

    data = json.loads(Path(path).read_text())
    table = {int(entry["id"]): entry for entry in data.get("licenses", [])}

    records = []
    for image in data.get("images", []):
        licence_id = image.get("license")
        entry = table.get(int(licence_id)) if licence_id is not None else None
        record = {
            "id": int(image["id"]),
            "file_name": image["file_name"],
            "license_id": int(licence_id) if licence_id is not None else None,
            "license_name": entry["name"] if entry else None,
            "license_url": entry.get("url") if entry else None,
            # s.4(a) requires crediting the original author; COCO records no
            # author name, so the Flickr URL is the attribution route.
            "flickr_url": image.get("flickr_url"),
            "coco_url": image.get("coco_url"),
        }
        # Provenance travels with the image, not with the URL it was found at.
        record["source"] = "coco"
        # The annotations file has no per-image checksum; the fetch stage fills
        # this in once the file is downloaded, if it is not already present.
        record.setdefault("expected_sha256", None)
        records.append(record)
    return records, table
