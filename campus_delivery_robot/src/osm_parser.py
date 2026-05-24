"""Small OSM XML parser for campus path-planning data."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def _local_name(tag: str) -> str:
    """Return a tag name without an optional XML namespace."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _extract_tags(element: ET.Element) -> dict[str, str]:
    """Extract OSM key/value tags from a node or way element."""
    tags: dict[str, str] = {}
    for child in element:
        if _local_name(child.tag) != "tag":
            continue
        key = child.attrib.get("k")
        value = child.attrib.get("v")
        if key and value is not None:
            tags[key] = value
    return tags


def _extract_node_refs(element: ET.Element) -> list[str]:
    """Extract referenced node ids from a way element."""
    refs: list[str] = []
    for child in element:
        if _local_name(child.tag) != "nd":
            continue
        ref = child.attrib.get("ref")
        if ref:
            refs.append(ref)
    return refs


def parse_osm_file(osm_path: str | Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Parse an .osm XML file into node and way dictionaries.

    Returns:
        nodes: node_id -> {"id", "lat", "lon", "tags"}
        ways: list of {"id", "node_refs", "tags"}
        metadata: lightweight counts and optional map bounds
    """
    path = Path(osm_path)
    if not path.exists():
        raise FileNotFoundError(f"OSM file not found: {path}")

    nodes: dict[str, dict[str, Any]] = {}
    ways: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "bounds": None,
        "node_count": 0,
        "way_count": 0,
        "source_file": str(path),
    }

    for _, element in ET.iterparse(path, events=("end",)):
        tag = _local_name(element.tag)

        if tag == "bounds":
            try:
                metadata["bounds"] = {
                    "minlat": float(element.attrib["minlat"]),
                    "minlon": float(element.attrib["minlon"]),
                    "maxlat": float(element.attrib["maxlat"]),
                    "maxlon": float(element.attrib["maxlon"]),
                }
            except (KeyError, ValueError):
                metadata["bounds"] = None
            element.clear()

        elif tag == "node":
            node_id = element.attrib.get("id")
            lat = element.attrib.get("lat")
            lon = element.attrib.get("lon")
            if node_id and lat is not None and lon is not None:
                try:
                    nodes[node_id] = {
                        "id": node_id,
                        "lat": float(lat),
                        "lon": float(lon),
                        "tags": _extract_tags(element),
                    }
                    metadata["node_count"] += 1
                except ValueError:
                    pass
            element.clear()

        elif tag == "way":
            way_id = element.attrib.get("id")
            if way_id:
                ways.append(
                    {
                        "id": way_id,
                        "node_refs": _extract_node_refs(element),
                        "tags": _extract_tags(element),
                    }
                )
                metadata["way_count"] += 1
            element.clear()

        elif tag == "relation":
            element.clear()

    return nodes, ways, metadata

