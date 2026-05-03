"""
format_demo.py
==============
Demonstrates Response Format Negotiation between two AutoGen-style agents
using a four-step protocol:

    STEP 1 — DECLARE:   Each agent advertises the formats it supports and
                        its preferred ordering (e.g. XML > JSON).
    STEP 2 — NEGOTIATE: The sender and receiver find a mutually supported
                        format, honouring the sender's preference first.
    STEP 3 — CONVERT:   If the sender's chosen encoding differs from the
                        agreed format, the payload is transparently converted.
    STEP 4 — DELIVER:   The converted payload is handed to the receiver with
                        the correct Content-Type header.  If no common format
                        exists the exchange is aborted and logged (FALLBACK).

No external packages are required — only the Python standard library.
"""

import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Serialisation helpers  (Step 3: JSON ↔ XML round-trip converters)
# ---------------------------------------------------------------------------

def dict_to_json_bytes(d: Dict) -> bytes:
    """Serialise a dict to compact, UTF-8-encoded JSON bytes."""
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def json_bytes_to_dict(b: bytes) -> Dict:
    """Deserialise UTF-8-encoded JSON bytes back to a dict."""
    return json.loads(b.decode("utf-8"))


def dict_to_xml_bytes(d: Dict) -> bytes:
    """Serialise a flat dict to a UTF-8-encoded XML <message> document.

    Each key becomes a child element whose text content is the string
    representation of the corresponding value.
    """
    root = ET.Element("message")
    for k, v in d.items():
        child = ET.SubElement(root, str(k))
        child.text = str(v)
    return ET.tostring(root, encoding="utf-8")


def xml_bytes_to_dict(b: bytes) -> Dict:
    """Parse a UTF-8-encoded XML <message> document into a dict.

    Numeric text values are automatically coerced to int or float.
    """
    root = ET.fromstring(b.decode("utf-8"))
    out = {}
    for child in root:
        text = child.text or ""
        try:
            num = float(text)
            out[child.tag] = int(num) if num.is_integer() else num
        except ValueError:
            out[child.tag] = text
    return out


def convert_bytes(src_fmt: str, dst_fmt: str, payload: bytes) -> bytes:
    """Convert *payload* from *src_fmt* to *dst_fmt* (JSON or XML).

    Returns the original payload unchanged when both formats are identical.
    Raises ValueError for unsupported format strings.
    """
    src_fmt, dst_fmt = src_fmt.upper(), dst_fmt.upper()
    if src_fmt == dst_fmt:
        return payload

    # Decode source payload to an intermediate dict
    if src_fmt == "JSON":
        d = json_bytes_to_dict(payload)
    elif src_fmt == "XML":
        d = xml_bytes_to_dict(payload)
    else:
        raise ValueError(f"Unsupported source format: {src_fmt!r}")

    # Re-encode to the destination format
    if dst_fmt == "JSON":
        return dict_to_json_bytes(d)
    elif dst_fmt == "XML":
        return dict_to_xml_bytes(d)
    else:
        raise ValueError(f"Unsupported destination format: {dst_fmt!r}")


# ---------------------------------------------------------------------------
# Format negotiation  (Step 2)
# ---------------------------------------------------------------------------

def negotiate(
    sender_supported: List[str],
    receiver_supported: List[str],
    sender_pref: List[str],
    receiver_pref: List[str],
) -> Optional[str]:
    """Return the best mutually supported format, or None if no overlap exists.

    Selection priority:
      1. Sender's preference list (first match wins).
      2. Receiver's preference list (first match wins).
      3. Any remaining format from the intersection (arbitrary order).

    Note: With the Agent-to-Agent (A2A) protocol this negotiation step can
    be handled automatically via capability advertisements, removing the
    need for manual overlap detection.
    """
    overlap = {x.upper() for x in sender_supported} & {x.upper() for x in receiver_supported}
    if not overlap:
        return None

    for p in (x.upper() for x in sender_pref):
        if p in overlap:
            return p
    for p in (x.upper() for x in receiver_pref):
        if p in overlap:
            return p
    return next(iter(overlap))


# ---------------------------------------------------------------------------
# Agent model
# ---------------------------------------------------------------------------

class Agent:
    """Minimal agent that can send and receive messages in multiple formats.

    Attributes:
        name:      Human-readable identifier used in log output.
        supported: Formats this agent can encode/decode (e.g. ["XML", "JSON"]).
        pref:      Preferred format ordering, most-preferred first.
    """

    def __init__(self, name: str, supported: List[str], pref: List[str]) -> None:
        self.name = name
        self.supported = supported
        self.pref = pref

    def encode(self, fmt: str, message: Dict[str, Any]) -> bytes:
        """Encode *message* into *fmt* bytes.  Raises ValueError for unknown formats."""
        if fmt.upper() == "JSON":
            return dict_to_json_bytes(message)
        elif fmt.upper() == "XML":
            return dict_to_xml_bytes(message)
        else:
            raise ValueError(f"Unsupported encode format: {fmt!r}")

    def receive(self, payload: bytes, fmt: str) -> bool:
        """Return True if this agent can accept a payload in *fmt*."""
        return fmt.upper() in (x.upper() for x in self.supported)

    def send(self, receiver: "Agent", message: Dict[str, Any], src_format: str) -> Dict[str, Any]:
        """Execute the full four-step protocol and deliver *message* to *receiver*.

        Steps performed:
          1. (Declared externally — each agent's supported/pref lists.)
          2. NEGOTIATE  — find the agreed format.
          3. CONVERT    — re-encode if the agreed format differs from src_format.
          4. DELIVER    — pass the payload to the receiver (or FALLBACK on failure).

        Returns a result dict with keys: ok, content_type, agreed_format
        (or ok=False and a reason string on the fallback path).
        """
        # STEP 2: NEGOTIATE — find the best mutually supported format
        agreed = negotiate(self.supported, receiver.supported, self.pref, receiver.pref)
        print(
            f"NEGOTIATE: sender={self.name}, receiver={receiver.name}, "
            f"sender_supported={self.supported}, "
            f"receiver_supported={receiver.supported}, agreed={agreed}"
        )

        if not agreed:
            # STEP 4 (FALLBACK) — no common format; abort and log
            print("FALLBACK: No common format. Action=abort_and_log")
            return {"ok": False, "reason": "no_common_format"}

        # STEP 3: CONVERT — encode in src_format first, then convert if necessary
        payload_before = self.encode(src_format, message)

        if src_format.upper() != agreed.upper():
            print(f"CONVERT: {src_format.upper()} -> {agreed.upper()}")
            payload_after = convert_bytes(src_format, agreed, payload_before)
        else:
            payload_after = payload_before

        # Log the raw payloads before and after conversion for inspection
        try:
            print(f"PAYLOAD BEFORE ({src_format.upper()}):")
            print(payload_before.decode("utf-8"))
            print(f"PAYLOAD AFTER  ({agreed.upper()}):")
            print(payload_after.decode("utf-8"))
        except Exception:
            pass  # non-fatal; keep the demo robust if decoding fails

        # STEP 4: DELIVER — hand the payload to the receiver
        content_type = "application/json" if agreed.upper() == "JSON" else "application/xml"
        ok = receiver.receive(payload_after, agreed)
        print(f"DELIVER: content_type={content_type}, ok={ok}")
        return {"ok": ok, "content_type": content_type, "agreed_format": agreed}


# ---------------------------------------------------------------------------
# STEP 1: Declare agent capabilities
# ---------------------------------------------------------------------------

# Agent A (sender)  — supports XML and JSON, prefers XML
A = Agent("A", ["XML", "JSON"], ["XML", "JSON"])

# Agent B (receiver) — supports JSON only
B = Agent("B", ["JSON"], ["JSON"])

# Sample message payload
message = {"id": 101, "title": "Quarterly Report", "amount": 123.45}


# ---------------------------------------------------------------------------
# Entry point — run both the success path and the fallback path
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Success path: A wants to send XML but B only understands JSON.
    # The protocol auto-negotiates JSON and converts the payload before delivery.
    print("=== Success path (format mismatch auto-heals) ===")
    result1 = A.send(B, message, src_format="XML")
    print("RESULT:", result1, "\n")

    # Fallback path: C speaks only XML, D speaks only JSON — no overlap.
    # The protocol detects the deadlock and returns an error without crashing.
    print("=== Fallback path (no overlapping format) ===")
    C = Agent("C", ["XML"], ["XML"])   # XML-only sender
    D = Agent("D", ["JSON"], ["JSON"]) # JSON-only receiver
    result2 = C.send(D, message, src_format="XML")
    print("RESULT:", result2)
