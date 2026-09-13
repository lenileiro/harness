"""Bounded protobuf wire primitives for Tencent's published Yuanbao Bot schema.

Field numbers follow Tencent/yuanbao-openclaw-plugin/src/access/ws/proto.
This implements text messages only, without generated code or loading schemas.
"""

from __future__ import annotations

from dataclasses import dataclass

Fields = dict[int, list[int | bytes]]


def varint(value: int) -> bytes:
    if not 0 <= value < 2**64:
        raise ValueError("Invalid protobuf integer")
    output = bytearray()
    while value > 127:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def field(number: int, value: str | bytes | int) -> bytes:
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    data = value.encode() if isinstance(value, str) else value
    return varint((number << 3) | 2) + varint(len(data)) + data


def parse(data: bytes) -> Fields:
    if len(data) > 2**20:
        raise ValueError("Protobuf message exceeds size limit")
    offset = 0

    def integer() -> int:
        nonlocal offset
        value = 0
        for shift in range(0, 70, 7):
            if offset >= len(data):
                raise ValueError("Truncated protobuf integer")
            byte = data[offset]
            offset += 1
            if shift == 63 and byte > 1:
                raise ValueError("Protobuf integer overflow")
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("Invalid protobuf integer")

    result: Fields = {}
    count = 0
    while offset < len(data):
        count += 1
        tag = integer()
        number, wire = tag >> 3, tag & 7
        if not 0 < number < 2**29 or count > 4096:
            raise ValueError("Invalid protobuf field")
        if wire == 0:
            value = integer()
        elif wire in {1, 2, 5}:
            length = integer() if wire == 2 else (8 if wire == 1 else 4)
            if length > len(data) - offset:
                raise ValueError("Truncated protobuf field")
            value = data[offset : offset + length]
            offset += length
        else:
            raise ValueError("Unsupported protobuf wire type")
        result.setdefault(number, []).append(value)
    return result


def binary(fields: Fields, number: int) -> bytes:
    values = fields.get(number, [])
    if not values:
        return b""
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise ValueError("Invalid singular protobuf bytes field")
    return values[0]


def string(fields: Fields, number: int) -> str:
    return binary(fields, number).decode("utf-8")


def number(fields: Fields, key: int) -> int:
    values = fields.get(key, [])
    if not values:
        return 0
    if len(values) != 1 or not isinstance(values[0], int):
        raise ValueError("Invalid singular protobuf integer field")
    return values[0]


@dataclass(frozen=True)
class Frame:
    kind: int
    command: str
    identifier: str
    module: str
    data: bytes = b""
    sequence: int = 0
    need_ack: bool = False
    status: int = 0

    def encode(self) -> bytes:
        head = b"".join(
            field(key, value)
            for key, value in (
                (1, self.kind),
                (2, self.command),
                (3, self.sequence),
                (4, self.identifier),
                (5, self.module),
                (6, int(self.need_ack)),
                (10, self.status),
            )
        )
        return field(1, head) + field(2, self.data)

    @classmethod
    def decode(cls, data: bytes) -> Frame:
        envelope = parse(data)
        head = parse(binary(envelope, 1))
        return cls(
            number(head, 1),
            string(head, 2),
            string(head, 4),
            string(head, 5),
            binary(envelope, 2),
            number(head, 3),
            bool(number(head, 6)),
            number(head, 10),
        )
