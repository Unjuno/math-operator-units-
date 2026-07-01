from __future__ import annotations

import re
from collections.abc import Iterable

_RESERVED_OPERATOR_RE = re.compile(r"^<OP_RESERVED_\d{4,}>$")


def is_reserved_operator_token(token: str) -> bool:
    return bool(_RESERVED_OPERATOR_RE.match(token))


def reserved_operator_tokens(vocab: Iterable[str]) -> list[str]:
    return [token for token in vocab if is_reserved_operator_token(token)]


def build_output_allow_list(vocab: Iterable[str], assigned_reserved_tokens: Iterable[str] = ()) -> list[bool]:
    assigned = set(assigned_reserved_tokens)
    allowed: list[bool] = []
    for token in vocab:
        allowed.append(not (is_reserved_operator_token(token) and token not in assigned))
    return allowed
