"""Structured library search expressions.

Mopidy's legacy search API is a dict mapping field names to lists of values,
plus a global `exact` flag. That cannot represent mixed operators, negation, or
an unconstrained search. This module is the additive replacement: an immutable
expression tree that frontends (MPD filter expressions, JSON-RPC) and backends
(SQL, remote search) share.

The tree is used by [LibraryController.search_with_expr][] only. Legacy
[LibraryController.search][] keeps the dict + `exact` contract and is never
converted into an expression. Backends that do not implement
`search_with_expr` are skipped.

The special field `any` matches if any metadata field satisfies the operator,
e.g. `Compare("any", "contains", "zz")`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeGuard, Union, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from mopidy.models import Artist, Track
    from mopidy.types import Query, SearchField

CompareOp = Literal["eq", "ne", "contains", "starts_with"]
"""Comparison operator used in [Compare][mopidy.query.Compare]."""

COMPARE_OPS: frozenset[str] = frozenset(
    ("eq", "ne", "contains", "starts_with"),
)

_SERIALIZED_KEYS = frozenset((*COMPARE_OPS, "and", "not", "match_all"))


@dataclass(frozen=True)
class Compare:
    """A comparison of one field against a single string value.

    `field` is a [SearchField][mopidy.types.SearchField], including the special
    field `any` (match if *any* metadata value satisfies the operator).

    Operators:

    - `eq` / `ne`: case-sensitive exact match. `ne` is true if *no* value
      equals the operand.
    - `contains` / `starts_with`: case-insensitive substring / prefix match.

    A multi-valued field (e.g. several artists) matches if any value satisfies
    `eq`/`contains`/`starts_with`.
    """

    field: str
    op: CompareOp
    value: str


@dataclass(frozen=True)
class Not:
    """Logical negation of a sub-expression."""

    expr: SearchExpr


@dataclass(frozen=True)
class And:
    """Conjunction of one or more sub-expressions."""

    exprs: tuple[SearchExpr, ...]


@dataclass(frozen=True)
class MatchAll:
    """Unconstrained query: every track matches.

    Distinct from an empty legacy dict, which core still treats as "do not
    search" for backward compatibility with broken clients.
    """


SearchExpr = Union[Compare, Not, And, MatchAll]
"""A library search expression."""


def is_search_expr(obj: object) -> TypeGuard[SearchExpr]:
    """Return whether `obj` is a [SearchExpr][mopidy.query.SearchExpr]."""
    return isinstance(obj, Compare | Not | And | MatchAll)


def is_serialized_expr(obj: object) -> TypeGuard[dict[str, object]]:
    """Return whether `obj` looks like a JSON-RPC serialized expression.

    Serialized form is a single-key dict, e.g. `{"eq": {"field": "album",
    "value": "Exciter"}}`, `{"and": [...]}`, `{"not": {...}}`, or
    `{"match_all": true}`.
    """
    if not isinstance(obj, dict) or len(obj) != 1:
        return False
    key = next(iter(obj))
    return key in _SERIALIZED_KEYS


def dict_to_expr(
    query: Query[SearchField],
    *,
    exact: bool = False,
) -> SearchExpr:
    """Compile a legacy field→values dict into an expression.

    Multiple values for one field, and multiple fields, are ANDed. `exact=True`
    becomes `eq`; otherwise `contains`. An empty dict becomes [MatchAll][].
    """
    op: CompareOp = "eq" if exact else "contains"
    clauses: list[SearchExpr] = []
    for field, values in query.items():
        for value in values:
            clauses.append(Compare(field=str(field), op=op, value=str(value)))
    if not clauses:
        return MatchAll()
    if len(clauses) == 1:
        return clauses[0]
    return And(tuple(clauses))


_PUSH_DOWN_OPS: frozenset[str] = frozenset(("eq", "contains", "starts_with"))


def to_legacy_query(expr: SearchExpr) -> dict[str, list[str]]:
    """Best-effort dict over-approximation of an expression.

    Collects `eq`/`contains`/`starts_with` comparisons that are not under a
    negation. `ne` and `Not` are omitted so the dict never excludes a true
    match. The result may be empty (e.g. a bare negation); callers must not
    treat that as "do not search" when talking to a backend directly.
    """
    query: dict[str, list[str]] = {}
    _collect_pushdown(expr, negated=False, query=query)
    return query


def _collect_pushdown(
    expr: SearchExpr,
    *,
    negated: bool,
    query: dict[str, list[str]],
) -> None:
    if isinstance(expr, Compare):
        if not negated and expr.op in _PUSH_DOWN_OPS:
            query.setdefault(expr.field, []).append(expr.value)
    elif isinstance(expr, Not):
        _collect_pushdown(expr.expr, negated=not negated, query=query)
    elif isinstance(expr, And):
        for sub in expr.exprs:
            _collect_pushdown(sub, negated=negated, query=query)


def to_dict(expr: SearchExpr) -> dict[str, object]:
    """Serialize an expression to a JSON-RPC-friendly nested dict."""
    if isinstance(expr, MatchAll):
        return {"match_all": True}
    if isinstance(expr, Compare):
        return {expr.op: {"field": expr.field, "value": expr.value}}
    if isinstance(expr, Not):
        return {"not": to_dict(expr.expr)}
    if isinstance(expr, And):
        return {"and": [to_dict(sub) for sub in expr.exprs]}
    raise TypeError(f"unknown expression node {expr!r}")


def from_dict(data: Mapping[str, object]) -> SearchExpr:
    """Deserialize the JSON-RPC form produced by [to_dict][]."""
    if not is_serialized_expr(dict(data)):
        msg = f"not a serialized search expression: {data!r}"
        raise ValueError(msg)
    key, val = next(iter(data.items()))
    if key == "match_all":
        return MatchAll()
    if key == "and":
        if not isinstance(val, list) or not val:
            msg = "and must be a non-empty list"
            raise ValueError(msg)
        return And(tuple(from_dict(cast("Mapping[str, object]", item)) for item in val))
    if key == "not":
        if not isinstance(val, Mapping):
            msg = "not must be a mapping"
            raise ValueError(msg)
        return Not(from_dict(val))
    if key in COMPARE_OPS:
        if not isinstance(val, Mapping):
            msg = f"{key} must be a mapping"
            raise ValueError(msg)
        field = val.get("field")
        value = val.get("value")
        if not isinstance(field, str) or not isinstance(value, str):
            msg = f"{key} requires string field and value"
            raise ValueError(msg)
        return Compare(field=field, op=cast("CompareOp", key), value=value)
    msg = f"unknown expression key {key!r}"
    raise ValueError(msg)


def coerce_query(
    query: Query[SearchField] | SearchExpr | Mapping[str, object],
    *,
    exact: bool = False,
) -> SearchExpr | None:
    """Normalize a search argument to a [SearchExpr][] or `None`.

    `None` means an empty legacy dict: callers should skip the search, matching
    historical `library.search({})` behaviour. [MatchAll][] is returned only
    when requested explicitly (or via `{"match_all": true}`).
    """
    if is_search_expr(query):
        return query
    if not isinstance(query, Mapping):
        msg = f"Expected a query dictionary or SearchExpr, not {query!r}"
        raise TypeError(msg)
    if is_serialized_expr(query):
        return from_dict(query)
    if not query:
        return None
    return dict_to_expr(cast("Query[SearchField]", query), exact=exact)


def _artist_names(artists: Iterable[Artist]) -> tuple[str, ...]:
    return tuple(a.name for a in artists if a.name)


def _mbid(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    return (str(value),)


_FIELD_EXTRACTORS: dict[str, Callable[[Track], tuple[str, ...]]] = {
    "uri": lambda t: (t.uri,) if t.uri else (),
    "track_name": lambda t: (t.name,) if t.name else (),
    "album": lambda t: (t.album.name,) if t.album and t.album.name else (),
    "artist": lambda t: _artist_names(t.artists),
    "albumartist": lambda t: _artist_names(t.album.artists) if t.album else (),
    "composer": lambda t: _artist_names(t.composers),
    "performer": lambda t: _artist_names(t.performers),
    "track_no": lambda t: (str(t.track_no),) if t.track_no is not None else (),
    "genre": lambda t: (t.genre,) if t.genre else (),
    "date": lambda t: (str(t.date),) if t.date else (),
    "comment": lambda t: (t.comment,) if t.comment else (),
    "disc_no": lambda t: (str(t.disc_no),) if t.disc_no is not None else (),
    "musicbrainz_albumid": lambda t: (
        _mbid(t.album.musicbrainz_id) if t.album else ()
    ),
    "musicbrainz_artistid": lambda t: tuple(
        str(a.musicbrainz_id) for a in t.artists if a.musicbrainz_id
    ),
    "musicbrainz_trackid": lambda t: _mbid(t.musicbrainz_id),
}

_FIELD_NAMES = tuple(_FIELD_EXTRACTORS)


def field_values(track: Track, field: str) -> tuple[str, ...]:
    """Return every string value `field` yields for `track`.

    `any` is the union of all other fields.
    """
    if field == "any":
        values: list[str] = []
        for other in _FIELD_NAMES:
            values.extend(field_values(track, other))
        return tuple(values)
    extractor = _FIELD_EXTRACTORS.get(field)
    if extractor is None:
        return ()
    return extractor(track)


def matches(expr: SearchExpr, track: Track) -> bool:
    """Evaluate `expr` exactly against a single track."""
    if isinstance(expr, MatchAll):
        return True
    if isinstance(expr, Compare):
        values = field_values(track, expr.field)
        if expr.op == "eq":
            return any(v == expr.value for v in values)
        if expr.op == "ne":
            return not any(v == expr.value for v in values)
        if expr.op == "contains":
            needle = expr.value.lower()
            return any(needle in v.lower() for v in values)
        if expr.op == "starts_with":
            needle = expr.value.lower()
            return any(v.lower().startswith(needle) for v in values)
        raise AssertionError(f"unknown operator {expr.op!r}")
    if isinstance(expr, Not):
        return not matches(expr.expr, track)
    if isinstance(expr, And):
        return all(matches(sub, track) for sub in expr.exprs)
    raise AssertionError(f"unknown expression node {expr!r}")
