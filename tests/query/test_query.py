from mopidy.models import Album, Artist, Track
from mopidy.query import (
    And,
    Compare,
    MatchAll,
    Not,
    coerce_query,
    dict_to_expr,
    field_values,
    from_dict,
    is_search_expr,
    is_serialized_expr,
    matches,
    to_dict,
    to_legacy_query,
)


def _track(**kwargs):
    kwargs.setdefault("uri", "dummy:track")
    return Track(**kwargs)


class TestDictToExpr:
    def test_single_contains(self):
        assert dict_to_expr({"artist": ["ABBA"]}) == Compare(
            "artist",
            "contains",
            "ABBA",
        )

    def test_exact_uses_eq(self):
        assert dict_to_expr({"artist": ["ABBA"]}, exact=True) == Compare(
            "artist",
            "eq",
            "ABBA",
        )

    def test_multiple_values_and_fields_are_anded(self):
        expr = dict_to_expr({"any": ["a", "b"], "artist": ["xyz"]})
        assert expr == And(
            (
                Compare("any", "contains", "a"),
                Compare("any", "contains", "b"),
                Compare("artist", "contains", "xyz"),
            )
        )

    def test_empty_dict_is_match_all(self):
        assert dict_to_expr({}) == MatchAll()


class TestToLegacyQuery:
    def test_and_of_eq_is_pushed_down(self):
        expr = And(
            (
                Compare("album", "eq", "Exciter"),
                Compare("albumartist", "eq", "Depeche Mode"),
            )
        )
        assert to_legacy_query(expr) == {
            "album": ["Exciter"],
            "albumartist": ["Depeche Mode"],
        }

    def test_bare_ne_is_empty(self):
        assert to_legacy_query(Compare("artist", "ne", "ABBA")) == {}

    def test_bare_not_is_empty(self):
        assert to_legacy_query(Not(Compare("artist", "eq", "ABBA"))) == {}

    def test_ne_alongside_positive_is_dropped(self):
        expr = And(
            (
                Compare("album", "eq", "Exciter"),
                Compare("artist", "ne", "ABBA"),
            )
        )
        assert to_legacy_query(expr) == {"album": ["Exciter"]}

    def test_contains_and_starts_with_are_pushed_down(self):
        expr = And(
            (
                Compare("artist", "contains", "AB"),
                Compare("album", "starts_with", "Gold"),
            )
        )
        assert to_legacy_query(expr) == {"artist": ["AB"], "album": ["Gold"]}

    def test_match_all_is_empty(self):
        assert to_legacy_query(MatchAll()) == {}


class TestSerialize:
    def test_roundtrip_nested(self):
        expr = And(
            (
                Compare("album", "eq", "Exciter"),
                Not(Compare("artist", "contains", "xx")),
            )
        )
        assert from_dict(to_dict(expr)) == expr

    def test_match_all_roundtrip(self):
        assert from_dict(to_dict(MatchAll())) == MatchAll()

    def test_is_serialized_expr(self):
        assert is_serialized_expr({"eq": {"field": "album", "value": "A"}})
        assert is_serialized_expr({"match_all": True})
        assert not is_serialized_expr({"artist": ["ABBA"]})
        assert not is_serialized_expr({"any": ["a"]})


class TestCoerceQuery:
    def test_empty_dict_is_none(self):
        assert coerce_query({}) is None

    def test_legacy_dict(self):
        assert coerce_query({"artist": ["A"]}, exact=True) == Compare(
            "artist",
            "eq",
            "A",
        )

    def test_passthrough_expr(self):
        expr = Compare("any", "contains", "xx")
        assert coerce_query(expr) is expr

    def test_serialized_dict(self):
        assert coerce_query({"contains": {"field": "any", "value": "xx"}}) == Compare(
            "any",
            "contains",
            "xx",
        )

    def test_is_search_expr(self):
        assert is_search_expr(MatchAll())
        assert is_search_expr(Compare("a", "eq", "b"))
        assert not is_search_expr({"artist": ["A"]})


class TestMatches:
    def test_equals_is_case_sensitive(self):
        track = _track(name="Exciter")
        assert matches(Compare("track_name", "eq", "Exciter"), track)
        assert not matches(Compare("track_name", "eq", "exciter"), track)

    def test_not_equals_is_negation_of_equals(self):
        track = _track(name="Exciter")
        assert not matches(Compare("track_name", "ne", "Exciter"), track)
        assert matches(Compare("track_name", "ne", "Something Else"), track)

    def test_contains_is_case_insensitive_substring(self):
        track = _track(name="Exciter")
        assert matches(Compare("track_name", "contains", "cit"), track)
        assert matches(Compare("track_name", "contains", "CIT"), track)
        assert not matches(Compare("track_name", "contains", "zzz"), track)

    def test_starts_with_is_case_insensitive_prefix(self):
        track = _track(name="Exciter")
        assert matches(Compare("track_name", "starts_with", "exc"), track)
        assert not matches(Compare("track_name", "starts_with", "cit"), track)

    def test_negation(self):
        track = _track(name="Exciter")
        assert not matches(Not(Compare("track_name", "eq", "Exciter")), track)
        assert matches(Not(Compare("track_name", "eq", "Something Else")), track)

    def test_and(self):
        track = _track(name="Exciter", genre="Rock")
        assert matches(
            And(
                (
                    Compare("track_name", "eq", "Exciter"),
                    Compare("genre", "eq", "Rock"),
                )
            ),
            track,
        )
        assert not matches(
            And(
                (
                    Compare("track_name", "eq", "Exciter"),
                    Compare("genre", "eq", "Pop"),
                )
            ),
            track,
        )

    def test_match_all(self):
        assert matches(MatchAll(), _track(name="Anything"))

    def test_multi_valued_artist_field(self):
        track = _track(artists=frozenset([Artist(name="Alice"), Artist(name="Bob")]))
        assert matches(Compare("artist", "eq", "Alice"), track)
        assert matches(Compare("artist", "eq", "Bob"), track)
        assert not matches(Compare("artist", "eq", "Carol"), track)

    def test_albumartist_comes_from_album(self):
        track = _track(
            album=Album(name="Exciter", artists=frozenset([Artist(name="DM")])),
        )
        assert matches(Compare("albumartist", "eq", "DM"), track)
        assert not matches(Compare("albumartist", "eq", "Other"), track)

    def test_any_matches_across_fields(self):
        track = _track(name="Exciter", genre="Rock")
        assert matches(Compare("any", "eq", "Exciter"), track)
        assert matches(Compare("any", "eq", "Rock"), track)
        assert not matches(Compare("any", "eq", "Pop"), track)

    def test_any_contains(self):
        track = _track(name="Exciter", genre="Rock")
        assert matches(Compare("any", "contains", "cit"), track)
        assert not matches(Compare("any", "contains", "zzz"), track)

    def test_field_values_any_is_union_of_all_fields(self):
        track = _track(name="Exciter", genre="Rock")
        values = field_values(track, "any")
        assert "Exciter" in values
        assert "Rock" in values
