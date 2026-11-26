"""Pure functions that are easy to break without noticing."""
from app.services import exclusions, local_view
from app.services.apparatus import fold, merge_variants
from app.services.enrichment import host_of, registration_candidates


# ---- exclusions ----

def test_exclusion_parse_ignores_blanks_and_comments():
    assert exclusions.parse("/wp-admin/\n\n# note\n*.css \n") == ["/wp-admin/", "*.css"]


def test_exclusion_folder_matches_with_and_without_slash():
    pats = ["/feed/"]
    assert exclusions.match("http://x.com/feed", pats) == "/feed/"
    assert exclusions.match("http://x.com/feed/", pats) == "/feed/"
    assert exclusions.match("http://x.com/feedback", pats) is None


def test_exclusion_glob_and_case():
    assert exclusions.match("http://x.com/STYLE.CSS", ["*.css"]) == "*.css"
    assert exclusions.match("http://x.com/a/wp-admin/x.php", ["*/wp-admin/*"])
    assert exclusions.match("http://x.com/index.html", ["*.css", "*/wp-admin/*"]) is None


def test_exclusion_query_only_when_pattern_has_one():
    assert exclusions.match("http://x.com/?p=1", ["*?p=*"]) == "*?p=*"
    assert exclusions.match("http://x.com/page?p=1", ["/page"]) == "/page"


def test_suggested_list_parses():
    pats = exclusions.parse(exclusions.SUGGESTED)
    assert "/wp-admin/" in pats or any("wp-admin" in p for p in pats)


# ---- local viewer ----

def test_url_variants_cover_www_port_and_index():
    v = local_view.url_variants("http://example.com/dir/")
    assert "http://www.example.com/dir/" in v
    assert "http://example.com:80/dir/index.html" in v
    assert "https://example.com/dir" in v


def test_rewrite_html_points_links_at_viewer_and_drops_base():
    body = (b'<html><head><base href="http://example.com/sub/"></head>'
            b'<body><a href="page.html#top">x</a><img src=\'/img/a.gif\'>'
            b'<a href="mailto:a@b.c">m</a><a href="javascript:void(0)">j</a></body></html>')
    out = local_view.rewrite_html(body, "http://example.com/", "20010101000000", 7)
    assert b"<base" not in out
    assert b"/project/7/local?ts=20010101000000&amp;url=http%3A%2F%2Fexample.com%2Fsub%2Fpage.html#top" in out
    assert b"url=http%3A%2F%2Fexample.com%2Fimg%2Fa.gif" in out
    assert b'href="mailto:a@b.c"' in out and b'href="javascript:void(0)"' in out


def test_rewrite_html_keeps_non_utf8_bytes():
    body = "<p>Ni\xf1o <a href='x.html'>x</a></p>".encode("latin-1")
    out = local_view.rewrite_html(body, "http://e.com/", "2001", 1)
    assert "Ni\xf1o".encode("latin-1") in out


# ---- registered domain walk ----

def test_host_of_strips_scheme_port_path():
    assert host_of("HTTPS://www.Example.com:8080/a/b?c") == "www.example.com"
    assert host_of("example.com.") == "example.com"


def test_registration_candidates():
    assert registration_candidates("info.cern.ch") == ["info.cern.ch", "cern.ch"]
    assert registration_candidates("www.bbc.co.uk") == ["www.bbc.co.uk", "bbc.co.uk"]
    assert registration_candidates("a.b.example.com.au")[-1] == "example.com.au"
    assert registration_candidates("example.com") == ["example.com"]


# ---- names ----

def test_fold_is_accent_and_case_insensitive():
    assert fold("César  Faz") == fold("cesar faz") == "cesar faz"


def test_merge_variants_is_conservative():
    m = merge_variants(["pena nieto", "enrique pena nieto", "lopez", "andres lopez", "jose lopez"])
    assert m["pena nieto"] == "enrique pena nieto"
    # Two longer Lopezes: ambiguous, left alone.
    assert m["lopez"] == "lopez"
