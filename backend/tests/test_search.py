from app.services.search_index import fts_query, render_snippet


def test_words_are_quoted_and_anded():
    assert fts_query("tim berners-lee") == '"tim" "berners-lee"'


def test_phrases_operators_and_prefix():
    assert fts_query('"world wide web" OR hypertext cern*') == '"world wide web" OR "hypertext" "cern"*'


def test_dangling_operators_and_junk_are_dropped():
    assert fts_query("OR NOT") == ""
    assert fts_query('" "') == ""
    assert fts_query("c++ AND") == '"c++"'


def test_snippet_is_escaped_but_marks_survive():
    out = render_snippet("<script>\x02x\x03</script>")
    assert out == "&lt;script&gt;<mark>x</mark>&lt;/script&gt;"


async def test_index_and_search_roundtrip(client, data_dir):
    import os
    from sqlalchemy import text
    from app.database import async_session_maker
    from app.services import search_index
    page = os.path.join(data_dir, "p.html")
    with open(page, "w", encoding="utf-8") as f:
        f.write("<html><body><p>La canción del pirata, de Espronceda.</p></body></html>")
    async with async_session_maker() as db:
        await db.execute(text("INSERT INTO projects (id, name, domain, status) VALUES (900, 's', 'x.com', 'CREATED')"))
        await db.execute(text(
            "INSERT INTO archived_files (id, project_id, original_url, timestamp, mimetype, local_path, status, in_scope) "
            "VALUES (9001, 900, 'http://x.com/a', '20010101000000', 'text/html', :p, 'DOWNLOADED', 1)"), {"p": page})
        await db.commit()
    assert (await search_index.index_project(900))["indexed"] == 1
    assert (await search_index.index_project(900))["indexed"] == 0     # incremental
    async with async_session_maker() as db:
        res = await search_index.search(db, 900, "cancion")             # diacritics folded
        assert res["total"] == 1 and "<mark>canción</mark>" in res["rows"][0]["snippet"]
        assert (await search_index.search(db, 900, "nothing-here"))["total"] == 0
    r = client.get("/project/900/_fulltext", params={"q": 'pirata "de Espronceda"'})
    assert r.status_code == 200 and "<mark>" in r.text
    assert client.get("/project/900/_fulltext", params={"q": 'AND "'}).status_code == 200
    # Leaving the scope removes a page from the index.
    async with async_session_maker() as db:
        await db.execute(text("UPDATE archived_files SET in_scope = 0 WHERE id = 9001"))
        await db.commit()
    await search_index.index_project(900)
    async with async_session_maker() as db:
        assert (await search_index.search(db, 900, "cancion"))["total"] == 0
        assert await search_index.pending_count(db, 900) == 0


def test_url_key_merges_spellings():
    from app.services.search_index import url_key
    assert url_key("http://www.x.com:80/a/") == url_key("https://x.com/a") == "x.com/a"
    assert url_key("http://x.com/a") != url_key("http://x.com/b")
