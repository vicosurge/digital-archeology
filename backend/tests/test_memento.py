import httpx
import pytest

from app.services import memento
from app.services.llm_client import RateLimiter

TIMEMAP = '''<http://a.org/x>; rel="original",
<https://wayback.archive-it.org/all/timemap/link/http://a.org/x>; rel="self"; type="application/link-format",
<https://wayback.archive-it.org/all/20100101000000/http://a.org/x>; rel="first memento"; datetime="Fri, 01 Jan 2010 00:00:00 GMT",
<https://wayback.archive-it.org/all/20120601000000/http://www.a.org/x>; rel="memento"; datetime="Fri, 01 Jun 2012 00:00:00 GMT",
<https://wayback.archive-it.org/all/20200101000000/http://a.org/x>; rel="last memento"; datetime="Wed, 01 Jan 2020 00:00:00 GMT"'''


def test_parse_timemap_and_order_by_distance():
    ms = memento.parse_timemap(TIMEMAP)
    assert [m["ts"] for m in ms] == ["20100101000000", "20120601000000", "20200101000000"]
    assert ms[1]["original"] == "http://www.a.org/x"
    assert memento.by_distance(ms, "20120301000000")[0]["ts"] == "20120601000000"


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


async def test_find_takes_the_closest_and_names_the_archive():
    def handler(request):
        u = str(request.url)
        if "archive-it.org/all/timemap" in u:
            return httpx.Response(200, text=TIMEMAP)
        if "archive-it.org/all/20120601000000id_/" in u:
            return httpx.Response(200, content=b"<html>2012</html>", headers={"content-type": "text/html"})
        return httpx.Response(404)
    async with _client(handler) as c:
        hit = await memento.find(c, RateLimiter(100000), "http://a.org/x", "20120301000000")
    assert hit["archive"] == "Archive-It" and hit["timestamp"] == "20120601000000"
    assert hit["content"] == b"<html>2012</html>"
    assert hit["view_url"] == "https://wayback.archive-it.org/all/20120601000000/http://www.a.org/x"


async def test_redirect_out_of_the_archive_is_never_followed():
    def handler(request):
        u = str(request.url)
        if "timemap" in u and "archive-it" in u:
            return httpx.Response(200, text=TIMEMAP)
        if "archive-it.org/all/" in u and "id_" in u:
            # The capture is of a redirect to the live, now-parked domain.
            return httpx.Response(302, headers={"location": "http://a.org/lander"})
        if u.startswith("http://a.org/"):
            raise AssertionError("fetched the live web")
        return httpx.Response(404)
    async with _client(handler) as c:
        assert await memento.find(c, RateLimiter(100000), "http://a.org/x", "20120301000000") is None


async def test_busy_archive_raises():
    async with _client(lambda r: httpx.Response(429)) as c:
        with pytest.raises(memento.ArchiveBusy):
            await memento.find(c, RateLimiter(100000), "http://a.org/x", None)


async def test_images_need_an_image_type():
    def handler(request):
        u = str(request.url)
        if "archive-it.org/all/timemap" in u:
            return httpx.Response(200, text=TIMEMAP)
        if "id_" in u:
            return httpx.Response(200, content=b"<html>not found</html>", headers={"content-type": "text/html"})
        return httpx.Response(404)
    async with _client(handler) as c:
        assert await memento.find(c, RateLimiter(100000), "http://a.org/x", None, want_image=True) is None


async def test_recover_links_records_the_source(client, monkeypatch, data_dir):
    from sqlalchemy import text
    from app.database import async_session_maker
    from app.services import memento as m

    async def fake_find(c, pace, url, ts, want_image=False):
        if url.endswith("/found"):
            return {"archive": "Arquivo.pt", "timestamp": "20141126131617", "content": b"<html>ok</html>",
                    "mimetype": "text/html", "original": url,
                    "view_url": "https://arquivo.pt/wayback/20141126131617/" + url}
        return None
    monkeypatch.setattr(m, "find", fake_find)
    monkeypatch.setattr(m, "LINKED_DIR", data_dir)
    async with async_session_maker() as db:
        await db.execute(text("INSERT INTO projects (id, name, domain, status) VALUES (59906, 'm', 'm.org', 'CREATED')"))
        for i, u in enumerate(["http://x.org/found", "http://x.org/lost"]):
            await db.execute(text(
                "INSERT INTO link_targets (id, project_id, url, url_key, domain, is_internal, kind, recovery_status, "
                "first_cited_ts, analysis_status) VALUES (:id, 59906, :u, :u, 'x.org', 0, 'article', 'not_archived', "
                "'20140903000000', 'not_selected')"), {"id": 599060 + i, "u": u})
        await db.commit()
        assert await m.pending(db, 59906) == 2
    result = await m.recover_links(59906)
    assert result == {"checked": 2, "found": 1}
    async with async_session_maker() as db:
        rows = (await db.execute(text(
            "SELECT url, recovery_status, source_archive, wayback_url, analysis_status, capture_distance_days "
            "FROM link_targets WHERE project_id = 59906 ORDER BY id"))).fetchall()
        assert rows[0][1:5] == ("recovered", "Arquivo.pt", "https://arquivo.pt/wayback/20141126131617/http://x.org/found", "pending")
        assert rows[0][5] == 84
        assert rows[1][1] == "not_archived" and rows[1][2] is None
        assert await m.pending(db, 59906) == 0          # each target is asked once
