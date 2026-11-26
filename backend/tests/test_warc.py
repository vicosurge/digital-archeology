import os

from app.services.warc_export import warc_date


def test_warc_date():
    assert warc_date("19930422153000") == "1993-04-22T15:30:00Z"
    assert warc_date("20010101") == "2001-01-01T00:00:00Z"
    assert warc_date("") is None


async def test_warc_roundtrip(client, data_dir):
    import json
    from sqlalchemy import text
    from warcio.archiveiterator import ArchiveIterator
    from app.database import async_session_maker
    page = os.path.join(data_dir, "w.html")
    with open(page, "wb") as f:
        f.write(b"<html><body>Hello 1993</body></html>")
    async with async_session_maker() as db:
        await db.execute(text("INSERT INTO projects (id, name, domain, status) VALUES (59905, 'w', 'w.org', 'CREATED')"))
        for i, (ts, scope) in enumerate([("19930422153000", 1), ("19930422153000", 1), ("20200101000000", 0)]):
            await db.execute(text(
                "INSERT INTO archived_files (id, project_id, original_url, timestamp, mimetype, local_path, status, in_scope) "
                "VALUES (:id, 59905, 'http://w.org/', :ts, 'text/html', :p, 'PROCESSED', :s)"),
                {"id": 99500 + i, "ts": ts, "p": page, "s": scope})
        await db.execute(text("INSERT INTO file_analyses (file_id, description, tags) VALUES (99500, 'A greeting', :t)"),
                         {"t": json.dumps(["hello"])})
        await db.commit()
    r = client.post("/project/59905/warc", data={})
    assert r.status_code == 200 and "1 captures and 1 analyses" in r.text
    r = client.get("/project/59905/warc/download")
    assert r.status_code == 200
    path = os.path.join(data_dir, "out.warc.gz")
    with open(path, "wb") as f:
        f.write(r.content)
    with open(path, "rb") as f:
        records = [(rec.rec_type, rec.rec_headers.get_header("WARC-Target-URI"),
                    rec.rec_headers.get_header("WARC-Date"), rec.content_stream().read(), rec.rec_headers)
                   for rec in ArchiveIterator(f)]
    assert [r[0] for r in records] == ["warcinfo", "resource", "metadata"]
    _, uri, date, body, headers = records[1]
    assert uri == "http://w.org/" and date == "1993-04-22T15:30:00Z" and body.startswith(b"<html>")
    assert headers.get_header("WARC-Source-URI") == "https://web.archive.org/web/19930422153000id_/http://w.org/"
    assert records[2][4].get_header("WARC-Concurrent-To") == headers.get_header("WARC-Record-ID")
    assert json.loads(records[2][3])["description"] == "A greeting"
    # Out-of-scope captures on request.
    assert "2 captures" in client.post("/project/59905/warc", data={"all_captures": "1"}).text
