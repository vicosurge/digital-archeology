from app.services import history


def test_word_change_counts():
    assert history.word_change("a b c", "a b c") == {"added": 0, "removed": 0}
    assert history.word_change("a b c", "a x c d") == {"added": 2, "removed": 1}


def test_sentences_split_and_wrap():
    lines = history.sentences("One. Two! " + "w" * 400)
    assert lines[:2] == ["One.", "Two!"] and all(len(x) <= 160 for x in lines)


def test_side_by_side_escapes_markup():
    table = history.side_by_side("<b>old</b> text.", "<b>new</b> text.", "a", "b")
    assert "<b>new</b>" not in table and "&lt;b&gt;" in table


async def test_history_page(client, data_dir):
    import os
    from sqlalchemy import text
    from app.database import async_session_maker
    paths = []
    for i, body in enumerate(["Hello world.", "Hello world.", "Hello brave new world."]):
        p = os.path.join(data_dir, f"h{i}.html")
        with open(p, "w") as f:
            f.write(f"<p>{body}</p>")
        paths.append(p)
    async with async_session_maker() as db:
        await db.execute(text("INSERT INTO projects (id, name, domain, status) VALUES (901, 'h', 'h.com', 'CREATED')"))
        for i, (p, url) in enumerate(zip(paths, ["http://h.com/", "http://www.h.com/", "http://h.com:80/"])):
            await db.execute(text(
                "INSERT INTO archived_files (id, project_id, original_url, timestamp, mimetype, local_path, status, in_scope) "
                "VALUES (:id, 901, :u, :ts, 'text/html', :p, 'DOWNLOADED', 1)"),
                {"id": 9100 + i, "u": url, "ts": f"200{i}0101000000", "p": p})
        await db.commit()
        tl = await history.timeline(db, 901, "http://h.com/")
    assert [c["change"] for c in tl["captures"]] == ["first", "same", "changed"]
    assert tl["versions"] == 2
    r = client.get("/project/901/history", params={"url": "http://h.com/"})
    assert r.status_code == 200 and "brave" in r.text
    r = client.get("/project/901/history", params={"url": "http://h.com/", "a": 9100, "b": 9101})
    assert "identical" in r.text
