from app.services.linked_sites import is_platform, site_of


def test_site_of_groups_by_registered_domain():
    assert site_of("www.ambulante.com.mx") == "ambulante.com.mx"
    assert site_of("b1mas4.us5.list-manage2.com") == "list-manage2.com"
    assert site_of("http://info.cern.ch/hypertext/") == "cern.ch"


def test_hosting_subdomains_are_their_own_sites():
    assert site_of("somebody.blogspot.com") == "somebody.blogspot.com"
    assert site_of("www.blogspot.com") == "blogspot.com"


def test_platforms():
    assert is_platform("facebook.com") and not is_platform("ambulante.com.mx")


async def test_follow_creates_once_with_provenance(client):
    from sqlalchemy import text
    from app.database import async_session_maker
    async with async_session_maker() as db:
        await db.execute(text("INSERT INTO projects (id, name, domain, status, ollama_model) "
                              "VALUES (9904, 'src', 'src.com', 'CREATED', 'm:1')"))
        await db.execute(text("INSERT INTO archived_files (id, project_id, original_url, timestamp, status, in_scope) "
                              "VALUES (99400, 9904, 'http://src.com/', '20100101000000', 'PROCESSED', 1)"))
        await db.execute(text("INSERT INTO link_targets (id, project_id, url, url_key, domain, is_internal, kind, recovery_status) "
                              "VALUES (99401, 9904, 'http://www.friend.org/a', 'friend.org/a', 'www.friend.org', 0, 'page', 'recovered')"))
        await db.execute(text("INSERT INTO page_links (project_id, source_file_id, target_id) VALUES (9904, 99400, 99401)"))
        await db.commit()
    r = client.get("/project/9904/_linked-sites")
    assert "friend.org" in r.text and "Start project" in r.text
    r = client.post("/project/9904/follow", data={"site": "friend.org"})
    new = r.headers["HX-Redirect"]
    assert r.status_code == 204 and new.startswith("/project/")
    assert client.post("/project/9904/follow", data={"site": "www.friend.org"}).headers["HX-Redirect"] == new
    page = client.get(new)
    assert "Started from the links of" in page.text
    assert "open project" in client.get("/project/9904/_linked-sites").text
