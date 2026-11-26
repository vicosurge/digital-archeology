from app.services import takeover


def page(words, spam=(), parked=(), cjk=0.0, length=200):
    return {"title": "", "generator": "", "spam": list(spam), "parked": list(parked),
            "words": set(words), "cjk": cjk, "length": length}


SITE = ["tijuana", "frontera", "cultura", "musica", "politica", "ciudad"]
SPAM = ["comprar", "cialis", "viagra", "precio", "envio", "pastillas"]


def test_detects_a_lasting_spam_takeover_after_a_lapse():
    years = {str(y): [page(SITE) for _ in range(10)] for y in range(2012, 2017)}
    years["2017"] = [page([], length=2) for _ in range(3)]              # placeholders
    years["2019"] = [page(SPAM, spam=["cialis", "viagra"]) for _ in range(10)]
    years["2020"] = [page(SPAM, spam=["cialis", "viagra"]) for _ in range(10)]
    stats = takeover.score_years(years)
    verdict = {s["year"]: ("bad" if s["bad"] else "thin" if s["thin"] else "ok") for s in stats}
    assert verdict["2016"] == "ok" and verdict["2017"] == "thin" and verdict["2019"] == "bad"
    p = takeover.propose(stats)
    assert p["first_bad_year"] == "2019" and p["last_good_year"] == "2016"
    assert p["gap_years"] == ["2017"] and p["confidence"] == "high"


def test_no_proposal_for_a_site_that_only_evolves():
    years = {str(y): [page(SITE + [f"topic{y}"]) for _ in range(10)] for y in range(1996, 2002)}
    assert takeover.propose(takeover.score_years(years)) is None


def test_a_single_bad_year_that_does_not_stick_is_ignored():
    years = {str(y): [page(SITE) for _ in range(10)] for y in range(2000, 2006)}
    years["2003"] = [page(SPAM, spam=["casino", "poker"]) for _ in range(10)]   # hacked, then fixed
    assert takeover.propose(takeover.score_years(years)) is None


def test_script_switch_counts():
    years = {"2010": [page(SITE) for _ in range(5)], "2011": [page(SITE) for _ in range(5)],
             "2012": [page(["博彩"], cjk=0.8) for _ in range(5)], "2013": [page(["博彩"], cjk=0.8) for _ in range(5)]}
    assert takeover.propose(takeover.score_years(years))["first_bad_year"] == "2012"


def test_spam_terms_match_whole_words():
    assert takeover._terms("the pokerface album", ["poker"]) == []
    assert takeover._terms("Online Poker room", ["poker"]) == ["poker"]


async def test_scope_apply_and_clear(client):
    from sqlalchemy import text
    from app.database import async_session_maker
    async with async_session_maker() as db:
        await db.execute(text("INSERT INTO projects (id, name, domain, status) VALUES (9903, 't', 't.com', 'CREATED')"))
        for i, ts in enumerate(["20100101000000", "20200101000000"]):
            await db.execute(text("INSERT INTO archived_files (id, project_id, original_url, timestamp, status, in_scope) "
                                  "VALUES (:id, 9903, :u, :ts, 'PENDING', 1)"), {"id": 99300 + i, "u": f"http://t.com/{i}", "ts": ts})
        await db.commit()
    assert client.get("/project/9903/_takeover").status_code == 200
    r = client.post("/project/9903/scope", data={"action": "apply", "scope_end": "2015-01-01", "scope_reason": "sold"})
    assert r.status_code == 200 and "1 captures in scope, 1 kept aside" in r.text
    r = client.post("/project/9903/scope", data={"action": "clear"})
    assert "all 2 captures are in scope" in r.text
    assert client.post("/project/9903/scope", data={"scope_start": "2020-01-01", "scope_end": "2010-01-01"}).status_code == 400
