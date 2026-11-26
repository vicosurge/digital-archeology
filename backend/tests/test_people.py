from collections import Counter

from app.services import people


def test_key_folds_case_accents_hyphens():
    assert people.key("Jean-François Groff") == people.key("jean francois GROFF") == "jean francois groff"


def test_auto_rule_merges_surname_into_unique_full_name():
    canon = people.resolve(Counter({"Tim Berners-Lee": 5, "Berners-Lee": 2, "Tim Berners Lee": 1}), {}, set())
    assert canon.name("Berners-Lee") == canon.name("tim berners lee") == "Tim Berners-Lee"


def test_ambiguous_surname_stays_alone_and_is_suggested():
    canon = people.resolve(Counter({"Ana López": 3, "Luis López": 2, "López": 1}), {}, set())
    assert canon.name("López") == "López"
    reasons = {(s["alias_name"], s["into_name"]): s["reason"] for s in people.suggestions(canon, set())}
    assert reasons[("López", "Ana López")].startswith("fits 2")


def test_initials_are_suggested_not_merged():
    canon = people.resolve(Counter({"Tim Berners-Lee": 9, "Tim BL": 2, "T. Berners-Lee": 1}), {}, set())
    assert canon.name("Tim BL") == "Tim BL"
    pairs = {(s["alias_name"], s["into_name"]) for s in people.suggestions(canon, set())}
    assert ("Tim BL", "Tim Berners-Lee") in pairs and ("T. Berners-Lee", "Tim Berners-Lee") in pairs
    assert not people.is_initials_variant("tim cook", "tim berners lee")


def test_manual_merge_chains_and_separate_blocks_auto():
    names = Counter({"Tim Berners-Lee": 9, "Tim BL": 2, "TimBL": 1, "Berners-Lee": 1})
    canon = people.resolve(names, {"timbl": "tim bl", "tim bl": "tim berners lee"}, set())
    assert canon.name("TimBL") == "Tim Berners-Lee"
    assert set(canon.variants("tim berners lee")) == set(names)
    canon = people.resolve(names, {}, {("berners lee", "tim berners lee")})
    assert canon.name("Berners-Lee") == "Berners-Lee"


async def test_people_page_merge_flow(client):
    import json
    from sqlalchemy import text
    from app.database import async_session_maker
    async with async_session_maker() as db:
        await db.execute(text("INSERT INTO projects (id, name, domain, status) VALUES (902, 'p', 'p.com', 'CREATED')"))
        for i, ppl in enumerate([["Tim Berners-Lee"], ["Tim BL"], ["Tim BL", "Robert Cailliau"]]):
            await db.execute(text(
                "INSERT INTO archived_files (id, project_id, original_url, timestamp, status, in_scope) "
                "VALUES (:id, 902, :u, '19930101000000', 'PROCESSED', 1)"), {"id": 9200 + i, "u": f"http://p.com/{i}"})
            await db.execute(text(
                "INSERT INTO file_analyses (file_id, people_mentioned) VALUES (:id, :p)"),
                {"id": 9200 + i, "p": json.dumps(ppl)})
        await db.commit()
    r = client.get("/project/902/people")
    assert r.status_code == 200 and "Same person" in r.text
    r = client.post("/project/902/people/merge", data={"key": "tim bl", "into": "tim berners lee"})
    assert r.status_code == 200 and "undo merge" in r.text
    r = client.get("/project/902/_results", params={"person": "Tim Berners-Lee"})
    assert "3 results" in r.text
    client.post("/project/902/people/unmerge", data={"key": "tim berners lee"})
    r = client.get("/project/902/_results", params={"person": "Tim Berners-Lee"})
    assert "1 result" in r.text
