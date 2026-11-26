from app.services import estimate


def test_summarize_without_prices_has_no_dollars():
    e = estimate.summarize(100, 2000, 250, (0, 0), "x")
    assert e == {"calls": 100, "in_tokens": 200000, "out_tokens": 25000, "dollars": None, "basis": "x"}


def test_summarize_with_prices():
    e = estimate.summarize(1000, 2000, 250, (0.5, 2.0), "x")
    # 2M in * $0.5 + 0.25M out * $2
    assert e["dollars"] == 1.5


def test_input_from_size_caps_at_what_is_sent():
    small = estimate._input_from_size(4000, 8000, "analysis")
    huge = estimate._input_from_size(10_000_000, 8000, "analysis")
    assert small < huge
    assert huge == 8000 / estimate.CHARS_PER_TOKEN + estimate.PROMPT_OVERHEAD["analysis"]


def test_describe():
    assert estimate.describe({"calls": 0}) == ""
    line = estimate.describe(estimate.summarize(1240, 1700, 234, (0.4, 1.6), "past replies"))
    assert line.startswith("About 1,240 calls")
    assert "≈ $" in line and "2.1M in" in line


async def test_estimate_on_empty_project(client):
    from app.database import async_session_maker
    async with async_session_maker() as db:
        for job in estimate.JOBS:
            e = await estimate.estimate(db, 1, job, "m")
            assert e["calls"] == 0
