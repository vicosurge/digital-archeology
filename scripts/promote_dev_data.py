"""Copy one project's link, reading and comment results from one database into another.

For a setup with a development copy of the database (made from production)
where link recovery, close readings and comment extraction were run: the
project's archived_files ids are then identical in both. Only rows belonging
to that project are copied, since other projects' ids may differ between the
two databases.

    python3 scripts/promote_dev_data.py --dev DEV.db --prod PROD.db --project 4          # dry run
    python3 scripts/promote_dev_data.py --dev DEV.db --prod PROD.db --project 4 --apply

Run with the prod app stopped. Refuses to run if prod already holds rows for the
project in any of the copied tables, and if the project's archived_files differ.
Does not touch the project's filters or archived_files.
"""
import argparse
import sqlite3
import sys

COPIED_TABLES = ("link_targets", "page_links", "page_readings", "page_comments")
CARD_COLUMNS = ("findings", "findings_model", "findings_generated_at")


def columns(conn, schema, table):
    return [r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({table})")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--prod", required=True)
    ap.add_argument("--project", type=int, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    pid = args.project

    conn = sqlite3.connect(args.prod)
    conn.execute("ATTACH DATABASE ? AS dev", (f"file:{args.dev}?mode=ro",))

    # 1. The corpus must be the same capture-for-capture, or file_id references break.
    mismatch = conn.execute("""
        SELECT COUNT(*) FROM dev.archived_files d
        LEFT JOIN main.archived_files p ON p.id = d.id
        WHERE d.project_id = ? AND (p.id IS NULL OR p.project_id != d.project_id
              OR p.original_url != d.original_url OR p.timestamp != d.timestamp)
    """, (pid,)).fetchone()[0]
    dev_n = conn.execute("SELECT COUNT(*) FROM dev.archived_files WHERE project_id=?", (pid,)).fetchone()[0]
    prod_n = conn.execute("SELECT COUNT(*) FROM main.archived_files WHERE project_id=?", (pid,)).fetchone()[0]
    print(f"archived_files: dev={dev_n} prod={prod_n} mismatched={mismatch}")
    if mismatch or dev_n != prod_n:
        sys.exit("archived_files differ between dev and prod for this project; refusing.")

    # 2. Nothing of this project may already be in the target tables.
    plan = []
    for table in COPIED_TABLES:
        prod_cols = set(columns(conn, "main", table))
        if not prod_cols:
            sys.exit(f"prod has no table {table}; start the new app version once first.")
        cols = [c for c in columns(conn, "dev", table) if c in prod_cols]
        existing = conn.execute(f"SELECT COUNT(*) FROM main.{table} WHERE project_id=?", (pid,)).fetchone()[0]
        if existing:
            sys.exit(f"prod already has {existing} {table} rows for project {pid}; refusing.")
        clash = conn.execute(
            f"SELECT COUNT(*) FROM dev.{table} d JOIN main.{table} p ON p.id = d.id WHERE d.project_id=?",
            (pid,)).fetchone()[0]
        if clash:
            sys.exit(f"{clash} {table} ids already used in prod; refusing.")
        n = conn.execute(f"SELECT COUNT(*) FROM dev.{table} WHERE project_id=?", (pid,)).fetchone()[0]
        plan.append((table, cols, n))
        print(f"{table}: copy {n} rows ({len(cols)} columns)")

    # 3. Analyses re-run in dev (same file, different reply).
    fa_cols = [c for c in columns(conn, "dev", "file_analyses")
               if c in set(columns(conn, "main", "file_analyses")) and c not in ("id", "file_id")]
    changed = [r[0] for r in conn.execute("""
        SELECT d.file_id FROM dev.file_analyses d
        JOIN main.file_analyses p ON p.file_id = d.file_id
        JOIN dev.archived_files a ON a.id = d.file_id
        WHERE a.project_id = ? AND COALESCE(d.raw_response,'') != COALESCE(p.raw_response,'')
    """, (pid,))]
    print(f"file_analyses: update {len(changed)} re-analyzed rows")

    card = conn.execute(
        f"SELECT {', '.join(CARD_COLUMNS)} FROM dev.archival_cards WHERE project_id=?", (pid,)).fetchone()
    print(f"archival_cards: findings {'present' if card and card[0] else 'absent'} in dev")

    if not args.apply:
        print("Dry run; pass --apply to write.")
        return

    with conn:
        for table, cols, _ in plan:
            col_list = ", ".join(cols)
            conn.execute(f"INSERT INTO main.{table} ({col_list}) "
                         f"SELECT {col_list} FROM dev.{table} WHERE project_id=?", (pid,))
        set_clause = ", ".join(f"{c} = (SELECT d.{c} FROM dev.file_analyses d "
                               f"WHERE d.file_id = main.file_analyses.file_id)" for c in fa_cols)
        for i in range(0, len(changed), 500):
            chunk = changed[i:i + 500]
            conn.execute(f"UPDATE main.file_analyses SET {set_clause} "
                         f"WHERE file_id IN ({','.join('?' * len(chunk))})", chunk)
        if card and card[0]:
            conn.execute(f"UPDATE main.archival_cards SET {', '.join(c + ' = ?' for c in CARD_COLUMNS)} "
                         f"WHERE project_id = ?", (*card, pid))

    for table, _, n in plan:
        got = conn.execute(f"SELECT COUNT(*) FROM main.{table} WHERE project_id=?", (pid,)).fetchone()[0]
        print(f"{table}: prod now has {got} (expected {n}) {'OK' if got == n else 'MISMATCH'}")


if __name__ == "__main__":
    main()
