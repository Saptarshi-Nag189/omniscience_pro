"""SQLite natural-language querying with comprehensive read-only safety restrictions."""
import logging
import os
import re
import sqlite3
import time

from security import check_rate_limit, sanitize_error_message

logger = logging.getLogger(__name__)

# Word-like SQL keywords — matched on word boundaries so legitimate identifiers
# that merely contain a keyword as a substring (e.g. "created_at" -> CREATE,
# "update_time" -> UPDATE) are not falsely rejected.
_DANGEROUS_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE',
    'ATTACH', 'DETACH', 'PRAGMA', 'LOAD_EXTENSION',
    'INTO OUTFILE', 'INTO DUMPFILE',
    'VACUUM', 'REINDEX', 'ANALYZE',
    'UNION',
]

# Comment / statement-separator sequences — matched as raw substrings because
# they contain non-word characters that word boundaries can't anchor to.
_DANGEROUS_SEQUENCES = [';--', '/*', '*/', '--', '#']

_KEYWORD_RE = re.compile(r'\b(' + '|'.join(re.escape(k) for k in _DANGEROUS_KEYWORDS) + r')\b')

_QUERY_TIMEOUT = 5
_MAX_ROWS = 1000


def query_sqlite_db(db_path: str, query: str, llm) -> str:
    """Translate a natural-language query to SQL and execute it read-only."""
    if not os.path.exists(db_path):
        return "Database file not found."

    if not check_rate_limit("sql_query"):
        return "Rate limit exceeded. Please wait before making more queries."

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=_QUERY_TIMEOUT)
        try:
            conn.execute("PRAGMA query_only = ON")
            # The connect() timeout only bounds lock waits — a runaway query
            # (e.g. a huge cross join) would otherwise hang the UI forever.
            # The progress handler aborts execution once the deadline passes.
            deadline = time.monotonic() + _QUERY_TIMEOUT
            conn.set_progress_handler(
                lambda: 1 if time.monotonic() > deadline else 0, 10_000
            )
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            schema_str = str(cursor.fetchall())

            prompt = f"""Generate a SQLite SELECT query.

SCHEMA:
{schema_str}

QUESTION:
{query}

RULES:
- SELECT only
- No INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, ATTACH, PRAGMA
- Single statement
- Simple query

Return ONLY the SQL query."""

            sql_query = llm.invoke(prompt).strip().replace("```sql", "").replace("```", "").strip()
            sql_upper = sql_query.upper().strip()

            if not sql_upper.startswith('SELECT'):
                logger.warning(f"Blocked non-SELECT SQL query: {sql_query[:100]}")
                return "Only SELECT queries are allowed for security reasons."

            kw_match = _KEYWORD_RE.search(sql_upper)
            if kw_match:
                keyword = kw_match.group(1)
                logger.warning(f"Blocked dangerous SQL keyword '{keyword}': {sql_query[:100]}")
                return f"Query contains prohibited keyword: {keyword}"

            for seq in _DANGEROUS_SEQUENCES:
                if seq in sql_upper:
                    logger.warning(f"Blocked dangerous SQL sequence '{seq}': {sql_query[:100]}")
                    return f"Query contains prohibited keyword: {seq}"

            # Any semicolon other than a single trailing one implies a second
            # statement (sqlite3 would reject it anyway; fail early and clearly).
            if ';' in sql_query.strip().rstrip(';'):
                return "Multiple SQL statements are not allowed."

            cursor.execute(sql_query)
            results = cursor.fetchmany(_MAX_ROWS)
            has_more = cursor.fetchone() is not None

            result_text = f"**SQL:** `{sql_query}`\n\n**Results ({len(results)} rows"
            if has_more:
                result_text += f", limited to {_MAX_ROWS}"
            result_text += f"):**\n{results}"

            logger.info(f"Executed SQL query: {sql_query[:100]}")
            return result_text
        finally:
            conn.close()

    except sqlite3.OperationalError as e:
        err = str(e).lower()
        if "locked" in err or "timeout" in err or "interrupt" in err:
            return "Query timeout exceeded. Please simplify your query."
        return f"SQL Error: {sanitize_error_message(e)}"
    except Exception as e:
        return f"SQL Error: {sanitize_error_message(e)}"
