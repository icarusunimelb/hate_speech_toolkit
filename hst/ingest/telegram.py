"""Telegram ingest from channel-scrape exports.

Input:  ``<paths.raw>/telegram/telegram_posts.csv`` and ``telegram_comments.csv`` with the
        columns produced by the channel scraper (posts: entity_name, entity_username, message_id,
        date, text, message_link, views, forwards, reactions, media_types, forwarded_from,
        reply_to_msg_id, comments_count; comments: entity_name, entity_username, message_id,
        parent_message_id, date, text, author, author_user_id, message_link, reactions,
        media_types, reply_to_msg_id).  Dates are local time followed by AEST/AEDT.
Output: a frame in the record contract via ``build(cfg)``.  Channel posts have the channel as
        author; comments have the commenting user (numeric user id, else a name key) as author,
        the post as parent and, when reply_to_msg_id is set, the replied-to message's author.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import Config
from ..schema import extract_mentions, norm_handle
from .common import blank_to_none, parse_count, parse_local_with_suffix, read_csv_any

FORWARD_HANDLE_RE = re.compile(r"\(@([A-Za-z0-9_]+)\)\s*$")


def forwarded_key(value: object) -> str | None:
    """'Some Channel (@somechannel)' -> 'somechannel'; a bare name -> lower-cased name."""
    if not isinstance(value, str) or not value.strip():
        return None
    m = FORWARD_HANDLE_RE.search(value)
    if m:
        return m.group(1).lower()
    return re.sub(r"\s+", " ", value.strip()).lower()


def name_key(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return "name:" + re.sub(r"\s+", " ", value.strip()).lower()


def load(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = cfg.path("raw") / "telegram"
    posts = read_csv_any(raw / "telegram_posts.csv")
    comments_path = raw / "telegram_comments.csv"
    comments = read_csv_any(comments_path) if comments_path.exists() else pd.DataFrame()
    return posts, comments


def build(cfg: Config) -> pd.DataFrame:
    posts, comments = load(cfg)
    col = lambda df, name: df[name] if name in df.columns else pd.Series("", index=df.index, dtype=object)  # noqa: E731

    # ---- posts ----------------------------------------------------------------------------
    channel = col(posts, "entity_username").map(norm_handle)
    channel = channel.where(channel.notna(), col(posts, "entity_name").map(name_key))
    mid = col(posts, "message_id").astype(str).str.strip()
    text = col(posts, "text").astype(str)
    p = pd.DataFrame({
        "record_id": channel.astype(str) + ":" + mid,
        "content_type": "post",
        "author": channel,
        "author_name": blank_to_none(col(posts, "entity_name")),
        "timestamp": parse_local_with_suffix(col(posts, "date")),
        "text": text,
        "url": blank_to_none(col(posts, "message_link")),
        "reply_to_record_id": None,
        "reply_to_author": None,
        "parent_record_id": None,
        "parent_author": None,
        "mentions": pd.Series([extract_mentions(t, "telegram") for t in text.tolist()], index=posts.index, dtype=object),
        "forwarded_from": col(posts, "forwarded_from").map(forwarded_key),
        "likes": parse_count(col(posts, "reactions")),
        "reposts": parse_count(col(posts, "forwards")),
        "replies": parse_count(col(posts, "comments_count")),
        "views": parse_count(col(posts, "views")),
    }, index=posts.index)
    reply_id = col(posts, "reply_to_msg_id").astype(str).str.strip()
    has_reply = reply_id != ""
    p.loc[has_reply, "reply_to_record_id"] = (channel[has_reply].astype(str) + ":" + reply_id[has_reply])
    p.loc[has_reply, "content_type"] = "reply"

    # ---- comments -------------------------------------------------------------------------
    if len(comments):
        cchannel = col(comments, "entity_username").map(norm_handle)
        cchannel = cchannel.where(cchannel.notna(), col(comments, "entity_name").map(name_key))
        cmid = col(comments, "message_id").astype(str).str.strip()
        pmid = col(comments, "parent_message_id").astype(str).str.strip()
        uid = col(comments, "author_user_id").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        author = pd.Series(np.where(uid != "", "user:" + uid, col(comments, "author").map(name_key)), index=comments.index, dtype=object)
        ctext = col(comments, "text").astype(str)
        c = pd.DataFrame({
            "record_id": cchannel.astype(str) + ":" + cmid,
            "content_type": "comment",
            "author": author.where(author.notna(), None),
            "author_name": blank_to_none(col(comments, "author")),
            "timestamp": parse_local_with_suffix(col(comments, "date")),
            "text": ctext,
            "url": blank_to_none(col(comments, "message_link")),
            "reply_to_record_id": None,
            "reply_to_author": None,
            "parent_record_id": cchannel.astype(str) + ":" + pmid,
            "parent_author": cchannel,
            "mentions": pd.Series([extract_mentions(t, "telegram") for t in ctext.tolist()], index=comments.index, dtype=object),
            "forwarded_from": None,
            "likes": parse_count(col(comments, "reactions")),
            "reposts": np.nan,
            "replies": np.nan,
            "views": np.nan,
        }, index=comments.index)
        creply = col(comments, "reply_to_msg_id").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        has = creply != ""
        c.loc[has, "reply_to_record_id"] = cchannel[has].astype(str) + ":" + creply[has]
        out = pd.concat([p, c], ignore_index=True)
    else:
        out = p.reset_index(drop=True)

    # resolve reply targets to authors (post or comment in the same channel)
    author_by_record = dict(zip(out["record_id"], out["author"]))
    target = out["reply_to_record_id"]
    resolved = target.map(author_by_record)
    out["reply_to_author"] = resolved.where(resolved.notna(), None)
    for c in ("region", "city"):
        out[c] = None
    for c in ("latitude", "longitude"):
        out[c] = np.nan
    return out
