"""Instagram ingest from account-scrape exports.

Input:  ``<paths.raw>/instagram/instagram_posts.csv`` (username, shortcode, date, text, post_url,
        post_type, hashtags, comments_count) and ``instagram_comments.csv`` (username, shortcode,
        parent_post_url, date, text, author, is_hidden).  Dates are local time followed by a zone
        name such as "Australia/Sydney".
Output: a frame in the record contract via ``build(cfg)``.  Posts have the posting account as
        author; comments have the commenting account as author and the post as parent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..schema import extract_mentions, norm_handle
from .common import blank_to_none, parse_count, parse_local_with_suffix, read_csv_any


def load(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = cfg.path("raw") / "instagram"
    posts = read_csv_any(raw / "instagram_posts.csv")
    comments_path = raw / "instagram_comments.csv"
    comments = read_csv_any(comments_path) if comments_path.exists() else pd.DataFrame()
    return posts, comments


def build(cfg: Config) -> pd.DataFrame:
    posts, comments = load(cfg)
    col = lambda df, name: df[name] if name in df.columns else pd.Series("", index=df.index, dtype=object)  # noqa: E731

    user = col(posts, "username").map(norm_handle)
    code = col(posts, "shortcode").astype(str).str.strip()
    text = col(posts, "text").astype(str)
    p = pd.DataFrame({
        "record_id": code,
        "content_type": "post",
        "author": user,
        "author_name": blank_to_none(col(posts, "username")),
        "timestamp": parse_local_with_suffix(col(posts, "date")),
        "text": text,
        "url": blank_to_none(col(posts, "post_url")),
        "reply_to_record_id": None,
        "reply_to_author": None,
        "parent_record_id": None,
        "parent_author": None,
        "mentions": pd.Series([extract_mentions(t, "instagram") for t in text.tolist()], index=posts.index, dtype=object),
        "forwarded_from": None,
        "likes": np.nan,
        "reposts": np.nan,
        "replies": parse_count(col(posts, "comments_count")),
        "views": np.nan,
    }, index=posts.index)

    if len(comments):
        ccode = col(comments, "shortcode").astype(str).str.strip()
        cuser = col(comments, "username").map(norm_handle)
        cauthor = col(comments, "author").map(norm_handle)
        ctext = col(comments, "text").astype(str)
        c = pd.DataFrame({
            "record_id": ccode + ":" + pd.Series(range(len(comments)), index=comments.index).astype(str),
            "content_type": "comment",
            "author": cauthor,
            "author_name": blank_to_none(col(comments, "author")),
            "timestamp": parse_local_with_suffix(col(comments, "date")),
            "text": ctext,
            "url": blank_to_none(col(comments, "parent_post_url")),
            "reply_to_record_id": ccode,
            "reply_to_author": cuser,
            "parent_record_id": ccode,
            "parent_author": cuser,
            "mentions": pd.Series([extract_mentions(t, "instagram") for t in ctext.tolist()], index=comments.index, dtype=object),
            "forwarded_from": None,
            "likes": np.nan,
            "reposts": np.nan,
            "replies": np.nan,
            "views": np.nan,
        }, index=comments.index)
        out = pd.concat([p, c], ignore_index=True)
    else:
        out = p.reset_index(drop=True)
    for c in ("region", "city"):
        out[c] = None
    for c in ("latitude", "longitude"):
        out[c] = np.nan
    return out
