from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flatten_categories(value: Any) -> list[str]:
    if not value:
        return []
    output: list[str] = []
    seen: set[str] = set()

    def add(category: Any) -> None:
        text = _text(category)
        if text and text not in seen:
            seen.add(text)
            output.append(text)

    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, list):
                for category in entry:
                    add(category)
            else:
                add(entry)
    else:
        add(value)
    return output


def amazon_review(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": _text(raw.get("reviewerID")),
        "item_id": _text(raw.get("asin")),
        "timestamp": _int_or_none(raw.get("unixReviewTime")),
        "rating": _float_or_none(raw.get("overall")),
        "review_title": _text(raw.get("summary")),
        "review": _text(raw.get("reviewText")),
        "statements": [],
    }


def amazon_user_from_review(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": _text(raw.get("reviewerID")),
        "name": _text(raw.get("reviewerName")),
    }


def amazon_item(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": _text(raw.get("asin")),
        "name": _text(raw.get("title")),
        "brand": _text(raw.get("brand")),
        "categories": _flatten_categories(raw.get("categories")),
        "description": _text(raw.get("description")),
        "price": _float_or_none(raw.get("price")),
        "image_url": _text(raw.get("imUrl")),
        "sales_rank": raw.get("salesRank"),
    }


def yelp_review(raw: dict[str, Any]) -> dict[str, Any]:
    timestamp = None
    date_text = _text(raw.get("date"))
    if date_text:
        timestamp = int(datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    return {
        "user_id": _text(raw.get("user_id")),
        "item_id": _text(raw.get("business_id")),
        "timestamp": timestamp,
        "rating": _float_or_none(raw.get("stars")),
        "review": _text(raw.get("text")),
        "statements": [],
    }


def yelp_item(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": _text(raw.get("business_id")),
        "name": _text(raw.get("name")),
        "categories": _flatten_categories(raw.get("categories")),
        "address": _text(raw.get("full_address")),
        "city": _text(raw.get("city")),
        "state": _text(raw.get("state")),
        "latitude": raw.get("latitude"),
        "longitude": raw.get("longitude"),
        "is_open": raw.get("open"),
        "review_count": _int_or_none(raw.get("review_count")),
        "average_rating": _float_or_none(raw.get("stars")),
        "neighborhoods": raw.get("neighborhoods") or [],
    }


def yelp_user(raw: dict[str, Any]) -> dict[str, Any]:
    votes = raw.get("votes") or {}
    return {
        "user_id": _text(raw.get("user_id")),
        "name": _text(raw.get("name")),
        "average_rating": _float_or_none(raw.get("average_stars")),
        "review_count": _int_or_none(raw.get("review_count")),
        "useful_votes": _int_or_none(votes.get("useful")),
        "funny_votes": _int_or_none(votes.get("funny")),
        "cool_votes": _int_or_none(votes.get("cool")),
    }


ADAPTERS = {
    "amazon2014": {
        "review_format": "json",
        "item_format": "python_literal",
        "user_format": None,
        "review": amazon_review,
        "item": amazon_item,
        "user_from_review": amazon_user_from_review,
    },
    "yelp2013": {
        "review_format": "json",
        "item_format": "json",
        "user_format": "json",
        "review": yelp_review,
        "item": yelp_item,
        "user": yelp_user,
    },
}
