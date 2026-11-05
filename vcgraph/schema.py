"""Pydantic schemas shared by the collection and reasoning agents."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------ collected records
class FounderInfo(BaseModel):
    name: str
    title: Optional[str] = None
    linkedin: Optional[str] = None


class CompanyProfile(BaseModel):
    """What the extraction LLM must produce from a crawled company page."""

    name: str
    website: Optional[str] = None
    one_liner: Optional[str] = None
    description: Optional[str] = None
    batch: Optional[str] = None
    sectors: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    team_size: Optional[int] = None
    founded_year: Optional[int] = None
    status: Optional[str] = None
    founders: list[FounderInfo] = Field(default_factory=list)
    investors: list[str] = Field(default_factory=list)


class SourceRecord(BaseModel):
    """A normalized record from one source, before entity resolution."""

    source: Literal["yc", "sec", "opencorporates", "demo"]
    source_id: str
    kind: Literal["company", "filing", "legal_entity"]
    name: str
    data: dict = Field(default_factory=dict)


# ------------------------------------------------------------ reasoning agent
class SubQuery(BaseModel):
    text: str
    target_types: list[str] = Field(default_factory=list,
                                    description="Node labels likely to hold the answer")
    relation_hints: list[str] = Field(default_factory=list,
                                      description="Relationship types worth following")
    anchors: list[str] = Field(default_factory=list,
                               description="Named entities explicitly mentioned")


class Decomposition(BaseModel):
    intent: Literal["competitors", "investors", "founders", "market_signal", "lookup", "other"] = "other"
    sub_queries: list[SubQuery]


class Sufficiency(BaseModel):
    sufficient: bool
    missing: Optional[str] = None
    follow_up: Optional[str] = None
