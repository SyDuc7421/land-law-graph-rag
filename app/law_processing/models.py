"""
Data models for Vietnamese Law document hierarchy.
Structure: LawDocument -> Chapter (Chương) -> Section (Mục) -> Article (Điều) -> Clause (Khoản)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Clause:
    """Khoản - numbered sub-item within an Article"""
    number: int           # e.g., 1, 2, 3
    content: str          # plain text content of the clause

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "content": self.content,
        }


@dataclass
class Article:
    """Điều - the primary law unit used for chunking"""
    article_number: int             # e.g., 1, 2, ... 212
    title: str                      # e.g., "Phạm vi điều chỉnh"
    content: str                    # full raw text of the article (including all khoản)
    clauses: List[Clause]           # parsed Khoản list

    # Hierarchy metadata
    law_year: int                   # 2013 or 2024
    chapter_number: str             # e.g., "I", "II", "3"
    chapter_name: str               # e.g., "QUY ĐỊNH CHUNG"
    section_number: Optional[str]   # e.g., "1", "2" — None if directly in chapter
    section_name: Optional[str]     # e.g., "QUYỀN CỦA NHÀ NƯỚC ĐỐI VỚI ĐẤT ĐAI"

    @property
    def chunk_id(self) -> str:
        return f"{self.law_year}_dieu_{self.article_number}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "law_year": self.law_year,
            "article_number": self.article_number,
            "title": self.title,
            "content": self.content,
            "clauses": [c.to_dict() for c in self.clauses],
            "chapter_number": self.chapter_number,
            "chapter_name": self.chapter_name,
            "section_number": self.section_number,
            "section_name": self.section_name,
        }


@dataclass
class Section:
    """Mục - groups Articles within a Chapter"""
    section_number: str             # e.g., "1", "2"
    name: str                       # e.g., "QUYỀN CỦA NHÀ NƯỚC ĐỐI VỚI ĐẤT ĐAI"
    articles: List[Article] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "section_number": self.section_number,
            "name": self.name,
            "article_count": len(self.articles),
        }


@dataclass
class Chapter:
    """Chương - top-level grouping of Sections and Articles"""
    chapter_number: str             # e.g., "I", "II", "III"
    name: str                       # e.g., "QUY ĐỊNH CHUNG"
    sections: List[Section] = field(default_factory=list)
    articles: List[Article] = field(default_factory=list)   # articles not under any section

    @property
    def all_articles(self) -> List[Article]:
        """Return all articles in this chapter, across all sections"""
        result = list(self.articles)
        for section in self.sections:
            result.extend(section.articles)
        return sorted(result, key=lambda a: a.article_number)

    def to_dict(self) -> dict:
        return {
            "chapter_number": self.chapter_number,
            "name": self.name,
            "section_count": len(self.sections),
            "article_count": len(self.all_articles),
        }


@dataclass
class LawDocument:
    """Root document representing a full law"""
    law_year: int                   # 2013 or 2024
    title: str                      # e.g., "LUẬT ĐẤT ĐAI 2013"
    chapters: List[Chapter] = field(default_factory=list)

    @property
    def all_articles(self) -> List[Article]:
        """Return all articles across all chapters and sections"""
        result = []
        for chapter in self.chapters:
            result.extend(chapter.all_articles)
        return sorted(result, key=lambda a: a.article_number)

    def to_dict(self) -> dict:
        return {
            "law_year": self.law_year,
            "title": self.title,
            "chapter_count": len(self.chapters),
            "article_count": len(self.all_articles),
        }
