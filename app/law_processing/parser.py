"""
Parser for Vietnamese Land Law documents.
Parses hierarchical structure: LawDocument → Chapter (Chương) → Section (Mục) → Article (Điều) → Clause (Khoản)

Handles both LandLaw2013.txt and LandLaw2024.txt format variations.
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

from app.law_processing.models import Article, Chapter, Clause, LawDocument, Section


# ─── Regex Patterns ────────────────────────────────────────────────────────────

# Match "Chương I" / "Chương 2." / "Chương II\n" etc.
CHAPTER_PATTERN = re.compile(
    r'(?:^|\n)Chương\s+([IVXLCDM\d]+)\.?\s*\n+\s*([^\n]+)',
    re.MULTILINE
)

# Match "MỤC 1. QUYỀN CỦA ..." / "Mục 1. ..." / "MỤC 1. ..."
SECTION_PATTERN = re.compile(
    r'(?:^|\n)(?:MỤC|Mục)\s+([\d\.]+)\.?\s+([^\n]+)',
    re.MULTILINE
)

# Match "Điều 1. Phạm vi điều chỉnh" — captures article number and title
ARTICLE_START_PATTERN = re.compile(
    r'(?:^|\n)Điều\s+(\d+)\.\s*([^\n]*)',
    re.MULTILINE
)

# Match "1. content", "2. content" etc. for clauses
CLAUSE_PATTERN = re.compile(
    r'^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s|\Z)',
    re.MULTILINE | re.DOTALL
)


def normalize_whitespace(text: str) -> str:
    """Normalize Vietnamese text whitespace"""
    # Collapse multiple blank lines to one
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing whitespace on each line
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    return text.strip()


def normalize_chapter_number(raw: str) -> str:
    """Normalize chapter numbers: 'I', 'II', '1', '2' → canonical form"""
    return raw.strip().upper()


def parse_clauses(article_content: str) -> List[Clause]:
    """
    Parse numbered clauses (Khoản) from article body text.
    Handles multi-line clause content.
    """
    # Remove the article header line (Điều N. Title)
    body = re.sub(r'^Điều\s+\d+\.[^\n]*\n', '', article_content.strip(), count=1)

    # If too short or no numbered items, treat as single clause
    body = body.strip()
    if not body:
        return []

    # Try to find numbered items "1. ...", "2. ..."
    clauses = []
    # Split on clause boundaries
    # Pattern: line starting with a number followed by period
    parts = re.split(r'\n(?=\s*\d+\.\s)', body)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check if it starts with a clause number
        m = re.match(r'^(\d+)\.\s+(.+)', part, re.DOTALL)
        if m:
            clause_num = int(m.group(1))
            clause_content = m.group(2).strip()
            # Clean up sub-items (a), b), c)...) — keep them as is within content
            clauses.append(Clause(number=clause_num, content=clause_content))
        # else: unnumbered preamble text — skip or embed in first clause

    return clauses


def _split_into_article_blocks(text: str) -> List[Tuple[int, str, str]]:
    """
    Split document text into article blocks.
    Returns list of (article_number, title, full_content).
    """
    blocks = []
    # Find all article starts
    matches = list(ARTICLE_START_PATTERN.finditer(text))

    for i, match in enumerate(matches):
        article_num = int(match.group(1))
        title = match.group(2).strip()

        start = match.start()
        # Content goes from this match to the next article/chapter/section
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()
        blocks.append((article_num, title, content))

    return blocks


def _find_context_for_article(
    article_num: int,
    chapters_data: List[dict]
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Given an article number and chapters_data (list of {chapter, sections, article_ranges}),
    return (chapter_number, chapter_name, section_number, section_name).
    """
    for ch_data in chapters_data:
        ch_art_range = ch_data.get('article_range', (0, 99999))
        if ch_art_range[0] <= article_num <= ch_art_range[1]:
            # Found the chapter, now find the section
            for sec_data in ch_data.get('sections', []):
                sec_art_range = sec_data.get('article_range', (0, 99999))
                if sec_art_range[0] <= article_num <= sec_art_range[1]:
                    return (
                        ch_data['chapter_number'],
                        ch_data['chapter_name'],
                        sec_data['section_number'],
                        sec_data['section_name'],
                    )
            # In chapter but not in any section
            return (
                ch_data['chapter_number'],
                ch_data['chapter_name'],
                None,
                None,
            )
    return ('UNKNOWN', 'UNKNOWN', None, None)


def _extract_structure_metadata(text: str) -> List[dict]:
    """
    Pre-scan the text to build a chapter/section structure map.
    Each entry: {chapter_number, chapter_name, start_pos, sections: [{section_number, section_name, start_pos}]}
    We also calculate article_range per chapter/section afterward.
    """
    chapters_data = []

    chapter_matches = list(CHAPTER_PATTERN.finditer(text))
    section_matches = list(SECTION_PATTERN.finditer(text))
    article_matches = list(ARTICLE_START_PATTERN.finditer(text))

    def get_article_num_at_pos(pos: int, next_pos: int) -> Tuple[int, int]:
        """Get the first and last article number in a text range"""
        nums = []
        for am in article_matches:
            if pos <= am.start() < next_pos:
                nums.append(int(am.group(1)))
        if nums:
            return (min(nums), max(nums))
        return (999999, 0)  # empty range

    for i, ch_match in enumerate(chapter_matches):
        ch_start = ch_match.start()
        ch_end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)

        ch_number = normalize_chapter_number(ch_match.group(1))
        # Chapter name might span two lines — take the second if present
        ch_name_raw = ch_match.group(2).strip()

        # Sometimes the chapter title is on the NEXT line after the number
        # Look for additional content right after
        after_header = text[ch_match.end():ch_match.end() + 200]
        next_line_match = re.match(r'\s*\n+\s*([A-ZĐÀÁẢÃẠĂẮẶẶẰẴÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰ][^\n]+)', after_header)
        if next_line_match and not re.match(r'Điều|Mục|MỤC', next_line_match.group(1)):
            if len(ch_name_raw) < 5 or ch_name_raw.isupper() is False:
                ch_name_raw = next_line_match.group(1).strip()

        # Find sections within this chapter
        ch_sections = []
        sec_in_chapter = [s for s in section_matches if ch_start <= s.start() < ch_end]

        for j, sec_match in enumerate(sec_in_chapter):
            sec_start = sec_match.start()
            sec_end = sec_in_chapter[j + 1].start() if j + 1 < len(sec_in_chapter) else ch_end

            sec_art_range = get_article_num_at_pos(sec_start, sec_end)
            ch_sections.append({
                'section_number': sec_match.group(1).strip(),
                'section_name': sec_match.group(2).strip(),
                'start_pos': sec_start,
                'end_pos': sec_end,
                'article_range': sec_art_range,
            })

        ch_art_range = get_article_num_at_pos(ch_start, ch_end)
        chapters_data.append({
            'chapter_number': ch_number,
            'chapter_name': ch_name_raw,
            'start_pos': ch_start,
            'end_pos': ch_end,
            'article_range': ch_art_range,
            'sections': ch_sections,
        })

    return chapters_data


def parse_law_file(file_path: str, law_year: int) -> LawDocument:
    """
    Main parser entry point.
    Reads a law text file and returns a fully populated LawDocument.

    Args:
        file_path: Path to the .txt law file
        law_year: 2013 or 2024

    Returns:
        LawDocument with all chapters, sections, articles, and clauses populated
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Law file not found: {file_path}")

    text = path.read_text(encoding='utf-8')
    text = normalize_whitespace(text)

    # Extract title from first non-empty line
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    title = first_line or f"LUẬT ĐẤT ĐAI {law_year}"

    doc = LawDocument(law_year=law_year, title=title)

    # Pre-scan structure
    chapters_data = _extract_structure_metadata(text)

    # Parse all article blocks
    article_blocks = _split_into_article_blocks(text)

    print(f"  → Found {len(chapters_data)} chapters, {len(article_blocks)} articles in law {law_year}")

    # Build chapter/section lookup maps for the LawDocument
    chapter_map: dict[str, Chapter] = {}
    section_map: dict[Tuple, Section] = {}  # (chapter_number, section_number) -> Section

    for ch_data in chapters_data:
        chapter = Chapter(
            chapter_number=ch_data['chapter_number'],
            name=ch_data['chapter_name'],
        )
        chapter_map[ch_data['chapter_number']] = chapter
        doc.chapters.append(chapter)

        for sec_data in ch_data['sections']:
            section = Section(
                section_number=sec_data['section_number'],
                name=sec_data['section_name'],
            )
            chapter.sections.append(section)
            key = (ch_data['chapter_number'], sec_data['section_number'])
            section_map[key] = section

    # Assign articles to chapters and sections
    for art_num, art_title, art_content in article_blocks:
        ch_num, ch_name, sec_num, sec_name = _find_context_for_article(art_num, chapters_data)

        clauses = parse_clauses(art_content)

        article = Article(
            article_number=art_num,
            title=art_title,
            content=art_content,
            clauses=clauses,
            law_year=law_year,
            chapter_number=ch_num,
            chapter_name=ch_name,
            section_number=sec_num,
            section_name=sec_name,
        )

        # Add to the correct container
        if ch_num in chapter_map:
            chapter = chapter_map[ch_num]
            if sec_num is not None:
                key = (ch_num, sec_num)
                if key in section_map:
                    section_map[key].articles.append(article)
                else:
                    chapter.articles.append(article)
            else:
                chapter.articles.append(article)
        else:
            # Fallback: create an UNKNOWN chapter
            if 'UNKNOWN' not in chapter_map:
                unknown_ch = Chapter(chapter_number='UNKNOWN', name='Không xác định')
                chapter_map['UNKNOWN'] = unknown_ch
                doc.chapters.append(unknown_ch)
            chapter_map['UNKNOWN'].articles.append(article)

    return doc


def parse_both_laws(
    file_2013: str = "LandLaw2013.txt",
    file_2024: str = "LandLaw2024.txt",
) -> Tuple[LawDocument, LawDocument]:
    """
    Convenience function to parse both law files.
    Returns (law_2013, law_2024).
    """
    print(f"📖 Parsing LandLaw 2013 from {file_2013}...")
    law_2013 = parse_law_file(file_2013, 2013)
    print(f"   ✓ {len(law_2013.all_articles)} articles parsed")

    print(f"📖 Parsing LandLaw 2024 from {file_2024}...")
    law_2024 = parse_law_file(file_2024, 2024)
    print(f"   ✓ {len(law_2024.all_articles)} articles parsed")

    return law_2013, law_2024
