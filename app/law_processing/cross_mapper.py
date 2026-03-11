"""
Cross-Mapper: 4-step pipeline to match and diff articles between LandLaw 2013 and 2024.

Step 1 – Number Anchor:  exact article number match
Step 2 – Sliding Window: match nearby article numbers with title similarity
Step 3 – Embedding Fallback: OpenAI cosine similarity for still-unmatched articles
Step 4 – LLM Diff: GPT-4o-mini summarizes changes for each matched pair
"""

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.law_processing.models import Article


# ─── Match Result Dataclass ────────────────────────────────────────────────────

@dataclass
class ArticleMatch:
    """Represents a matched pair of articles across two laws"""
    article_2013: Optional[dict]    # chunk dict (may be None if NEW in 2024)
    article_2024: Optional[dict]    # chunk dict (may be None if DELETED in 2024)
    match_method: str               # "number_anchor" | "sliding_window" | "embedding" | "unmatched"
    similarity_score: float = 0.0   # 0–1 score for sliding window / embedding matches
    change_type: Optional[str] = None        # "UNCHANGED" | "MODIFIED" | "SUPERSEDES" | "EXPANDED" | "DELETED" | "NEW" | "RENAMED"
    diff_summary: Optional[str] = None       # LLM-generated diff summary
    embedding_2013: Optional[list] = None    # cached embedding (not serialized)
    embedding_2024: Optional[list] = None    # cached embedding (not serialized)

    def to_dict(self) -> dict:
        return {
            "article_2013_num": self.article_2013["article_number"] if self.article_2013 else None,
            "article_2013_title": self.article_2013["title"] if self.article_2013 else None,
            "article_2024_num": self.article_2024["article_number"] if self.article_2024 else None,
            "article_2024_title": self.article_2024["title"] if self.article_2024 else None,
            "match_method": self.match_method,
            "similarity_score": round(self.similarity_score, 4),
            "change_type": self.change_type,
            "diff_summary": self.diff_summary,
        }


# ─── Text Normalization Helpers ────────────────────────────────────────────────

def _normalize_vi(text: str) -> str:
    """
    Normalize Vietnamese text: lowercase, remove diacritics for fuzzy comparison.
    Keeps alphanumeric and spaces.
    """
    text = text.lower().strip()
    # Remove diacritics using NFD decomposition
    nfd = unicodedata.normalize('NFD', text)
    ascii_text = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    # Remove non-alphanumeric
    return re.sub(r'[^a-z0-9\s]', ' ', ascii_text).strip()


def _title_similarity(t1: str, t2: str) -> float:
    """
    Simple normalized token overlap similarity between two titles.
    Returns 0.0–1.0.
    """
    tokens1 = set(_normalize_vi(t1).split())
    tokens2 = set(_normalize_vi(t2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


# ─── Step 1: Number Anchor ─────────────────────────────────────────────────────

def step1_number_anchor(
    chunks_2013: List[dict],
    chunks_2024: List[dict],
) -> Tuple[List[ArticleMatch], List[dict], List[dict]]:
    """
    Match articles by exact article_number.
    Returns: (matched_pairs, unmatched_2013, unmatched_2024)
    """
    map_2013 = {c["article_number"]: c for c in chunks_2013}
    map_2024 = {c["article_number"]: c for c in chunks_2024}

    matched = []
    used_2013 = set()
    used_2024 = set()

    for num, chunk_2024 in map_2024.items():
        if num in map_2013:
            chunk_2013 = map_2013[num]
            matched.append(ArticleMatch(
                article_2013=chunk_2013,
                article_2024=chunk_2024,
                match_method="number_anchor",
                similarity_score=1.0,
            ))
            used_2013.add(num)
            used_2024.add(num)

    unmatched_2013 = [c for c in chunks_2013 if c["article_number"] not in used_2013]
    unmatched_2024 = [c for c in chunks_2024 if c["article_number"] not in used_2024]

    print(f"  Step 1 (Number Anchor): {len(matched)} matched, "
          f"{len(unmatched_2013)} unmatched in 2013, {len(unmatched_2024)} unmatched in 2024")
    return matched, unmatched_2013, unmatched_2024


# ─── Step 2: Sliding Window ────────────────────────────────────────────────────

def step2_sliding_window(
    unmatched_2013: List[dict],
    unmatched_2024: List[dict],
    window: int = 5,
    threshold: float = 0.45,
) -> Tuple[List[ArticleMatch], List[dict], List[dict]]:
    """
    For each unmatched 2024 article, look within ±window article numbers in 2013
    and match if title similarity exceeds threshold.
    Greedy matching: highest similarity wins.
    """
    matched = []
    used_2013 = set()
    used_2024 = set()

    map_2013_by_num = {c["article_number"]: c for c in unmatched_2013}

    # Build candidates sorted by similarity
    candidates = []
    for chunk_2024 in unmatched_2024:
        num_2024 = chunk_2024["article_number"]
        for delta in range(-window, window + 1):
            candidate_num = num_2024 + delta
            if candidate_num in map_2013_by_num:
                chunk_2013 = map_2013_by_num[candidate_num]
                score = _title_similarity(chunk_2013["title"], chunk_2024["title"])
                if score >= threshold:
                    candidates.append((score, chunk_2013["article_number"], num_2024, chunk_2013, chunk_2024))

    # Sort by score descending
    candidates.sort(key=lambda x: -x[0])

    for score, num_2013, num_2024, chunk_2013, chunk_2024 in candidates:
        if num_2013 in used_2013 or num_2024 in used_2024:
            continue
        matched.append(ArticleMatch(
            article_2013=chunk_2013,
            article_2024=chunk_2024,
            match_method="sliding_window",
            similarity_score=score,
        ))
        used_2013.add(num_2013)
        used_2024.add(num_2024)

    still_unmatched_2013 = [c for c in unmatched_2013 if c["article_number"] not in used_2013]
    still_unmatched_2024 = [c for c in unmatched_2024 if c["article_number"] not in used_2024]

    print(f"  Step 2 (Sliding Window ±{window}): {len(matched)} additional matches, "
          f"{len(still_unmatched_2013)} unmatched in 2013, {len(still_unmatched_2024)} unmatched in 2024")
    return matched, still_unmatched_2013, still_unmatched_2024


# ─── Step 3: Embedding Fallback ────────────────────────────────────────────────

def _get_embeddings(texts: List[str], client) -> List[List[float]]:
    """Call OpenAI embeddings API in batches"""
    from openai import OpenAI  # lazy import

    results = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        results.extend([item.embedding for item in response.data])
    return results


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    import math
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a ** 2 for a in v1))
    norm2 = math.sqrt(sum(b ** 2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def step3_embedding_fallback(
    unmatched_2013: List[dict],
    unmatched_2024: List[dict],
    threshold: float = 0.75,
    openai_api_key: Optional[str] = None,
) -> Tuple[List[ArticleMatch], List[dict], List[dict]]:
    """
    Compute OpenAI embeddings for unmatched articles and match by cosine similarity.
    Requires OPENAI_API_KEY.
    """
    if not unmatched_2013 or not unmatched_2024:
        return [], unmatched_2013, unmatched_2024

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  Step 3 (Embedding Fallback): SKIPPED — no OPENAI_API_KEY found")
        return [], unmatched_2013, unmatched_2024

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Build text inputs: title + first 300 chars of content
        def chunk_text(c: dict) -> str:
            return f"{c['title']}\n{c['content'][:300]}"

        texts_2013 = [chunk_text(c) for c in unmatched_2013]
        texts_2024 = [chunk_text(c) for c in unmatched_2024]

        print(f"  Step 3 (Embedding Fallback): computing {len(texts_2013) + len(texts_2024)} embeddings...")
        embeds_2013 = _get_embeddings(texts_2013, client)
        embeds_2024 = _get_embeddings(texts_2024, client)

        # Greedy matching by cosine similarity
        matched = []
        used_2013 = set()
        used_2024 = set()

        # Build similarity matrix
        candidates = []
        for i, (chunk_2013, emb_2013) in enumerate(zip(unmatched_2013, embeds_2013)):
            for j, (chunk_2024, emb_2024) in enumerate(zip(unmatched_2024, embeds_2024)):
                score = _cosine_similarity(emb_2013, emb_2024)
                if score >= threshold:
                    candidates.append((score, i, j, chunk_2013, chunk_2024))

        candidates.sort(key=lambda x: -x[0])

        for score, i, j, chunk_2013, chunk_2024 in candidates:
            num_2013 = chunk_2013["article_number"]
            num_2024 = chunk_2024["article_number"]
            if num_2013 in used_2013 or num_2024 in used_2024:
                continue
            matched.append(ArticleMatch(
                article_2013=chunk_2013,
                article_2024=chunk_2024,
                match_method="embedding",
                similarity_score=score,
            ))
            used_2013.add(num_2013)
            used_2024.add(num_2024)

        still_unmatched_2013 = [c for c in unmatched_2013 if c["article_number"] not in used_2013]
        still_unmatched_2024 = [c for c in unmatched_2024 if c["article_number"] not in used_2024]

        print(f"  Step 3 done: {len(matched)} semantic matches, "
              f"{len(still_unmatched_2013)} unmatched in 2013, {len(still_unmatched_2024)} unmatched in 2024")
        return matched, still_unmatched_2013, still_unmatched_2024

    except Exception as e:
        print(f"  Step 3 (Embedding Fallback): ERROR — {e}")
        return [], unmatched_2013, unmatched_2024


# ─── Step 4: LLM Diff ──────────────────────────────────────────────────────────

_DIFF_SYSTEM_PROMPT = """Bạn là chuyên gia pháp lý chuyên về luật đất đai Việt Nam.
Nhiệm vụ: So sánh một Điều từ Luật Đất Đai 2013 và phiên bản sửa đổi tương ứng trong Luật Đất Đai 2024.
Trả lời theo định dạng JSON như sau:
{
  "change_type": "UNCHANGED | MODIFIED | SUPERSEDES | EXPANDED | DELETED | NEW | RENAMED",
  "summary": "Tóm tắt ngắn gọn (<=100 từ) những thay đổi quan trọng nhất"
}

Phân loại:
- UNCHANGED: nội dung gần như giống nhau, chỉ thay đổi văn phong
- MODIFIED: có thay đổi nội dung đáng kể, nhưng điều khoản tương tự tồn tại
- SUPERSEDES: điều luật 2024 thay thế hoàn toàn điều luật 2013 với nội dung mới đáng kể
- EXPANDED: 2024 mở rộng/thêm nhiều quy định hơn 2013
- RENAMED: chỉ đổi tên/tiêu đề, nội dung gần như không đổi
- DELETED: điều luật 2013 bị loại bỏ hoàn toàn trong 2024
- NEW: điều luật chỉ xuất hiện trong 2024"""

_DIFF_USER_TEMPLATE = """
## Điều {num_2013} (Luật 2013) — {title_2013}
{content_2013}

## Điều {num_2024} (Luật 2024) — {title_2024}
{content_2024}

So sánh hai điều luật trên và trả lời JSON.
"""


def step4_llm_diff(
    matched_pairs: List[ArticleMatch],
    openai_api_key: Optional[str] = None,
    max_content_chars: int = 1500,
) -> List[ArticleMatch]:
    """
    Use GPT-4o-mini to generate a diff summary for each matched pair.
    Adds change_type and diff_summary to each ArticleMatch.
    Handles unmatched (DELETED / NEW) articles as well.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  Step 4 (LLM Diff): SKIPPED — no OPENAI_API_KEY found")
        for m in matched_pairs:
            m.change_type = "UNKNOWN"
            m.diff_summary = "LLM diff not run (no API key)"
        return matched_pairs

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        print(f"  Step 4 (LLM Diff): classifying {len(matched_pairs)} article pairs...")

        for i, match in enumerate(matched_pairs):
            if match.article_2013 is None:
                match.change_type = "NEW"
                match.diff_summary = f"Điều {match.article_2024['article_number']} là điều khoản mới trong Luật 2024"
                continue
            if match.article_2024 is None:
                match.change_type = "DELETED"
                match.diff_summary = f"Điều {match.article_2013['article_number']} đã bị loại bỏ trong Luật 2024"
                continue

            try:
                prompt = _DIFF_USER_TEMPLATE.format(
                    num_2013=match.article_2013["article_number"],
                    title_2013=match.article_2013["title"],
                    content_2013=match.article_2013["content"][:max_content_chars],
                    num_2024=match.article_2024["article_number"],
                    title_2024=match.article_2024["title"],
                    content_2024=match.article_2024["content"][:max_content_chars],
                )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": _DIFF_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                match.change_type = result.get("change_type", "UNKNOWN")
                match.diff_summary = result.get("summary", "")

            except Exception as e:
                match.change_type = "ERROR"
                match.diff_summary = f"LLM error: {e}"

            if (i + 1) % 20 == 0:
                print(f"    ... processed {i+1}/{len(matched_pairs)}")

        print(f"  Step 4 done: {len(matched_pairs)} pairs classified")
        return matched_pairs

    except Exception as e:
        print(f"  Step 4 (LLM Diff): ERROR — {e}")
        return matched_pairs


# ─── Main Cross-Mapper Orchestrator ────────────────────────────────────────────

def run_cross_mapping(
    chunks_2013: List[dict],
    chunks_2024: List[dict],
    output_path: str = "output/cross_mapping.json",
    sliding_window: int = 5,
    sliding_threshold: float = 0.45,
    embedding_threshold: float = 0.75,
    openai_api_key: Optional[str] = None,
    skip_embeddings: bool = False,
    skip_llm_diff: bool = False,
) -> List[ArticleMatch]:
    """
    Full 4-step cross-mapping pipeline.

    Args:
        chunks_2013: List of article chunks from LandLaw 2013
        chunks_2024: List of article chunks from LandLaw 2024
        output_path: Where to save the JSON output
        sliding_window: ±window for step 2
        sliding_threshold: title similarity threshold for step 2
        embedding_threshold: cosine similarity threshold for step 3
        openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
        skip_embeddings: Skip step 3 (useful for testing or no API key)
        skip_llm_diff: Skip step 4 (useful for testing or no API key)

    Returns:
        Full list of ArticleMatch objects
    """
    print("\n🔗 Starting Cross-Mapping Pipeline...")

    # Step 1
    matched, unmatched_2013, unmatched_2024 = step1_number_anchor(chunks_2013, chunks_2024)

    # Step 2
    sw_matched, unmatched_2013, unmatched_2024 = step2_sliding_window(
        unmatched_2013, unmatched_2024,
        window=sliding_window,
        threshold=sliding_threshold,
    )
    matched.extend(sw_matched)

    # Step 3
    if not skip_embeddings:
        emb_matched, unmatched_2013, unmatched_2024 = step3_embedding_fallback(
            unmatched_2013, unmatched_2024,
            threshold=embedding_threshold,
            openai_api_key=openai_api_key,
        )
        matched.extend(emb_matched)

    # Mark remaining as DELETED (in 2013, not in 2024) or NEW (in 2024, not in 2013)
    for chunk_2013 in unmatched_2013:
        matched.append(ArticleMatch(
            article_2013=chunk_2013,
            article_2024=None,
            match_method="unmatched",
            similarity_score=0.0,
            change_type="DELETED",
        ))
    for chunk_2024 in unmatched_2024:
        matched.append(ArticleMatch(
            article_2013=None,
            article_2024=chunk_2024,
            match_method="unmatched",
            similarity_score=0.0,
            change_type="NEW",
        ))

    # Step 4
    if not skip_llm_diff:
        # Only run LLM diff on actually matched pairs (not unmatched stubs)
        paired = [m for m in matched if m.article_2013 and m.article_2024]
        paired = step4_llm_diff(paired, openai_api_key=openai_api_key)

    # Summary stats
    change_type_counts: dict = {}
    for m in matched:
        ct = m.change_type or "UNKNOWN"
        change_type_counts[ct] = change_type_counts.get(ct, 0) + 1

    print(f"\n📊 Cross-Mapping Summary ({len(matched)} total):")
    for ct, count in sorted(change_type_counts.items()):
        print(f"   {ct}: {count}")

    # Save to JSON
    output_dicts = [m.to_dict() for m in matched]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output_dicts, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Cross-mapping saved → {path}")

    return matched
