# coding=utf-8
"""Score fixed FIM eval prediction JSONL files.

Expected prediction fields are produced by ``fim_batch_infer_mcore.py``:

  eval_id, language, repo, span_kind, checkpoint_label,
  raw_output, cleaned_output, gold_middle

The scorer truncates leaked special tokens before comparing predictions with
the gold middle span. This keeps over-generation visible while avoiding a
single leaked file separator from destroying the local similarity score.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean


STOP_MARKERS = (
    "<|file_sep|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|fim_pad|>",
    "<|im_end|>",
    "<|endoftext|>",
)


def truncate_prediction(text: str) -> str:
    text = text or ""
    cut = len(text)
    for marker in STOP_MARKERS:
        index = text.find(marker)
        if index >= 0:
            cut = min(cut, index)
    return text[:cut]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def similarity(prediction: str, gold: str) -> float:
    return SequenceMatcher(None, normalize_text(prediction), normalize_text(gold)).ratio()


def code_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||[^\s]", text or "")


def token_similarity(prediction: str, gold: str) -> float:
    return SequenceMatcher(None, code_tokens(prediction), code_tokens(gold)).ratio()


def line_similarity(prediction: str, gold: str) -> float:
    pred_lines = [line.strip() for line in (prediction or "").splitlines() if line.strip()]
    gold_lines = [line.strip() for line in (gold or "").splitlines() if line.strip()]
    return SequenceMatcher(None, pred_lines, gold_lines).ratio()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_eval_context(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    contexts = {}
    for row in load_rows(path):
        eval_id = str(row.get("eval_id") or row.get("id") or "")
        if eval_id:
            contexts[eval_id] = row
    return contexts


def bracket_balance(text: str) -> tuple[bool, str]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {value: key for key, value in pairs.items()}
    stack = []
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    index = 0

    text = text or ""
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char in "\r\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue

        if char == "/" and next_char == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char in ("'", '"', "`"):
            quote = char
        elif char in pairs:
            stack.append((char, index))
        elif char in closers:
            if not stack or stack[-1][0] != closers[char]:
                return False, f"unmatched_closer:{char}@{index}"
            stack.pop()
        index += 1

    if quote:
        return False, "unclosed_string"
    if block_comment:
        return False, "unclosed_block_comment"
    if stack:
        opener, index = stack[-1]
        return False, f"unclosed_opener:{opener}@{index}"
    return True, ""


def suffix_overlap(prediction: str, suffix: str) -> bool:
    pred = normalize_text(prediction)
    suffix_lines = [normalize_text(line) for line in (suffix or "").splitlines() if normalize_text(line)]
    meaningful = [line for line in suffix_lines[:6] if len(line) >= 16]

    for window_size in (3, 2):
        if len(meaningful) < window_size:
            continue
        for index in range(0, len(meaningful) - window_size + 1):
            snippet = " ".join(meaningful[index : index + window_size])
            if len(snippet) >= 48 and snippet in pred:
                return True

    long_single_lines = [line for line in meaningful[:3] if len(line) >= 48]
    return any(line in pred for line in long_single_lines)


def repetition_issue(text: str) -> bool:
    lines = [normalize_text(line) for line in (text or "").splitlines() if normalize_text(line)]
    if len(lines) >= 6 and Counter(lines).most_common(1)[0][1] >= 3:
        return True
    tokens = code_tokens(text)
    if len(tokens) < 32:
        return False
    windows = [" ".join(tokens[index : index + 8]) for index in range(0, len(tokens) - 7)]
    return bool(windows and Counter(windows).most_common(1)[0][1] >= 4)


def syntax_quality(language: str, prediction: str, context: dict | None) -> dict:
    language = (language or "unknown").lower()
    prefix = (context or {}).get("prefix") or ""
    suffix = (context or {}).get("suffix") or ""
    candidate = f"{prefix}{prediction}{suffix}" if context else prediction
    balance_ok, balance_error = bracket_balance(candidate)

    result = {
        "syntax_checked": True,
        "syntax_ok": balance_ok,
        "syntax_mode": "bracket_balance",
        "syntax_error": balance_error,
        "bracket_balance_ok": balance_ok,
    }

    if not prediction.strip():
        result.update(
            {
                "syntax_ok": False,
                "syntax_error": "empty_prediction",
            }
        )
        return result

    if language == "python" and context:
        result["syntax_mode"] = "python_ast"
        try:
            ast.parse(candidate)
        except SyntaxError as exc:
            result.update(
                {
                    "syntax_ok": False,
                    "syntax_error": f"python_syntax:{exc.msg}",
                }
            )
        else:
            result.update(
                {
                    "syntax_ok": balance_ok,
                    "syntax_error": balance_error,
                }
            )

    return result


def boundary_quality(raw: str, prediction: str, context: dict | None) -> dict:
    leaked_special = prediction != (raw or "")
    suffix_repeated = suffix_overlap(prediction, (context or {}).get("suffix") or "")
    repeated = repetition_issue(prediction)
    empty = not prediction.strip()
    boundary_ok = not leaked_special and not suffix_repeated and not repeated and not empty
    reasons = []
    if leaked_special:
        reasons.append("special_token_leak")
    if suffix_repeated:
        reasons.append("suffix_overlap")
    if repeated:
        reasons.append("repetition")
    if empty:
        reasons.append("empty_prediction")
    return {
        "boundary_ok": boundary_ok,
        "suffix_overlap": suffix_repeated,
        "repetition_issue": repeated,
        "boundary_reasons": reasons,
    }


def is_length_outlier(pred_len: int, gold_len: int) -> bool:
    if gold_len <= 0:
        return pred_len > 0
    return pred_len < max(1, int(gold_len * 0.2)) or pred_len > max(16, int(gold_len * 3.0))


def score_file(path: Path, eval_context: dict[str, dict] | None = None) -> dict:
    rows = []
    special_leaks = 0
    raw_rows = load_rows(path)
    eval_context = eval_context or {}

    for row in raw_rows:
        eval_id = str(row.get("eval_id") or "")
        gold = row.get("gold_middle") or ""
        raw = row.get("cleaned_output") or row.get("raw_output") or ""
        pred = truncate_prediction(raw)
        leaked = pred != raw
        special_leaks += int(leaked)
        context = eval_context.get(eval_id)
        quality = syntax_quality(row.get("language") or "unknown", pred, context)
        boundary = boundary_quality(raw, pred, context)
        length_outlier = is_length_outlier(len(pred), len(gold))
        char_sim = similarity(pred, gold)
        token_sim = token_similarity(pred, gold)
        line_sim = line_similarity(pred, gold)
        failure_reasons = []
        failure_reasons.extend(boundary["boundary_reasons"])
        if length_outlier:
            failure_reasons.append("length_outlier")
        if not quality["syntax_ok"]:
            failure_reasons.append("structure_error")
        if char_sim < 0.15 and normalize_text(pred) != normalize_text(gold):
            failure_reasons.append("low_similarity")
        if not failure_reasons and normalize_text(pred) != normalize_text(gold):
            failure_reasons.append("semantic_or_reference_mismatch")

        rows.append(
            {
                "eval_id": eval_id,
                "language": row.get("language") or "unknown",
                "repo": row.get("repo") or "",
                "span_kind": row.get("span_kind") or "",
                "checkpoint_label": row.get("checkpoint_label") or path.stem,
                "exact": pred == gold,
                "normalized_exact": normalize_text(pred) == normalize_text(gold),
                "similarity": char_sim,
                "char_similarity": char_sim,
                "token_similarity": token_sim,
                "line_similarity": line_sim,
                "gold_len": len(gold),
                "pred_len": len(pred),
                "length_ratio": (len(pred) / len(gold)) if len(gold) else None,
                "leaked_special": leaked,
                "empty_prediction": not pred.strip(),
                "length_outlier": length_outlier,
                **quality,
                **boundary,
                "failure_reasons": failure_reasons,
                "gold": gold,
                "prediction": pred,
            }
        )

    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)

    def aggregate(items: list[dict]) -> dict:
        if not items:
            return {
                "n": 0,
                "exact": 0,
                "normalized_exact": 0,
                "avg_similarity": 0.0,
                "leaked_special": 0,
                "syntax_checked": 0,
                "syntax_ok": 0,
                "syntax_ok_rate": 0.0,
                "empty_prediction": 0,
                "length_outlier": 0,
                "boundary_ok": 0,
                "boundary_ok_rate": 0.0,
                "failure_reasons": {},
                "composite_score": 0.0,
            }
        syntax_checked = sum(1 for item in items if item["syntax_checked"])
        syntax_ok = sum(1 for item in items if item["syntax_ok"])
        normalized_exact = sum(1 for item in items if item["normalized_exact"])
        special_leak = sum(1 for item in items if item["leaked_special"])
        empty_prediction = sum(1 for item in items if item["empty_prediction"])
        length_outlier = sum(1 for item in items if item["length_outlier"])
        boundary_ok = sum(1 for item in items if item["boundary_ok"])
        failure_reasons = Counter(reason for item in items for reason in item["failure_reasons"])
        n = len(items)
        avg_similarity = mean(item["similarity"] for item in items)
        avg_token_similarity = mean(item["token_similarity"] for item in items)
        avg_line_similarity = mean(item["line_similarity"] for item in items)
        syntax_ok_rate = syntax_ok / syntax_checked if syntax_checked else 0.0
        boundary_ok_rate = boundary_ok / n
        length_ok_rate = 1.0 - (length_outlier / n)
        text_score = 0.50 * avg_similarity + 0.30 * avg_token_similarity + 0.20 * avg_line_similarity
        composite_score = (
            0.35 * text_score
            + 0.20 * syntax_ok_rate
            + 0.20 * boundary_ok_rate
            + 0.10 * length_ok_rate
            + 0.15 * (normalized_exact / n)
        )
        return {
            "n": n,
            "exact": sum(1 for item in items if item["exact"]),
            "normalized_exact": normalized_exact,
            "avg_similarity": avg_similarity,
            "avg_token_similarity": avg_token_similarity,
            "avg_line_similarity": avg_line_similarity,
            "leaked_special": special_leak,
            "syntax_checked": syntax_checked,
            "syntax_ok": syntax_ok,
            "syntax_ok_rate": syntax_ok_rate,
            "empty_prediction": empty_prediction,
            "length_outlier": length_outlier,
            "boundary_ok": boundary_ok,
            "boundary_ok_rate": boundary_ok_rate,
            "suffix_overlap": sum(1 for item in items if item["suffix_overlap"]),
            "repetition_issue": sum(1 for item in items if item["repetition_issue"]),
            "length_ok_rate": length_ok_rate,
            "failure_reasons": dict(sorted(failure_reasons.items(), key=lambda item: (-item[1], item[0]))),
            "text": {
                "score": text_score,
                "char_similarity": avg_similarity,
                "token_similarity": avg_token_similarity,
                "line_similarity": avg_line_similarity,
                "normalized_exact_rate": normalized_exact / n,
            },
            "structure": {
                "checked": syntax_checked,
                "ok": syntax_ok,
                "ok_rate": syntax_ok_rate,
                "mode": "python_ast_else_bracket_balance",
            },
            "boundary": {
                "ok": boundary_ok,
                "ok_rate": boundary_ok_rate,
                "special_leak": special_leak,
                "suffix_overlap": sum(1 for item in items if item["suffix_overlap"]),
                "repetition": sum(1 for item in items if item["repetition_issue"]),
                "empty": empty_prediction,
            },
            "composite_score": composite_score,
        }

    failure_examples = sorted(
        rows,
        key=lambda item: (
            not item["failure_reasons"],
            item["boundary_ok"],
            item["syntax_ok"],
            item["similarity"],
        ),
    )[:10]
    return {
        "path": str(path),
        "label": rows[0]["checkpoint_label"] if rows else path.stem,
        "score_version": "fim-local-eval-v2",
        "overall": aggregate(rows),
        "by_language": {language: aggregate(items) for language, items in sorted(by_language.items())},
        "worst": sorted(rows, key=lambda item: (item["syntax_ok"], item["similarity"]))[:5],
        "failure_examples": failure_examples,
    }


def print_report(result: dict, worst_chars: int) -> None:
    overall = result["overall"]
    print("=" * 80)
    print(f"file: {result['path']}")
    print(f"label: {result['label']}")
    print(
        "overall: "
        f"n={overall['n']} "
        f"exact={overall['exact']} "
        f"normalized_exact={overall['normalized_exact']} "
        f"avg_similarity={overall['avg_similarity']:.4f} "
        f"token_similarity={overall.get('avg_token_similarity', 0.0):.4f} "
        f"structure_ok={overall['syntax_ok']}/{overall['syntax_checked']} "
        f"boundary_ok={overall.get('boundary_ok', 0)}/{overall['n']} "
        f"empty={overall['empty_prediction']} "
        f"length_outlier={overall['length_outlier']} "
        f"special_leak={overall['leaked_special']} "
        f"composite={overall['composite_score']:.4f}"
    )
    if overall.get("failure_reasons"):
        print("failure reasons:")
        for reason, count in overall["failure_reasons"].items():
            print(f"  {reason}: {count}")
    print("by language:")
    for language, metrics in result["by_language"].items():
        print(
            f"  {language}: "
            f"n={metrics['n']} "
            f"exact={metrics['exact']} "
            f"normalized_exact={metrics['normalized_exact']} "
            f"avg_similarity={metrics['avg_similarity']:.4f} "
            f"syntax_ok={metrics['syntax_ok']}/{metrics['syntax_checked']} "
            f"special_leak={metrics['leaked_special']} "
            f"composite={metrics['composite_score']:.4f}"
        )
    print("worst examples:")
    for row in result["worst"]:
        print("-" * 80)
        print(
            f"{row['eval_id']} {row['language']} "
            f"sim={row['similarity']:.4f} "
            f"gold_len={row['gold_len']} pred_len={row['pred_len']} "
            f"syntax_ok={row['syntax_ok']} "
            f"syntax_mode={row['syntax_mode']} "
            f"syntax_error={row['syntax_error']!r} "
            f"length_outlier={row['length_outlier']} "
            f"leak={row['leaked_special']}"
        )
        print(f"GOLD: {row['gold'][:worst_chars]!r}")
        print(f"PRED: {row['prediction'][:worst_chars]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_jsonl", nargs="+", help="Prediction JSONL file(s) to score.")
    parser.add_argument("--eval-jsonl", help="Optional eval JSONL with prefix/suffix context.")
    parser.add_argument("--output-json", help="Optional machine-readable JSON report path.")
    parser.add_argument("--worst-chars", type=int, default=400)
    args = parser.parse_args()

    eval_context = load_eval_context(Path(args.eval_jsonl)) if args.eval_jsonl else {}
    results = []
    for item in args.prediction_jsonl:
        result = score_file(Path(item), eval_context=eval_context)
        results.append(result)
        print_report(result, worst_chars=args.worst_chars)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
