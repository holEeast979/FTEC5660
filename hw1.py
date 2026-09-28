#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You extract totals from Hong Kong supermarket receipt images.
Return exactly one JSON object with these fields:
final_payment_after_rounding, subtotal_after_discounts_before_rounding, discount_total.
Use decimal numbers with two digits after the decimal point and no currency symbols.

Rules:
- final_payment_after_rounding is the amount actually paid after any ROUNDING line. Use the final amount charged, not the subtotal or amount tendered before change.
- subtotal_after_discounts_before_rounding is the receipt subtotal after discounts and before rounding.
- discount_total is the positive sum of all discounts, promotions, coupons, membership or app savings, and other price reductions on the receipt.
- Exclude ROUNDING, payment/tender, change, stored-value balance, and loyalty-point information from discount_total.
- Count each discount once. If a discount is repeated in a summary, do not add that duplicate again.
- If there are no discounts, use 0.00. Read the printed amounts carefully; do not estimate.
- Return JSON only, with no markdown or explanation.""",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Extract the three amounts from this receipt image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            ),
        ]
    )
    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_tokens=8192,
    )
    return prompt | model | StrOutputParser()


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    if not images:
        raise ValueError("At least one receipt image is required")

    inputs = [{"image_url": image_data_url(path)} for path in images]
    outputs = chain.batch(
        inputs,
        config={"max_concurrency": min(3, len(inputs))},
    )
    if len(outputs) != len(images):
        raise RuntimeError(
            f"Expected {len(images)} receipt responses, received {len(outputs)}"
        )

    def read_amount(data: dict[str, Any], key: str, image: Path) -> Decimal:
        if key not in data:
            raise ValueError(f"{image.name}: model response is missing {key}")
        raw_value = str(data[key])
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", raw_value)
        if match is None:
            raise ValueError(f"{image.name}: invalid amount for {key}")
        try:
            return Decimal(match.group(0).replace(",", "")).quantize(
                Decimal("0.01")
            )
        except InvalidOperation as exc:
            raise ValueError(f"{image.name}: invalid amount for {key}") from exc

    def parse_response(output: Any, image: Path) -> dict[str, Any]:
        text = str(output).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"{image.name}: model response did not contain JSON")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"{image.name}: model returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{image.name}: model response must be a JSON object")
        return data

    spent_total = Decimal("0.00")
    pre_discount_total = Decimal("0.00")
    for image, output in zip(images, outputs):
        for attempt in range(3):
            try:
                data = parse_response(output, image)
                break
            except ValueError:
                if attempt == 2:
                    raise
                output = chain.invoke({"image_url": image_data_url(image)})

        paid = read_amount(data, "final_payment_after_rounding", image)
        subtotal = read_amount(
            data, "subtotal_after_discounts_before_rounding", image
        )
        discounts = abs(read_amount(data, "discount_total", image))
        spent_total += paid
        pre_discount_total += subtotal + discounts

    return {
        QUERY_1: f"HK${spent_total:.2f}",
        QUERY_2: f"HK${pre_discount_total:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
