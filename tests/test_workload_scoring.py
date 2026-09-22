import pytest

from modellab.workloads.models import ExpectedAnswer
from modellab.workloads.scoring import score_answer


@pytest.mark.parametrize(
    ("output", "expected", "score"),
    [
        ("Paris", ExpectedAnswer(scorer="exact_match", value="Paris"), 1.0),
        (" Paris! ", ExpectedAnswer(scorer="normalized_exact_match", value="paris"), 1.0),
        ("B.", ExpectedAnswer(scorer="multiple_choice", value="B"), 1.0),
        ("TTFT and E2E", ExpectedAnswer(scorer="contains_all", value=["ttft", "e2e"]), 1.0),
        ("42.01", ExpectedAnswer(scorer="numeric_tolerance", value=42, tolerance=0.02), 1.0),
        ('{"ready": true}', ExpectedAnswer(scorer="valid_json"), 1.0),
        (
            '```json\n{"ready": true}\n```',
            ExpectedAnswer(
                scorer="json_schema",
                schema={
                    "type": "object",
                    "required": ["ready"],
                    "properties": {"ready": {"type": "boolean", "const": True}},
                },
            ),
            1.0,
        ),
        ("not json", ExpectedAnswer(scorer="valid_json"), 0.0),
    ],
)
def test_deterministic_scorers(
    output: str, expected: ExpectedAnswer, score: float
) -> None:
    assert score_answer(output, expected) == score


def test_unscored_case_returns_none() -> None:
    assert score_answer("anything", None) is None
