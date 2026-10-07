"""The benchmark grades the final answer, never a reasoning trace."""

def final_answer(completion: str | None, *, thinking: bool = False) -> str:
    text = completion or ""
    if thinking and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return ""
    return text.strip()


def is_abstention(answer: str, marker: str) -> bool:
    return answer.strip().casefold() == marker.strip().casefold()
