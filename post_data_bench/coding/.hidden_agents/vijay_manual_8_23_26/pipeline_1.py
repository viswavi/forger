import json
import os

from pyforger.acquire import sample_from_LLM
from pyforger.acquire import retrieve_from_web
from pyforger import op

from extraction import (
    extract_problems_from_github,
    extract_problems_from_pdf,
    extract_problems_from_webpage,
    extract_problems_from_search_results,
    write_jsonl,
)

BASE = "/home/vijayv/forger/post_data_bench/coding"
PIPELINE_DIR = os.path.join(BASE, ".hidden_agents", "vijay_manual_8_23_26")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_validation_data():
    """Load validation data from a JSONL file."""
    validation_file = os.path.join(BASE, "validation_tasks_de_obfuscated.jsonl")
    if not os.path.exists(validation_file):
        print(f"Validation data file not found: {validation_file}")
        return []
    return read_jsonl(validation_file)

def web_search(query: str, n=20):
    return retrieve_from_web(queries=[query], n=n, api_key=os.environ["BRAVE_API_KEY"], cache=True)

# asyncify in next iteration
def impossible_lcb_pipeline():
    # First attempt: retrieve new Python programming contest problems
    basic_search_results = web_search("Very hard programming challenges in Python", n=20)
    icpc_problems = web_search("ICPC programming problems in Python", n=20)
    icpc_problems_2 = web_search("ICPC programming problems full list", n=20)
    icpc_solutions = web_search("ICPC programming problem solutions full list", n=20)
    ioi_solutions = web_search("IOI programming problem solutions full list", n=20)
    all_search_results = (
        basic_search_results
        + icpc_problems
        + icpc_problems_2
        + icpc_solutions
        + ioi_solutions
    )
    extracted = extract_problems_from_search_results(all_search_results)
    out_path = os.path.join(PIPELINE_DIR, "raw_extracted_problems.jsonl")
    write_jsonl(extracted, out_path)
    print(f"Extracted {len(extracted)} problem rows from {len(all_search_results)} search results -> {out_path}")
    return extracted


# asyncify in next iteration
if __name__ == "__main__":
    validation_data = load_validation_data()
    if not validation_data:
        print("No validation data available. Exiting.")
    else:
        impossible_lcb_pipeline()