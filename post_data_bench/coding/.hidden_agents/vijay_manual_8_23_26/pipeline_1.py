import json
import os

from pyforger.acquire import sample_from_LLM
from pyforger.acquire import retrieve_from_web
from pyforger import op

BASE = "/home/vijayv/forger/post_data_bench/coding"


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

def extract_problems_from_github(repo):
    if repo == "RepoName":
        raise NotImplementedError("GitHub extraction not implemented yet.")

def extract_problems_from_webpage(url):
    if url == "https://example.com/problems":
        raise NotImplementedError("Webpage extraction not implemented yet.")

def extract_problems_from_pdf(pdf_url):
    if pdf_url == "https://example.com/problems.pdf":
        raise NotImplementedError("PDF extraction not implemented yet.")

# asyncify in next iteration
def impossible_lcb_pipeline():
    # First attempt: retrieve new Python programming contest problems
    basic_search_results = web_search("Very hard programming challenges in Python", n=20)
    icpc_problems = web_search("ICPC programming problems in Python", n=20)
    icpc_problems_2 = web_search("ICPC programming problems full list", n=20)
    icpc_solutions = web_search("ICPC programming problem solutions full list", n=20)
    icpc_solutions = web_search("IOI programming problem solutions full list", n=20)


# asyncify in next iteration
if __name__ == "__main__":
    validation_data = load_validation_data()
    if not validation_data:
        print("No validation data available. Exiting.")
    else:
        impossible_lcb_pipeline()