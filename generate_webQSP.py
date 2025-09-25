import argparse
import csv
import json
import logging
import tarfile
from pathlib import Path
from typing import List, Dict, Tuple, Any
import networkx as nx
from tqdm import tqdm
from KG_LM.utils.SparqlQueryEngine import get_pagerank_map, fetch_neighbors, get_entity_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_entity_boundaries(sentence: str, token_ids: List[int] | None):
    words = sentence.split()
    try:
        start_index = sum(len(words[i]) + 1 for i in range(token_ids[0]))  # +1 for spaces
        start_index -= 1
        entity_length = sum(len(words[i]) for i in token_ids) + (len(token_ids) - 1)
        end_index = start_index + entity_length
    except IndexError:
        return None
    return [start_index, end_index]

def to_sentence_format(
    pageranks: Dict[str, float],
    utterance: str,
    answer_ids: List[str],
    boundaries: List[int],
    G: nx.Graph,
):
    central_node_id = G.graph['central_node']
    central_node = G.nodes[central_node_id]
    central_node_label = central_node.get('label')
    central_node_rank = central_node.get('rank')
    answers = []
    for answer_id in answer_ids:
        neighbor_node = G.nodes.get(answer_id)
        if neighbor_node:
            object_label = neighbor_node.get('label')
            object_rank = neighbor_node.get('rank')
        else:
            object_label = get_entity_label(answer_id)
            object_rank = pageranks.get(answer_id, 0.5)
        if object_label is None:
            print(f"Entity '{answer_id}' no longer exists - skipping")
            continue
        neighbor_edge = G[central_node_id].get(answer_id)
        if neighbor_edge:
            predicate_id = neighbor_edge.get('id')
            predicate_label = neighbor_edge.get('label')
        else:
            predicate_id = "Unknown"
            predicate_label = "Unknown"
        prefix = 'Question: '
        question = utterance
        answer = object_label
        subject_boundary_start = boundaries[0] + len(prefix)
        subject_boundary_end = boundaries[1] + len(prefix)
        answers.append({
            "question": question,
            "answer": answer,
            "subject_id": central_node_id,
            "subject_label": central_node_label,
            "subject_rank": central_node_rank,
            "subject_boundary_start": subject_boundary_start,
            "subject_boundary_end": subject_boundary_end,
            "predicate_id": predicate_id,
            "predicate_label": predicate_label,
            "object_id": answer_id,
            "object_label": object_label,
            "object_rank": object_rank,
            "k": len(answer_ids),
        })
    return answers


def create_sentence_tar(data_directory: Path, output_tar_file_path: Path):
    # Check if the csv directory exists
    if not data_directory.exists():
        raise Exception(f"Directory {data_directory} does not exist.")
    # Creating a tar file
    with tarfile.open(output_tar_file_path, "w") as tar:
        # Loop through the subdirectories "test", "train", and "validation"
        for subdirectory in ["test", "train", "validation"]:
            subdirectory_path = data_directory / subdirectory
            if subdirectory_path.exists():
                for file_path in subdirectory_path.glob('*.csv'):
                    print(f"Adding {file_path} to tar")
                    # Add each file to the tar, preserving the subdirectory structure
                    tar.add(file_path, arcname=str(file_path.relative_to(data_directory)))
    print(f"Tar file created at {output_tar_file_path}")

def create_graph_tar(json_directory:Path, output_tar_file_path:Path):
    """
    Saves the generated json files into a tar that is later used by HF Dataset
    :return:
    """
    # Check if the json directory exists
    if not json_directory.exists():
        raise Exception(f"Directory {json_directory} does not exist.")
    # Creating a tar file
    with tarfile.open(output_tar_file_path, "w") as tar:
        for file_path in json_directory.glob('*.json'):
            print(f"Adding {file_path} to tar")
            tar.add(file_path, arcname=file_path.name)
    print(f"Tar file created at {output_tar_file_path}")


def generate_webqsp(base_path: str | Path, version: int = 1) -> bool:
    """
    Generate WebQSP artifacts (sentences CSVs and star graphs JSONs) and pack them into tar files
    placed under the provided base_path. If artifacts already exist, it's a no-op.

    Returns True when artifacts are present (created or already existed). Returns False if the
    required raw dataset file is missing.
    """
    base = Path(base_path)
    # Input raw file expected to be provided by the user
    raw_input_file = base / "webqsp.examples.test.wikidata.json"

    # Output artifact locations (config-driven)
    artifacts_sentence_directory = base / f"WebQSP_sentences_v{version}"
    csv_sentence_directory = artifacts_sentence_directory / "csv"
    publish_sentence_directory = artifacts_sentence_directory / "publish"
    output_tar_sentence_file_path = publish_sentence_directory / f"WebQSP_sentences_v{version}.tar"

    artifacts_star_directory = base / f"WebQSP_star_v{version}"
    json_star_directory = artifacts_star_directory / "json"
    publish_star_directory = artifacts_star_directory / "publish"
    output_tar_star_file_path = publish_star_directory / f"WebQSP_star_v{version}.tar"

    # If both tar files already exist, nothing to do
    if output_tar_sentence_file_path.exists() and output_tar_star_file_path.exists():
        logging.info("WebQSP artifacts already present. Skipping generation.")
        return True

    # Ensure directories
    csv_sentence_directory.mkdir(parents=True, exist_ok=True)
    publish_sentence_directory.mkdir(parents=True, exist_ok=True)
    json_star_directory.mkdir(parents=True, exist_ok=True)
    publish_star_directory.mkdir(parents=True, exist_ok=True)

    if not raw_input_file.exists():
        logging.warning(
            f"Missing raw WebQSP file: {raw_input_file}. Provide it to enable dataset generation."
        )
        return False

    pageranks = get_pagerank_map(base_path)

    # Load the dataset format
    with open(raw_input_file, "r") as f:
        web_qsp_datapoints = json.load(f)

    star_folder_path = json_star_directory
    sentence_folder_path = csv_sentence_directory / "test"
    sentence_folder_path.mkdir(parents=True, exist_ok=True)

    for datapoint in tqdm(web_qsp_datapoints, desc=f"Generating WebQSP benchmark"):
        # Get the question text
        question_text = datapoint.get('utterance', '')
        question_id = datapoint.get('questionid', '')  # Assuming there's an ID field

        # Skip if no entities
        if not datapoint.get('entities'):
            print("NO ENTITIES FOUND")
            continue

        # Process each entity in the question
        for entity in datapoint['entities']:
            # Skip if no linkings
            if not entity.get('linkings') or not entity['linkings'][0]:
                print("NO LINKINGS FOUND")
                continue

            entity_id = entity['linkings'][0][0]  # Get the first linking
            if not entity_id:  # Skip if entity_id is null
                print("NO ENTITY ID FOUND")
                continue

            token_ids = entity.get('token_ids', [])
            if not token_ids:
                print("NO TOKEN IDS")
                continue

            # Find entity boundaries
            boundaries = find_entity_boundaries(question_text, token_ids)
            if boundaries is None:
                print("No boundaries found")
                continue

            output_path = sentence_folder_path / f"{question_id}_{entity_id}.csv"
            if output_path.exists():
                print(f"File {output_path} already exists, skipping")
                continue

            # Get or create graph
            G = fetch_neighbors(pageranks, entity_id, edge_limit=10_000)

            if not G:
                print("Graph could not be fetched or created")
                continue

            # Save graph as JSON
            star_entity_path = star_folder_path / f"{entity_id}.json"
            graph_json_data = nx.node_link_data(G)
            with open(star_entity_path, 'w') as f:
                json.dump(graph_json_data, f, indent=4)

            # Extract answer IDs
            answer_ids = datapoint.get('answers', [])

            # Create sentence format
            sentences = to_sentence_format(pageranks, question_text, answer_ids, boundaries, G)
            # Write to CSV
            with open(output_path, "w") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "question", "answer", "subject_id", "subject_label", "subject_rank",
                        "subject_boundary_start", "subject_boundary_end",
                        "predicate_id", "predicate_label", "object_id",
                        "object_label", "object_rank", "k",
                    ],
                )
                writer.writeheader()
                writer.writerows(sentences)

    # Always recreate tars to ensure consistency
    create_graph_tar(json_star_directory, output_tar_star_file_path)
    create_sentence_tar(csv_sentence_directory, output_tar_sentence_file_path)

    logging.info("WebQSP generation complete.")
    return True
def _main():
    parser = argparse.ArgumentParser(description='Generate WebQSP artifacts using a base path.')
    parser.add_argument('--version', type=int, default=1, help='Version number of the dataset (e.g., 1)')
    parser.add_argument('--base-path', type=str, default=None, help='Base path for dataset artifacts', required=True)
    args = parser.parse_args()

    base_path = args.base_path

    ok = generate_webqsp(base_path=base_path, version=args.version)
    if not ok:
        print("WebQSP raw file missing; nothing generated.")


if __name__ == "__main__":
    _main()