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

def find_entity_boundaries_in_question(question: str, entity_label: str) -> List[int]:
    """Find the start and end boundaries of an entity mention in the question."""
    # First try exact match (case insensitive)
    start_index = question.lower().find(entity_label.lower())
    if start_index != -1:
        end_index = start_index + len(entity_label)
        return [start_index, end_index]
    
    # Try to find partial matches or handle tokenization differences
    words = question.split()
    entity_words = entity_label.split()
    
    for i in range(len(words) - len(entity_words) + 1):
        if all(word.lower() in words[i + j].lower() for j, word in enumerate(entity_words)):
            start_char = sum(len(words[k]) + 1 for k in range(i))
            end_char = start_char + sum(len(entity_words[j]) for j in range(len(entity_words))) + len(entity_words) - 1
            return [start_char, end_char]
    
    # Fallback: return the beginning of the question
    return [0, len(entity_label) if entity_label else 1]


def parse_simplequestions_line(line: str) -> Dict[str, str]:
    """Parse a line from the SimpleQuestions dataset format."""
    parts = line.strip().split('\t')
    if len(parts) != 4:
        return None
    
    subject_id, property_id, object_id, question = parts
    
    return {
        'subject_id': subject_id,
        'property_id': property_id,
        'object_id': object_id,
        'question': question
    }


def to_sentence_format(
    pageranks: Dict[str, float],
    question: str,
    subject_id: str,
    property_id: str,
    object_id: str,
    boundaries: List[int],
    G: nx.Graph,
) -> List[Dict]:
    """Convert SimpleQuestions format to sentence format."""
    central_node_id = G.graph['central_node']
    if central_node_id != subject_id:
        print(f"Subject node {subject_id} not found in graph")
        return []

    central_node = G.nodes[central_node_id]
    central_node_label = central_node.get('label')
    central_node_rank = central_node.get('rank')
    
    # Get object information
    object_node = G.nodes.get(object_id)
    if object_node:
        object_label = object_node.get('label')
        object_rank = object_node.get('rank')
    else:
        object_label = get_entity_label(object_id)
        object_rank = pageranks.get(object_id, 0.5)
    
    if object_label is None:
        print(f"Entity '{object_id}' no longer exists - skipping")
        return []
    
    # Get predicate information
    neighbor_edge = G[central_node_id].get(object_id)
    if neighbor_edge:
        predicate_id = neighbor_edge.get('id', property_id)
        predicate_label = neighbor_edge.get('label', property_id)
    else:
        predicate_id = property_id
        predicate_label = property_id
    
    prefix = 'Question: '
    subject_boundary_start = boundaries[0] + len(prefix)
    subject_boundary_end = boundaries[1] + len(prefix)
    
    return [{
        "question": question,
        "answer": object_label,
        "subject_id": central_node_id,
        "subject_label": central_node_label,
        "subject_rank": central_node_rank,
        "subject_boundary_start": subject_boundary_start,
        "subject_boundary_end": subject_boundary_end,
        "predicate_id": predicate_id,
        "predicate_label": predicate_label,
        "object_id": object_id,
        "object_label": object_label,
        "object_rank": object_rank,
        "k": 1,  # SimpleQuestions has single answers
    }]


def create_sentence_tar(data_directory: Path, output_tar_file_path: Path):
    """Create tar file from CSV directory."""
    if not data_directory.exists():
        raise Exception(f"Directory {data_directory} does not exist.")
    
    with tarfile.open(output_tar_file_path, "w") as tar:
        # Loop through the subdirectories "test", "train", and "valid"
        for subdirectory in ["test", "train", "valid"]:
            subdirectory_path = data_directory / subdirectory
            if subdirectory_path.exists():
                for file_path in subdirectory_path.glob('*.csv'):
                    print(f"Adding {file_path} to tar")
                    tar.add(file_path, arcname=str(file_path.relative_to(data_directory)))
    print(f"Tar file created at {output_tar_file_path}")


def create_graph_tar(json_directory: Path, output_tar_file_path: Path):
    """Create tar file from JSON directory."""
    if not json_directory.exists():
        raise Exception(f"Directory {json_directory} does not exist.")
    
    with tarfile.open(output_tar_file_path, "w") as tar:
        for file_path in json_directory.glob('*.json'):
            print(f"Adding {file_path} to tar")
            tar.add(file_path, arcname=file_path.name)
    print(f"Tar file created at {output_tar_file_path}")


def process_simplequestions_split(
    split_file: Path,
    split_name: str,
    pageranks: Dict[str, float],
    sentence_folder_path: Path,
    star_folder_path: Path,
) -> None:
    """Process a single split of the SimpleQuestions dataset."""
    
    with open(split_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_num, line in enumerate(tqdm(lines, desc=f"Processing {split_name}", total=len(lines))):
            data = parse_simplequestions_line(line)
            if not data:
                print(f"Skipping malformed line {line_num} in {split_name}")
                continue
            
            subject_id = data['subject_id']
            property_id = data['property_id']
            object_id = data['object_id']
            question = data['question']
            
            # Skip if subject_id is empty or invalid
            if not subject_id or subject_id == 'None':
                print(f"Skipping line {line_num}: invalid subject_id")
                continue
            
            # Create unique identifier for this question
            question_id = f"{split_name}_{line_num}_{subject_id}"
            
            output_path = sentence_folder_path / f"{question_id}.csv"
            graph_output_path = star_folder_path / f"{subject_id}.json"
            if output_path.exists() and graph_output_path.exists():
                # print(f"Files {output_path} and {graph_output_path} already exist, skipping")
                continue
            
            # Get or create graph
            G = fetch_neighbors(pageranks, subject_id, edge_limit=10_000)
            
            if not G:
                print(f"Graph could not be fetched for entity {subject_id}")
                continue
            
            # Get subject label for boundary detection
            subject_node = G.nodes.get(subject_id)
            if subject_node:
                subject_label = subject_node.get('label', subject_id)
            else:
                print(f"Subject node {subject_id} not found in graph")
                continue
            
            # Find entity boundaries
            boundaries = find_entity_boundaries_in_question(question, subject_label)
            
            # Create sentence format
            sentences = to_sentence_format(
                pageranks, question, subject_id, property_id, 
                object_id, boundaries, G
            )
            
            if not sentences:
                print(f"No valid sentences generated for line {line_num}")
                continue
            
            # Save graph as JSON
            star_entity_path = star_folder_path / f"{subject_id}.json"
            if not star_entity_path.exists():
                graph_json_data = nx.node_link_data(G)
                with open(star_entity_path, 'w') as f:
                    json.dump(graph_json_data, f, indent=4)
            
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


def generate_simplequestions(base_path: str | Path, version: int = 1) -> bool:
    """
    Generate SimpleQuestions artifacts (sentences CSVs and star graphs JSONs) and pack them into tar files
    placed under the provided base_path. If artifacts already exist, it's a no-op.

    Returns True when artifacts are present (created or already existed). Returns False if the
    required raw dataset files are missing.
    """
    base = Path(base_path)
    
    # Input raw files expected to be provided by the user
    train_file = base / "annotated_wd_data_train_answerable.txt"
    valid_file = base / "annotated_wd_data_valid_answerable.txt"
    test_file = base / "annotated_wd_data_test_answerable.txt"

    # Check if at least one input file exists
    input_files = [train_file, valid_file, test_file]
    if not any(f.exists() for f in input_files):
        logging.warning(
            f"Missing SimpleQuestions files in {base}. "
            f"Expected files: {[f.name for f in input_files]}. "
            "Please download them from https://github.com/askplatypus/wikidata-simplequestions"
        )
        return False
    
    # Output artifact locations (config-driven)
    artifacts_sentence_directory = base / f"SimpleQuestions_sentences_v{version}"
    csv_sentence_directory = artifacts_sentence_directory / "csv"
    publish_sentence_directory = artifacts_sentence_directory / "publish"
    output_tar_sentence_file_path = publish_sentence_directory / f"SimpleQuestions_sentences_v{version}.tar"

    artifacts_star_directory = base / f"SimpleQuestions_star_v{version}"
    json_star_directory = artifacts_star_directory / "json"
    publish_star_directory = artifacts_star_directory / "publish"
    output_tar_star_file_path = publish_star_directory / f"SimpleQuestions_star_v{version}.tar"

    # If both tar files already exist, nothing to do
    if output_tar_sentence_file_path.exists() and output_tar_star_file_path.exists():
        logging.info("SimpleQuestions artifacts already present. Skipping generation.")
        return True

    # Ensure directories
    csv_sentence_directory.mkdir(parents=True, exist_ok=True)
    publish_sentence_directory.mkdir(parents=True, exist_ok=True)
    json_star_directory.mkdir(parents=True, exist_ok=True)
    publish_star_directory.mkdir(parents=True, exist_ok=True)

    pageranks = get_pagerank_map(base)

    # Process each split
    splits = [
        (train_file, "train"),
        (valid_file, "valid"),
        (test_file, "test")
    ]
    
    for split_file, split_name in splits:
        if not split_file.exists():
            print(f"Skipping {split_name} split - file {split_file} not found")
            continue
            
        sentence_split_path = csv_sentence_directory / split_name
        sentence_split_path.mkdir(parents=True, exist_ok=True)
        
        process_simplequestions_split(
            split_file, split_name, pageranks, 
            sentence_split_path, json_star_directory
        )

    # Always recreate tars to ensure consistency
    create_graph_tar(json_star_directory, output_tar_star_file_path)
    create_sentence_tar(csv_sentence_directory, output_tar_sentence_file_path)

    logging.info("SimpleQuestions generation complete.")
    return True


def _main():
    parser = argparse.ArgumentParser(description='Generate SimpleQuestions artifacts using a base path.')
    parser.add_argument('--version', type=int, default=1, help='Version number of the dataset (e.g., 1)')
    parser.add_argument('--base-path', type=str, default=None, help='Base path for dataset artifacts', required=True)
    args = parser.parse_args()

    base_path = args.base_path

    ok = generate_simplequestions(base_path=base_path, version=args.version)
    if not ok:
        print("SimpleQuestions raw files missing; nothing generated.")


if __name__ == "__main__":
    _main()
