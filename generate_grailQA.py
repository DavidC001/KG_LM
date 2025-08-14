import argparse
import csv
import json
import logging
import tarfile
from pathlib import Path
from typing import List, Dict, Tuple, Any
import networkx as nx
from tqdm import tqdm
from KG_LFM.utils.SparqlQueryEngine import get_pagerank_map, fetch_neighbors, get_entity_label, mid_to_qid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_entity_boundaries_in_question(question: str, entity_mention: str) -> List[int]:
    """Find the start and end boundaries of an entity mention in the question."""
    start_index = question.lower().find(entity_mention.lower())
    if start_index == -1:
        # Try to find partial matches or handle tokenization differences
        words = question.split()
        entity_words = entity_mention.split()
        
        for i in range(len(words) - len(entity_words) + 1):
            if all(word.lower() in words[i + j].lower() for j, word in enumerate(entity_words)):
                start_char = sum(len(words[k]) + 1 for k in range(i))
                end_char = start_char + sum(len(entity_words[j]) for j in range(len(entity_words))) + len(entity_words) - 1
                return [start_char, end_char]
        
        # Fallback: return the beginning of the question
        return [0, len(entity_mention)]
    
    end_index = start_index + len(entity_mention)
    return [start_index, end_index]


def grailqa_to_sentence_format(
    pageranks: Dict[str, float],
    question: str,
    answer_entities: List[str],
    central_entity_id: str,
    central_entity_label: str,
    qid: str,
    level: str,
    function: str,
    G: nx.Graph,
) -> List[Dict]:
    """Convert GrailQA format to sentence format similar to WebQSP."""
    
    central_node_rank = pageranks.get(central_entity_id, 0.5)
    boundaries = find_entity_boundaries_in_question(question, central_entity_label)
    
    sentences = []
    for answer_entity in answer_entities:
        # Convert Freebase ID to Wikidata if needed
        if answer_entity.startswith('m.'):
            wikidata_id = mid_to_qid(answer_entity)
            if not wikidata_id:
                continue
            answer_entity = wikidata_id
        
        # Get answer entity information
        if G and G.has_node(answer_entity):
            neighbor_node = G.nodes[answer_entity]
            object_label = neighbor_node.get('label')
            object_rank = neighbor_node.get('rank')
        else:
            object_label = get_entity_label(answer_entity)
            object_rank = pageranks.get(answer_entity, 0.5)
        
        if not object_label:
            print(f"Could not get label for answer entity {answer_entity}")
            continue
        
        # Find the relationship between central entity and answer
        predicate_id = "Unknown"
        predicate_label = "Unknown"
        
        if G and G.has_edge(central_entity_id, answer_entity):
            edge_data = G[central_entity_id][answer_entity]
            predicate_id = edge_data.get('id', 'Unknown')
            predicate_label = edge_data.get('label', 'Unknown')
        
        prefix = 'Question: '
        subject_boundary_start = boundaries[0] + len(prefix)
        subject_boundary_end = boundaries[1] + len(prefix)
        
        sentences.append({
            "question": question,
            "answer": object_label,
            "subject_id": central_entity_id,
            "subject_label": central_entity_label,
            "subject_rank": central_node_rank,
            "subject_boundary_start": subject_boundary_start,
            "subject_boundary_end": subject_boundary_end,
            "predicate_id": predicate_id,
            "predicate_label": predicate_label,
            "object_id": answer_entity,
            "object_label": object_label,
            "object_rank": object_rank,
            "k": len(answer_entities),
            "level": level,
            "function": function,
            "qid": qid
        })
    
    return sentences


def create_sentence_tar(data_directory: Path, output_tar_file_path: Path):
    """Create a tar file from the CSV sentence data."""
    if not data_directory.exists():
        raise Exception(f"Directory {data_directory} does not exist.")
    
    with tarfile.open(output_tar_file_path, "w") as tar:
        for subdirectory in ["test", "train", "validation"]:
            subdirectory_path = data_directory / subdirectory
            if subdirectory_path.exists():
                for file_path in subdirectory_path.glob('*.csv'):
                    print(f"Adding {file_path} to tar")
                    tar.add(file_path, arcname=str(file_path.relative_to(data_directory)))
    print(f"Tar file created at {output_tar_file_path}")


def create_graph_tar(json_directory: Path, output_tar_file_path: Path):
    """Create a tar file from the JSON graph data."""
    if not json_directory.exists():
        raise Exception(f"Directory {json_directory} does not exist.")
    
    with tarfile.open(output_tar_file_path, "w") as tar:
        for file_path in json_directory.glob('*.json'):
            print(f"Adding {file_path} to tar")
            tar.add(file_path, arcname=file_path.name)
    print(f"Tar file created at {output_tar_file_path}")


def generate_grailqa(base_path: str | Path, version: int = 1) -> bool:
    """
    Generate GrailQA artifacts (sentences CSVs and star graphs JSONs) and pack them into tar files.
    
    Returns True when artifacts are present (created or already existed). 
    Returns False if the required raw dataset file is missing.
    """
    base = Path(base_path)
    
    # Input raw file expected to be provided by the user
    raw_input_file = base / f"grailqa_v{version}.0_train.json"
    raw_dev_file = base / f"grailqa_v{version}.0_dev.json"
    # raw_test_file = base / f"grailqa_v{version}.0_test_public.json" # lacks important fields for preprocessing

    # Output artifact locations
    artifacts_sentence_directory = base / f"GrailQA_sentences_v{version}"
    csv_sentence_directory = artifacts_sentence_directory / "csv"
    publish_sentence_directory = artifacts_sentence_directory / "publish"
    output_tar_sentence_file_path = publish_sentence_directory / f"GrailQA_sentences_v{version}.tar"

    artifacts_star_directory = base / f"GrailQA_star_v{version}"
    json_star_directory = artifacts_star_directory / "json"
    publish_star_directory = artifacts_star_directory / "publish"
    output_tar_star_file_path = publish_star_directory / f"GrailQA_star_v{version}.tar"

    # If both tar files already exist, nothing to do
    if output_tar_sentence_file_path.exists() and output_tar_star_file_path.exists():
        logging.info("GrailQA artifacts already present. Skipping generation.")
        return True

    # Ensure directories
    csv_sentence_directory.mkdir(parents=True, exist_ok=True)
    publish_sentence_directory.mkdir(parents=True, exist_ok=True)
    json_star_directory.mkdir(parents=True, exist_ok=True)
    publish_star_directory.mkdir(parents=True, exist_ok=True)

    # Check if raw files exist
    input_files = [
        (raw_input_file, "train"),
        (raw_dev_file, "validation"),
        # (raw_test_file, "test")
    ]
    
    available_files = [(f, split) for f, split in input_files if f.exists()]
    
    if not available_files:
        logging.warning(
            f"Missing raw GrailQA files. Provide grailqa_v1.0_train.json, grailqa_v1.0_dev.json, "
            f"and/or grailqa_v1.0_test_public.json to enable dataset generation."
        )
        return False

    pageranks = get_pagerank_map(base)

    for raw_file, split in available_files:
        logging.info(f"Processing {split} split from {raw_file}")
        
        with open(raw_file, "r") as f:
            grailqa_datapoints = json.load(f)

        sentence_folder_path = csv_sentence_directory / split
        sentence_folder_path.mkdir(parents=True, exist_ok=True)

        successful_conversions = 0
        progress = tqdm(grailqa_datapoints, desc=f"Generating GrailQA {split} split")
        for datapoint in progress:
            qid = str(datapoint.get('qid', ''))
            question_text = datapoint.get('question', '')
            level = datapoint.get('level', '')
            function = datapoint.get('function', 'none')
            
            # Skip if no graph query or nodes
            graph_query = datapoint.get('graph_query', {})
            nodes = graph_query.get('nodes', [])
            if not nodes:
                tqdm.write(f"No nodes found for question {qid}")
                continue

            # Find the central entity (usually the one that's not a question node and is an entity)
            central_entity_id = None
            central_entity_label = None
            
            for node in nodes:
                if (node.get('node_type') == 'entity' and 
                    node.get('question_node', 0) == 0 and 
                    node.get('id', '').startswith('m.')):
                    
                    freebase_id = node.get('id', '')
                    # Convert Freebase ID to Wikidata
                    wikidata_id = mid_to_qid(freebase_id)
                    if wikidata_id:
                        central_entity_id = wikidata_id
                        central_entity_label = node.get('friendly_name', '')
                        break
            
            if not central_entity_id:
                tqdm.write(f"No suitable central entity found for question {qid}")
                continue

            # Extract answer entities
            answer_entities = []
            answers = datapoint.get('answer', [])
            for answer in answers:
                if answer.get('answer_type') == 'Entity':
                    answer_arg = answer.get('answer_argument', '')
                    if answer_arg.startswith('m.'):
                        wikidata_id = mid_to_qid(answer_arg)
                        if wikidata_id:
                            answer_entities.append(wikidata_id)

            if not answer_entities:
                tqdm.write(f"No valid answer entities found for question {qid}")
                continue

            output_path = sentence_folder_path / f"{qid}_{central_entity_id}.csv"
            if output_path.exists():
                tqdm.write(f"File {output_path} already exists, skipping")
                successful_conversions += 1
                progress.set_description(f"Generated {successful_conversions} samples")
                continue

            # Get or create graph
            G = fetch_neighbors(pageranks, central_entity_id, edge_limit=10_000)
            
            if not G:
                tqdm.write(f"Graph could not be fetched for entity {central_entity_id}")
                G = None

            # Save graph as JSON if it exists
            if G:
                star_entity_path = json_star_directory / f"{central_entity_id}.json"
                if not star_entity_path.exists():
                    graph_json_data = nx.node_link_data(G)
                    with open(star_entity_path, 'w') as f:
                        json.dump(graph_json_data, f, indent=4)

            # Create sentence format
            sentences = grailqa_to_sentence_format(
                pageranks, question_text, answer_entities, central_entity_id, 
                central_entity_label, qid, level, function, G
            )
            
            if not sentences:
                tqdm.write(f"No sentences generated for question {qid}")
                continue
            
            successful_conversions += 1
            
            # Write to CSV
            with open(output_path, "w") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "question", "answer", "subject_id", "subject_label", "subject_rank",
                        "subject_boundary_start", "subject_boundary_end",
                        "predicate_id", "predicate_label", "object_id",
                        "object_label", "object_rank", "k", "level", "function", "qid"
                    ],
                )
                writer.writeheader()
                writer.writerows(sentences)

            progress.set_description(f"Generated {successful_conversions} samples")

    # Always recreate tars to ensure consistency
    create_graph_tar(json_star_directory, output_tar_star_file_path)
    create_sentence_tar(csv_sentence_directory, output_tar_sentence_file_path)

    logging.info("GrailQA generation complete.")
    return True


def _main():
    parser = argparse.ArgumentParser(description='Generate GrailQA artifacts using a base path.')
    parser.add_argument('--version', type=int, default=1, help='Version number of the dataset (e.g., 1)')
    parser.add_argument('--base-path', type=str, default=None, help='Base path for dataset artifacts', required=True)
    args = parser.parse_args()

    base_path = args.base_path

    ok = generate_grailqa(base_path=base_path, version=args.version)
    if not ok:
        print("GrailQA raw files missing; nothing generated.")


if __name__ == "__main__":
    _main()
