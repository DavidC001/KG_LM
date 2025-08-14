import csv
import logging
from pathlib import Path
from typing import List, Any
from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value, Sequence

logger = logging.getLogger(__name__)

class GrailQA(GeneratorBasedBuilder):
    """GrailQA dataset factory for the KG_LFM project."""
    VERSION = "1.0.0"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="GrailQA",
            version=VERSION,
            description="GrailQA dataset for knowledge graph question answering"
        )
    ]
    
    def __init__(self, base_path, **kwargs):
        """
        Initializes the GrailQA dataset builder.

        Args:
        - base_path: Base path for dataset storage
        - **kwargs: Additional keyword arguments.
        """
        self.data_base_path = Path(base_path)
        super().__init__(**kwargs)

    def _info(self) -> DatasetInfo:
        """
        Specifies the datasets.DatasetInfo object.
        """
        return DatasetInfo(
            features=Features({
                "question": Value("string"),
                "answer": Value("string"),
                "k": Value("int32"),
                "subject": {
                    "id": Value("string"),
                    "label": Value("string"),
                    "rank": Value("float64"),
                    "boundaries": Sequence(Value("int32"))
                },
                "predicate": {
                    "id": Value("string"),
                    "label": Value("string")
                },
                "object": {
                    "id": Value("string"),
                    "label": Value("string"),
                    "rank": Value("float64"),
                },
                "level": Value("string"),
                "function": Value("string"),
                "qid": Value("string")
            })
        )

    def _split_generators(self, dl_manager: Any) -> List[SplitGenerator]:
        """
        Downloads the data and defines splits of the data.
        """
        grailqa_sentences_path = self.data_base_path / 'GrailQA_sentences_v1' / 'publish' / 'GrailQA_sentences_v1.tar'
        
        urls = {
            "grailqa_sentences_dir": str(grailqa_sentences_path),
        }
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "grailqa_sentences_dir": download_dir["grailqa_sentences_dir"],
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "grailqa_sentences_dir": download_dir["grailqa_sentences_dir"],
                    "split": "validation",
                },
            )
        ]

    def _generate_examples(self, grailqa_sentences_dir: str, split: str):
        """Generate examples from the GrailQA CSV files."""
        import glob
        selected_data_points = glob.glob(f"{grailqa_sentences_dir}/{split}/*.csv")

        for csv_file in selected_data_points:
            question_id = Path(csv_file).stem
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for idx, row in enumerate(reader):
                    datapoint = {
                        "question": row['question'],
                        "answer": row['answer'],
                        "k": int(row['k']),
                        "subject": {
                            "id": row['subject_id'],
                            "label": row['subject_label'],
                            "rank": float(row['subject_rank']),
                            "boundaries": [
                                int(row['subject_boundary_start']),
                                int(row['subject_boundary_end']),
                            ]
                        },
                        "predicate": {
                            "id": row['predicate_id'],
                            "label": row['predicate_label'],
                        },
                        "object": {
                            "id": row['object_id'],
                            "label": row['object_label'],
                            "rank": float(row['object_rank']),
                        },
                        "level": row.get('level', ''),
                        "function": row.get('function', ''),
                        "qid": row.get('qid', question_id)
                    }
                    yield f'{question_id}-{idx}', datapoint
