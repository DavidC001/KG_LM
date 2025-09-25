import csv
import glob
from typing import List, Dict, Any
from pathlib import Path

from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value, Sequence


class SimpleQuestionsSentences(GeneratorBasedBuilder):
    VERSION = "1.0.0"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="SimpleQuestionsSentences",
            version=VERSION,
            description=("SimpleQuestions dataset with Wikidata entities")
        )
    ]
    
    def __init__(self, base_path, **kwargs):
        """
        Initializes the SimpleQuestionsSentences dataset builder.

        Args:
        - base_path: Base path for dataset artifacts
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
                    "rank": Value("float32"),
                    "boundaries": Sequence(Value("int32"))  # List of integers
                },
                "predicate": {
                    "id": Value("string"),
                    "label": Value("string")
                },
                "object": {
                    "id": Value("string"),
                    "label": Value("string"),
                    "rank": Value("float32"),
                }
            })
        )

    def _split_generators(self, dl_manager: Any) -> List[SplitGenerator]:
        """
        Downloads the data and defines splits of the data.

        Args:
        - dl_manager: datasets.download.DownloadManager object

        Returns:
        - List of SplitGenerator objects for each data split.
        """
        simplequestions_path = self.data_base_path / 'SimpleQuestions_sentences_v1' / 'publish' / 'SimpleQuestions_sentences_v1.tar'
        
        urls = {
            "simplequestions_sentences_dir": str(simplequestions_path),
        }
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "simplequestions_sentences_dir": download_dir["simplequestions_sentences_dir"],
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "simplequestions_sentences_dir": download_dir["simplequestions_sentences_dir"],
                    "split": "valid",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "simplequestions_sentences_dir": download_dir["simplequestions_sentences_dir"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, simplequestions_sentences_dir: str, split: str) -> Dict[str, Any]:
        selected_data_points = glob.glob(f"{simplequestions_sentences_dir}/{split}/*.csv")

        for csv_file in selected_data_points:
            question_id = Path(csv_file).stem
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                # Yielding each row
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
                        }
                    }
                    yield f'{question_id}-{idx}', datapoint
