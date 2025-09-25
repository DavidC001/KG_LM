import json
from typing import List, Dict, Any
from pathlib import Path

from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value


class GrailQAStar(GeneratorBasedBuilder):
    """GrailQA Star graph dataset builder."""
    VERSION = "1.0.0"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="GrailQAStar",
            version=VERSION,
            description="GrailQA star graphs dataset"
        )
    ]
    
    def __init__(self, base_path, **kwargs):
        """
        Initializes the GrailQA Star dataset builder.

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
                "entity": Value("string"),
                "json": Value("string")
            })
        )

    def _split_generators(self, dl_manager: Any) -> List[SplitGenerator]:
        """
        Downloads the data and defines splits of the data.
        """
        grailqa_star_path = self.data_base_path / 'GrailQA_star_v1' / 'publish' / 'GrailQA_star_v1.tar'
        
        urls = {
            "grailqa_star_dir": str(grailqa_star_path),
        }
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "grailqa_star_dir": download_dir["grailqa_star_dir"],
                    "split": "all"
                },
            ),
        ]

    def _generate_examples(self, grailqa_star_dir: str) -> Dict[str, Any]:
        """Generate examples from the GrailQA star graphs."""
        import glob
        json_files = glob.glob(f"{grailqa_star_dir}/*.json")

        for json_file in json_files:
            entity_id = Path(json_file).stem
            with open(json_file, 'r', encoding='utf-8') as file:
                json_content = file.read()
                
                datapoint = {
                    "entity": entity_id,
                    "json": json_content
                }
                yield entity_id, datapoint
