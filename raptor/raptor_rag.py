# raptor_rag.py Description: This file contains the Raptor_RAG_Wrapper class that wraps the Raptor Retrieval Augmentation module.
import re
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from raptor.EmbeddingModels import BaseEmbeddingModel
from raptor.QAModels import GPT3TurboQAModel
from raptor.RetrievalAugmentation import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
)
from raptor.SummarizationModels import GPT3TurboSummarizationModel
import asyncio
import json
from typing import Set
import logging
from typing import List, Callable, Any, Dict
from tqdm import tqdm
import time
from torch import Tensor
from typing import Optional
from typing import List, Union

logging.basicConfig(
    level=logging.INFO,
    filename="raptor_wrapper.log",
    filemode="a",  # Use 'w' to overwrite the log file each time, 'a' to append to it
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1",
        device="mps",
    ):
        self.model = SentenceTransformer(model_name, device=device)
        logging.info(
            f"Initialized SBertEmbeddingModel with model {model_name} on device {device}"
        )

    def create_embedding(self, text) -> Union[List[Tensor], np.ndarray, Tensor]:
        logging.info("Creating embedding for text")
        return self.model.encode(text)

    def embed_query(self, query):
        logging.info(f"Embedding query: {query}")
        return self.create_embedding(query)


class Raptor_RAG_Wrapper:
    def __init__(self, tree=None):
        self.RAC = RetrievalAugmentationConfig(
            summarization_model=GPT3TurboSummarizationModel(),
            qa_model=GPT3TurboQAModel(),
            embedding_model=SBertEmbeddingModel(),
        )
        self.RA = (
            RetrievalAugmentation(tree=tree, config=self.RAC)
            if tree
            else RetrievalAugmentation(config=self.RAC)
        )
        self.tree_cache = {}
        logging.info("Initialized Raptor_RAG_Wrapper")

    @lru_cache(maxsize=1)
    def set_tree(self, tree):
        if tree not in self.tree_cache:
            self.tree_cache[tree] = RetrievalAugmentation(tree=tree, config=self.RAC)
        self.RA = self.tree_cache[tree]
        logging.info(f"Tree set: {tree}")

    def add_documents(self, text):
        """Add a document to the internal model for augmentation."""
        self.RA.add_documents(text)
        logging.info("Added document to the internal model for augmentation")

    def save(self, augmented_file_path):
        """Save the augmented document to a specified path."""
        self.RA.save(augmented_file_path)
        logging.info(f"Saved augmented document to: {augmented_file_path}")

    def answer_question(self, question, context=None):
        """Answer a question based on the augmented document's context"""
        logging.info(f"Answering question: {question}")
        return self.RA.answer_question(question=question)


    async def process_file(
        self, file: str, search_term: str, query: str
    ) -> Tuple[Tuple[str, str], str]:
        try:
            self.set_tree(file)

            answer = await asyncio.to_thread(
                self.answer_question,
                f"Does the document contain any information on {query}? If so, please provide all relevant details. If not, answer 'No'.",
            )
            
            # Ensure answer is a string before using strip()
            if isinstance(answer, str) and answer.strip().lower() != 'no':
                title: Union[str, None, Exception] = await asyncio.to_thread(
                    self.answer_question,
                    "What is the title of the document?"
                )
                
                # Ensure title is a string before appending
                if isinstance(title, str):
                    answer = f"Title: {title}\nAnswer: {answer}"
            return (search_term, file), answer
        except Exception as e:
            logging.error(f"Error processing {file}: {e}")
            return (search_term, file), "Error"
            return (search_term, file), answer
        except Exception as e:
            logging.error(f"Error processing {file}: {e}")
            return (search_term, file), "Error"

    async def analyze_executable_files_for_query(
        self, results: Dict[str, Dict[str, List[str]]], query: str
    ) -> Dict[Tuple[str, str], str]:
        tasks = []
        for search_term, file_types in results.items():
            for file in file_types["executable_files"]:
                tasks.append(self.process_file(file, search_term, query))

        answers = {}
        for task in asyncio.as_completed(tasks):
            key, answer = await task
            answers[key] = answer

        return answers

    async def process_files_and_save(
        self,
        file_paths: List[str],
        output_file: str,
        processed_files: Set[str],
        batch_size: int = 50
    ) -> List[str]:
        total_files = len(file_paths)
        print(f"Total files to process: {total_files}")
        print(f"Already processed files: {len(processed_files)}")
        
        files_to_process = [f for f in file_paths if f not in processed_files]
        print(f"Files that need processing: {len(files_to_process)}")
        
        if not files_to_process:
            print("No new files to process.")
            return list(processed_files)

        async def process_file(filepath: str):
            try:
                self.set_tree(filepath)
                if self.RA.tree:
                    data = []
                    for node_id, node in self.RA.tree.all_nodes.items():
                        data.append({
                            "file_path": filepath,
                            "node_id": node_id,
                            "text": node.text,
                            "embeddings": {
                                key: val.tolist() if isinstance(val, np.ndarray) else val
                                for key, val in node.embeddings.items()
                            },
                        })
                    return data
            except Exception as e:
                logging.error(f"Error processing {filepath}: {e}")
            return []

        newly_processed_files = set()
        start_time = time.time()

        async with asyncio.TaskGroup() as tg:
            with open(output_file, "a") as outfile:
                for i in tqdm(range(0, len(files_to_process), batch_size), desc="Processing batches"):
                    batch = files_to_process[i:i+batch_size]
                    tasks = [tg.create_task(process_file(file_path)) for file_path in batch]
                    
                    for task in asyncio.as_completed(tasks):
                        result = await task
                        for data in result:
                            json.dump(data, outfile)
                            outfile.write("\n")
                            if "file_path" in data:
                                newly_processed_files.add(data["file_path"])

        duration = time.time() - start_time
        files_processed = len(newly_processed_files)
        remaining_files = total_files - (len(processed_files) + files_processed)
        
        print(f"Final Progress: {files_processed} new files processed, {remaining_files} remaining.")
        print(f"Processing took {duration:.2f} seconds.")
        # Delete all unused variables to free up memory
        del start_time
        del total_files
        del files_to_process
        del file_paths
        del batch_size
        del duration
        del remaining_files
        del files_processed

        return list(processed_files.union(newly_processed_files))


# Example usage:
# async def main():
#     file_paths = ["path1.txt", "path2.txt", "path3.txt"]
#     output_file = "output.ndjson"
#     processed_files = set()
#
#     def process_func(file_path: str) -> Dict[str, Any]:
#         # Your processing logic here
#         return {"file_path": file_path, "content": "processed content"}
#
#     updated_processed_files = await process_files_and_save(
#         file_paths, output_file, process_func, processed_files
#     )
#     print(f"Total files processed: {len(updated_processed_files)}")
#
# if __name__ == "__main__":
#     asyncio.run(main())


# Running instructions:
# Instantiate the Raptor_RAG_Wrapper and use its methods to perform desired operations.
# Example usage:
# async def main():
#     raptor_wrapper = Raptor_RAG_Wrapper()
#     results = {...}  # Your results dictionary
#     query = "Your query here"
#     answers = await raptor_wrapper.analyze_executable_files_for_query(results, query)
#     print(answers)

# if __name__ == "__main__":
#     asyncio.run(main())
