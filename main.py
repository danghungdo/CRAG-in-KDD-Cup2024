import os
import bz2
import json
from tqdm import tqdm
from loguru import logger

from models.load_model import load_model, load_model_ollama
from models.router.router import SequenceClassificationRouter, OpenAISequenceClassifier
from models.retrieve.retriever import Retriever, Retriever_Milvus
from models.model import RAGModel, FalsePremiseDetector
from dotenv import load_dotenv

BATCH_SIZE = 1

def load_data_in_batches(dataset_path, batch_size, last_index=0):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.
    
    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    last_index (int): The last index of the dataset that has been processed.
    
    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": [], "question_type": [], "static_or_dynamic": [], "domain": []}

    try:
        if dataset_path.endswith(".bz2"):
            with bz2.open(dataset_path, "rt") as file:
                batch = initialize_batch()
                for i, line in enumerate(file):
                    if i < last_index:
                        continue
                    try:
                        item = json.loads(line)
                        
                        # if item["question_type"] == "simple" and item["split"] == 1:
                            
                        for key in batch:
                            batch[key].append(item[key])
                        
                        if len(batch["query"]) == batch_size:
                            yield batch
                            batch = initialize_batch()
                    except json.JSONDecodeError:
                        logger.warn("Warning: Failed to decode a line.")
                # Yield any remaining data as the last batch
                if batch["query"]:
                    yield batch
        else:
            with open(dataset_path, "r") as file:
                batch = initialize_batch()
                for line in file:
                    try:
                        item = json.loads(line)
                        for key in batch:
                            batch[key].append(item[key])
                        
                        if len(batch["query"]) == batch_size:
                            yield batch
                            batch = initialize_batch()
                    except json.JSONDecodeError:
                        logger.warn("Warning: Failed to decode a line.")
                # Yield any remaining data as the last batch
                if batch["query"]:
                    yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e
    
def generate_predictions(dataset_path, participant_model, false_premise_detector, last_index=0):
    """
    Processes batches of data from a dataset to generate predictions using a model.
    
    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.
    last_index (int): The last index of the dataset that has been processed.
    
    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    questions, dynamics, domains = [], [], []
    batch_size = BATCH_SIZE
    
    try:
        # for batch in tqdm(load_data_in_batches(dataset_path, batch_size, last_index), desc="Detect false premises"):
        #     batch_ground_truths = batch.pop("question_type")  # Remove answers from batch and store them
        #     batch_predictions = false_premise_detector.batch_generate_answer(batch)
        #     for i, pred in enumerate(batch_predictions):
        #         if str(pred).lower() == "true":
        #             false_premise_queries.append(batch["query"][i])
        #             logger.info(f"False premise queries: {false_premise_queries}")
            
            
            
        for batch in tqdm(load_data_in_batches(dataset_path, batch_size, last_index), desc="Generating predictions"):
            batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
            querry = batch["query"]
            false_premise_info = false_premise_detector.batch_generate_answer(batch)
            logger.info(f"False premise info: {false_premise_info}")
            if str(false_premise_info[0]).strip().lower() == "true":
                queries.extend(querry)
                ground_truths.extend(batch_ground_truths)
                predictions.append("invalid question")
                questions.extend(batch["question_type"])
                dynamics.extend(batch["static_or_dynamic"])
                domains.extend(batch["domain"])
                logger.info(f"Detected false premise: {querry}")
                logger.info(f"Ground Truth Example: {ground_truths[-1]}")
                logger.info(f"Prediction Example: {predictions[-1]}")
                continue
            batch_predictions = participant_model.batch_generate_answer(batch)
            queries.extend(batch["query"])
            ground_truths.extend(batch_ground_truths)
            predictions.extend(batch_predictions)
            questions.extend(batch["question_type"])
            dynamics.extend(batch["static_or_dynamic"])
            domains.extend(batch["domain"])
            logger.info(f"Static or Dynamic: {batch['static_or_dynamic']}")
            logger.info(f"Query Example: {queries[-1]}")
            logger.info(f"Ground Truth Example: {ground_truths[-1]}")
            logger.info(f"Prediction Example: {predictions[-1]}")
    except Exception as e:
        logger.error(f"Error: An error occurred while generating predictions. {e}")
    finally:
        return queries, ground_truths, predictions, questions, dynamics, domains
    return queries, ground_truths, predictions, questions, dynamics, domains

if __name__ == "__main__":
    # Set the environment variable for the mock API
    load_dotenv()
    os.environ["CRAG_MOCK_API_URL"] = "https://demo3.kbs.uni-hannover.de"

    # Load the model
    # api_key = "<your-api-key>"
    api_key = os.getenv("INTERWEB_APIKEY")
    # base_url = "<your-base-url>"
    base_url = "https://interweb.l3s.uni-hannover.de"
    # base_url = "http://gpunode04.kbs:11434/v1/"
    model_name = "llama3.3:70b"
    chat_model = load_model(model_name=model_name, api_key=api_key, base_url=base_url, temperature=0)
    # chat_model = load_model(model_name="llama3.3:70b", api_key=api_key, base_url="http://gpunode04.kbs:11434/v1/", temperature=0)
    # chat_model = load_model_ollama(model_name=model_name, temperature=0)

    # Load the retriever
    embedding_model_path = "models/retrieve/embedding_models/bge-m3"
    reranker_model_path = "models/retrieve/reranker_models/bge-reranker-v2-m3"

    # retriever = Retriever(10, 5, embedding_model_path, reranker_model_path, rerank=True)
    # To use the retriever with Milvus, uncomment the following lines and comment the previous line
    # collection_name = "bge_m3_crag_task_3_dev_v3_llamaindex"
    collection_name = "bge_m3_crag_dev_v3_llamaindex"
    # uri = "http://localhost:19530"
    uri = "milvus.db"
    retriever = Retriever_Milvus(10, 5, collection_name, uri, embedding_model_path, reranker_model_path, rerank=True)

    # Load the domain router
    domain_router = SequenceClassificationRouter(
        # model_path="models/llm/Meta-Llama-3-8B-Instruct-hf",
        # model_path="/home/dang.hung.do/models//Meta-Llama-3-8B-Instruct",
        model_path="models/router/bge-m3/domain",
        classes=["finance", "music", "movie", "sports", "open"],
        # classes=["finance", "movie", "music", "open", "sports"],
        device_map="auto",
        # peft_path="models/router/domain",
        # peft_path="/home/dang.hung.do/workspace/finetune-llama-8b/sequence_classification/checkpoint-1355",
        # use_bits_and_bytes=True,
        # use_peft=True,
    )
    # domain_router = OpenAISequenceClassifier(api_key, ["finance", "music", "movie", "sports", "open"], model="llama3.1:8b-instruct-q8_0")

    # Load the dynamic router
    use_kg = True
    use_dynamic = True
    if use_dynamic:
        dynamic_router = SequenceClassificationRouter(
            # model_path="/home/dang.hung.do/models/Meta-Llama-3-8B-Instruct",
            # model_path = "meta-llama/Llama-3.1-8B-Instruct"
            model_path="models/router/bge-m3/dynamic",
            classes=['static', 'slow-changing', 'fast-changing', 'real-time'],
            # classes=["fast-changing", "real-time", "slow-changing", "static"],
            device_map="auto",
            # peft_path="models/router/dynamic",
            # peft_path="/home/dang.hung.do/workspace/finetune-llama-8b/dynamic_classification/checkpoint-408",
            # use_bits_and_bytes=True,
            # use_peft=True,
        )
        # dynamic_router = OpenAISequenceClassifier(api_key, ['static', 'slow-changing', 'fast-changing', 'real-time'], model="llama3.1:8b-instruct-q8_0")
    
    # Initialize the RAG model
        rag_model = RAGModel(chat_model, retriever, domain_router, dynamic_router, use_kg=use_kg)
    else:
        rag_model = RAGModel(chat_model, retriever, domain_router, use_kg=use_kg)
    false_premise_detector = FalsePremiseDetector(chat_model)
    # Generate predictions
    # dataset_path = "example_data/crag_task_1_dev_v4_release.jsonl.bz2"
    # dataset_path = "/home/dang.hung.do/workspace/meta-kdd-cup24/example_data/crag_task_3_dev_v4/crag_task_3_dev_v4_0.jsonl"
    dataset_path = "example_data/dev_data.jsonl.bz2" # 10 samples dataset
    
    # output_path = f"results/llama3.3_70b_predictions_task2_milvus_bge_fpd.jsonl"
    output_path = f"results/llama3.3_70b_predictions_dev_data.jsonl"
    last_index = 0
    if os.path.exists(output_path):
        # read the last line to get the last interaction_id
        with open(output_path, "r") as file:
            lines = file.readlines()
            last_index = len(lines)
    
    if last_index > 0:
       logger.info(f"Resuming from index {last_index}")
    
    queries, ground_truths, predictions, questions, dynamics, domains = generate_predictions(dataset_path, rag_model, false_premise_detector, last_index)
    
    # Save the predictions

    if last_index > 0:
        with open(output_path, "a") as file:
            for query, ground_truth, prediction, question_type, dynamic, domain in zip(queries, ground_truths, predictions, questions, dynamics, domains):
                item = {"query": query, "ground_truth": ground_truth, "prediction": prediction, "question_type": question_type, "static_or_dynamic": dynamic, "domain": domain}
                file.write(json.dumps(item) + "\n")
    else:
        with open(output_path, "w") as file:
            for query, ground_truth, prediction, question_type, dynamic, domain in zip(queries, ground_truths, predictions, questions, dynamics, domains):
                item = {"query": query, "ground_truth": ground_truth, "prediction": prediction, "question_type": question_type, "static_or_dynamic": dynamic, "domain": domain}
                file.write(json.dumps(item) + "\n")
   
    logger.info(f"Predictions saved to {output_path}.")
    
    