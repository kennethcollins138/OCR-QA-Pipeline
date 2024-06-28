import sys
import pathlib
import logging
import re
import argparse
import os
import json
import atexit
import time
from functools import partial
import requests
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device, default_batch_size
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
import pypdf
import jsonschema
from jsonschema import validate
from flask import Flask, request, jsonify

# Initialize Flask app
nougat_ocr = Flask(__name__)
logger = logging.getLogger(__name__)

def setup_logging():
    config_file = pathlib.Path("logging_configs/queued-stderr-json-file.json")
    with open(config_file) as f_in:
        config = json.load(f_in)
    logging.config.dictConfig(config)
    queue_handler = logging.getLogger().handlers[0]
    if queue_handler:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)

def validate_json(data):
    schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "document": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "file_path": {"type": "string"},
                    "document_type": {"type": "string"},
                    "uploaded_at": {"type": "string"}
                },
                "required": ["document_id", "file_path", "document_type", "uploaded_at"]
            },
            "process": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "last_updated": {"type": "string"}
                },
                "required": ["status", "last_updated"]
            },
            "pages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page_number": {"type": "integer"},
                        "raw_content": {"type": "string"},
                        "ocr_result": {"type": "string"}
                    },
                    "required": ["page_number", "raw_content", "ocr_result"]
                }
            },
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "message": {"type": "string"}
                    },
                    "required": ["timestamp", "message"]
                }
            },
            "model_config": {
                "type": "object",
                "properties": {
                    "selected_pages": {"type": "string"},
                    "other_options": {"type": "object"}
                },
                "required": ["selected_pages"]
            }
        },
        "required": ["user_id", "document", "process", "pages", "model_config"]
    }
    validate(instance=data, schema=schema)

def get_args(pdf_path: str, small: bool = False):
    """
    Configure and return arguments for running the Nougat model.

    Parameters:
    pdf_path (str): Path to the PDF file to process.
    small (bool): Flag to indicate if the small model should be used.

    Returns:
    argparse.Namespace: Parsed arguments for the Nougat model.
    """
    parser = argparse.ArgumentParser(description="Configure Nougat model arguments.")
    parser.add_argument(
        "--batchsize", "-b", type=int, default=default_batch_size(),
        help="Batch size to use."
    )
    parser.add_argument(
        "--checkpoint", "-c", type=Path, default=None,
        help="Path to checkpoint directory."
    )
    parser.add_argument(
        "--model", "-m", type=str, default="0.1.0-small" if small else "0.1.0-base",
        help="Model tag to use."
    )
    parser.add_argument("--out", "-o", type=Path, help="Output directory.")
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute already computed PDF, discarding previous predictions."
    )
    parser.add_argument(
        "--full-precision", action="store_true",
        help="Use float32 instead of bfloat16. Can speed up CPU conversion for some setups."
    )
    parser.add_argument(
        "--no-markdown", dest="markdown", action="store_false",
        help="Do not add postprocessing step for markdown compatibility."
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Add postprocessing step for markdown compatibility (default)."
    )
    parser.add_argument(
        "--no-skipping", dest="skipping", action="store_false",
        help="Don't apply failure detection heuristic."
    )
    parser.add_argument(
        "--pages", "-p", type=str,
        help="Provide page numbers like '1-4,7' for pages 1 through 4 and page 7. Only works for single PDF input."
    )
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF(s) to process.")
    
    args_list = [
        "--checkpoint", "/root/weights/small" if small else "/root/weights/base",
        "--model", "0.1.0-small" if small else "0.1.0-base",
        "--out", "/root",
        pdf_path
    ]

    args = parser.parse_args(args_list)
    
    if args.checkpoint is None or not args.checkpoint.exists():
        args.checkpoint = get_checkpoint(args.checkpoint, model_tag=args.model)
    
    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be a directory.")
            sys.exit(1)
    
    if len(args.pdf) == 1 and not args.pdf[0].suffix == ".pdf":
        try:
            pdfs_path = args.pdf[0]
            if pdfs_path.is_dir():
                args.pdf = list(pdfs_path.rglob("*.pdf"))
            else:
                args.pdf = [Path(l) for l in open(pdfs_path).read().split("\n") if l]
            logging.info(f"Found {len(args.pdf)} files.")
        except Exception as e:
            logging.error(f"Error processing PDF path: {e}")
    
    if args.pages and len(args.pdf) == 1:
        pages = []
        for p in args.pages.split(","):
            if "-" in p:
                start, end = map(int, p.split("-"))
                pages.extend(range(start - 1, end))
            else:
                pages.append(int(p) - 1)
        args.pages = pages
    else:
        args.pages = None
    
    return args

@nougat_ocr.route('/run_nougat', methods=['POST'])
async def run_nougat():
    start_time = time.time()
    setup_logging()
    
    document_config = request.get_json()
    try:
        validate_json(document_config)
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"JSON validation error: {e}")
        return jsonify({"error": f"Invalid JSON format: {e.message}"}), 400

    file_path = document_config['document']['file_path']
    document_id = document_config['document']['document_id']
    directory = os.path.dirname(file_path)

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if file exists in the shared volume
    if not os.path.exists(file_path):
        logger.error(f"File not found in shared volume: {file_path}")
        document_config['errors'].append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "message": f"File not found: {file_path}"
        })
        return jsonify(document_config), 404

    # Process the PDF with Nougat
    try:
        args = get_args(file_path, small=False)
        base_model = NougatModel.from_pretrained(args.checkpoint)
        base_model = move_to_device(base_model, bf16=not args.full_precision, cuda=args.batchsize > 0)
        base_model.eval()

        small_args = get_args(file_path, small=True)
        small_model = NougatModel.from_pretrained(small_args.checkpoint)
        small_model = move_to_device(small_model, bf16=not small_args.full_precision, cuda=small_args.batchsize > 0)
        small_model.eval()

        dataset = LazyDataset(
            args.pdf[0],
            partial(base_model.encoder.prepare_input, random_padding=False),
            args.pages
        )
        dataloader = torch.utils.data.DataLoader(
            ConcatDataset([dataset]),
            batch_size=1,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate
        )

        predictions = []
        failed_pages = []

        def process_page(sample, model, use_markdown):
            try:
                output = model.inference(image_tensors=sample, early_stopping=False)
                page_predictions = []
                for j, prediction in enumerate(output["predictions"]):
                    if use_markdown:
                        prediction = markdown_compatible(prediction)
                    page_predictions.append(prediction)
                return {"success": True, "predictions": page_predictions}
            except Exception as e:
                logger.error(f"Error processing page: {e}")
                return {"success": False}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_page, sample, base_model, args.markdown): sample for sample, _ in dataloader}

            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    predictions.extend(result["predictions"])
                else:
                    failed_pages.append(futures[future])

        # Process failed pages with small model
        if failed_pages:
            logger.info(f"Retrying {len(failed_pages)} failed pages with the small model.")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_page, sample, small_model, args.markdown): sample for sample in failed_pages}

                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        predictions.extend(result["predictions"])
                    else:
                        logger.error(f"Page failed even with the small model.")
                        document_config['errors'].append({
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "message": "Page failed even with the small model."
                        })

        # Update the document_config with predictions
        for page_number, prediction in enumerate(predictions, start=1):
            document_config['pages'].append({
                "page_number": page_number,
                "raw_content": "",  # Haven't decided this yet, will model want to be able to pull images in convo?
                "ocr_result": prediction
            })

        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds.")
        document_config['process']['status'] = "completed"
        document_config['process']['last_updated'] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        return jsonify(document_config)

    except Exception as e:
        logger.error(f"Error during Nougat processing: {e}")
        document_config['errors'].append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "message": f"Error during processing: {e}"
        })
        return jsonify(document_config), 500

if __name__ == "__main__":
    nougat_ocr.run(host='0.0.0.0', port=5000)