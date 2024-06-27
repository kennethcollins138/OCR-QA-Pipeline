import sys
from pathlib import Path
import logging
import re
import argparse
import re
import os
from functools import partial
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

logging.basicConfig(level=logging.INFO)


'''
This code is taken from the predict.py method from Nougat github. Automating the CLI interface with python to handle data ingestion and computation.
Other than the automation, functionality is the same.
'''


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


async def run_nougat(document_config: dict) -> dict:
    pass


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