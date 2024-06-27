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

logging.basicConfig(level=logging.INFO)


'''
This code is taken from the predict.py method from Nougat github. Automating the CLI interface with python to handle data ingestion and computation.
'''


dontKnowWhatToNameJSon = [
    {
        User: '123458912',
        Document: {
            documentID: '123455512457890',
            filepath: 'sharedvolume logic',
            document_type: 'scientific',
        },
        Process: {
            status: 'uploaded',
            time: '12:24 PST'
        },
        Pages: {
            1: {raw: "could be like the page itself",
                extracted: "This is where ocr result for page goes"}
        },
        Errors: "Not sure what error handling would be good",
        model-config: {
            selected_pages: '1-4',
            ignore_this: 'i just realized selcted pages might be useless for the since i can extract the pages in the data ingestion step'
        }
    }
]