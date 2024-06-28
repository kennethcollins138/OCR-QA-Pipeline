# OCR-QA-Pipeline

## Overview

OCR-QA-Pipeline is a comprehensive project that automates text extraction from various documents and images and utilizes a large language model (LLM) for advanced question answering. The pipeline is designed to handle multiple forms of text-based images, process them through Optical Character Recognition (OCR), and provide accurate answers to user queries.

## Initial Plan

### Phase One: OCR

- Starting out with OCR pipeline. This consists of a lot of management for containers.
- First issue, How will I pass information between the containers? For testing purposes, I will start out with a shared volume. I will transition to a NFS once fleshed out and OCR is tested. After testing, I will move to the NFS system like stated and transition to the LLM.

videos/135_modern_logging
