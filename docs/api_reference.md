# Ocr Config json

'''
{
    "user_id": "123458912",
    "document": {
        "document_id": "123455512457890",
        "file_path": "sharedvolume/path/to/document.pdf",
        "document_type": "scientific",
        "uploaded_at": "2022-11-15T12:24:00Z"
    },
    "process": {
        "status": "uploaded",
        "last_updated": "2022-11-15T12:24:00Z"
    },
    "pages": [
        {
            "page_number": 1,
            "raw_content": "Base64_encoded_page_content_or_link",
            "ocr_result": "Extracted text from OCR for page 1"
        },
        {
            "page_number": 2,
            "raw_content": "Base64_encoded_page_content_or_link",
            "ocr_result": "Extracted text from OCR for page 2"
        }
    ],
    "errors": [
        {
            "timestamp": "2022-11-15T12:25:00Z",
            "message": "Error message describing what went wrong"
        }
    ],
    "model_config": {
        "selected_pages": "1-4",
        "other_options": {
            "example_option": "example_value"
        }
    }
}
'''

## Metadata

- user_id: The unique identifier for the user.
- document: Contains information about the document.
  - document_id: Unique identifier for the document.
  - file_path: Path to the document in the shared volume.
  - document_type: Type of document (e.g., "scientific", "simple_text").
  - uploaded_at: Timestamp indicating when the document was uploaded.

### Processing Status

- process: Contains the current status and timestamps.
  - status: Current status of the document processing (e.g., "uploaded", "classified", "ocr_processing", "completed", "error").
  - last_updated: Timestamp of the last update to the processing status.

### Content Information

- pages: An array of page objects.
  - page_number: Page number in the document.
  - raw_content: Raw content of the page, which can be the Base64-encoded content or a link to the page image.
  - ocr_result: The OCR result for the page.

### Error Handling

- errors: An array of error objects to log any issues encountered during processing.
  - timestamp: Timestamp when the error occurred.
  - message: Description of the error.

### Configuration Parameters

- model_config: Configuration settings for the OCR models.
  - selected_pages: Pages selected for OCR processing (if applicable).
  - other_options: A dictionary to include any other configuration options that might be needed.

### Considerations

- Consistency: Ensure that all timestamps follow a consistent format (e.g., ISO 8601).
- Scalability: The array structures for pages and errors allow for easy addition of new pages and error logs.
- Flexibility: The model_config dictionary can be expanded with additional options as needed.
