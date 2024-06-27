import jsonschema
from jsonschema import validate

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
