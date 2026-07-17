import os
from us_visa.constants import SCHEMA_FILE_PATH, MODEL_FILE_NAME
from us_visa.utils.main_utils import read_yaml_file

def test_constants_loading():
    # Verify file name constants
    assert SCHEMA_FILE_PATH == os.path.join("config", "schema.yaml")
    assert MODEL_FILE_NAME == "model.pkl"

def test_schema_file_exists():
    # Verify the schema yaml can be read and contains targets
    assert os.path.exists(SCHEMA_FILE_PATH)
    schema = read_yaml_file(SCHEMA_FILE_PATH)
    assert "columns" in schema
    assert "target_column" in schema
    assert schema["target_column"] == "case_status"
