import pandas as pd
import os


def test_integration():
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ['OUTPUT_DIR'] = 's3://biopay/output/'
    os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566"

    os.system("python ../../src/predict/predict.py --output_path OUTPUT_FILE_PATTERN")
    options = {
            "client_kwargs": {
                "endpoint_url": os.getenv('S3_ENDPOINT_URL')
            },
            "key": os.getenv('AWS_ACCESS_KEY_ID'),
            "secret": os.getenv('AWS_SECRET_ACCESS_KEY')
        }

    test_output = pd.read_csv(os.getenv('OUTPUT_DIR'), storage_options=options)

    assert isinstance(test_output, pd.DataFrame)
    assert "prediction" in test_output.columns