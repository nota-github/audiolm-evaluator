# Audiolm Evaluator
Audio Language Model Evaluator

## Install dependencies
```bash
pip install -r requirements.txt
```

## Generate submission file
```python
python evaluate_salmonn.py
```

## Validate submission file
```python
python submission_validator.py /path/to/submission.csv
```

위 스크립트는 파일의 형식만 확인하며, 샘플의 개수는 validation하지 않습니다.