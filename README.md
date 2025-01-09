# Audiolm Evaluator
Audio Language Model Evaluator

## Install dependencies
```bash
git clone --recursive https://github.com/nota-github/audiolm-evaluator
pip install -r audiolm-trainer/requirements.txt
pip install -r requirements.txt
```

## Generate submission file
`salmonn_eval_config.yaml` 에서 데이터셋 경로, 모델 경로 등을 적절히 수정한 후 아래 스크립트를 실행합니다.
```python
python evaluate_salmonn.py
```

## Validate submission file
```python
python submission_validator.py /path/to/submission.csv
```

위 스크립트는 파일의 형식만 확인하며, 샘플의 개수는 validation하지 않습니다.
