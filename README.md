# Audiolm Evaluator
Audio Language Model Evaluator

## Install dependencies
```bash
git clone --recursive https://github.com/nota-github/audiolm-evaluator
pip install -r audiolm-trainer/requirements.txt
pip install -r requirements.txt
```

## Evaluate
`salmonn_eval_config.yaml` 에서 데이터셋 경로, 모델 경로 등을 적절히 수정한 후 아래 스크립트를 실행합니다.
```python
python evaluate_salmonn.py --task {asr, aac}
```

위 파일을 실행하면 기본적으로 `submission.csv`가 생성됩니다.

자체적인 평가를 진행하고자 한다면 아래 형식으로 자체 평가용 json 파일을 만들고 평가하고자 하는 task를 인자로 주면 됩니다.
```
{
  "annotation": [
    {
      "testset_id": "any_id_for_test",
      "path": "/path/to/audio_file",
      "task": {asr or audiocaption_v2},
      "test": "Ground truth for sample"
    },
    ...
```

## Validate submission file
```python
python submission_validator.py /path/to/submission.csv
```

위 스크립트는 파일의 형식만 확인하며, 샘플의 개수는 validation하지 않습니다.
