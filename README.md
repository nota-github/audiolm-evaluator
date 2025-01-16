# Audiolm Evaluator
Audio Language Model Evaluator

## Install dependencies
```bash
git clone --recursive https://github.com/nota-github/audiolm-evaluator
pip install -r audiolm-trainer/requirements.txt
pip install -r requirements.txt
aac-metrics-download
```

## Evaluate
`salmonn_eval_config.yaml` 에서 데이터셋 경로, 모델 경로 등을 적절히 수정한 후 아래 스크립트를 실행합니다.
```python
python evaluate_salmonn.py --mode {submission_asr, submission_aac, valid_asr, valid_aac}
```
- submission mode는 제출용인 csv를 만들기 위한 모드입니다.
- valid mode는 자체적인 평가를 진행하고자 할 때 사용하며 text 라벨이 있는 json 파일이 필요합니다.
- 두 모드는 서로 다른 디렉토리에 csv 파일이 저장됩니다.

```
{
  "annotation": [
    {
      "testset_id": "any_id_for_test",
      "path": "/path/to/audio_file",
      "task": {asr or audiocaption_v2},
      "text": "Ground truth for sample" # valid 시 필요
    },
    ...
```

## Validate submission file
```python
python submission_validator.py /path/to/submission.csv
```

위 스크립트는 파일의 형식만 확인하며, 샘플의 개수는 validation하지 않습니다.
