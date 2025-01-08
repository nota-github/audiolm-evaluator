# np-app-audiolm-evaluator
Audio Language Model Evaluator

## Install dependencies
```
bash install_dependencies.sh
```

## Data setting
* ASR task
  * corpus = Librispeech/test-other 
  * path = `/mnt/nas/geonmin.kim/AICA326/audio_eval_datasets/LibriSpeech/test-other`
* AAC task
  * corpus = AudioCaps/test
  * path = `@3090m > /ssd5/geonminkim_data/audiocaps-download/audiocaps/test`
* Speech QA task
  * corpus = AIR-Bench/Chat
  * path = `/mnt/nas/geonmin.kim/AICA326/audio_eval_datasets/AIR-Bench-Dataset/Chat`

## Evaluating ASR task
### run
```
bash evaluate_qwen2audio_asr.sh
```

### example results (Qwen2-Audio-7B)
* inference: TBA 
* scoring
```
source: librispeech_test_other  cnt: 2939 wer: 0.0498 
```

## Evaluating AAC task
### run
```
bash evaluate_qwen2audio_aac.sh
```

### example results (Qwen2-Audio-7B)
* inference
```json
[
    {
        "gt": "Constant rattling noise and sharp vibrations",
        "response": "The sound of coins dropping into a machine.",
        "source": "AudioCaps",
        "audio_path": "/ssd1/geonmin.kim/audiocaps-download//audiocaps/test/103549.webm"
    },
    {
        "gt": "A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle",
        "response": "A loud boom followed by a hissing sound.",
        "source": "AudioCaps",
        "audio_path": "/ssd1/geonmin.kim/audiocaps-download//audiocaps/test/103548.m4a"
    },
    {
        "gt": "Humming and vibrating with a man and children speaking and laughing",
        "response": "A child is speaking and there are sounds of a vehicle, paper rustling, and a door tap.",
        "source": "AudioCaps",
        "audio_path": "/ssd1/geonmin.kim/audiocaps-download//audiocaps/test/103541.m4a"
    },
```

* scoring
```
source: AudioCaps  SPIDEr: 17.25
```

## (Hidden) Evaluating SpeechQA task
* run
```
bash evaluate_chat_qwen2audio.sh
```

* example result
```json
[
    {
        "idx": 0,
        "response": "The first speaker implies that the national debt is very large, exceeding six trillion dollars.",
        "audio_path": "/data/gmkim/AIR-Bench-Dataset/Chat//speech_dialogue_QA_fisher/568.38_596.46.wav"
    },
    {
        "idx": 1,
        "response": "To see if they are compatible.",
        "audio_path": "/data/gmkim/AIR-Bench-Dataset/Chat//speech_dialogue_QA_fisher/57.81_86.38.wav"
    },
    {
        "idx": 2,
        "response": "A continuation from week to week.",
        "audio_path": "/data/gmkim/AIR-Bench-Dataset/Chat//speech_dialogue_QA_fisher/235.73_259.24.wav"
    },
```

* construct L-Eval prompt
```
python construct_L-Eval_prompt_for_audiolm.py
```

* L-Eval score example (model = Qwen2-audio-7B)

|         | AudioLM |     |     | Reference |     |
|---------|---------|-----|-----|-----------|-----|
| uniq_id | AVG     | 1st | 2nd | 1st       | 2nd |
| 0       | 7.5     | 9   | 6   | 8         | 8   |
| 1       | 10      | 10  | 10  | 10        | 10  |
| 2       | 9       | 9   | 9   | 8         | 8   |
| …       | …       | …   | …   | …         | …   |
| 2197    | 5.5     | 9   | 2   | 8         | 8   |
| 2198    | 8       | 8   | 8   | 8         | 8   |
| 2199    | 5.5     | 2   | 9   | 9         | 2   |


* Reported G-Eval score
```
-------------------- Official Qwen2-audio-7B Results --------------------
Speech 800 7.18
Sound 400 6.99
Music 400 6.79
Mixed Audio 400 6.77
Average: 2000 6.93
```
