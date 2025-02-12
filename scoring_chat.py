import argparse
import json
import ollama
import numpy as np

from tqdm import tqdm

category_map = {
    "speech_dialogue_QA": "speech",
    "speech_QA": "speech",
    "sound_QA": "sound",
    "sound_generation_QA": "sound",
    "music_QA": "music",
    "music_generation_analysis_QA": "music",
    "speech_and_sound_QA": "mixed_audio",
    "speech_and_music_QA": "mixed_audio"
}

def main(args):
    # LLM inference API
    # llama.cpp model info (hardcoded for now)
    ollama_model_info = {
        "endpoint": args.endpoint_uri,  # "http://0.0.0.0:11434",  # allow request from external
        "model": args.model_name,  # "llama3.3:70b",
    }
    ollama_client = ollama.Client(host=ollama_model_info["endpoint"])
    # load with warmup model
    ollama_client.generate(
        model=ollama_model_info["model"],
        prompt="dummy test",
        options={
            "num_predict": 0,
            "temperature": 0.0,
        },
    )

    ## Setup LLM judge

    ## Setup data
    ## load inference result with chat meta data
    # required fields
    # "meta_info", "question", "answer_gt", "respnose", "path", "task_name", "datset_name", "uniq_id"

    result_json = args.result_json # results/salmonn_result_chat.json

    with open(result_json, "r") as fr:
        results = json.load(fr)

    system_prompt = (
        "You are a helpful and precise assistant for checking the quality of the answer.\n"
        "[Detailed Audio Description]\nXAudioX\n[Question]\nXQuestionX\n"
        "[The Start of Assistant 1s Answer]\nXAssistant1X\n[The End of Assistant 1s Answer]\n"
        "[The Start of Assistant 2s Answer]\nXAssistant2X\n[The End of Assistant 2s Answer]\n[System]\n"
        "We would like to request your feedback on the performance of two AI assistants in response to the user question "
        "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
        "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
        "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
        "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
        "The two scores are separated by a space."
        "Do not generate any text except two scores."
    )

    
    scores = np.zeros((len(results), 4)) # forward/reverse X GT/AudioLM
    
    for idx, result in tqdm(enumerate(results), total=len(results)):

        question = result["Q"]
        answer_gt = result["ref"]
        response = result["hyp"]        
        meta_info = result["meta_info"]

        # construct prompt         
        # for forward (assistant1 = GT, assistant2 = AudioLM)
        LEval_prompt = system_prompt.replace("XAudioX", meta_info)
        LEval_prompt = LEval_prompt.replace("XQuestionX", question)
        
        LEval_prompt_forward = LEval_prompt.replace("XAssistant1X", answer_gt)
        LEval_prompt_forward = LEval_prompt_forward.replace("XAssistant2X", response)

        # for reverse (assistant1 = AudioLM, assistant2 = GT)
        LEval_prompt_reverse = LEval_prompt.replace("XAssistant1X", response)
        LEval_prompt_reverse = LEval_prompt_reverse.replace("XAssistant2X", answer_gt)

        # Call LLM judge for forward/reverse
        try:
            api_output_forward = ollama_client.generate(
                model=ollama_model_info["model"],
                prompt=LEval_prompt_forward,
                options={"num_predict": 64},  # allow large enough maximum length
            )
            response_forward = api_output_forward["response"]

            api_output_reverse = ollama_client.generate(
                model=ollama_model_info["model"],
                prompt=LEval_prompt_reverse,
                options={"num_predict": 64},  # allow large enough maximum length
            )
            response_reverse = api_output_reverse["response"]
        except Exception as e:
            print("API response fail")
            print(f"Error datails: {e}")
            continue  # just skip writing results

        # Parse judge's response for forward/reverse
        scores_forward = response_forward.split(' ')
        assert len(scores_forward) == 2
        assert scores_forward[0].isnumeric()
        assert scores_forward[1].isnumeric()

        scores_reverse = response_reverse.split(' ')
        assert len(scores_reverse) == 2
        assert scores_reverse[0].isnumeric()
        assert scores_reverse[1].isnumeric()
        
        scores[idx, 0] = scores_forward[0] # gt, forward
        scores[idx, 1] = scores_forward[1] # audiolm, forward
        scores[idx, 2] = scores_reverse[0] # audiolm, reverse
        scores[idx, 3] = scores_reverse[1] # gt, reverse


    ### Aggregate score
    scores_per_category = {
        "speech": 0.0,
        "sound": 0.0,
        "music": 0.0,
        "mixed_audio": 0.0
    }
    num_samples_per_category = {
        "speech": 0,
        "sound": 0,
        "music": 0,
        "mixed_audio": 0
    }

    assert scores.shape[0] == len(results)

    for idx, result in enumerate(results):
        sample_path = result["path"]

        category_name = sample_path.split('/')[1]

        category = None
        for cat_key in category_map.keys():
            if category_name.startswith(cat_key):
                category = category_map[cat_key]
                break

        assert category is not None

        # Get average (forward/reverse) score of Audio LM
        score = (scores[idx, 1] + scores[idx, 2])/2.0

        # Save score
        scores_per_category[category] += score
        num_samples_per_category[category] += 1

    # Normalize score
    for category in scores_per_category.keys():
        scores_per_category[category] /= num_samples_per_category[category]

    print(f"scores_per_category = {scores_per_category}")
    print(f"num_samples_per_category = {num_samples_per_category}")

    # Final averaged score
    avg_score = 0.0
    for category in scores_per_category.keys():
        avg_score += scores_per_category[category]
    avg_score /= len(scores_per_category.keys())
    print(f"average score = {avg_score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_json",
        type=str,
        default="results/salmonn_result_chat.json",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.3:70b",
        help="model name for ollama API",
    )

    parser.add_argument(
        "--endpoint_uri",
        type=str,
        default="http://0.0.0.0:11434",
        help="endpoint URI for ollama API",
    )

    args = parser.parse_args()
    main(args)
