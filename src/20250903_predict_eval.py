import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from pathlib import Path

# Step 1: Initialize the model and tokenizer
# モデルとトークナイザを読み込む（GPUが利用可能ならto("cuda")を追加）
try:
    model_name = "Joylimjy/LecEval"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print("モデルが正常に読み込まれました。")
except Exception as e:
    print(f"モデルの読み込みに失敗しました: {e}")
    exit()

# 評価項目ごとのプロンプトを定義
evaluation_prompts = {
    "content_relevance": "このスライドとトランスクリプトはどのくらい関連していますか？1から5のスケールで評価してください。",
    "expressive_clarity": "トランスクリプトは明確で理解しやすいですか？1から5のスケールで評価してください。",
    "logical_structure": "スライドの構成は論理的ですか？1から5のスケールで評価してください。",
    "audience_engagement": "このプレゼンテーションは聴衆の興味を引きますか？1から5のスケールで評価してください。"
}

# Step 2: Define the prediction function
def get_predictions(image_path, transcript):
    """
    指定された画像とトランスクリプトに対する予測スコアを生成します。
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"警告: 画像 {image_path} の読み込みに失敗しました。このサンプルはスキップします。エラー: {e}")
        return None
    
    predictions = {}
    for rubric, prompt in evaluation_prompts.items():
        messages = [{'role': 'user', 'content': prompt, 'images': [image]}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        ).to(model.device)
        
        with torch.no_grad():
            output_tokens = model.generate(inputs, max_new_tokens=20)
            generated_text = tokenizer.decode(output_tokens[0][inputs.shape[1]:], skip_special_tokens=True)
            
        try:
            score = float(generated_text.strip())
            predictions[rubric] = score
        except ValueError:
            print(f"警告: '{rubric}' のスコアをパースできませんでした。出力: '{generated_text}'")
            predictions[rubric] = 0.0
            
    return predictions

# Step 3: Extract and process the data
def main():
    """メイン関数：データの抽出と推論を実行"""
    
    # ファイルパスを適切に設定
    metadata_file_path = "../dataset/ml-1/metadata.jsonl"
    images_base_path = "../images"
    
    ml_01_samples = []
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    if sample.get('id', '').startswith('ml-1_01'):
                        ml_01_samples.append(sample)
    except FileNotFoundError:
        print(f"エラー: データファイル '{metadata_file_path}' が見つかりません。")
        return
    
    if not ml_01_samples:
        print("ml-1_01 に該当するサンプルは見つかりませんでした。")
        return
    
    print(f"ml-1_01 のサンプルを {len(ml_01_samples)} 個見つけました。推論を開始します。")

    # 推論結果を格納するリスト
    all_predictions = []

    # スライドごとに推論を実行
    for sample in ml_01_samples:
        sample_id = sample['id']
        slide_path = os.path.join(images_base_path, sample['slide'].lstrip('/'))
        transcript = sample['transcript']
        
        print(f"\n--- ID: {sample_id} の推論を開始 ---")
        
        predicted_scores = get_predictions(slide_path, transcript)
        
        if predicted_scores:
            all_predictions.append({
                "id": sample_id,
                "predictions": predicted_scores
            })
            
            print("予測された評価スコア:")
            for rubric, score in predicted_scores.items():
                print(f"- {rubric}: {score}")
    
    # 予測結果をファイルに保存 (オプション)
    # output_file_path = "predictions_ml-01.jsonl"
    # with open(output_file_path, 'w') as f:
    #     for item in all_predictions:
    #         f.write(json.dumps(item) + '\n')
    # print(f"\n予測結果を {output_file_path} に保存しました。")

if __name__ == "__main__":
    main()