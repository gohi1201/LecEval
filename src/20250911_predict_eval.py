# 20250910_predict_eval_combined_clean.py
from typing import Dict, Any, List
import os, json, re, time, csv, datetime, collections

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Joylimjy/LecEval"

# === 入力データ ===
metadata_name = "ml-1"
# metadata_file_path = f"../dataset/{metadata_name}/metadata.jsonl"
metadata_file_path = f"../dataset/{metadata_name}/metadata_test.jsonl"

# === 出力（ログ）設定 ===
dt_str_common = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file_path = f"../log/{dt_str_common}_predict_eval.csv"

# === モデル/トークナイザ/プロセッサ初期化 ===
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
).to(device)
model.eval()

# 1〜5 の数字（少数可）を先頭から最大4つ抽出
_score_regex = re.compile(r"(?<!\d)([1-5](?:\.\d+)?)")

def extract_four_scores(text: str) -> List[float]:
    nums = _score_regex.findall(text)
    return [float(x) for x in nums[:4]]

def remove_duplicates(lst): 
    return list(set(lst))

def get_predictions(slide_path: str, prompt_msgs: List[Dict[str, str]]) -> str:
    """画像+与えられたprompt（そのまま）で chat 推論し、生テキストを返す"""
    image = Image.open(slide_path).convert("RGB")
    with torch.inference_mode():
        answer = model.chat(
            image=image,
            msgs=prompt_msgs,            # ★ 指示通り：与えられた prompt をそのまま送る
            tokenizer=tokenizer,
            processor=processor,
            max_new_tokens=64,
            sampling=False,
        )
    return answer

def parse_sample_id(sample_id: str):
    # 想定形式: "ml-1_10_slide_058"
    parts = sample_id.split("_")
    theme = parts[0] if len(parts) > 0 else ""
    presenter_id = parts[1] if len(parts) > 1 else ""
    slide_num = parts[3] if len(parts) > 3 else ""
    return theme, presenter_id, slide_num

def main():
    start_all = time.time()

    if not os.path.exists(metadata_file_path):
        print(f"エラー: データファイル '{metadata_file_path}' が見つかりません。")
        return

    # presenter_id ごとにサンプルを束ねる
    data_by_presenter = collections.defaultdict(list)
    presenter_ids = []

    with open(metadata_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            sid = sample.get("id", "")
            if not sid.startswith(metadata_name):
                continue
            parts = sid.split("_")
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            pid = int(parts[1])
            presenter_ids.append(pid)
            data_by_presenter[pid].append(sample)

    if not data_by_presenter:
        print(f"{metadata_name} に該当するサンプルは見つかりませんでした。")
        return

    
    print(f"各発表のスライド枚数：{collections.Counter(presenter_ids)}個見つけました。推論を開始します。")

    presenter_ids = sorted(set(presenter_ids))
    results: List[Dict[str, Any]] = []
    rubric_keys = ["content_relevance", "expressive_clarity", "logical_structure", "audience_engagement"]

    for pid in presenter_ids:
        for sample in data_by_presenter[pid]:
            sid = sample["id"]
            slide_path = os.path.join(".."+sample["slide"])
            prompt_msgs = sample.get("prompt", [])
            t0 = time.time()
            print(f"\n--- ID: {sid} の推論を開始 ---")

            if not os.path.exists(slide_path):
                print(f"警告: 画像が見つかりません: {slide_path}")
                continue

            try:
                answer_text = get_predictions(slide_path, prompt_msgs)
            except Exception as e:
                print(f"エラー: {sid} の推論に失敗しました: {e}")
                continue

            scores = extract_four_scores(answer_text)
            # 4つ揃わない場合は欠損を None で埋める
            while len(scores) < 4:
                scores.append(None)

            theme, presenter_id, slide_num = parse_sample_id(sid)
            record = {
                "theme": theme,
                "presenter_id": presenter_id,
                "slide_num": slide_num,
            }
            for k, v in zip(rubric_keys, scores):
                record[k] = v
            results.append(record)

            print(f"ID {sid}")
            for k, v in zip(rubric_keys, scores):
                print(f"  - {k}: {v}")
            print(f"{sid}処理完了: {time.time()-t0:.4f}秒")

    # === CSV 保存 ===
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    fieldnames = ["theme", "presenter_id", "slide_num"] + rubric_keys
    with open(log_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results:
            writer.writerow(rec)

    print(f"\n全サンプルの処理完了: {time.time()-start_all:.4f}秒")
    print(f"予測結果を {log_file_path} に保存しました。")

if __name__ == "__main__":
    main()