# 20250903_predict_eval.py — MiniCPM-V (LecEval) 対応：最新版シンプル版
from typing import Dict, Any, List
import os, json, re
from pathlib import Path

import torch
from PIL import Image, ImageTk
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import time
import tkinter as tk
from tkinter import ttk

# ====================================================================
# 元のスクリプトからGUI表示のためのコードを追加
# ====================================================================

# === モデル/トークナイザ/プロセッサ初期化（モンキーパッチ不要） ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Joylimjy/LecEval"  # HF のモデルID

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
).to(device)
model.eval()

# 評価プロンプト
evaluation_prompts: Dict[str, str] = {
    "content_relevance": "このスライドとトランスクリプトはどのくらい関連していますか？1から5のスケールで評価してください。",
    "expressive_clarity": "トランスクリプトは明確で理解しやすいですか？1から5のスケールで評価してください。",
    "logical_structure": "スライドの構成は論理的ですか？1から5のスケールで評価してください。",
    "audience_engagement": "このプレゼンテーションは聴衆の興味を引きますか？1から5のスケールで評価してください。"
}

_num_regex = re.compile(r"(?<!\d)([1-5](?:\.\d+)?)")

def parse_first_score(text: str) -> float:
    """
    出力テキストから 1〜5 の最初の数値を抽出（小数も可）。
    見つからなければ None を返す。
    """
    m = _num_regex.search(text)
    return float(m.group(1)) if m else None

def build_user_content(prompt: str, transcript: str) -> str:
    # 画像の挿入位置は (./) が仕様。先頭に1つ入れる。
    return f"(./)\n{prompt}\nTranscript: {transcript}"

def get_predictions(slide_path: str, transcript: str) -> Dict[str, Any]:
    # 画像をロード
    image = Image.open(slide_path).convert("RGB")

    predictions: Dict[str, Any] = {}
    for rubric, jp_prompt in evaluation_prompts.items():
        # ユーザー入力（画像タグ (./) を1つ）
        user_content = build_user_content(jp_prompt, transcript)

        # === 推奨の chat() で実行 ===
        with torch.inference_mode():
            answer = model.chat(
                image=image,
                msgs=[{"role": "user", "content": user_content}],
                tokenizer=tokenizer,
                processor=processor,
                max_new_tokens=256,
                sampling=False,
            )

        score = parse_first_score(answer)
        predictions[rubric] = {
            "raw_text": answer,
            "score": score,
        }
    return predictions

# ====================================================================
# GUIアプリケーションクラス
# ====================================================================

class MiniCPMEvalApp(tk.Tk):
    def __init__(self, samples: List[dict]):
        super().__init__()
        self.title("MiniCPM-V (LecEval) Viewer")
        self.geometry("1200x800")

        self.samples = samples
        self.current_index = 0

        # メインフレーム
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左右に分割
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左側：画像とトランスクリプト
        self.image_label = ttk.Label(left_frame, text="スライド画像", relief="solid", borderwidth=1)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        ttk.Label(left_frame, text="トランスクリプト:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(10, 2))
        self.transcript_text = tk.Text(left_frame, wrap="word", height=10, font=('Helvetica', 10))
        self.transcript_text.pack(fill=tk.X, padx=5)

        # 右側：推論結果
        ttk.Label(right_frame, text="評価結果:", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.results_text = tk.Text(right_frame, wrap="word", font=('Helvetica', 10), state="disabled")
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # ボタン
        self.next_button = ttk.Button(self, text="Next Slide", command=self.load_next_slide)
        self.next_button.pack(pady=10)

        # 最初のスライドをロード
        self.load_slide()

    def load_slide(self):
        """現在のインデックスのスライドをロードして表示する"""
        if self.current_index >= len(self.samples):
            self.image_label.configure(text="全スライドの評価が完了しました。")
            self.transcript_text.delete(1.0, tk.END)
            self.results_text.configure(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "すべてのスライドを完了しました。")
            self.results_text.configure(state="disabled")
            self.next_button.config(state="disabled")
            return

        sample = self.samples[self.current_index]
        slide_path = os.path.join(".." + sample["slide"])
        transcript = sample["transcript"]

        # 画像を表示
        try:
            pil_image = Image.open(slide_path)
            pil_image = pil_image.resize((600, int(600 * pil_image.height / pil_image.width)), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # ガベージコレクション回避
        except Exception as e:
            self.image_label.config(text=f"画像ロードエラー: {e}")
            self.image_label.image = None

        # トランスクリプトを表示
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.insert(tk.END, transcript)
        
        # 推論を実行して結果を表示
        self.next_button.config(state="disabled") # 推論中はボタンを無効化
        self.after(100, self.run_prediction, slide_path, transcript)

    def run_prediction(self, slide_path, transcript):
        """推論を実行し、結果をGUIに表示する"""
        try:
            predictions = get_predictions(slide_path, transcript)
            self.update_results_display(predictions)
        except Exception as e:
            self.results_text.configure(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"推論エラー: {e}")
            self.results_text.configure(state="disabled")
        finally:
            self.next_button.config(state="normal") # 推論完了後にボタンを有効化

    def update_results_display(self, predictions):
        """推論結果をテキストエリアに表示する"""
        self.results_text.configure(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "評価結果:\n\n")
        for rubric, data in predictions.items():
            score = data['score'] if data['score'] is not None else "評価不可"
            self.results_text.insert(tk.END, f"- {rubric}:\n")
            self.results_text.insert(tk.END, f"  スコア: {score}\n")
            self.results_text.insert(tk.END, f"  生のテキスト: {data['raw_text']}\n\n")
        self.results_text.configure(state="disabled")

    def load_next_slide(self):
        """次のスライドに切り替える"""
        self.current_index += 1
        self.load_slide()

def main_app():
    metadata_file_path = "../dataset/ml-1/metadata.jsonl"
    if not os.path.exists(metadata_file_path):
        print(f"エラー: データファイル '{metadata_file_path}' が見つかりません。")
        return

    ml_01_samples: List[dict] = []
    with open(metadata_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("id", "").startswith("ml-1_01"):
                ml_01_samples.append(sample)

    if not ml_01_samples:
        print("ml-1_01 に該当するサンプルは見つかりませんでした。")
        return

    print(f"ml-1_01 のサンプルを {len(ml_01_samples)} 個見つけました。")
    app = MiniCPMEvalApp(ml_01_samples)
    app.mainloop()

if __name__ == "__main__":
    main_app()