# 20250911_predict_eval_GUI_autorun_end_save.py
# 要件:
# - 起動後に自動で全件推論開始（GUI操作不要）
# - 保存は最後に1回だけ（途中保存なし）
# - GUIは進行状況の可視化＆巻き戻し閲覧用

from typing import Dict, Any, List
import os, json, re, time, csv, datetime, collections, threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

# ====== 設定 ======
metadata_name = "ml-1"
# metadata_file_path = f"../dataset/{metadata_name}/metadata.jsonl"
metadata_file_path = f"../dataset/{metadata_name}/metadata_test.jsonl"

dt_str_common = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file_path = f"../log/{dt_str_common}_predict_eval.csv"

model_id = "Joylimjy/LecEval"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== モデル初期化 ======
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
).to(device)
model.eval()

# ====== ユーティリティ ======
_score_regex = re.compile(r"(?<!\d)([1-5](?:\.\d+)?)")
RUBRICS = ["content_relevance", "expressive_clarity", "logical_structure", "audience_engagement"]

def extract_four_scores(text: str) -> List[float]:
    nums = _score_regex.findall(text)
    out = [float(x) for x in nums[:4]]
    while len(out) < 4:
        out.append(None)
    return out

def parse_sample_id(sample_id: str):
    parts = sample_id.split("_")
    theme = parts[0] if len(parts) > 0 else ""
    presenter_id = parts[1] if len(parts) > 1 else ""
    slide_num = parts[3] if len(parts) > 3 else ""
    return theme, presenter_id, slide_num

def get_predictions(slide_path: str, prompt_msgs: List[Dict[str, str]]) -> str:
    image = Image.open(slide_path).convert("RGB")
    with torch.inference_mode():
        answer = model.chat(
            image=image,
            msgs=prompt_msgs,
            tokenizer=tokenizer,
            processor=processor,
            max_new_tokens=64,
            sampling=False,
        )
    return answer

def load_dataset(metadata_path: str, theme_prefix: str):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"データファイルが見つかりません: {metadata_path}")
    data_by_presenter = collections.defaultdict(list)
    order = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sid = sample.get("id", "")
            if not sid.startswith(theme_prefix):
                continue
            parts = sid.split("_")
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            pid = int(parts[1])
            data_by_presenter[pid].append(sample)
            order.append((pid, sid))
    order.sort(key=lambda x: (x[0], x[1]))
    flat = []
    for pid, sid in order:
        for s in data_by_presenter[pid]:
            if s["id"] == sid:
                flat.append(s)
                break
    return flat

# ====== GUI アプリ ======
class LecEvalGUI(tk.Tk):
    def __init__(self, samples: List[Dict[str, Any]]):
        super().__init__()
        self.title("LecEval Auto Runner (View-Only)")
        self.geometry("1180x740")
        self.samples = samples
        self.index = 0
        self.results = []  # CSV 保存用
        self.delay_ms = tk.IntVar(value=120)   # 自動実行の表示ウェイト（見やすさ用）
        self.running = False                   # 単発実行中
        self.auto_running = False              # 連続実行中
        self.stop_flag = False
        self.saved_at_end = False              # 最終保存済みフラグ
        self._image_cache = None

        # 上部：コントロール群（手動でも触れるが、基本は自動スタート）
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self.idx_label = ttk.Label(top, text=f"0 / {len(self.samples)}")
        self.idx_label.pack(side=tk.LEFT, padx=(0,10))

        self.id_label = ttk.Label(top, text="-")
        self.id_label.pack(side=tk.LEFT, padx=(0,10))

        ttk.Label(top, text="time(ms)").pack(side=tk.LEFT, padx=(10,2))
        self.delay_entry = ttk.Entry(top, width=6, textvariable=self.delay_ms)
        self.delay_entry.pack(side=tk.LEFT)

        self.auto_btn = ttk.Button(top, text="▶ Execute inference", command=self.run_all_async)
        self.auto_btn.pack(side=tk.LEFT, padx=8)

        self.stop_btn = ttk.Button(top, text="■ Stop", command=self.request_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)

        self.prev_btn = ttk.Button(top, text="◀ Back", command=self.go_prev)
        self.prev_btn.pack(side=tk.RIGHT, padx=5)

        self.next_btn = ttk.Button(top, text="Next ▶", command=self.go_next)
        self.next_btn.pack(side=tk.RIGHT, padx=5)

        self.time_label = ttk.Label(top, text=f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.save_btn = ttk.Button(top, text="Save CSV", command=self.save_csv)
        self.save_btn.pack(side=tk.RIGHT, padx=5)

        # 中央：画像 + transcript
        center = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        center.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # 左：画像
        self.img_frame = ttk.LabelFrame(center, text="Slide Image")
        center.add(self.img_frame, weight=3)
        self.img_label = ttk.Label(self.img_frame)
        self.img_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 右：transcript と結果
        right_frame = ttk.Panedwindow(center, orient=tk.VERTICAL)
        center.add(right_frame, weight=2)

        self.tr_frame = ttk.LabelFrame(right_frame, text="Transcript")
        right_frame.add(self.tr_frame, weight=3)
        self.tr_text = tk.Text(self.tr_frame, wrap="word", height=10)
        self.tr_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.res_frame = ttk.LabelFrame(right_frame, text="Prediction / Scores")
        right_frame.add(self.res_frame, weight=1)

        self.answer_var = tk.StringVar(value="")
        self.answer_label = ttk.Label(self.res_frame, textvariable=self.answer_var, anchor="w", justify="left", wraplength=520)
        self.answer_label.pack(fill=tk.X, padx=8, pady=(8,4))

        self.score_vars = {k: tk.StringVar(value=f"{k}: -") for k in RUBRICS}
        for k in RUBRICS:
            ttk.Label(self.res_frame, textvariable=self.score_vars[k], anchor="w", justify="left").pack(fill=tk.X, padx=8)

        # 最初のサンプル表示
        self.show_current()

        # 起動後に自動で全件実行スタート（200ms遅延してUI準備完了を待つ）
        self.after(200, self.run_all_async)

        # ウィンドウクローズ時のフック
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ====== ナビゲーション ======
    def show_current(self):
        n = len(self.samples)
        if n == 0:
            messagebox.showerror("エラー", "サンプルが空です。")
            return
        self.index = max(0, min(self.index, n-1))
        sample = self.samples[self.index]
        sid = sample.get("id","")
        self.idx_label.config(text=f"{self.index+1} / {n}")
        self.id_label.config(text=sid)

        slide_rel = sample.get("slide","")
        slide_path = os.path.join(".."+slide_rel)
        if os.path.exists(slide_path):
            self.display_image(slide_path)
        else:
            self.img_label.config(text=f"(画像が見つかりません)\n{slide_path}")
            self._image_cache = None

        self.tr_text.delete("1.0", tk.END)
        self.tr_text.insert(tk.END, sample.get("transcript",""))

        self.answer_var.set("")
        for k in RUBRICS:
            self.score_vars[k].set(f"{k}: -")

    def display_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            w = max(500, self.img_label.winfo_width())
            h = max(360, self.img_label.winfo_height())
            img.thumbnail((w-16, h-16), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            self.img_label.config(image=imgtk, text="")
            self._image_cache = imgtk
        except Exception as e:
            self.img_label.config(text=f"画像読み込みエラー: {e}")
            self._image_cache = None

    def go_next(self):
        self.index += 1
        if self.index >= len(self.samples):
            self.index = len(self.samples)-1
        self.show_current()

    def go_prev(self):
        self.index -= 1
        if self.index < 0:
            self.index = 0
        self.show_current()

    # ====== 連続実行（全部） ======
    def run_all_async(self):
        if self.running or self.auto_running:
            return
        self.auto_running = True
        self.stop_flag = False
        self.auto_btn.config(state=tk.DISABLED, text="連続実行中…")
        self.stop_btn.config(state=tk.NORMAL)
        t = threading.Thread(target=self._run_all_worker, daemon=True)
        t.start()

    def request_stop(self):
        self.stop_flag = True

    def _run_all_worker(self):
        n = len(self.samples)
        t_start = time.time()
        self.results = []  # 最初から作り直す（最後に1回保存のため）

        for i in range(n):
            if self.stop_flag:
                break

            self.index = i
            self._after(self.show_current)

            sample = self.samples[i]
            sid = sample["id"]
            slide_path = os.path.join(".."+sample["slide"])
            prompt_msgs = sample.get("prompt", [])
            t0 = time.time()
            print(f"\n--- ID: {sid} の推論を開始 ---")

            try:
                if not os.path.exists(slide_path):
                    # 画像が無くてもスキップして続行
                    answer_text = "(image missing)"
                    scores = [None, None, None, None]
                else:
                    answer_text = get_predictions(slide_path, prompt_msgs)
                    scores = extract_four_scores(answer_text)
            except Exception as e:
                answer_text = f"(error: {e})"
                scores = [None, None, None, None]

            theme, presenter_id, slide_num = parse_sample_id(sid)
            record = {"theme": theme, "presenter_id": presenter_id, "slide_num": slide_num}
            for k, v in zip(RUBRICS, scores):
                record[k] = v
            self.results.append(record)

            def update_ui():
                # self.answer_var.set(f"Answer: {answer_text}")
                scores = extract_four_scores(answer_text)
                for k, v in zip(RUBRICS, scores):
                    self.score_vars[k].set(f"{k}: {v}")
            self._after(update_ui)

            # 見やすさのための少しの待ち（推論自体はもう終わっている）
            delay = max(0, int(self.delay_ms.get())) / 1000.0
            time.sleep(delay)
            print(f"{sid}処理完了: {time.time()-t0:.4f}秒")

        # 終了時保存（最後に1回だけ）
        def finish_all():
            self.auto_running = False
            self.stop_btn.config(state=tk.DISABLED)
            self.auto_btn.config(state=tk.NORMAL, text="▶ Automatically predict all samples")
            if (not self.stop_flag) and self.results:
                try:
                    self._save_csv_to_path(log_file_path)
                    self.saved_at_end = True
                    messagebox.showinfo("完了", f"全サンプルの推論が完了し、CSV を保存しました:\n{log_file_path}")
                except Exception as e:
                    messagebox.showerror("保存エラー", str(e))
            elif self.stop_flag:
                messagebox.showinfo("停止", "連続実行を停止しました（保存は行っていません）。")
        self._after(finish_all)
        print(f"全処理 {time.time()-t_start:.2f} 秒")

    # ====== CSV 保存 ======
    def _save_csv_to_path(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["theme","presenter_id","slide_num"]+RUBRICS)
            writer.writeheader()
            for rec in self.results:
                writer.writerow(rec)

    def save_csv(self):
        if not self.results:
            messagebox.showwarning("注意", "保存する結果がまだありません。")
            return
        try:
            self._save_csv_to_path(log_file_path)
            self.saved_at_end = True
            messagebox.showinfo("保存", f"CSV を保存しました:\n{log_file_path}")
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))

    # ====== UIスレッドに投げる ======
    def _after(self, fn):
        self.after(0, fn)

    # ====== 終了時の確認 ======
    def on_close(self):
        if self.auto_running:
            if not messagebox.askokcancel("確認", "連続実行中です。停止して閉じますか？（保存は行いません）"):
                return
        elif (not self.saved_at_end) and self.results:
            if messagebox.askyesno("保存確認", "未保存の結果があります。保存しますか？"):
                try:
                    self._save_csv_to_path(log_file_path)
                except Exception as e:
                    messagebox.showerror("保存エラー", str(e))
        self.destroy()

# ====== エントリーポイント ======
def main():
    try:
        samples = load_dataset(metadata_file_path, metadata_name)
    except Exception as e:
        print("データ読み込みエラー:", e)
        return
    app = LecEvalGUI(samples)
    app.mainloop()

if __name__ == "__main__":
    main()
