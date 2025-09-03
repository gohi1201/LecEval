import json
import sys
import os
import traceback
from pathlib import Path

# 予測に必要なモジュールをインポートします
from dataset import LecEvalDataset
from leceval import LecEvalMetric # READMEに記載された正しいインポート文

def run_predictions(data_path, images_path):
    """
    データセットの各スライドを処理し、予測を生成して保存します。
    """
    try:
        # データセットをロード
        dataset = LecEvalDataset(data_path=data_path, images_path=images_path)
        
        # モデルを初期化
        metric = LecEvalMetric()
        
        # 予測結果を保存するリスト
        all_predictions = []
        
        print("\n--- 予測の生成を開始します ---")
        
        # データセット内の各サンプルをループ処理
        for sample in dataset.get_all_samples():
            slide_path = sample.get('slide')
            explanation_text = sample.get('transcript')
            
            # 画像パスを補完（images_path引数を使用）
            full_slide_path = os.path.join(images_path, slide_path.lstrip('/'))
            
            print(f"処理中: {sample['id']}")
            
            try:
                # 1つのスライドとテキストを評価（予測）
                scores = metric.evaluate(full_slide_path, explanation_text)
                
                # 結果をリストに追加
                result = {
                    "id": sample['id'],
                    "predictions": scores
                }
                all_predictions.append(result)
                
                print(f"  - 予測スコア: {scores}")
                
            except Exception as e:
                print(f"  - エラー: {sample['id']}の予測中に問題が発生しました: {e}")
                
    except Exception as e:
        print(f"\nエラー：処理中に問題が発生しました: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 予測結果をJSONファイルに保存
    output_filename = "predictions.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
        
    print(f"\n--- 予測の生成が完了しました ---")
    print(f"すべての予測は '{output_filename}' に保存されました。")


if __name__ == "__main__":
    # このスクリプトがあるディレクトリからの相対パスで指定します。
    # LecEval/dataset と LecEval/images が、このスクリプトの親ディレクトリに存在する必要があります。
    run_predictions(
        data_path="../dataset/ml-1/metadata.jsonl",
        images_path="../images"
    )