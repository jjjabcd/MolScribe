import argparse
import json
import torch
from molscribe import MolScribe
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def predict_folder(model, folder_path, return_atoms_bonds=False, return_confidence=False):
    # 폴더 내의 모든 PNG 파일 목록 생성
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    predictions = []

    # CUDA가 활성화되었는지 확인
    # cuda_enabled = torch.cuda.is_available()
    # if cuda_enabled:
    #     print("CUDA가 활성화되었습니다.")
    # else:
    #     print("CUDA를 사용할 수 없습니다. CPU를 사용합니다.")

    # 예측 수행
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        output = model.predict_image_file(
            image_path, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)
        output['image_file'] = image_file
        predictions.append(output)

        # 100개의 예측마다 로그 출력
        if idx % 100 == 0:
            print(f"{idx}개 이미지 예측 완료")

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--image_folder', type=str, default=None, required=True)
    parser.add_argument('--return_confidence', action='store_true')
    parser.add_argument('--return_atoms_bonds', action='store_true')
    parser.add_argument('--output_csv', type=str, default='predictions.csv')
    args = parser.parse_args()

    device = torch.device('cuda')
    model = MolScribe(args.model_path, device)
    predictions = predict_folder(
        model, args.image_folder, 
        return_atoms_bonds=args.return_atoms_bonds, 
        return_confidence=args.return_confidence)
    
    # 예측 결과를 DataFrame으로 변환
    df_predictions = pd.DataFrame(predictions)
    
    # DataFrame을 CSV 파일로 저장
    df_predictions.to_csv(args.output_csv, index=False)
