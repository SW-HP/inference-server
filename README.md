# Multi-HMR SMPLX Measurement Application

이 프로젝트는 네이버랩의 Multi-HMR을 사용하여 SMPLX 모델의 측정값을 활용하는 애플리케이션입니다. Multi-HMR은 여러 사람의 전체 신체 메쉬를 단일 이미지에서 추론하는 강력한 도구입니다. 이 프로젝트는 SMPL-Anthropometry를 사용하여 SMPLX 모델의 측정값을 계산하고 시각화합니다.

## 주요 기능

- **Multi-HMR**: 단일 RGB 이미지를 입력으로 받아 여러 사람의 3D 메쉬를 카메라 공간에서 효율적으로 재구성합니다.
- **SMPL-Anthropometry**: SMPL 및 SMPLX 모델의 신체 측정값을 계산합니다.

## 설치

1. 이 저장소를 클론합니다.
2. 가상 환경을 설정하고 필요한 패키지를 설치합니다.

```bash
python3.9 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

3. SMPLX 모델을 다운로드하고 `./models/smplx/` 디렉토리에 `SMPLX_NEUTRAL.npz` 파일을 배치합니다.

## 사용법

1. Multi-HMR을 사용하여 이미지를 처리합니다.

```bash
python3.9 demo.py --img_folder example_data --out_folder demo_out --model_name multiHMR_896_L
```

2. SMPL-Anthropometry를 사용하여 측정값을 계산합니다.

```bash
python measure.py --measure_neutral_smplx_with_mean_shape
```

## 라이센스

이 프로젝트는 CC BY-NC-SA 4.0 라이센스에 따라 배포됩니다. 자세한 내용은 [LICENSE](./LICENSE)를 참조하세요.

## 인용

이 프로젝트가 연구에 유용하다면 다음 논문을 인용해 주세요:

```bibtex
@inproceedings{multi-hmr2024,
    title={Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot},
    author={Baradel*, Fabien and 
            Armando, Matthieu and 
            Galaaoui, Salma and 
            Br{\'e}gier, Romain and 
            Weinzaepfel, Philippe and 
            Rogez, Gr{\'e}gory and
            Lucas*, Thomas
            },
    booktitle={ECCV},
    year={2024}
}
```

@misc{SMPL-Anthropometry,
  author = {Bojani\'{c}, D.},
  title = {SMPL-Anthropometry},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DavidBoja/SMPL-Anthropometry}},
}