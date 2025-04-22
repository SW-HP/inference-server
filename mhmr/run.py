# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os 
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ['EGL_DEVICE_ID'] = '0'

import numpy as np
import time
import torch
import logging
import math

from PIL import Image, ImageOps
from .utils import MeasureBody
from .utils.measurement_definitions import STANDARD_LABELS, MeasurementType

from .utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, demo_color as color, print_distance_on_image, render_side_views, create_scene, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
from .model import Model

# 로깅 설정 (기본 포맷 사용)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

import math

def cm_to_inch(cm):
    return cm / 2.54

def body_fat_percentage(sex: str, neck: float, abdomen: float = None, waist: float = None, hip: float = None, height: float = None) -> float:
    if sex == "male":
        if abdomen is None or neck is None or height is None:
            raise ValueError("남성의 경우 복부, 목, 키 치수가 필요합니다.")
        return round(86.010 * math.log10(abdomen - neck) - 70.041 * math.log10(height) + 36.76, 2)

    elif sex == "female":
        if waist is None or hip is None or neck is None or height is None:
            raise ValueError("여성의 경우 허리, 엉덩이, 목, 키 치수가 필요합니다.")
        return round(163.205 * math.log10(waist + hip - neck) - 97.684 * math.log10(height) - 78.387, 2)

    else:
        raise ValueError("성별은 'male' 또는 'female'로 입력하세요.")

def estimate_body_fat_from_cm_data(data: dict, sex: str) -> float:
    height_in = cm_to_inch(data['height'])
    neck_in = cm_to_inch(data['neck circumference'])

    if sex == "male":
        abdomen_in = cm_to_inch(data['waist circumference'])
        return body_fat_percentage(sex=sex, neck=neck_in, abdomen=abdomen_in, height=height_in)

    elif sex == "female":
        waist_in = cm_to_inch(data['waist circumference'])
        hip_in = cm_to_inch(data['hip circumference'])
        return body_fat_percentage(sex=sex, neck=neck_in, waist=waist_in, hip=hip_in, height=height_in)

    else:
        raise ValueError("성별은 'male' 또는 'female'이어야 합니다.")

class HumanMeshRecovery:
    def __init__(self, model_name='multiHMR_896_L', det_thresh=0.3, nms_kernel_size=3, device=torch.device('cuda')):
        self.device = device
        self.model_name = model_name
        self.det_thresh = det_thresh
        self.nms_kernel_size = nms_kernel_size
        self.model = self.load_model()
        self.measurer = MeasureBody('smplx')
        self.check_smplx_model()
        self.check_mean_params()

    def preprocessing_image(self, img_path, img_size):
        """ 경로에서 이미지를 열고, 크기를 조정하고 패딩합니다. """
    
        # 이미지 열기 및 크기 조정
        img_pil = Image.open(img_path).convert('RGB')
        img_pil = ImageOps.pad(ImageOps.contain(img_pil, (img_size, img_size)), size=(img_size, img_size))
    
        # numpy로 변환 및 정규화 후 torch로 변환
        resize_img = normalize_rgb(np.asarray(img_pil))
        x = torch.from_numpy(resize_img).unsqueeze(0).to(self.device)
        return x
    
    def get_camera_parameters(self, img_size, fov, p_x=None, p_y=None):
        """ 이미지 크기, fov 및 주점 좌표를 주어지면 카메라 매개변수 행렬 K를 반환합니다. """
        K = torch.eye(3)
        # 초점 거리 가져오기
        focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
        K[0,0], K[1,1] = focal, focal
    
        # 주점 설정
        if p_x is not None and p_y is not None:
                K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
        else:
                K[0,-1], K[1,-1] = img_size//2, img_size//2
    
        # 배치 차원 추가
        K = K.unsqueeze(0).to(self.device)
        return K
    
    def load_model(self):
        """ 체크포인트를 열고, 저장된 인수를 사용하여 Multi-HMR을 빌드하고, 모델 가중치를 로드합니다. """
        # 모델
        ckpt_path = os.path.join(CACHE_DIR_MULTIHMR, self.model_name+ '.pt')
        if not os.path.isfile(ckpt_path):
            os.makedirs(CACHE_DIR_MULTIHMR, exist_ok=True)
            logging.warning(f"{ckpt_path} 파일을 찾을 수 없습니다...")
            logging.info("데모 코드를 처음 실행하는 것 같습니다.")
            logging.info("NAVER LABS Europe 웹사이트에서 체크포인트를 다운로드 중입니다...")
            
            try:
                os.system(f"wget -O {ckpt_path} https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{self.model_name}.pt")
                logging.info(f"체크포인트가 {ckpt_path}에 다운로드되었습니다.")
            except:
                logging.error("fabien.baradel@naverlabs.com에 문의하거나 GitHub 저장소에 이슈를 열어주세요.")
                return 0
    
        # 가중치 로드
        logging.info("모델 로딩 중")
        ckpt = torch.load(ckpt_path, map_location=self.device)
    
        # 체크포인트에 저장된 인수를 가져와 모델을 재구성
        kwargs = {}
        for k,v in vars(ckpt['args']).items():
                kwargs[k] = v
    
        # 모델 빌드
        kwargs['type'] = ckpt['args'].train_return_type
        kwargs['img_size'] = ckpt['args'].img_size[0]
        model = Model(**kwargs).to(self.device)
    
        # 모델에 가중치 로드
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logging.info("가중치가 로드되었습니다")
    
        return model
    
    def forward_model(self, input_image, camera_parameters):
            
        """ 입력 이미지와 카메라 매개변수에 대한 순방향 패스를 수행합니다. """
        
        # 모델 순방향
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                humans = self.model(input_image, 
                               is_training=False, 
                               nms_kernel_size=int(self.nms_kernel_size),
                               det_thresh=self.det_thresh,
                               K=camera_parameters)
    
        return humans

    def check_smplx_model(self):
        smplx_fn = os.path.join(SMPLX_DIR, 'smplx', 'SMPLX_NEUTRAL.npz')
        if not os.path.isfile(smplx_fn):
            logging.warning(f"\033[95m[{time.strftime('%H:%M:%S')}] [경고] {smplx_fn} 파일을 찾을 수 없습니다. SMPLX_NEUTRAL.npz 파일을 다운로드하세요.\033[0m")
            logging.info("https://smpl-x.is.tue.mpg.de에서 계정을 생성해야 합니다.")
            logging.info("'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use this for SMPL-X Python codebase'를 다운로드하세요.")
            logging.info(f"압축 파일을 풀고 SMPLX_NEUTRAL.npz를 {smplx_fn}로 이동하세요.")
            logging.warning("\033[95m[{time.strftime('%H:%M:%S')}] [경고] 불편을 드려 죄송합니다. SMPLX 모델을 재배포할 라이선스가 없습니다.\033[0m")
            raise NotImplementedError
        else:
            logging.info('SMPLX 모델이 발견되었습니다.')

    def check_mean_params(self):
        if not os.path.isfile(MEAN_PARAMS):
            logging.warning(f"\033[95m[{time.strftime('%H:%M:%S')}] [경고] SMPL 평균 매개변수 다운로드 시작\033[0m")
            os.system(f"wget -O {MEAN_PARAMS}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
            logging.info('SMPL 평균 매개변수가 성공적으로 다운로드되었습니다.')
        else:
            logging.info('SMPL 평균 매개변수가 이미 존재합니다.')

    def process_image(self, img_path, fov):
        img_size = self.model.img_size
        x = self.preprocessing_image(img_path, img_size)
        K = self.get_camera_parameters(self.model.img_size, fov=fov)
        start = time.time()
        humans = self.forward_model(x, K)
        duration = time.time() - start
        logging.info(f"Multi-HMR 추론 시간={int(1000 * duration)}ms, 장치: {torch.cuda.get_device_name()}")
        return humans

    def measure_human(self, humans):
        betas = humans[0]['shape'].unsqueeze(0)
        self.measurer.from_body_model(gender="NEUTRAL", shape=betas)
        measurement_names = self.measurer.all_possible_measurements
        self.measurer.measure(measurement_names)
        self.measurer.label_measurements(STANDARD_LABELS)
        return(self.measurer.measurements)
