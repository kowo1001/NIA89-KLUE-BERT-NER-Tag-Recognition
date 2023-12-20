# NIA 89번 공적말하기데이터 구축 과제 

■ 모델: KLUE-BERT (BERT)

■ 개요
 - KLUE BERT base는 한국어로 사전 훈련된 BERT 모델
 - 한국어 이해 평가(KLUE,  Korean Language Understanding Evaluation)벤치마크 개발의 맥락에서 모델 개발
 - 벤치마크 데이터인 KLUE에서 베이스라인으로 사용되었던 모델로 모두의 말뭉치, CC-100-Kor, 나무위키, 뉴스, 청원 등 문서에서 추출한 63GB의 데이터로 학습되었음
 - Morpheme-based Subword Tokenizer를 사용하였으며, vocab size는 32,000이고 모델의 크기는 111M
 - Fine-tuning을 통해 한국어 개체명(태그명)인식 다운스트림 태스크를 진행
 - 참고자료1: https://github.com/KLUE-benchmark/KLUE
 - 참고자료2: https://huggingface.co/klue/bert-base#model-details

■ How to Get Started With the Model
```python
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
model = TFBertModel.from_pretrained("klue/bert-base")

```

■ 구축 환경   \
`CPU` : Xeon Gold 6348 CPU 2.60GHz 28core * 2EA \
`Memory` : 527GB  \
`GPU` : Nvidia TESLA A100 (80G) * 4EA \
`Storage` : 1.8TB HDD  \
`OS` : Ubuntu 22.04.1 LTS  \
`개발언어` : Python 3.8.10  \
`프레임워크` : Tensorflow 2.13.0  


■ Architecture

![model_architecture_image](https://github.com/kowo1001/KLUE-BERT-NER/assets/37354978/65a85c46-5abd-4a63-8520-4f0e257821d8)


# 유효성 검증 실행방법

도커 이미지 : nia89_lang_workspace.tar \
도커이미지명: nia89_workspace_publicspeak_gpu 


## 1. 실행 방법
(1. 데이터전처리)
언어적 태깅 탐지(인식) 데이터전처리 방법은 다음과 같습니다.

첨부한 Docker 파일을 다운받습니다.

파일이 있는 경로에서 아래 명령어를 통해 docker load를 실행합니다.

```bash
$ docker load -i nia89_lang_workspace.tar 
```

Load가 완료되면 Docker를 실행시킵니다.

```bash
$ docker run -it --gpus all nia89_workspace_publicspeak_gpu bash
```

(또는) \

```bash
$ docker run -it \
  --name speaking \
  --gpus all \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -p 7676:9096 \
  --restart always \
  --ipc=host \
  -v /home/udas/nado/essay/workspace/aes/jwjang_workspace/tag-recognition/KLUE-BERT-NER/nia89_workspace:/app \
  -e TZ=Asia/Seoul \
  -e LC_ALL=C.UTF-8 \
  nia89_workspace_publicspeak_gpu:latest bash  )
```

명령어를 수행하게 되면 실행중인 Docker 내부로 진입하게 됩니다.

아래 명령어를 통해 마침표 기준으로 문장 분리, 태그시퀀스 라벨링을 위한 데이터전처리를 수행 합니다.

```bash
$ sudo python data_preprocessor.py
```

dataset 폴더에 있는 라벨링데이터들(json)을 모듈(data_preprocessor.ppy)로 뽑아낸 결과물은 다음과 같습니다.
학습데이터셋, 검증데이터셋, 테스트데이터셋을 8:1:1 비율로 분리하였습니다.
1) tag_train_data_17290e.csv 2) tag_valid_data_2164e.csv 3) tag_test_data_2163e.csv


(2. 모델학습 및 평가)
언어적 태깅 탐지(인식) 모델의 학습 및 평가 방법은 다음과 같습니다.

dataset_nerlabel 폴더에 태그들이 정의된 텍스트 파일(ner_label_v2.txt)이 있는지 확인 합니다.

아래 명령어를 통해 dataset_tagsequence 폴더에 있는 데이터전처리가 완료된 데이터들을 입력으로 
모델학습 및 평가(테스트)를 합니다.

```bash
$ sudo python main.py
```

학습이 완료되면 학습된 모델과 체크포인트 저장 후, 학습을 종료합니다.

해당 결과물을 통해 유효성 검증이 되었음을 확인할 수 있습니다.

마지막으로, 테스트데이터셋에서 지정원고 대상으로 발화된 문장 2개를 테스트한 결과, 
발표평가요소인 태그들(REP-B: 반복어구, FIL-B: 간투어, WR-B: 발음오류, PS-B: 긴쉼)이 인식됨을 확인할 수 있습니다.


## 2. 코드 파일 설명
- 모델학습 전 데이터전처리 할 시에는 data_preprocessor.py 실행
- 유효성 검증을 위해 모델학습 및 모델평가할 시에는 main.py 실행

```bash
# docker-compose.yml 마운트경로 수정 후 이 파일이 있는 경로에서
docker-compose up -d --build
docker exec -it nia89_workspace_publicspeak_gpu bash

# @컨테이너 내부에서 데이터셋전처리시
sudo chmod 777 -R /dataset/
python data_preprocessor.py

# @컨테이너 내부에서 모델 학습시
sudo chmod 777 -R /dataset_tagsequence/
python main.py

# 컨테이너 내부 실행파일들이 있는 경로는 다음과 같다
/app
├── dataset
│   ├── A00_S01_F_C_01_029_02_WA_MO_presentation.json
│   ├── A00_S01_F_C_01_030_02_WA_MO_presentation.json
│   ├── A00_S01_F_C_01_031_02_WA_MO_presentation.json
│   └── ...
├── dataset_nerlabel
│   └── ner_label_v2.txt
├── dataset_tagsequence
│   ├── tag_train_data_17290e.csv
│   ├── tag_valid_data_2164e.csv
│   └── tag_test_data_2163e.csv
├── data_preprocessor.py
├── main.py
├── Dockerfile
├── docker-compose.yml
├── saved_model
└── training_1
```

### 2.1 (학습용) 데이터 전처리 파일
- data_preprocessor.py
    - 실행시 dataset 폴더에 있는 라벨링데이터(json) 전처리 작업
    - 학습데이터, 검증데이터, 테스트데이터 8:1:1 비율로 분리 
    - 마침표 기준으로 문장 분리, 각 문장에 말하기평가요소(간투어, 긴쉼, 반복어구,발음오류 혹은 본문과 다른 표현)인 태그들에 대해 태그 시퀀스화 적용
    - dataset_tagsequence 폴더에 Sentence(문장), Tag(태그 시퀀스)로 구성된 학습데이터, 검증데이터, 테스트데이터 csv 파일 저장

### 2.2 (학습용) 학습 실행 파일 및 유효성 검증 관련 실행 파일
- main.py
    - 데이터 전처리부터 학습까지 필요한 코드 한번에 실행하는 파일
    - dataset_tagsequence 폴더의 학습데이터, 검증데이터, 테스트데이터를 로드하여 학습에 이용
    - BertTokenizer 를 활용하여 모델의 입력으로 들어가도록 데이터전처리 수행
    - training_1 폴더에 체크포인트(checkpoint) 파일 생성
    - 학습 실행 후 산출물 파일들이 saved_model 폴더 내에 생성
    - 학습 후 유효성 검증시에 사용됨
    - 학습용 데이터로 학습 후 저장된 모델을 통해 각 에포크(epoch)마다 테스트데이터셋을 통해 모델 성능 측정
    - 산출물이 saved_model 폴더내에 생성

-----------------------------------------------------------------------------

## 3. 데이터셋 및 산출물 경로 설명
- 마운트된 도커 컨테이너 내 데이터셋 경로는 아래와 같다.
```
/dataset
    ├── A00_S01_F_C_01_029_02_WA_MO_presentation.json
    ├── A00_S01_F_C_01_030_02_WA_MO_presentation.json
    ├── A00_S01_F_C_01_031_02_WA_MO_presentation.json
    └── A00_S01_F_C_02_032_02_WA_MO_presentation.json
        ...

/dataset_tagsequence
    ├── tag_train_data_17290e.csv
    ├── tag_valid_data_2164e.csv
    └── tag_test_data_2163e.csv

/dataset_nerlabel
    └── ner_label_v2.txt

```

### 3.1 마운트할 데이터셋의 로컬호스트내 경로 설명
- docker-compose.yml파일을 참고한다
- 데이터전처리할 시, 마운트할 (로컬경로)에 dataset 폴더가 있기만 하면 된다. 
- 모델학습할 시, 마운트할 (로컬경로)에 dataset_tagsequence 폴더와 dataset_nerlabel 폴더가 있고 각 폴더에 아래 예시와 같이 파일을 넣기만 하면 된다. 
(예시) 
/dataset_tagsequence
    ├── tag_train_data_17290e.csv
    ├── tag_valid_data_2164e.csv
    └── tag_test_data_2163e.csv

/dataset_nerlabel
    └── ner_label_v2.txt

- /home/nia89_workspace/docker-compose.yml
    - (로컬경로):(도커 컨테이너 내 경로)
    - 자유롭게 docker-compose.yml 의 (로컬경로)를 수정한다(예시: /nia89_workspace:/app).

### 3.2 학습 또는 유효성 검증 후 관련 산출물 경로 설명
- 실행파일이 있는 /app 폴더와 같은 레벨에 있는 /saved_model폴더
- /home/nia89_workspace/saved_model

### 3.3 학습시 필요한 마운트 경로 설명
- 전처리전 데이터는 폴더까지 명시를 해야한다
    - 예시:
        - /dataset_tagsequence/tag_train_data_17290e.csv
        - /dataset_tagsequence/tag_valid_data_2164e.csv
        - /dataset_tagsequence/tag_test_data_2163e.csv

- 전처리후 데이터는 dataset_tagsequence 폴더의 위치와 동일

----------------------------------------------------------------------------- 

## 4. 실행 순서 (실행방법2)
```bash
# docker-compose.yml파일이 있는 경로로 이동하여 다음 명령어를 실행한다.
# 도커 올리기
(base) admin@labtest-XPS-8950:~/home/NIA$ docker-compose up -d --build
# 컨테이너 접속
(base) admin@labtest-XPS-8950:~/home/NIA$ docker exec -it nia89_workspace_publicspeak_gpu bash

# /app 경로에서 다음 명령어를 실행한다.
admin@6752c5e31641:/app$ sudo chmod 777 -R /saved_model
admin@6752c5e31641:/app$ python main.py

# 도거 내리기
(base) admin@labtest-XPS-8950:~/home/NIA$ docker-compose down -v
```

-----------------------------------------------------------------------------

## 5. 모델 평가 산출물 확인
- main.py 실행시 /saved_model에 산출물이 저장된다.
```
/saved_model/
└── kluebert_base_new
	├── assets
	├── variables 
	│	├──variables.data-00000-of-00001
	│	└──variables.index
	├── fingerprint.pb
	├── keras_metadata.pb
	└── saved_model.pb
```
