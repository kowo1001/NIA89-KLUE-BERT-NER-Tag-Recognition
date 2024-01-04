# NIA 89번 공적말하기데이터 구축 과제 


■ 과제 개요
 - 공적 말하기 실습 및 평가를 위한 발표 영상 및 음성, 발표 자료 텍스트, 발표 평가 텍스트 데이터셋 구축과 공적 말하기 인식 및 분류, 수준 평가
 
■ 추진목적
 - 공적 말하기 실습 및 평가 분석과 초거대 AI 및 말하기 교육 AI의 인프라 확충을 목적으로, 공적 말하기의 영상과 발표자 전사 스크립트를 한 세트로 구성한 데이터셋 확보
 - 전문가를 통한 공적 말하기의 필수적인 평가 항목 지표 설정 및 데이터 라벨링 설계, 공적 말하기 데이터셋 구축에 특화된 저작도구 개발
 - 데이터셋 전체 공정(수집, 정제, 가공, 품질관리) 과정에 신규 인력 고용으로 일자리 창출

■ 과제 목표
- 공적 말하기 실습 및 평가를 위한 발표 영상 및 음성, 발표 자료 텍스트, 발표 평가 텍스트 데이터셋 구축과 공적 말하기 인식 및 분류, 수준 평가
- 인공지능 학습용 데이터 구축량 
- 800명 이상의 발표자의 음성이 포함된 발표 동영상 2,400 클립 이상
- 공적 말하기 발표 원고 280건 (전사 스크립트800건) 이상(발표 평가 포함)
- 공적 말하기를 교육 및 평가하는 객관적인 교육용 자료로 활용하고 실시간 공적 말하기 관련 코칭 및 피드백 솔루션 제공

■ 학습모델 품질기준  \
〇 선정된 학습모델 임무(TASK) 개념 및 적정성
- 임무 : 태깅 탐지(태그 인식 및 분류)
- 개념 : 발표자의 발화에서 언어적 평가항목 중 인식 가능한 태그를 탐지하여 언어적 평가에 활용

〇 학습모델 후보군
- 임무 : KLUE-BERT
- 선정사유 : 
1) KLUE는 한국어 자연어 이해 작업에 대한 평가와 벤치마크로서 신뢰성이 높고 한국어 텍스트 처리에 
특화되어 있음
2) KLUE는 한국어 이해를 위한 다양한 작업을 포함하고 있으며, 개체명 인식은 그 중 하나의 작업
3) KLUE의 개체명 인식 작업은 한국어 텍스트에서 중요한 정보를 추출하고 한국어 텍스트에서 개체명을 식별하고
분류하는 작업을 수행
4) KLUE-BERT 언어모델을 사용하여 한국어 자연어 처리 작업에 대한 평가와 성능 비교를 제공함
5) KLUE BERT base는 한국어로 사전 훈련된 BERT 모델
6) 한국어 이해 평가(KLUE, Korean Language Understanding Evaluation)벤치마크 개발의 맥락에서 모델 개발
7) 벤치마크 데이터인 KLUE에서 베이스라인으로 사용되었던 모델로 모두의 말뭉치, CC-100-Kor, 나무위키, 뉴스, 청원 등 문서에서 추출한 63GB의 데이터로 학습되었음
8) Morpheme-based Subword Tokenizer를 사용하였으며, vocab size는 32,000이고 모델의 크기는 111M
9) Fine-tuning을 통해 한국어 개체명(태그명)인식 다운스트림 태스크를 진행
    
〇 성능 지표 및 목표값
- F1-점수 (F1-Score) 64% 이상 \
〇 Data I/O
- Input data : 전사 스크립트(STT 발화 내용 텍스트 문장),특정 평가항목(휴지)에 대한 라벨링 데이터
- Output data :  입력된 텍스트 문장에서 인식된 태그명에 대한 정보

■ 모델: KLUE-BERT (BERT)

■ 모델 개요
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
