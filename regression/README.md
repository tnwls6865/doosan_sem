# doosan_gasturbin

<b> [1] extract_image_feature_from_seg/extract_image_feature.py </b>

: 이미지 피쳐값 추출

(1) args
  - "option" _원하는 데이터 지정
  - "feature_num" _이미지 피쳐값 설정
    
(2) dir
  - "data_root_path" _데이터셋 경로
  - "output_dir" _결과 저장 경로 (image_features_{option}_{feature_num}.pkl)

<b> [2] data_preprocessing.py </b>

: ① 시험정보 & 야금학적 특징 통합  ② 이미지 피쳐맵 간 통합  ③ 전체 통합 데이터 구축

(1) input:
  - gasturbin_data.csv (원본 시험정보 데이터 - in792sx, in792sx interrupt, cm939w 통합본)
  - in792sx_features.csv, in792sx_interrupt_features.csv, cm939w_features.csv (야금학적 물성정보 데이터)
  -  (image_features_{option}_{feature_num}.pkl 필요)
  -  
(2) output: data_all_features_add_image.csv (최종 통합 데이터)

(3) args
  - "data_dir" _시험정보 데이터 경로 (gasturbin_data.csv)
  - "in792sx_dir" _in792sx 야금학적 특징 데이터 경로
  - "in792sx_interrupt_dir"_in792sx interrupt 야금학적 특징 데이터 경로
  - "cm939w_dir" _cm939w 야금학적 특징 데이터 경로 
  - "save_dir"_결과 저장 경로 (data_all_feature.csv)

<b> [3] saint/train.py </b>

: regression 수행

 regression 독립변수 변경 → saint/data_openml.py 
   > data_prep_openml function → X, categorical_indicator, attribute_names 지정

 수정사항 : models/model.py
  > RoWColTransformer class → nfeats 값 설정, x=torch.cat((x,image_feature), dim=1) 설정
