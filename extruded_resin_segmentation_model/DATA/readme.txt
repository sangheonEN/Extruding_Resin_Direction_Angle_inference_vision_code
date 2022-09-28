데이터 넣는방법

# case마다 left, right class를 change 해줘야하는 프로세싱이 있었는데, image name에 caseN을 붙여서 나중에 flag로 사용.

1. IMAGE, MASK 폴더에 CASE_1, CASE_2 ... CASE 별로 각 IMAGE, MASK DATA를 넣음

2. data_utils.py 에서 CASE별로 나누어진 데이터를 하나의 all_data 폴더로 총 취합한다. (data_gather_one_dir function)

3. data_utils.py 에서 train, valid, test data로 8 : 1 : 1로 나누어 저장한다.

4. train_flip_data_extract.py 실행하여 train data에 대해 vertical, horizontal image 추출함 (라벨링 curve 일관성 있는지 확인 360도 로테이션했을때 똑같은 경우로 left, right curve 형성되는지 확인)

5. 나중에 data_loader.py에서 train_flip_aug_data, valid_data, test_data에서 데이터를 추출하여 사용한다.