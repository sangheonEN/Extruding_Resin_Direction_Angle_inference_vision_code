데이터 넣는방법

1. IMAGE, MASK 폴더에 CASE_1, CASE_2 ... CASE 별로 각 IMAGE, MASK DATA를 넣음

2. data_utils.py 에서 mask data를 255. 곱해서 다시 MASK 파일로 저장한다. (mask_to_255 function)

3. data_utils.py 에서 CASE별로 나누어진 데이터를 하나의 all_data 폴더로 총 취합한다. (data_gather_one_dir function)

4. data_utils.py 에서 train, valid, test data로 8 : 1 : 1로 나누어 저장한다.

5. 나중에 data_loader.py에서 train_data, valid_data, test_data에서 데이터를 추출하여 사용한다.