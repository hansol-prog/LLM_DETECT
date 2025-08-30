import pickle

# 열고 싶은 파일 경로를 지정합니다.
file_path = './data/train_ml_data.pkl'

# 'rb'(read binary) 모드로 파일을 엽니다.
# with 구문을 사용하면 파일을 사용한 후 자동으로 닫아주어 편리합니다.
with open(file_path, 'rb') as f:
    # pickle.load() 함수로 파일의 내용을 불러옵니다.
    data = pickle.load(f)

# 불러온 데이터를 출력하거나 사용합니다.
print("불러온 데이터:")
train_dataset = data

print(train_dataset)

