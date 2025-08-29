import csv
import json

def convert_csv_to_json(csv_file_path, json_file_path):
    """
    CSV 파일을 JSON 파일로 변환하는 함수입니다.

    Args:
        csv_file_path (str): 입력할 CSV 파일의 경로
        json_file_path (str): 출력할 JSON 파일의 경로
    """
    data = []
    
    try:
        # CSV 파일을 열고 각 줄을 딕셔너리로 읽어옵니다.
        # encoding='utf-8-sig'는 UTF-8 BOM을 처리하기 위함입니다.
        with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
            # DictReader는 첫 번째 줄을 헤더(key)로 자동 인식합니다.
            csv_reader = csv.DictReader(csv_file)
            
            # 각 줄(딕셔너리)을 data 리스트에 추가합니다.
            for row in csv_reader:
                data.append(row)

        # JSON 파일로 저장합니다.
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            # json.dump를 사용하여 데이터를 파일에 씁니다.
            # ensure_ascii=False: 한글이 깨지지 않도록 설정합니다.
            # indent=4: JSON 파일을 예쁘게 정렬하여 가독성을 높입니다.
            json.dump(data, json_file, ensure_ascii=False, indent=4)
            
        print(f"✅ 성공: '{csv_file_path}' 파일이 '{json_file_path}' 파일로 변환되었습니다.")

    except FileNotFoundError:
        print(f"❌ 오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 변환할 CSV 파일 이름과 저장할 JSON 파일 이름을 지정합니다.
    input_csv_file = './open/train.csv'
    output_json_file = 'train.json'
    
    # 함수를 호출하여 변환을 실행합니다.
    convert_csv_to_json(input_csv_file, output_json_file)