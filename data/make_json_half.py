import json

def split_json_half(input_filename, output_filename):
    """
    JSON 파일을 읽어 절반 크기의 새로운 JSON 파일로 저장합니다.
    JSON 데이터가 리스트 형태라고 가정합니다.
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 데이터가 리스트인지 확인
        if isinstance(data, list):
            # 데이터의 절반 지점 계산
            half_point = len(data) // 2
            half_data = data[:half_point]

            # 절반의 데이터를 새로운 파일에 저장
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(half_data, f, ensure_ascii=False, indent=4)
            
            print(f"성공: '{output_filename}'에 원본의 절반이 저장되었습니다.")

        # 데이터가 딕셔너리인 경우 (키-값 쌍의 절반을 저장)
        elif isinstance(data, dict):
            keys = list(data.keys())
            half_point = len(keys) // 2
            half_keys = keys[:half_point]
            
            half_data = {key: data[key] for key in half_keys}

            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(half_data, f, ensure_ascii=False, indent=4)

            print(f"성공: '{output_filename}'에 원본의 절반이 저장되었습니다.")

        else:
            print("오류: 지원되지 않는 JSON 형식입니다. 최상위 요소는 리스트 또는 딕셔너리여야 합니다.")

    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print(f"오류: '{input_filename}' 파일이 유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")


# --- 사용 예시 ---
input_file = 'hhalf_train.json'  # 원본 파일 경로
output_file = 'hhhalf_train.json'      # 저장할 파일 경로

# 함수 호출
split_json_half(input_file, output_file)