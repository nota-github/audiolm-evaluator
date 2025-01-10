import pandas as pd
import re
import sys

def validate_csv(file_path):
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 1. testset_id와 text 컬럼만 존재하는지 확인
        expected_columns = ['testset_id', 'text']
        if list(df.columns) != expected_columns:
            return "Error: The CSV file must contain only the columns 'testset_id' and 'text'."

        # 2. testset_id가 testXXXX 형식을 가지는지 확인
        if not df['testset_id'].apply(lambda x: isinstance(x, str) and re.match(r'^test\d{4}$', x)).all():
            return "Error: 'testset_id' must follow the format 'testXXXX' (e.g., 'test0000')."

        # 3. text 컬럼이 문자열(string)인지 확인
        if not df['text'].apply(lambda x: isinstance(x, str)).all():
            return "Error: All values in the 'text' column must be strings."

        return "Validation passed: The CSV file is valid."
    
    except FileNotFoundError:
        return "Error: The specified file was not found."
    except pd.errors.EmptyDataError:
        return "Error: The CSV file is empty or invalid."
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_csv.py <file_path>")
    else:
        file_path = sys.argv[1]
        result = validate_csv(file_path)
        print(result)
