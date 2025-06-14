from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
import pandas as pd
import re
import os

# Load environment variables
load_dotenv()

# Initialize LangChain LLM with max_tokens control
model = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=3000)

# Define output schema
class BaselTestCase(BaseModel):
    test_title: str = Field(description="Overall title describing the Basel 3.1 RWA requirement being tested")
    test_case_title: str = Field(description="Concise title of the specific test case")
    test_case_description: str = Field(description="Detailed description of what this test case is validating")
    test_steps: List[str] = Field(description="Step-by-step instructions to execute the test case")
    test_data: List[str] = Field(description="Test data inputs required for the test")
    expected_results: List[str] = Field(description="Expected outcome or result after performing the test steps")

structured_model = model.with_structured_output(BaselTestCase)

# --- Configuration ---
USE_CSV = False  # Set True to read from CSV
CSV_FILE_PATH = "requirements_input.csv"
REQUIREMENT_COLUMN = "requirement"
MAX_TEST_CASES_PER_REQ = 2

# --- Input Sources ---
if USE_CSV:
    df_input = pd.read_csv(CSV_FILE_PATH)
    requirements = df_input[REQUIREMENT_COLUMN].dropna().tolist()
else:
    requirements = [
        "For real estate exposures, Loan-to-Value (LTV) ratio must be accurately computed based on the current market value and the outstanding loan balance. If LTV > 80%, apply a risk weight of 75%. If LTV <= 80%, apply a risk weight of 35%.",
        "For SME retail exposures, apply a 75% risk weight if the total exposure is less than EUR 1 million and meets criteria for regulatory retail portfolio."
    ]

# --- Prompt Template ---
def build_prompt(requirement: str, max_cases: int = 1):
    return f"""
Generate {max_cases} test cases for the following Basel 3.1 RWA requirement:

\"\"\"{requirement}\"\"\"

Each test case should include:
- Test title
- Test case title and description
- At least 3 test steps
- Required test data
- Expected results
"""

# --- Processing ---
all_rows = []

for req in requirements:
    prompt = build_prompt(req, MAX_TEST_CASES_PER_REQ)
    results = structured_model.invoke(prompt)

    # In case of multiple test cases (list), make it iterable
    results = results if isinstance(results, list) else [results]

    for test_case in results:
        result_dict = test_case.dict()
        max_len = max(len(result_dict['test_steps']),
                      len(result_dict['test_data']),
                      len(result_dict['expected_results']))

        for i in range(max_len):
            raw_step = result_dict['test_steps'][i] if i < len(result_dict['test_steps']) else ""
            cleaned_step = re.sub(r"^Step\s*\d+\s*:\s*", "", raw_step, flags=re.IGNORECASE).strip()

            row = {
                'requirement': req,
                'test_title': result_dict['test_title'],
                'test_case_title': result_dict['test_case_title'],
                'test_case_description': result_dict['test_case_description'],
                'step_number': f"Step {i+1}",
                'test_step': cleaned_step,
                'test_data': result_dict['test_data'][i] if i < len(result_dict['test_data']) else "",
                'expected_result': result_dict['expected_results'][i] if i < len(result_dict['expected_results']) else "",
            }
            all_rows.append(row)

# --- Final Output ---
df_final = pd.DataFrame(all_rows)
df_final.to_csv("langchain/Output/2_all_normalized_test_cases.csv", index=False)
print("✅ Test case generation completed. Output saved to 'all_normalized_test_cases.csv'")
print(df_final.head(10))
