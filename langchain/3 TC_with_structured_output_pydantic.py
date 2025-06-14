from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
import pandas as pd
import re

# Load environment variables (OpenAI API Key, etc.)
load_dotenv()

# Initialize the model with controlled verbosity
model = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=3000)

# --- Define Schema for a Single Test Case ---
class BaselTestCase(BaseModel):
    test_title: str = Field(description="Overall title describing the Basel 3.1 RWA requirement being tested")
    test_case_title: str = Field(description="Concise title of the specific test case")
    test_case_description: str = Field(description="Detailed description of what this test case is validating")
    test_steps: List[str] = Field(description="Step-by-step instructions to execute the test case")
    test_data: List[str] = Field(description="Test data inputs required for the test")
    expected_results: List[str] = Field(description="Expected outcome or result after performing the test steps")

# --- Define Wrapper Schema (required by OpenAI for structured function calls) ---
class BaselTestCaseList(BaseModel):
    test_cases: List[BaselTestCase]

# Use LangChain model with structured output schema
structured_model = model.with_structured_output(BaselTestCaseList)

# --- Configuration ---
USE_CSV = False
CSV_FILE_PATH = "requirements_input.csv"
REQUIREMENT_COLUMN = "requirement"
MAX_TEST_CASES_PER_REQ = 2

# --- Load Requirements ---
if USE_CSV:
    df_input = pd.read_csv(CSV_FILE_PATH)
    requirements = df_input[REQUIREMENT_COLUMN].dropna().tolist()
else:
    requirements = [
        "For real estate exposures, Loan-to-Value (LTV) ratio must be accurately computed using the current market value of the property and the outstanding loan balance. If LTV ≤ 80%, assign a risk weight of 35%. If LTV > 80%, assign a risk weight of 75%.",
        "For SME retail exposures to qualify for a 75% risk weight: the total exposure of the obligor must not exceed EUR 1 million across all retail exposures, and it must be part of a well-diversified portfolio meeting regulatory retail criteria."
    ]

# --- Prompt Generator ---
def build_prompt(requirement: str, max_cases: int):
    return f"""
You are a Basel 3.1 compliance testing expert.

Generate {max_cases} different test cases for the following regulatory requirement:
\"\"\"{requirement}\"\"\"

Each test case must include:
- Test title
- Test case title and description
- At least 3 test steps
- Test data
- Expected results

Return the test cases in a JSON format under a key called "test_cases".
Ensure each test case handles a different logic branch, input range, or edge case.
"""

# --- Output Accumulator ---
all_rows = []

# --- Generate Test Cases and Flatten ---
for req in requirements:
    prompt = build_prompt(req, MAX_TEST_CASES_PER_REQ)
    results = structured_model.invoke(prompt)

    print(f"\n✅ Generated {len(results.test_cases)} test case(s) for requirement:\n{req}\n")

    for test_case in results.test_cases:
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

# --- Export Output ---
df_final = pd.DataFrame(all_rows)
df_final.to_csv("langchain/Output/3_all_normalized_test_cases.csv", index=False)

print("\n🎯 All test cases written to: all_normalized_test_cases.csv")
print(df_final.head(10))
