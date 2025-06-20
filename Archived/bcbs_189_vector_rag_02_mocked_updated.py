
import pandas as pd
import re

class BaselTestCase:
    def __init__(self, test_title, test_case_title, test_case_description, test_steps, test_data, expected_results):
        self.test_title = test_title
        self.test_case_title = test_case_title
        self.test_case_description = test_case_description
        self.test_steps = test_steps
        self.test_data = test_data
        self.expected_results = expected_results

    def dict(self):
        return {
            "test_title": self.test_title,
            "test_case_title": self.test_case_title,
            "test_case_description": self.test_case_description,
            "test_steps": self.test_steps,
            "test_data": self.test_data,
            "expected_results": self.expected_results,
        }

class BaselTestCaseList:
    def __init__(self, test_cases):
        self.test_cases = test_cases

def generate_dummy_test_case(requirement):
    return BaselTestCaseList([
        BaselTestCase(
            test_title="Validate Basel Requirement",
            test_case_title=f"Test: {requirement[:30]}...",
            test_case_description=f"Ensure requirement '{requirement}' is implemented.",
            test_steps=["Check input", "Run calculation", "Verify result"],
            test_data=["Sample input", "Expected logic", "Expected output"],
            expected_results=["Valid input", "Correct process", "Compliant result"]
        )
    ])

def estimate_tokens(text, model_name):
    return len(text.split())

def generate_test_cases_for_requirement(requirement: str):
    try:
        structured_result = generate_dummy_test_case(requirement)
        rag_tokens = estimate_tokens(requirement, "dummy-model")
        prompt_tokens = estimate_tokens(requirement, "dummy-model")

        result_rows = []
        for tc in structured_result.test_cases:
            d = tc.dict()
            max_len = max(len(d['test_steps']), len(d['test_data']), len(d['expected_results']))
            for i in range(max_len):
                result_rows.append({
                    'requirement': requirement,
                    'test_title': d['test_title'],
                    'test_case_title': d['test_case_title'],
                    'test_case_description': d['test_case_description'],
                    'step_number': f"Step {i+1}",
                    'test_step': re.sub(r"^Step\s*\d+\s*:\s*", "", d['test_steps'][i]) if i < len(d['test_steps']) else "",
                    'test_data': d['test_data'][i] if i < len(d['test_data']) else "",
                    'expected_result': d['expected_results'][i] if i < len(d['expected_results']) else ""
                })

        return pd.DataFrame(result_rows), rag_tokens, prompt_tokens
    except Exception as e:
        return f"Error: {e}", 0, 0

def init_rag_chain(test_mode=False):
    return "dummy-model"
