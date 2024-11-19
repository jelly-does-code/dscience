curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{
           "id": 0,
           "person_age": 37,
           "person_income": 35000,
           "person_home_ownership": "RENT",
           "person_emp_length": 0.0,
           "loan_intent": "EDUCATION",
           "loan_grade": "B",
           "loan_amnt": 6000,
           "loan_int_rate": 11.49,
           "loan_percent_income": 0.17,
           "cb_person_default_on_file": "N",
           "cb_person_cred_hist_length": 14
         }'
