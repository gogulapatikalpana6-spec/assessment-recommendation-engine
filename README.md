# Assessment Recommendation Engine

A simple recommendation system that suggests SHL-style assessments based on job role and skills.

## Tech Stack
- Python
- Flask
- Pandas
- Scikit-learn

## Run Instructions
```bash
pip install -r requirements.txt
python app.py
```

## API
POST /recommend

### Sample Input
```json
{
  "job_role": "Software Engineer",
  "skills": ["Python", "DSA", "Problem Solving"]
}
```
