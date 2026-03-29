import google.generativeai as genai #type: ignore
import os
from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3-flash-preview")

def learning_planner(skills):

    prompt = f"""
You are an AI career mentor.

Create a STRICT 8-week learning roadmap for:
{', '.join(skills)}

Follow this EXACT format. Do NOT deviate.

Rules:
- Exactly 8 weeks (no more, no less)
- Each week MUST have:
    1. Main Topic
    2. Subtopics (comma separated)
    3. Practical Task
- Do NOT merge lines
- Do NOT write paragraphs
- Do NOT skip labels

Format:

Week 1: <Main Topic>
Subtopics: <Subtopic 1>, <Subtopic 2>, <Subtopic 3>
Practical Task: <Task>

Week 2: <Main Topic>
Subtopics: <Subtopic 1>, <Subtopic 2>, <Subtopic 3>
Practical Task: <Task>

Week 3: <Main Topic>
Subtopics: <Subtopic 1>, <Subtopic 2>, <Subtopic 3>
Practical Task: <Task>

Week 4: <Main Topic>
Subtopics: <Subtopic 1>, <Subtopic 2>, <Subtopic 3>
Practical Task: <Task>

Week 5: <Main Topic>
Subtopics: <Subtopic 1>, <Subtopic 2>, <Subtopic 3>
Practical Task: <Task>

Week 6: <Main Topic>
Subtopics: <Subtopic 1>, <Subtopic 2>, <Subtopic 3>
Practical Task: <Task>

Week 7: <Main Topic>
Subtopics: <Subtopic 1>, <Subtopic 2>, <Subtopic 3>
Practical Task: <Task>

Week 8: <Main Topic>
Subtopics: <Subtopic 1>, <Subtopic 2>, <Subtopic 3>
Practical Task: <Task>
"""

    response = model.generate_content(prompt)

    return response.text