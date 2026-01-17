import google.generativeai as genai
import os
import json
import time
from dotenv import load_dotenv 

load_dotenv()
# REPLACE WITH YOUR KEY
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

class StoryValidator:
    def __init__(self, retrieval_func):
        self.retrieval_func = retrieval_func

    def decompose_backstory(self, text):
        """Extracts atomic claims from the backstory."""
        prompt = f"""
        Extract 3 concrete, verifiable facts from this backstory.
        Ignore feelings; focus on events, relationships, or physical constraints.
        Output JSON list of strings.
        Text: {text}
        """
        try:
            res = model.generate_content(prompt)
            clean = res.text.replace("```json", "").replace("```", "")
            return json.loads(clean)
        except:
            return [text]

    def verify(self, claim, evidence):
        """Checks consistency."""
        if not evidence: return "NEUTRAL (No evidence found)"
        
        prompt = f"""
        Premise: The backstory claims "{claim}".
        Story Evidence: {evidence}
        
        Task: Is the Premise consistent with the Story Evidence?
        Output strictly in this format:
        Verdict: [CONSISTENT/CONTRADICT/NEUTRAL]
        Reason: [1 sentence explanation]
        """
        try:
            return model.generate_content(prompt).text.strip()
        except:
            return "Verdict: NEUTRAL\nReason: Error in verification."

    def process_row(self, backstory, book_name):
        # 1. Decompose
        claims = self.decompose_backstory(backstory)
        contradictions = []
        
        # 2. Verify each claim
        for claim in claims:
            # Retrieve from specific book
            chunks = self.retrieval_func(query=claim, book_name=book_name)
            evidence_text = "\n...\n".join(chunks)
            
            # Check logic
            result = self.verify(claim, evidence_text)
            
            if "CONTRADICT" in result:
                # Capture the reasoning
                contradictions.append(f"Claim '{claim}' contradicted: {result}")

        # 3. Final Decision
        if contradictions:
            return 0, " | ".join(contradictions)
        else:
            return 1, "Backstory fits narrative constraints."