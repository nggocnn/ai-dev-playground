# AI-Powered Conversational Assistant with Advanced Prompting (Azure OpenAI)

This project demonstrates **effective prompt design and context management** for conversational AI using the **Azure OpenAI Chat Completions API**. It includes **few-shot prompting**, **reasoning-friendly (CoT-safe) prompting**, and **multi-turn context** to simulate an **event management assistant** that can produce conversation starters, analyze sentiment, and plan micro-agendas.

---

## Requirements

- Python 3.8+
- [Azure OpenAI resource](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) with a deployed chat model (e.g., `gpt-4o-mini`)

Install dependencies:

```bash
pip install openai python-dotenv
```

---

## Environment Variables

Put these in a `.env` file or export them in your shell:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment>      # e.g., gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-07-01-preview
```

---

## How to run

```bash
python ai_chatbot.py
```

The script runs **three mock examples** automatically:

1. **Conversation Starters** for a cloud networking mixer
2. **Sentiment Analysis** of three texts (few-shot guided)
3. **Context-Aware Micro-Agenda** for a delayed keynote scenario

---

## Output

Below is the output you provided (your run may differ slightly):

```text
================================================================================
A) Conversation Starters for a Cloud Networking Mixer
================================================================================
1. "What’s the most exciting cloud project you’ve worked on?"  
   *Why it works: Encourages sharing personal experiences.*

2. "How do you see AI influencing cloud technology in the next year?"  
   *Why it works: Sparks discussion on current trends.*

3. "What cloud tools do you find indispensable in your work?"  
   *Why it works: Invites practical insights and tool recommendations.*

4. "Have you encountered any surprising challenges with cloud migration?"  
   *Why it works: Opens up a conversation about problem-solving.*

5. "What’s a recent cloud-related innovation that impressed you?"  
   *Why it works: Highlights enthusiasm for new developments.*

6. "How do you balance security and accessibility in cloud solutions?"  
   *Why it works: Engages on a critical industry topic.*

7. "What role do you think edge computing will play in the future?"  
   *Why it works: Promotes forward-thinking dialogue.*

8. "Can you share a success story from your cloud initiatives?"  
   *Why it works: Encourages sharing of positive outcomes.*


================================================================================
B) Sentiment Analysis (Few-Shot + CoT-Safe Reasoning)
================================================================================
[Sample 1] Text: I met so many inspiring people—this event gave me a boost!
  Result:
    Sentiment  : Positive
    Confidence : 0.95
    Explanation: Expresses enthusiasm and inspiration gained from the event.

[Sample 2] Text: The venue's acoustics made it hard to hear. Kinda frustrating.
  Result:
    Sentiment  : Negative
    Confidence : 0.85
    Explanation: Expresses frustration due to poor acoustics affecting the experience.

[Sample 3] Text: Met a few contacts. Overall fine, but nothing stood out.
  Result:
    Sentiment  : Neutral
    Confidence : 0.75
    Explanation: Indicates a mixed experience with no strong feelings either way.

================================================================================
C) Micro-Agenda Plan: Keynote Delay Handling
================================================================================
**Micro-Agenda for Delayed Keynote Speaker**

**Duration:** 20 mins

**Steps:**
1. **Introduction by MCs** (2 mins)  
   - Welcome attendees back and explain the delay.
  
2. **Sponsor Spotlight** (5 mins)  
   - Invite one or two sponsors to share a brief message about their offerings (2-3 mins each).

3. **Engagement Activity** (8 mins)  
   - Conduct a quick interactive poll or quiz related to the conference theme using A/V tools.

4. **Networking Break** (5 mins)  
   - Encourage attendees to visit sponsor booths in the foyer and network with each other.

**Roles:**
- **MCs:** Introduce segments and facilitate the activities.
- **Sponsors:** Present their messages.
- **A/V Crew:** Support with necessary equipment for polls and presentations.

**Comms:**  
"Thank you for your patience. While we wait for our keynote speaker, we have some exciting updates from our sponsors and a quick interactive activity to engage everyone!"

**Rationale:** This agenda maintains attendee engagement and provides valuable exposure to sponsors, while also allowing for networking opportunities during the delay.
```

---

## Prompting strategy

### 1. Roles & context

- **System prompt** defines the assistant’s identity and behavior (event management expert, structured answers, **no chain-of-thought disclosure**).
- **User/Assistant turns** simulate realistic multi-turn context (e.g., “we’re hosting a mixer”, “understood—let’s keep energy up”).

### 2. Few-shot prompting (3 samples)

- Three labeled sentiment examples (Positive/Negative/Neutral) are **embedded before** the new classification.
- This **calibrates** the model on label names, tone, and output format (with `Confidence` and `Explanation` fields).

### 3. Reasoning-friendly (CoT-safe) prompting

- We **encourage internal reasoning** with a short instruction (e.g., “think privately; return only final answer + brief rationale”).
- Outputs remain **concise and structured**, avoiding chain-of-thought leakage while benefiting from better reasoning.

### 4. Structured outputs

- Each task specifies a **tight schema** (e.g., numbered lists, `Sentiment/Confidence/Explanation`, or a micro-agenda template).
- This reduces ambiguity and makes downstream parsing easy.

---
