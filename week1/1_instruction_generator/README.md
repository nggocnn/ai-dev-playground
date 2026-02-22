# Prompt-Driven Agent for Generating Work Instructions (Azure OpenAI)

This project is a **single-file Python script** that uses Azure OpenAI to generate **detailed, production-ready work instructions** from high-level task descriptions — ideal for new car model launches in automotive manufacturing.

The script now **creates a fresh output file each run** and **appends each row one-by-one** immediately after generating it, so results are saved even if the run is interrupted.

---

## Features

* **Reads** tasks from a CSV (`tasks.csv` by default, supports multiline descriptions).
* **Generates** shop-floor work instructions with:

  * Safety reminders
  * Required tools/materials
  * Preparation steps
  * Numbered procedure
  * Acceptance criteria & verification steps
  * Nonconformance handling
  * Documentation requirements
* **Writes** to CSV (`instructions.csv`) — **fresh file each run**, appending one task at a time.
* Deterministic output (`temperature=0`).
* Retries automatically on transient API errors.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install openai python-dotenv
```

Or use `requirements.txt`:

```txt
openai
python-dotenv
```

---

## Configure

Create a `.env` file in the project root:

```txt
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-07-01-preview
```

---

## Run

```bash
python instruction_generator.py
```

* Output file will be **deleted** if it already exists.
* Each task is processed and **immediately appended** to the CSV, so you can monitor progress live.

---

## Input Format (`tasks.csv`)

```csv
id,task_description
1,"Install the battery module in the rear compartment, connect to the high-voltage harness, and verify torque on fasteners."
2,"Calibrate the ADAS (Advanced Driver Assistance Systems) radar sensors on the front bumper using factory alignment targets."
3,"Apply anti-corrosion sealant to all exposed welds on the door panels before painting."
4,"Perform leak test on coolant system after radiator installation. Record pressure readings and verify against specifications."
5,"Program the infotainment ECU with the latest software package and validate connectivity with dashboard display."
```

---

## Output Format (`instructions.csv`)

```csv
Task Description Generated,Work Instruction
"Install the battery module in the rear compartment, connect to the high-voltage harness, and verify torque on fasteners.","1. **Safety**: Ensure high-voltage lockout..."
"Calibrate the ADAS (Advanced Driver Assistance Systems) radar sensors on the front bumper using factory alignment targets.","1. **Safety**: Wear ESD-protective gloves..."
...
```

---
