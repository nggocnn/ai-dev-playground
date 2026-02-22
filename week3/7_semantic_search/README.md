# Semantic Search for Clothing Products (Azure OpenAI + SciPy)

Build a simple, reliable **semantic search** engine for clothing products using **Azure OpenAI embeddings** and **cosine similarity** (SciPy). The script reads a CSV catalog, embeds product descriptions with `text-embedding-3-small`, embeds each query, ranks products by similarity, prints the **Top‑N** matches to the console, and writes results to a CSV.

---

## What you get

- **Production-ready Python script** (`semantic_search.py`) using the official `openai` Azure client.
- **No interactive input** (`input()` not used). Queries are loaded from a **text file** (or fall back to sane defaults).
- **File I/O**:
  - Input: `clothing_products.csv`
  - Queries: `queries.txt`
  - Output: `search_results.csv`
- **Exact schema** for products: `title, short_description, price, category`
- **Cosine similarity** via `scipy.spatial.distance.cosine`

---

## Example Output

Console (Top 3 per query) and CSV output preview:

```txt
Top 3 matches for query: 'warm cotton sweatshirt'
1. [0.5663] Cable Knit Sweater — Warm cable-knit sweater in cream wool blend. (Category: Sweaters, Price: $65.00)
2. [0.5622] White Crewneck T-Shirt — Soft cotton crewneck tee in bright white. (Category: T-Shirts, Price: $15.00)
3. [0.5237] Fleece Zip Hoodie — Warm fleece-lined zip-up hoodie with drawstring hood. (Category: Hoodies, Price: $44.99)

Top 3 matches for query: 'slim black leather jacket'
1. [0.7942] Black Leather Jacket — Stylish black leather jacket with a slim fit design. (Category: Jackets, Price: $120.00)
2. [0.5573] Black V-Neck Tee — Lightweight black V-neck t-shirt with slim silhouette. (Category: T-Shirts, Price: $17.50)
3. [0.4369] Little Black Dress — Classic black sheath dress suitable for evening events. (Category: Dresses, Price: $89.00)

Top 3 matches for query: 'breathable running shorts'
1. [0.7091] Athletic Mesh Shorts — Breathable mesh shorts with elastic waistband for sports. (Category: Shorts, Price: $29.00)
2. [0.6680] Breathable Running Shorts — Featherweight running shorts with quick-dry liner. (Category: Shorts, Price: $33.00)
3. [0.6013] Athletic Pullover Hoodie — Breathable pullover hoodie ideal for workouts and running. (Category: Hoodies, Price: $46.00)

Top 3 matches for query: 'office-ready white blouse'
1. [0.4537] White Crewneck T-Shirt — Soft cotton crewneck tee in bright white. (Category: T-Shirts, Price: $15.00)
2. [0.4071] Shirt Dress — Button-down shirt dress in breathable cotton poplin. (Category: Dresses, Price: $69.00)
3. [0.3818] Little Black Dress — Classic black sheath dress suitable for evening events. (Category: Dresses, Price: $89.00)

Top 3 matches for query: 'waterproof hiking jacket'
1. [0.6663] Hiking Shell Jacket — Breathable waterproof hiking shell with sealed seams. (Category: Jackets, Price: $159.00)
2. [0.6138] Waterproof Parka — Long waterproof parka with adjustable hood for rainy days. (Category: Jackets, Price: $149.00)
3. [0.5975] Rain Jacket — Lightweight rain jacket with water-repellent finish. (Category: Jackets, Price: $89.00)
```

CSV (`search_results.csv`) sample:

```csv
query,rank,similarity,title,short_description,price,category
warm cotton sweatshirt,1,0.566314,Cable Knit Sweater,Warm cable-knit sweater in cream wool blend.,65.0,Sweaters
warm cotton sweatshirt,2,0.562182,White Crewneck T-Shirt,Soft cotton crewneck tee in bright white.,15.0,T-Shirts
warm cotton sweatshirt,3,0.523675,Fleece Zip Hoodie,Warm fleece-lined zip-up hoodie with drawstring hood.,44.99,Hoodies
slim black leather jacket,1,0.794151,Black Leather Jacket,Stylish black leather jacket with a slim fit design.,120.0,Jackets
slim black leather jacket,2,0.557334,Black V-Neck Tee,Lightweight black V-neck t-shirt with slim silhouette.,17.5,T-Shirts
slim black leather jacket,3,0.436863,Little Black Dress,Classic black sheath dress suitable for evening events.,89.0,Dresses
breathable running shorts,1,0.709092,Athletic Mesh Shorts,Breathable mesh shorts with elastic waistband for sports.,29.0,Shorts
breathable running shorts,2,0.668022,Breathable Running Shorts,Featherweight running shorts with quick-dry liner.,33.0,Shorts
breathable running shorts,3,0.601303,Athletic Pullover Hoodie,Breathable pullover hoodie ideal for workouts and running.,46.0,Hoodies
office-ready white blouse,1,0.453713,White Crewneck T-Shirt,Soft cotton crewneck tee in bright white.,15.0,T-Shirts
office-ready white blouse,2,0.407139,Shirt Dress,Button-down shirt dress in breathable cotton poplin.,69.0,Dresses
office-ready white blouse,3,0.381759,Little Black Dress,Classic black sheath dress suitable for evening events.,89.0,Dresses
waterproof hiking jacket,1,0.666341,Hiking Shell Jacket,Breathable waterproof hiking shell with sealed seams.,159.0,Jackets
waterproof hiking jacket,2,0.613836,Waterproof Parka,Long waterproof parka with adjustable hood for rainy days.,149.0,Jackets
waterproof hiking jacket,3,0.597502,Rain Jacket,Lightweight rain jacket with water-repellent finish.,89.0,Jackets
```

> Scores will vary slightly by model updates and minor numeric differences across environments.

---

## How it works

1. **Data** — Load products from CSV with these exact columns:
   - `title` (string)
   - `short_description` (string)
   - `price` (float)
   - `category` (string)

2. **Embeddings** — For each product, the script sends `short_description` to Azure OpenAI **embeddings** and stores the resulting vector.

3. **Queries** — Reads one query per line from `queries.txt` (or falls back to a small built-in list). Each query is embedded the same way.

4. **Similarity** — Uses **SciPy** cosine *distance* and converts it to similarity:
   \f$ \text{similarity} = 1 - \text{cosine\_distance}(q, p) \f$

5. **Ranking** — Products are sorted by descending similarity for each query.

6. **Results** — Printed to console and written to `search_results.csv` with columns:
   - `query, rank, similarity, title, short_description, price, category`

---

## Requirements

- Python 3.9+
- Packages:
  - `openai>=1.35.0`
  - `python-dotenv`
  - `scipy`
  - `numpy`

Install:

```bash
pip install -r requirements.txt
```

---

## Azure OpenAI configuration

Set these environment variables (e.g., in your shell or a `.env` file):

## Environment Variables

Put these in a `.env` file or export them in your shell:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=<your-deployment>      # e.g., text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-07-01-preview
```

---

> **Important:** `AZURE_OPENAI_DEPLOYMENT` must be the **deployment name** you created in Azure for the **text-embedding-3-small** model.

---

## Input files & defaults

- **Products CSV**: defaults to `./clothing_products.csv`
- **Queries file**: defaults to `./queries.txt` (one query per line)

If the queries file is missing/empty, the script uses this safe default list:

```txt
warm cotton sweatshirt
slim black leather jacket
breathable running shorts
```

You can override any path via CLI flags (see below).

---

## Run

With defaults (expects files in current directory):

```bash
python semantic_search.py
```

Custom paths and Top‑N:

```bash
python semantic_search.py \
  --input /path/to/clothing_products.csv \
  --queries-file /path/to/queries.txt \
  --top-n 5 \
  --output /path/to/search_results.csv
```

---

## Data schema details

`clothing_products.csv` must contain **exactly these** headers:

```csv
title, short_description, price, category
```

- `title`: name of the product
- `short_description`: short, human‑readable description (used for embedding in this version)
- `price`: numeric (float)
- `category`: e.g., `Jeans`, `Hoodies`, `Jackets`, etc.

---

## Cosine similarity

Cosine measures the angle between two vectors. We use SciPy’s **cosine distance** and convert to similarity as `1 - distance`. A value closer to **1.0** means the query and product description are more semantically similar.

---

## Extending the script

- **Richer embedding text**: Swap the embedded field from just `short_description` to a combined string, e.g.:

  ```python
  text = f"{p['title']}. {p['short_description']}. Category: {p['category']}."
  ```

- **Persist product embeddings**: Cache vectors to a file (e.g., JSON, parquet) to avoid re‑embedding unchanged products.
- **Vector stores**: Replace the in‑memory list with a vector DB (Chroma, FAISS, pgvector) for large catalogs.
