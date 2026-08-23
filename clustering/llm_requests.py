from groq import Groq
import os
import time
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
client = Groq(api_key=API_KEY)

PAUSE_BETWEEN_QUERIES = 0

DOMAINS = {
    "General and regional statistics": "Introductory data and cross-cutting regional/summary indicators (e.g. NUTS classifications, city statistics).",
    "Economy and finance": "National accounts, GDP, prices, public finance, exchange and interest rates, balance of payments.",
    "Population and social conditions": "Demography, migration, health, education, labour market, income and living conditions, social protection.",
    "Industry, trade and services": "Structure and output of industrial sectors, domestic trade, services, business statistics.",
    "Agriculture, forestry and fisheries": "Agricultural production, forestry, fisheries, economic accounts for agriculture.",
    "Transport": "Freight and passenger transport by mode (road, rail, air, maritime, inland waterways).",
    "Environment and energy": "Emissions, waste, land use, energy production/consumption, renewable sources.",
    "Science, technology, digital society": "R&D, innovation, patents, ICT/internet use by businesses and citizens.",
}


# -----------------------
# TEXT-TO-SQL
# -----------------------

def build_message(terms):
    return [
        {
            "role": "system",
            "content": """You are a system that associates each term with EXACTLY ONE domain from the list below.

            RULES:
            - For EVERY list of terms you receive, choose EXACTLY ONE domain. Never assign more than one domain to the same list, even if it seems to fit several.
            - I'll give you a dictionary with the domains as keys and explanations of what each should contain as values.
            - Respond ONLY with single chosen domain name. No explanation, no markdown, no extra text.

            """
        },
        {
            "role": "user",
            "content": f"""
            Domains: {DOMAINS}
            list_of_terms: {terms}
            """
        }
    ]


def result_definer(message, result, cluster_id):

    try:
        asw = call_llm(message, "openai/gpt-oss-120b")

    except Exception as e:
        print("Error:", e)
        asw = "ERROR"

    result.append({
        "cluster_id": cluster_id,
        "domain": asw
    })

    return result


def call_llm(messages, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    return content


# -----------------------
# MAIN FUNCTION
# -----------------------

def prompt(input_data, llm_files_path):

    output_file = llm_files_path / "cluster_domain.csv"

    # Recreate the file with header if it does alredy exist
    if output_file.exists():
        output_file.unlink()

    pd.DataFrame(columns=["cluster", "domain"]).to_csv(output_file, index=False)


    result = []

    for _, item in input_data.iterrows():

        cluster_id = item["cluster"]
        terms = item["words"]

        message = build_message(terms)

        try:
            asw = call_llm(message, "openai/gpt-oss-120b")
            result.append({"cluster_id": cluster_id, "domain": asw})

            # Write immediately to CSV
            pd.DataFrame([{"cluster": cluster_id, "domain": asw}]).to_csv(llm_files_path / "cluster_domain.csv", mode="a", header=False, index=False)

            print(f"Cluster {cluster_id} -> {asw}")

        except Exception as e:

            print(f"Error for cluster {cluster_id}: {e}")

        time.sleep(PAUSE_BETWEEN_QUERIES)

    return pd.DataFrame(result)