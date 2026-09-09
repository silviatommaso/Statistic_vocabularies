from groq import Groq
import os
import time
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
client = Groq(api_key=API_KEY)

PAUSE_BETWEEN_QUERIES = 0

DOMAINS = {
    "Money & Markets": "Financial markets and instruments, national accounts, public and private debt, exchange/interest rates, international trade, business demography and enterprise/government activity.",
    "People, Work & Living Conditions": "Population structure, households, births, deaths and migration, housing conditions, employment, wages, working conditions, poverty, income inequality, social protection.",
    "Health & Safety": "Physical and mental health, healthcare, mortality and survival as health outcomes, workplace safety, crime, gender-based violence.",
    "Learning & Education": "Education at all levels, lifelong learning, skills acquired.",
    "Environment & Natural Resources and Production": "Environment, energy, land use, agriculture, fisheries, forestry, natural resource sustainability.",
    "Transports": "Transport of people and goods across all modes (air, maritime, rail, road, inland waterways).",
    "Tech & Innovation": "Digital technologies and ICT, including internet access, digital skills, ICT specialists, R&D, innovation, high-tech activities, patents, trademarks, designs, and intellectual property.",
    "Living Well": "Culture and cultural activities participation, sport, tourism, leisure, recreation, time use, well-being, and secondary activities."
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
            - Assign each statistic to the most appropriate domain based on what is being measured, not on its disaggregation variables.
            - Classify according to the substantive phenomenon being measured, not according to the generic context in which it appears.
            - I'll give you a dictionary with the domains as keys and explanations of what each represents.
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


def result_definer(message, result, cluster):

    try:
        asw = call_llm(message, "openai/gpt-oss-120b")

    except Exception as e:
        print("Error:", e)
        asw = "ERROR"

    result.append({
        "cluster": cluster,
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

    # # Recreate the file with header if it does alredy exist
    # if output_file.exists():
    #     output_file.unlink()

    # pd.DataFrame(columns=["cluster", "domain"]).to_csv(output_file, index=False)


    result = []

    total_clusters = len(input_data)

    for i, item in enumerate(input_data, start=1):

        cluster = item["cluster"]
        terms = item["words"]

        message = build_message(terms)

        try:
            asw = call_llm(message, "openai/gpt-oss-120b")
            result.append({"cluster": cluster, "domain": asw})

            # Write immediately to CSV
            pd.DataFrame([{"cluster": cluster, "domain": asw}]).to_csv(output_file, mode="a", header=False, index=False)

        except Exception as e:

            print(f"Error for cluster {cluster}: {e}")
            break

        percentage = (i / total_clusters) * 100
        print(f"\rClustering: {percentage:.1f}% of completion", end="", flush=True)

        time.sleep(PAUSE_BETWEEN_QUERIES)

    print("\nClustering completed.")

    return pd.DataFrame(result)



# prompt(pd.read_csv("clustering/llm_files/cluster_words.csv").to_dict("records"), Path("clustering/llm_files"))