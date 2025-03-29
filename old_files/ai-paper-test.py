import arxiv
import re
import re
from datetime import datetime, timedelta, timezone
import math
import google.generativeai as genai
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import fitz
import requests
import os
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Paper


genai.configure(api_key='AIzaSyA8KvHS_I58qHg8SP9zal1dH0cLikw5UmE')
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# Define a global arXiv client
client = arxiv.Client()

import arxiv
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, date

def ensure_datetime_utc(date_value):
    """Ensures that the given date value is a timezone-aware datetime object in UTC."""
    if isinstance(date_value, datetime):
        return date_value.astimezone(timezone.utc)  # Ensure UTC
    elif isinstance(date_value, date):  # If it's a date object, convert it to datetime at midnight UTC
        return datetime.combine(date_value, datetime.min.time(), tzinfo=timezone.utc)
    elif isinstance(date_value, str):  # If it's a string, try to parse it
        try:
            return datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"⚠️ Warning: Unexpected date format: {date_value}")
            return None
    else:
        print(f"⚠️ Warning: Unknown date type: {type(date_value)} - {date_value}")
        return None

def fetch_ai_papers(max_results=10):
    """Fetches recent AI papers from arXiv using both the API and web scraping (New Submissions section)."""

    # Step 1: Fetch from the API
    search = arxiv.Search(
        query="cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.CV OR cat:cs.RO OR cat:cs.MA OR cat:cs.HC OR cat:cs.IR OR cat:cs.NE OR cat:eess.SP",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    recent_papers = []
    today_utc = datetime.now(timezone.utc).date()  # Get today's date in UTC

    for result in client.results(search):   
            papers.append({
                'title': result.title,
                'summary': result.summary,
                'authors': [author.name for author in result.authors],
                'link': result.pdf_url,
                'published': result.published}
            )

    # Step 2: Scrape New Submissions from arXiv
    url = "https://arxiv.org/list/cs.AI/new"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the paper entries
    for dt in soup.find_all("dt"):
        link_tag = dt.find("a", {"title": "Abstract"})
        if link_tag:
            paper_id = link_tag.text.strip()
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            abs_url = f"https://arxiv.org/abs/{paper_id}"

            # Find corresponding title
            dd = dt.find_next_sibling("dd")
            title_tag = dd.find("div", class_="list-title mathjax")
            title = title_tag.text.replace("Title:", "").strip() if title_tag else "Unknown Title"

            # Find authors
            authors_tag = dd.find("div", class_="list-authors")
            authors = authors_tag.text.replace("Authors:", "").strip() if authors_tag else "Unknown Authors"

            # Find summary
            summary_tag = dd.find("p", class_="mathjax")
            summary = summary_tag.text.strip() if summary_tag else "No summary available."

            # Append to papers list
            recent_papers.append({
                'title': title,
                'summary': summary,
                'authors': authors,
                'link': pdf_url,
                'published': ensure_datetime_utc(today_utc)  # Ensure UTC datetime
            })

    return [papers, recent_papers]




def store_papers_in_db(papers, table="recent"):
    """
    Stores fetched AI papers into the specified database table.
    
    Options for `table`: "recent", "daily", "weekly".
    """
    from datetime import datetime, timezone
    from crud import create_paper  # Use existing CRUD function
    db = SessionLocal()

    model_map = {
        "recent": RecentPaper,
        "daily": DailyPaper,
        "weekly": WeeklyPaper
    }
    PaperModel = model_map.get(table, RecentPaper)

    for paper in papers:
        # Score paper
        score, _ = score_paper(paper)
        paper['score'] = score

        # Generate AI Summary
        ai_summ = ai_summary(paper['title'], paper['summary'])
        paper['ai_summary'] = ai_summ if ai_summ else "Summary not available."

        # Generate PDF Summary
        pdf_summ = pdf_summary(paper['link'])
        paper['pdf_summary'] = pdf_summ if pdf_summ else "Summary not available."

        # Convert authors list to a comma-separated string
        paper['authors'] = ", ".join(paper['authors'])

        # Ensure the date is timezone-aware
        if isinstance(paper['published'], datetime):
            paper['published'] = paper['published'].astimezone(timezone.utc)

        # Check if the paper already exists in the selected table (avoid duplicates)
        existing_paper = db.query(PaperModel).filter(PaperModel.title == paper['title']).first()
        if existing_paper:
            print(f"Skipping duplicate: {paper['title']}")
            continue

        # Store in the correct table
        db_paper = PaperModel(
            title=paper['title'],
            summary=paper['summary'],
            ai_summary=paper['ai_summary'],
            pdf_summary=paper['pdf_summary'],
            authors=paper['authors'],
            link=paper['link'],
            published=paper['published'],
            score=paper['score'],
            date=datetime.now(timezone.utc)  # Store the date when added
        )
        db.add(db_paper)

    db.commit()
    db.close()
    print(f"✅ Papers stored in `{table}` table successfully.")

import json

def load_top_ai_researchers():
    """Loads the curated list of top AI researchers from JSON."""
    try:
        with open("./backend/top_ai_researchers.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        with open("./top_ai_researchers.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading top AI researchers: {e}")
        return []
    
def evaluate_paper_with_gemini(title, summary):
    """Uses Gemini 2.0 Flash to evaluate a paper's relevance, insight, novelty, and impact."""
    prompt = f"""
    You are an expert AI researcher reviewing recent AI papers. 
    Given the following paper title and summary, rate it on the following factors (1-10):
    
    - **Relevance:** How relevant is this to cutting-edge AI?
    - **Insightfulness:** Does it provide new insights?
    - **Novelty:** Is this a genuinely new idea?
    - **Impact:** How likely is this to influence AI research significantly?

    Your scores should be based on a system with a normal distribution of scores, with a mean of 5 and a standard deviation of 3.
    If you are very suprised by a paper, you might give it a 9 or 10. If you are very unimpressed, you might give it a 1 or 2.
    Often, Large Language Models (LLMs) like yourself only provide scores in the 6-8 range for most papers.
    Ensure your scores are not all within the same range, but reflect the true quality of the paper.
    Ensure you only give scores, not feedback, comments, or explanation.

    Return your response as JSON:
    {{"relevance": <score>, "insightfulness": <score>, "novelty": <score>, "impact": <score>}}

    **Title:** {title}
    **Summary:** {summary}
    """

    try:
        
        response = llm_model.generate_content(prompt)
        
        if not response or not response.text:
            raise ValueError("Gemini returned an empty response.")

        response_text = response.text.strip().strip("```json").strip("```").strip()  # ✅ Clean JSON response

        # Ensure the response is valid JSON
        scores = json.loads(response_text)
        
        return scores
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from Gemini: {e}\nResponse: {response.text if response else 'None'}")
    except Exception as e:
        print(f"Error with Gemini evaluation: {e}")


def ai_summary(title, summary):
    """Generates a summary of an AI paper using Gemini 2.0 Flash."""
    prompt = f"""
    You are an AI researcher reviewing a recent AI paper. 
    Summarize the following paper using 3-5 sentences:
    
    **Title:** {title}
    **Summary:** {summary}
    """

    try:
        response = llm_model.generate_content(prompt)
        
        if not response or not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text.strip()
    except Exception as e:
        print(f"Error with Gemini summary generation: {e}")


def pdf_summary(pdf_path):
    """
    Extracts text from a given PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    if pdf_path.startswith("http"):
        local_pdf = "temp_paper.pdf"
        try:
            response = requests.get(pdf_path, stream=True)
            response.raise_for_status()  # Raise error if request fails
            
            with open(local_pdf, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            pdf_path = local_pdf  # Set path to downloaded file
        except requests.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return None

    # Extract text from the PDF
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()  # ✅ Properly close the document before returning
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None
        
    prompt = """
        You are an expert AI researcher and technical writer. Your task is to analyze and summarize the following AI research paper, focusing on new discoveries, theoretical advancements, practical applications, and potential impact.

        ### **Instructions:**
        1. **Identify Key Contributions:**
           - What are the paper's major findings or innovations?
           - How does it compare to prior work in the field?
           - What are the implications for future research or real-world applications?

        2. **Explain Complex Concepts Clearly:**
           - Break down difficult ideas in an accessible way without oversimplifying.
           - Provide analogies or context when needed.
           - Reference specific passages or sentences from the paper to support explanations.

        3. **Evaluate Benchmarks & Experimental Results (if applicable):**
           - If benchmarks are tested, summarize performance improvements over previous models.
           - Highlight any significant progress made in areas such as accuracy, efficiency, or scalability.
           - Quantify improvements using specific metrics (e.g., "achieved a 5.2% increase in accuracy on ImageNet").

        4. **Discuss Limitations & Open Questions:**
           - What are the limitations acknowledged by the authors?
           - Are there any concerns or gaps in their methodology?
           - What follow-up research questions emerge from this work?

        5. **Assess Broader Impact:**
           - How might this research influence AI development, industry applications, or scientific understanding?
           - Are there ethical, societal, or safety concerns?
           - Could this be a stepping stone toward AGI, better interpretability, or more efficient computing?

        ---
        ### **Response Format (Example Structure)**
        1. **Title & Authors:**  
           - "Title of the Paper" by [Authors]

        2. **Summary of Key Findings**  
           - [Summarize the most important contributions]  
           - Reference supporting sections: "According to section 3.2, the method introduced improves..."

        3. **Complex Concepts Explained**  
           - "The authors introduce X, which functions as... (Reference: Page 5, Section 4.1)"  
           - Provide accessible explanations.

        4. **Benchmark Results & Comparisons**  
           - "On the SuperGLUE benchmark, this model achieved X% improvement over previous state-of-the-art (Reference: Table 2)."

        5. **Limitations & Future Work**  
           - "The authors acknowledge that... (Reference: Section 6)"  
           - Discuss research gaps or unresolved questions.

        6. **Potential Impact**  
           - "This research could impact AI scaling laws by... (Reference: Section 7)"  
           - Consider long-term implications.

        ---
        Please structure your response following this format, referencing specific sections when possible.
        """
    
    try:
        response = llm_model.generate_content(prompt + text)
        
        if not response or not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text.strip()
    except Exception as e:
        print(f"Error with Gemini text extraction: {e}")
        return None


def print_papers(papers):
    for paper in papers:
        print(f"Title: {paper['title']} + Score: {paper['score']} + Date: {paper['published']}")
        print("\n--------------------------------------------------\n")
        print("Summary:", ai_summary(paper['title'], paper['summary']))
        print("Link:", paper['link'])
        print("\n--------------------------------------------------\n")
        print("PDF Summary:", pdf_summary(paper['link']))
        print("\n--------------------------------------------------\n")


def export_report(papers, filename="paper_report.md"):
    """Exports the list of AI papers to a Markdown file with proper formatting."""
    if not papers:
        print("⚠️ No papers to export!")
        return
    """Exports the list of AI papers to a Markdown file with proper formatting."""
    with open(filename, "w", encoding="utf-8") as file:
        for paper in papers:
            title = paper['title']
            score = paper['score']
            published = paper['published']
            summary = ai_summary(title, paper['summary'])  # Compute AI summary before writing
            pdf_summary_text = pdf_summary(paper['link'])  # Compute PDF summary
            link = paper['link']

            # Get date from published in the format Mar 10 2020
            published = published.strftime("%b %d, %Y")            


            file.write(f"# {title} ({link})\n")
            file.write(f"**Date:** {published}\n\n")
            file.write("---\n\n")
            file.write(f"### Summary\n{summary}\n\n")
            file.write("---\n\n")
            file.write(f"### AI Breakdown\n{pdf_summary_text}\n\n")
            file.write("---\n\n")

    print("✅ Report exported successfully.")


def get_top_papers(papers, num):
    for paper in papers:
        paper['score'] = score_paper(paper)[0]

    # Sort papers by score
    papers = sorted(papers, key=lambda x: x['score'], reverse=True)

    # Select the top 5 papers
    top_papers = papers[:num]

    return top_papers

TOP_RESEARCHERS = set(load_top_ai_researchers())  # Store as a set for fast lookup
FUTURE_KEYWORD_LIST = [# AI Enhancing AI (Meta AI Research)
    "AutoML", "automated machine learning", "neural architecture search", "NAS", 
    "AI-generated AI", "self-improving neural networks", "AI-optimized AI hardware", 
    "hyperparameter optimization", "evolutionary algorithms in AI", 
    "AI-augmented data labeling", "automated theorem proving in AI", 
    "synthetic data generation", "algorithm distillation",

    # The Road to AGI & Theoretical AI
    "recursive self-improvement", "agency in AI", "AI autonomy", "self-reflective AI", 
    "emergent behaviors in LLMs", "AI alignment", "AI safety", "AI consciousness", 
    "temporal memory in AI", "hierarchical reinforcement learning", 
    "self-supervised world models", "goal-oriented AI", "simulation-based learning", 
    "AI interpretability", "explainable AI", "XAI", "time-consistent AI", 
    "AI decision-making under uncertainty",

    # AI for Biotech & Genetic Research
    "AI in drug discovery", "AI in genomics", "gene editing AI", "AI for CRISPR", 
    "protein folding AI", "AlphaFold", "AI in synthetic biology", 
    "AI-driven medical diagnosis", "AI in computational chemistry", 
    "personalized medicine AI", "AI in neuroscience", "brain mapping AI", 
    "AI in epidemiology", "AI for disease prediction", "AI-augmented bionics", 
    "bioinformatics AI", "AI in biomedical imaging",

    # AI for Scientific Research & Physics Simulations
    "AI in quantum computing", "AI for theoretical physics", 
    "AI in high-energy particle physics", "AI in astrophysics", "AI in cosmology", 
    "AI in material science", "AI in climate science", "AI in weather forecasting", 
    "AI in computational fluid dynamics", "AI for CFD", "AI in fusion research", 
    "AI in space exploration", "AI in molecular simulations", "AI in dark matter research", 
    "AI in dark energy research", "AI in gravitational wave detection",

    # AI in Advanced Computing & Hardware Optimization
    "AI-optimized data centers", "AI-designed semiconductors", 
    "AI in neuromorphic computing", "neuromorphic AI", "AI in FPGA optimization", 
    "AI for optical computing", "AI in 3D chip design", "AI for quantum algorithms",

    # AI in Education & Programming
    "AI-generated code", "AI in software development", "AI in code optimization", 
    "AI-assisted debugging", "AI for automated code refactoring", "AI tutors", 
    "AI in personalized learning", "AI-driven curriculum design", 
    "AI in language learning", "AI-powered translation",

    # AI Improving AI (Advanced Concepts)
    "meta-learning", "self-taught AI", "AI designing AI", "AI-augmented AI research", 
    "AI self-replication", "self-evolving AI", "recursive AI development", 
    "transfer learning", "multi-agent reinforcement learning", "MARL", 
    "AI in theoretical computer science", "AI in complexity theory"]
TRENDING_KEYWORDS = [
    # Core AI & LLM Scaling
    "LLM", "large language model", "scaling laws", "sparse models", 
    "transformer efficiency", "neural network pruning", "model compression", 
    "parameter-efficient fine-tuning", "PEFT", "mixture of experts", "MoE", 
    "low-rank adaptation", "LoRA", "efficient training", "gradient accumulation", 
    "distributed training", "TPU optimization", "GPU optimization", 
    "adaptive computation", "RLHF", "reinforcement learning from human feedback", 
    "chain-of-thought", "CoT", "self-supervised learning", "continual learning", 
    "auto-regressive models", "zero-shot learning", "few-shot learning", 
    "memory-augmented neural networks", "attention mechanisms", 
    "transformer alternatives", "state space models", "SSM", "Mamba", 
    "retrieval-augmented generation", "RAG", "latent variable models", 
    "knowledge distillation", "contrastive learning", "tokenizer-free models", 
    "multi-modal AI", "multi-modal learning", "open source", "open-source", "open source model"
]

def score_paper(paper):
    """Assigns a score to an AI paper based on authorship, topic relevance, and clarity, 
       while tracking the breakdown of score components."""
    
    score = 0
    score_breakdown = {
        "Top Researcher Bonus": 0,
        "Trending Keywords Bonus": 0,
        "Recency Boost": 0,
        "Relevance Score": 0,
        "Insightfulness Score": 0,
        "Novelty Score": 0,
        "Impact Score": 0,
        "Final Score": 0
    }

    # 1️⃣ Boost if a top AI researcher is an author
    if any(author in TOP_RESEARCHERS for author in paper['authors']):
        score_breakdown["Top Researcher Bonus"] = 5
        score += 5  # High weight for top AI researchers

    # 2️⃣ Boost papers covering trending AI research topics
    for keyword in TRENDING_KEYWORDS:
        if re.search(rf"\b{keyword}\b", paper['title'], re.IGNORECASE) or \
           re.search(rf"\b{keyword}\b", paper['summary'], re.IGNORECASE):
            if score_breakdown["Trending Keywords Bonus"] < 12:
                score_breakdown["Trending Keywords Bonus"] += 3  # Accumulate topic relevance
                score += 3  # Medium weight for trending topics

    # 3️⃣ Boost newer papers (favor those in the last 2-3 weeks)
    try:
        today = datetime.now(pytz.utc)  # ✅ Ensure `today` is timezone-aware (UTC)
        
        # ✅ Convert `paper['published']` to a timezone-aware datetime (UTC)
        if isinstance(paper['published'], datetime):
            published_date = paper['published'].astimezone(pytz.utc)
        else:
            published_date = datetime.strptime(paper['published'], "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
        
        days_since_published = (today - published_date).days

        if days_since_published <= 100:
            recency_boost = math.log10(100/(days_since_published+1)) * 2
            score_breakdown["Recency Boost"] = recency_boost
            score += recency_boost
    except Exception as e:
        print(f"Error parsing date: {e}")

    # 4️⃣ Evaluate paper with Gemini 2.0 Flash
    gemini_scores = evaluate_paper_with_gemini(paper['title'], paper['summary'])

    if gemini_scores:
        score_breakdown["Relevance Score"] = gemini_scores["relevance"]
        score_breakdown["Insightfulness Score"] = gemini_scores["insightfulness"]
        score_breakdown["Novelty Score"] = gemini_scores["novelty"]
        score_breakdown["Impact Score"] = gemini_scores["impact"]

        # 5️⃣ Adjust score based on LLM evaluation
        llm_score = (
            gemini_scores["relevance"] * 1.2 +  
            gemini_scores["insightfulness"] * 1.3 +  
            gemini_scores["novelty"] * 1.5 +  
            gemini_scores["impact"] * 1.4  
        ) / 12  # Normalize scores
        
        score += llm_score

    score_breakdown["Final Score"] = score

    return score, score_breakdown


papers = fetch_ai_papers(200)

day_papers = papers[0]
recent_papers = papers[1]




day_papers = papers[0]
recent_papers = papers[1]


top_papers = get_top_papers(day_papers, 5)
top_recent_papers = get_top_papers(recent_papers, 5)

store_papers_in_db(day_papers, "daily")
store_papers_in_db(recent_papers, "recent")


# export_report(top_papers, "paper_report.md")

# export_report(top_recent_papers, "recent_paper_report.md")





visual_breakdown = False

if visual_breakdown:

    scoring_data = []

    for paper in papers:
        score, breakdown = score_paper(paper)
        breakdown["Title"] = paper["title"]
        scoring_data.append(breakdown)

    # Convert to DataFrame
    df_scoring = pd.DataFrame(scoring_data)

    # Display the DataFrame in a readable format
    print(df_scoring)

    # Visualization: Contribution Breakdown Per Paper
    plt.figure(figsize=(12, 6))
    df_scoring.set_index("Title")[["Top Researcher Bonus", "Trending Keywords Bonus", "Recency Boost",
                                "Relevance Score", "Insightfulness Score", "Novelty Score", "Impact Score"]].plot(
        kind="bar", stacked=True, figsize=(12, 6))
    plt.ylabel("Score Contribution")
    plt.title("Contribution of Factors to Paper Scores")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Scoring Factors", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y")

    plt.show()

    # Visualization 1: Distribution of Each Scoring Factor
    df_scoring_numeric = df_scoring.drop(columns=["Title", "Final Score"])

    plt.figure(figsize=(12, 6))
    df_scoring_numeric.plot(kind="box", figsize=(12, 6))
    plt.title("Distribution of Scoring Variables")
    plt.ylabel("Score Contribution")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.legend(title="Scoring Factors", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Visualization 2: Correlation Heatmap of Scoring Factors
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_scoring_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Scoring Factors")

    # Visualization 3: Contribution of Each Factor Across All Papers
    plt.figure(figsize=(12, 6))
    df_scoring_numeric.mean().plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Average Contribution to Score")
    plt.title("Average Contribution of Each Factor to Final Scores")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.legend(["Score Contribution"], bbox_to_anchor=(1.05, 1), loc="upper left")


    plt.show()