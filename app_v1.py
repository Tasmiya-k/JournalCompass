import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urlparse, urljoin
import time
import os
from datetime import datetime
import json
import subprocess

# Set page configuration
st.set_page_config(
    page_title="Journal Information Scraper",
    page_icon="📚",
    layout="wide"
)

def get_domain(url):
    """Extract domain from URL"""
    parsed_uri = urlparse(url)
    return '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)

def is_valid_url(url, base_domain):
    """Check if URL is valid and belongs to the same domain"""
    if not url:
        return False
    if url.startswith('#') or url.startswith('javascript:'):
        return False
    if url.startswith('http'):
        return base_domain in url
    return True

def clean_text(text):
    """Clean text by removing extra spaces and newlines"""
    if text:
        return re.sub(r'\s+', ' ', text).strip()
    return ""

def extract_dates(text):
    """Extract dates from text"""
    # Common date formats: DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, Month DD, YYYY
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return dates

def extract_costs(text):
    """Extract publication costs from text"""
    # Look for money amounts
    cost_patterns = [
        r'\$\s*\d+(?:,\d+)*(?:\.\d+)?',
        r'USD\s*\d+(?:,\d+)*(?:\.\d+)?',
        r'€\s*\d+(?:,\d+)*(?:\.\d+)?',
        r'EUR\s*\d+(?:,\d+)*(?:\.\d+)?',
        r'£\s*\d+(?:,\d+)*(?:\.\d+)?',
        r'GBP\s*\d+(?:,\d+)*(?:\.\d+)?',
        r'publication\s+fee\s+(?:of\s+)?(?:is\s+)?(?:USD|EUR|GBP|\$|€|£)?\s*\d+(?:,\d+)*(?:\.\d+)?'
    ]
    
    costs = []
    for pattern in cost_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        costs.extend(matches)
    
    return costs

def extract_topics(text):
    """Extract potential research topics from text"""
    # Look for common topic list patterns
    if "topics include" in text.lower() or "research areas" in text.lower() or "subject areas" in text.lower():
        # Try to find lists
        list_patterns = [
            r'(?:topics include|research areas|subject areas):?\s*((?:[^.]*?(?:,|;|\band\b)[^.]*)+)',
            r'(?:<li>.*?</li>)+',  # HTML list items
            r'(?:^|\n)\s*[-•*]\s+.*?(?=\n\s*[-•*]|\n\n|\Z)',  # Bullet points
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Clean and split topics
                topics = []
                for match in matches:
                    if '<li>' in match:
                        # Extract from HTML list
                        soup = BeautifulSoup(match, 'html.parser')
                        items = soup.find_all('li')
                        topics.extend([item.get_text().strip() for item in items])
                    else:
                        # Split by comma or semicolon
                        split_topics = re.split(r',|;|\band\b', match)
                        topics.extend([t.strip() for t in split_topics if t.strip()])
                return topics
    
    return []

def extract_journal_type(text):
    """Extract journal type information"""
    type_patterns = [
        r'open\s+access',
        r'subscription\s+based',
        r'peer\s+reviewed',
        r'quarterly',
        r'monthly',
        r'bimonthly',
        r'annual',
        r'biannual',
        r'hybrid\s+journal'
    ]
    
    types = []
    for pattern in type_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            types.append(re.search(pattern, text, re.IGNORECASE).group())
    
    return types

def analyze_with_llama(journal_name, all_text):
    """Use Ollama's Llama model to analyze the scraped text"""
    st.write("Analyzing scraped content with Llama 3.2...")
    
    # Define the prompt for Llama
    prompt = f"""
You are analyzing information about the academic journal "{journal_name}". 
As an expert academic advisor, extract the most important information for researchers who want to publish in this journal.

Focus on:
1. Submission guidelines and key requirements
2. Publication fees and any fee waiver policies
3. Acceptance rate and review process details
4. Impact factor and journal ranking information
5. Special issue opportunities
6. Preferred article types and formatting requirements
7. Author guidelines that might improve chances of acceptance
8. Word/page limits
9. Citation style and preferences
10. Editorial board information if notable

Your task is to extract only the most reliable and factual information from the provided text. 
Organize your response into clear sections, and provide specific details when available.
If certain important information appears to be missing, indicate this.

Here is the text from the journal website:

{all_text}
"""

    try:
        # Call Ollama using subprocess with stdin instead of file
        process = subprocess.Popen(
            ["ollama", "run", "llama3.2"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the prompt to the process
        stdout, stderr = process.communicate(input=prompt)
        
        if process.returncode != 0:
            st.error(f"Error calling Llama model: {stderr}")
            return "LLM analysis failed. Please check your Ollama installation and try again."
        
        print(stdout)
        return stdout
    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return "An unexpected error occurred during LLM analysis."

def scrape_journal(journal_name, journal_url, max_pages=50, delay=1):
    """Main function to scrape journal information"""
    base_domain = get_domain(journal_url)
    visited_urls = set()
    urls_to_visit = [journal_url]
    page_count = 0
    
    all_text = ""
    important_info = {
        "Journal Name": journal_name,
        "Journal URL": journal_url,
        "Dates": [],
        "Costs": [],
        "Topics": [],
        "Journal Type": [],
        "Scraped Pages": []
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while urls_to_visit and page_count < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            page_count += 1
            
            status_text.text(f"Scraping page {page_count}/{max_pages}: {current_url}")
            progress_bar.progress(page_count / max_pages)
            
            try:
                response = requests.get(current_url, timeout=10)
                if response.status_code != 200:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                page_text = soup.get_text()
                all_text += page_text
                
                # Extract information from the current page
                important_info["Scraped Pages"].append(current_url)
                important_info["Dates"].extend(extract_dates(page_text))
                important_info["Costs"].extend(extract_costs(page_text))
                
                # Extract topics if not already found
                if not important_info["Topics"]:
                    page_topics = extract_topics(page_text)
                    if page_topics:
                        important_info["Topics"] = page_topics
                
                # Extract journal type if not already found
                page_types = extract_journal_type(page_text)
                if page_types:
                    important_info["Journal Type"].extend(page_types)
                
                # Find additional URLs to crawl
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    
                    if is_valid_url(full_url, base_domain) and full_url not in visited_urls and full_url not in urls_to_visit:
                        urls_to_visit.append(full_url)
                
                # Sleep to be respectful to the website
                time.sleep(delay)
            
            except Exception as e:
                st.error(f"Error scraping {current_url}: {str(e)}")
                continue
    
    except Exception as e:
        st.error(f"An error occurred during scraping: {str(e)}")
    
    # Remove duplicates
    important_info["Dates"] = list(set(important_info["Dates"]))
    important_info["Costs"] = list(set(important_info["Costs"]))
    important_info["Journal Type"] = list(set(important_info["Journal Type"]))
    
    return important_info, all_text

def categorize_dates(dates):
    """Attempt to categorize dates into submission, review, and publication dates"""
    categorized = {
        "Submission Deadlines": [],
        "Review Dates": [],
        "Publication Dates": [],
        "Other Dates": []
    }
    
    for date in dates:
        lower_context = date.lower()
        if "submit" in lower_context or "deadline" in lower_context:
            categorized["Submission Deadlines"].append(date)
        elif "review" in lower_context:
            categorized["Review Dates"].append(date)
        elif "publish" in lower_context or "publication" in lower_context:
            categorized["Publication Dates"].append(date)
        else:
            categorized["Other Dates"].append(date)
    
    return categorized

def save_to_file(data, filename):
    """Save scraped data to a file"""
    directory = "scraped_journals"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(data)
    
    return filepath

# Function to check if Ollama is available
def check_ollama_availability():
    try:
        subprocess.run(["ollama", "list"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Function to clean up scraped text for LLM analysis
def clean_for_llm(text, max_length=8000):
    """Clean and truncate text for LLM analysis"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]
        
    # Attempt to truncate at a sentence boundary
    last_period = text.rfind('.')
    if last_period > max_length * 0.9:  # Only if we're relatively close to the end
        text = text[:last_period+1]
        
    return text

# Main app interface
st.title("📚 Academic Journal Information Scraper with LLM Analysis")

st.markdown("""
This application scrapes academic journal websites to extract key information and uses a Llama 3.2 model to analyze and summarize the information most valuable for researchers looking to publish.

The app extracts:
- Important dates (submission deadlines, publication dates)
- Publication costs
- Research topics
- Journal type (open access, peer-reviewed, etc.)
- Comprehensive publishing guidelines and requirements

Enter a journal name and URL below to begin.
""")

col1, col2 = st.columns(2)

with col1:
    journal_name = st.text_input("Journal Name", placeholder="e.g., Nature Communications")
    
with col2:
    journal_url = st.text_input("Journal URL", placeholder="https://www.nature.com/ncomms/")

with st.expander("Advanced Settings"):
    max_pages = st.slider("Maximum Pages to Scrape", min_value=1, max_value=100, value=20, 
                         help="Set the maximum number of pages to scrape from the journal website")
    delay = st.slider("Delay Between Requests (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                     help="Set a delay between requests to be respectful to the website's resources")
    use_llm = st.checkbox("Use Llama 3.2 Analysis", value=True, 
                        help="Use the local Llama 3.2 model to analyze the scraped content")

# Check for Ollama availability at startup
ollama_available = check_ollama_availability()
if not ollama_available and use_llm:
    st.warning("Ollama not detected or Llama 3.2 model not available. LLM analysis will be disabled.")
    use_llm = False

if st.button("Start Scraping", type="primary"):
    if not journal_name or not journal_url:
        st.error("Please enter both journal name and URL")
    else:
        with st.spinner("Scraping journal information..."):
            start_time = time.time()
            
            # Perform the scraping
            scraped_info, all_text = scrape_journal(journal_name, journal_url, max_pages, delay)
            
            # Process the dates
            categorized_dates = categorize_dates(scraped_info["Dates"])
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            st.success(f"Scraping completed in {elapsed_time:.2f} seconds!")
            
            # LLM Analysis
            llm_analysis = None
            if use_llm and ollama_available:
                with st.spinner("Analyzing with Llama 3.2..."):
                    # Clean and prepare text for LLM
                    cleaned_text = clean_for_llm(all_text)
                    llm_analysis = analyze_with_llama(journal_name, cleaned_text)
            
            # Display results
            st.header("Scraped Journal Information")
            
            if llm_analysis and use_llm:
                st.subheader("💡 LLM Analysis for Researchers")
                print(llm_analysis)
                st.markdown(llm_analysis)
                
                # Add a download button for the LLM analysis
                if st.button("Save LLM Analysis"):
                    filename = f"{journal_name.replace(' ', '_').lower()}_llm_analysis.txt"
                    filepath = save_to_file(llm_analysis, filename)
                    st.success(f"LLM analysis saved to {filepath}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Information")
                st.write(f"**Journal Name:** {scraped_info['Journal Name']}")
                st.write(f"**Journal URL:** {scraped_info['Journal URL']}")
                st.write(f"**Pages Scraped:** {len(scraped_info['Scraped Pages'])}")
            
            with col2:
                st.subheader("Journal Type")
                if scraped_info["Journal Type"]:
                    for j_type in scraped_info["Journal Type"]:
                        st.write(f"- {j_type}")
                else:
                    st.write("No specific journal type information found")
            
            st.subheader("Important Dates")
            date_tabs = st.tabs(["Submission Deadlines", "Review Dates", "Publication Dates", "Other Dates"])
            
            with date_tabs[0]:
                if categorized_dates["Submission Deadlines"]:
                    for date in categorized_dates["Submission Deadlines"]:
                        st.write(f"- {date}")
                else:
                    st.write("No submission deadlines found")
            
            with date_tabs[1]:
                if categorized_dates["Review Dates"]:
                    for date in categorized_dates["Review Dates"]:
                        st.write(f"- {date}")
                else:
                    st.write("No review dates found")
            
            with date_tabs[2]:
                if categorized_dates["Publication Dates"]:
                    for date in categorized_dates["Publication Dates"]:
                        st.write(f"- {date}")
                else:
                    st.write("No publication dates found")
            
            with date_tabs[3]:
                if categorized_dates["Other Dates"]:
                    for date in categorized_dates["Other Dates"]:
                        st.write(f"- {date}")
                else:
                    st.write("No other dates found")
            
            st.subheader("Publication Costs")
            if scraped_info["Costs"]:
                for cost in scraped_info["Costs"]:
                    st.write(f"- {cost}")
            else:
                st.write("No publication cost information found")
            
            st.subheader("Research Topics")
            if scraped_info["Topics"]:
                for topic in scraped_info["Topics"]:
                    st.write(f"- {topic}")
            else:
                st.write("No specific research topics found")
            
            # Export options
            st.header("Export Data")
            
            # Save structured data
            structured_data = f"""
Journal Information Report
==========================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Basic Information:
-----------------
Journal Name: {scraped_info['Journal Name']}
Journal URL: {scraped_info['Journal URL']}
Pages Scraped: {len(scraped_info['Scraped Pages'])}

Journal Type:
------------
{chr(10).join([f"- {j_type}" for j_type in scraped_info['Journal Type']]) if scraped_info['Journal Type'] else "No specific journal type information found"}

Important Dates:
--------------
Submission Deadlines:
{chr(10).join([f"- {date}" for date in categorized_dates['Submission Deadlines']]) if categorized_dates['Submission Deadlines'] else "No submission deadlines found"}

Review Dates:
{chr(10).join([f"- {date}" for date in categorized_dates['Review Dates']]) if categorized_dates['Review Dates'] else "No review dates found"}

Publication Dates:
{chr(10).join([f"- {date}" for date in categorized_dates['Publication Dates']]) if categorized_dates['Publication Dates'] else "No publication dates found"}

Other Dates:
{chr(10).join([f"- {date}" for date in categorized_dates['Other Dates']]) if categorized_dates['Other Dates'] else "No other dates found"}

Publication Costs:
----------------
{chr(10).join([f"- {cost}" for cost in scraped_info['Costs']]) if scraped_info['Costs'] else "No publication cost information found"}

Research Topics:
--------------
{chr(10).join([f"- {topic}" for topic in scraped_info['Topics']]) if scraped_info['Topics'] else "No specific research topics found"}

Scraped Pages:
------------
{chr(10).join([f"- {page}" for page in scraped_info['Scraped Pages']])}
"""

            if llm_analysis and use_llm:
                structured_data += f"""

LLM Analysis for Researchers:
---------------------------
{llm_analysis}
"""
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Save Structured Report"):
                    filename = f"{journal_name.replace(' ', '_').lower()}_report.txt"
                    filepath = save_to_file(structured_data, filename)
                    st.success(f"Report saved to {filepath}")
            
            with col2:
                if st.button("Save Raw Text"):
                    filename = f"{journal_name.replace(' ', '_').lower()}_raw_text.txt"
                    filepath = save_to_file(all_text, filename)
                    st.success(f"Raw text saved to {filepath}")
                    
            with col3:
                if st.button("Save as JSON"):
                    json_data = {
                        "journal_name": scraped_info['Journal Name'],
                        "journal_url": scraped_info['Journal URL'],
                        "journal_type": scraped_info['Journal Type'],
                        "dates": {
                            "submission": categorized_dates['Submission Deadlines'],
                            "review": categorized_dates['Review Dates'],
                            "publication": categorized_dates['Publication Dates'],
                            "other": categorized_dates['Other Dates']
                        },
                        "costs": scraped_info['Costs'],
                        "topics": scraped_info['Topics'],
                        "scraped_pages": scraped_info['Scraped Pages']
                    }
                    
                    if llm_analysis and use_llm:
                        json_data["llm_analysis"] = llm_analysis
                        
                    filename = f"{journal_name.replace(' ', '_').lower()}_data.json"
                    filepath = save_to_file(json.dumps(json_data, indent=4), filename)
                    st.success(f"JSON data saved to {filepath}")
            
            # Display all scraped pages
            with st.expander("View All Scraped Pages"):
                st.write("The following pages were scraped:")
                for page in scraped_info['Scraped Pages']:
                    st.write(f"- {page}")

st.markdown("""
### How to Use This Tool

1. Enter the journal name and URL in the fields above
2. Adjust the advanced settings if needed
3. Click "Start Scraping" to begin the process
4. Review the extracted information and LLM analysis
5. Save the reports for further reference

### Notes

- The LLM analysis uses your local Llama 3.2 model to provide insights specifically relevant to researchers
- This tool respects website resources by adding a delay between requests
- Some websites may have terms of service that prohibit scraping - always check before using
""")

st.sidebar.header("About")
st.sidebar.markdown("""
This application helps researchers and academics gather important information about journals quickly and efficiently.

**Features:**
- Extracts key dates, costs, and topics
- Uses Llama 3.2 to analyze content for publication insights
- Identifies specific requirements for authors
- Provides comprehensive reports for researchers

Developed as an advanced research tool for academic publishing.
""")