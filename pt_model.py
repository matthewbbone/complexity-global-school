## Portuguese Language Model ##
# This script trains and evaluates a Portuguese language model for
# detecting its two major dialects: European Portuguese and Brazilian Portuguese.
# It has three stages:
# 1. Data Preparation: Load and preprocess the dataset of matched European and Brazilian Portuguese sentences.
# 2. Model Training: Fine-tune a pre-trained Portuguese language model on the prepared dataset. Use CatBoost for binary classification.
# 3. Evaluation: Assess the model's performance on a validation set and output metrics

import os 
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
import polars as pl
import PythonTmx as tmx
import lxml.etree as etree
from tqdm import tqdm
from sentence_transformers import InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from scipy import stats

load_dotenv()

CACHE_DIR = os.getenv("CACHE")

def process_save_data():
    # Load and parse the TMX file
    tmx_file_path = CACHE_DIR + "/pt-pt_BR.tmx"
    
    print("Parsing TMX file...")
    # Parse the TMX file using lxml with iterparse for memory efficiency
    # Use recover=True to handle malformed XML more gracefully
    context = etree.iterparse(
        tmx_file_path, 
        events=('end',), 
        tag='tu',
        recover=True,
        encoding='utf-8'
    )
    
    # Pre-allocate lists for better performance
    pt_pt_texts = []
    pt_br_texts = []
    
    # Process in chunks for faster DataFrame creation
    chunk_size = 10000
    all_data = []
    
    print("Extracting translation pairs...")
    count = 0
    debug_count = 0
    for event, elem in tqdm(context, desc="Extracting pairs"):  # Adjust total based on expected number of <tu> elements
        debug_count += 1
        pt_pt_content = None
        pt_br_content = None
        
        # Find all tuv elements within this tu
        tuvs = elem.findall('.//tuv')
        
        # Debug: print first few to see what we're getting
        if debug_count <= 3:
            print(f"\nDebug TU #{debug_count}:")
            print(f"  Found {len(tuvs)} tuv elements")
        
        for tuv in tuvs:
            lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang')
            
            # Also try without namespace
            if not lang:
                lang = tuv.get('lang')
            
            # Debug output
            if debug_count <= 3:
                print(f"  Lang: {lang}")
            
            # Extract text from seg element
            seg = tuv.find('seg')
            if seg is not None and seg.text:
                text = seg.text.strip()
                
                if debug_count <= 3:
                    print(f"  Text: {text[:50]}...")
                
                # Match both pt and pt-PT for European Portuguese
                if lang in ("pt-PT", "pt"):
                    pt_pt_content = text
                # Match both pt-BR and pt_BR for Brazilian Portuguese
                elif lang in ("pt-BR", "pt_BR"):
                    pt_br_content = text
        
        # Only add pairs where both variants exist
        if pt_pt_content and pt_br_content:
            pt_pt_texts.append(pt_pt_content)
            pt_br_texts.append(pt_br_content)
            count += 1
            
            # Process in chunks to avoid memory issues
            if len(pt_pt_texts) >= chunk_size:
                # Create one row per pair with both pt-PT and pt-BR columns
                chunk_df = pl.DataFrame({
                    "id": list(range(count - len(pt_pt_texts), count)),
                    "pt-PT": pt_pt_texts,
                    "pt-BR": pt_br_texts
                })
                all_data.append(chunk_df)
                
                if len(all_data) % 10 == 0:
                    print(f"Processed {count} pairs...")
                
                pt_pt_texts = []
                pt_br_texts = []
        
        # Clear element to free memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    
    # Process remaining data
    if pt_pt_texts:
        chunk_df = pl.DataFrame({
            "id": list(range(count - len(pt_pt_texts), count)),
            "pt-PT": pt_pt_texts,
            "pt-BR": pt_br_texts
        })
        all_data.append(chunk_df)
    
    print(f"Total pairs extracted: {count}")
    
    # Check if we have any data
    if not all_data:
        print("ERROR: No valid translation pairs found in the TMX file!")
        print("Please check:")
        print("  1. The TMX file path is correct")
        print("  2. The file contains <tu> elements with both pt-PT and pt-BR variants")
        print("  3. The language codes in the file match 'pt-PT' and 'pt-BR'")
        return
    
    # Concatenate all chunks
    print("Combining data chunks...")
    data = pl.concat(all_data)
    
    # Save to parquet with compression
    print("Saving to parquet...")
    data.write_parquet(CACHE_DIR + "/pt_dialect_data.parquet", compression="zstd")
    print(f"Saved {len(data)} rows (translation pairs)")
    
def train_embedding_model():
    
    print("Loading data...")
    # Load the full dataset
    data = pl.read_parquet(CACHE_DIR + "/pt_dialect_data.parquet")
    
    # Sample a subset for training (adjust size based on your needs)
    print(f"Total pairs available: {len(data)}")
    sample_size = min(1_000_000, len(data))  # Use 1M pairs for training
    sample_df = data.sample(sample_size, seed=42)
    
    print(f"Using {len(sample_df)} pairs for training")
    
    # Split into train/validation/test
    train_size = int(0.9 * len(sample_df))
    val_size = int(0.1 * len(sample_df))
    
    train_df = sample_df[:train_size]
    val_df = sample_df[train_size:train_size + val_size]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Load the pre-trained model
    print("\nLoading pre-trained model: PORTULAN/serafim-100m-portuguese-pt-sentence-encoder")
    model = SentenceTransformer('PORTULAN/serafim-100m-portuguese-pt-sentence-encoder')

    # Create training examples
    # Goal: Train the model to distinguish between pt-PT and pt-BR dialects
    # Strategy: Same sentences in different dialects should be dissimilar (push apart)
    #           Same sentences in the same dialect should be similar (pull together)
    print("\nCreating training examples...")
    train_examples = []
    
    train_rows = train_df.to_dicts()  # Convert to list of dicts for easier access
    
    for i, row in enumerate(train_rows):
        # NEGATIVE pair: pt-PT and pt-BR versions of the SAME sentence (different dialects = dissimilar)
        train_examples.append(InputExample(texts=[row['pt-PT'], row['pt-BR']], label=0.0))
        
        # POSITIVE pair: pt-PT with another pt-PT (same dialect = similar)
        random_idx = random.randint(0, len(train_rows) - 1)
        if random_idx != i:  # Make sure it's a different sentence
            train_examples.append(InputExample(texts=[row['pt-PT'], train_rows[random_idx]['pt-PT']], label=1.0))
        
        # POSITIVE pair: pt-BR with another pt-BR (same dialect = similar)
        random_idx = random.randint(0, len(train_rows) - 1)
        if random_idx != i:  # Make sure it's a different sentence
            train_examples.append(InputExample(texts=[row['pt-BR'], train_rows[random_idx]['pt-BR']], label=1.0))
    
    print(f"Created {len(train_examples)} training examples")
    
    # Create validation evaluator
    print("\nCreating validation evaluator...")
    val_sentences1 = val_df['pt-PT'].to_list()
    val_sentences2 = val_df['pt-BR'].to_list()
    val_scores = [0.0] * len(val_df)  # Different dialects should be dissimilar
    
    evaluator = EmbeddingSimilarityEvaluator(
        val_sentences1, 
        val_sentences2, 
        val_scores,
        name='pt-dialect-validation'
    )
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Define loss function - CosineSimilarityLoss works well for this task
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Training parameters
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% warmup
    
    output_path = CACHE_DIR + "/pt-dialect-model"
    
    print(f"\nTraining model for {num_epochs} epochs...")
    print(f"Batch size: 16")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Output path: {output_path}")
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        evaluation_steps=1000,
        save_best_model=True,
        show_progress_bar=True
    )
    
    print(f"\nModel saved to: {output_path}")
    
    return model

def train_catboost_model():
    
    print("Loading data...")
    # Load the full dataset
    data = pl.read_parquet(CACHE_DIR + "/pt_dialect_data.parquet")
    
    # Sample a subset for training (adjust size based on your needs)
    print(f"Total pairs available: {len(data)}")
    sample_size = min(5_000_000, len(data))  # Use up to 5M pairs
    sample_df = data.sample(sample_size, seed=42)
    
    print(f"Using {len(sample_df)} pairs for classification")
    
    # Load the fine-tuned model
    model_path = CACHE_DIR + "/pt-dialect-model"
    print(f"\nLoading fine-tuned model from: {model_path}")
    model = SentenceTransformer(model_path)

    # Create dataset with both dialects
    # Each row will have: text, dialect_label (0 for pt-PT, 1 for pt-BR)
    print("\nPreparing data for classification...")
    texts = []
    labels = []
    
    for row in sample_df.iter_rows(named=True):
        # Add pt-PT example
        texts.append(row['pt-PT'])
        labels.append(0)  # 0 = European Portuguese
        
        # Add pt-BR example
        texts.append(row['pt-BR'])
        labels.append(1)  # 1 = Brazilian Portuguese
    
    print(f"Total samples: {len(texts)} ({len(texts)//2} from each dialect)")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Split into train/val/test (no dimensionality reduction - use embeddings directly)
    print("\nSplitting data into train/val/test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Train CatBoost classifier directly on 384-dimensional embeddings
    print("\nTraining CatBoost classifier on 384-dimensional embeddings...")
    catboost_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    
    catboost_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=100
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    y_pred = catboost_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['pt-PT', 'pt-BR']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted pt-PT  Predicted pt-BR")
    print(f"Actual pt-PT         {cm[0][0]:6d}         {cm[0][1]:6d}")
    print(f"Actual pt-BR         {cm[1][0]:6d}         {cm[1][1]:6d}")
    
    # Save the model
    catboost_model_path = CACHE_DIR + "/pt-dialect-catboost.cbm"
    
    print(f"\nSaving CatBoost model to: {catboost_model_path}")
    catboost_model.save_model(catboost_model_path)
    
    print("\nModel saved successfully!")
    
    return {
        'sentence_transformer': model,
        'catboost_classifier': catboost_model,
        'test_accuracy': accuracy
    }
    
def classify_news():
    """
    Test the trained embedding model + CatBoost classifier on out-of-sample news sentences.
    Load Brazilian and European Portuguese news, sample them, and evaluate classification accuracy.
    """
    print("\n" + "="*60)
    print("Testing on out-of-sample news sentences...")
    print("="*60)
    
    # Load news sentences
    br_news_path = CACHE_DIR + "/por-br_newscrawl_2011_1M-sentences.txt"
    pt_news_path = CACHE_DIR + "/por-pt_newscrawl_2011_1M-sentences.txt"
    
    print("\nLoading Brazilian Portuguese news...")
    with open(br_news_path, 'r', encoding='utf-8') as f:
        br_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(br_sentences)} Brazilian Portuguese sentences")
    
    print("\nLoading European Portuguese news...")
    with open(pt_news_path, 'r', encoding='utf-8') as f:
        pt_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(pt_sentences)} European Portuguese sentences")
    
    # Sample a large random subset for testing
    sample_size = min(1_000_000, len(br_sentences), len(pt_sentences))
    print(f"\nSampling {sample_size} sentences from each dialect...")
    
    random.seed(42)
    br_sample = random.sample(br_sentences, sample_size)
    pt_sample = random.sample(pt_sentences, sample_size)
    
    # Combine into test dataset
    test_texts = br_sample + pt_sample
    test_labels = [1] * len(br_sample) + [0] * len(pt_sample)  # 1 = pt-BR, 0 = pt-PT
    
    # Shuffle the test set
    combined = list(zip(test_texts, test_labels))
    random.shuffle(combined)
    test_texts, test_labels = zip(*combined)
    
    print(f"Total test samples: {len(test_texts)} ({len(br_sample)} pt-BR + {len(pt_sample)} pt-PT)")
    
    # Load the fine-tuned embedding model
    model_path = CACHE_DIR + "/pt-dialect-model"
    print(f"\nLoading fine-tuned embedding model from: {model_path}")
    embedding_model = SentenceTransformer(model_path)
    
    # Generate embeddings
    print("\nGenerating embeddings for test sentences...")
    test_embeddings = embedding_model.encode(
        list(test_texts), 
        show_progress_bar=True, 
        batch_size=32
    )
    print(f"Embeddings shape: {test_embeddings.shape}")
    
    # Load the CatBoost classifier
    catboost_path = CACHE_DIR + "/pt-dialect-catboost.cbm"
    print(f"\nLoading CatBoost classifier from: {catboost_path}")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(catboost_path)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = catboost_model.predict(test_embeddings)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    print("\n" + "="*60)
    print("NEWS CLASSIFICATION RESULTS")
    print("="*60)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(
        test_labels, 
        predictions, 
        target_names=['European Portuguese (pt-PT)', 'Brazilian Portuguese (pt-BR)'],
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, predictions)
    print(f"\n                    Predicted pt-PT  Predicted pt-BR")
    print(f"Actual pt-PT            {cm[0][0]:8d}         {cm[0][1]:8d}")
    print(f"Actual pt-BR            {cm[1][0]:8d}         {cm[1][1]:8d}")
    
    # Calculate per-class metrics
    pt_pt_precision = cm[0][0] / (cm[0][0] + cm[1][0]) if (cm[0][0] + cm[1][0]) > 0 else 0
    pt_pt_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    pt_br_precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    pt_br_recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    
    print(f"\nPer-Dialect Performance:")
    print(f"  European Portuguese (pt-PT):")
    print(f"    - Precision: {pt_pt_precision:.4f}")
    print(f"    - Recall:    {pt_pt_recall:.4f}")
    print(f"  Brazilian Portuguese (pt-BR):")
    print(f"    - Precision: {pt_br_precision:.4f}")
    print(f"    - Recall:    {pt_br_recall:.4f}")
    
    print("\n" + "="*60)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': test_labels,
        'confusion_matrix': cm
    }
    
def download_news_timeseries():
    """
    Download Portuguese news articles from Arquivo.pt (Portuguese Web Archive).
    Fetches articles from amensagem.pt and jn.pt across multiple years.
    Uses the Arquivo.pt Full-text Search API.
    """
    import requests
    from pathlib import Path
    import time
    from bs4 import BeautifulSoup
    import re
    
    print("="*60)
    print("Downloading Portuguese News from Arquivo.pt")
    print("="*60)
    
    # Create subdirectory for Arquivo.pt news
    news_dir = Path(CACHE_DIR) / "arquivo_pt_news"
    news_dir.mkdir(exist_ok=True)
    print(f"Saving to: {news_dir}")
    
    # API endpoint and rate limits
    api_endpoint = "https://arquivo.pt/textsearch"
    max_requests_per_minute = 200  # Stay well below 250 limit
    delay_between_requests = 60.0 / max_requests_per_minute  # seconds
    
    # News sources to download
    news_sites = [
        # {"name": "amensagem", "url": "amensagem.pt"},
        # {"name": "jn", "url": "jn.pt"},
        {"name": "expresso", "url": "expresso.pt"},
        {"name": "observador", "url": "observador.pt"},
        {"name": "abola", "url": "abola.pt"},
        {"name": "publico", "url": "publico.pt"},
        {"name": "visao", "url": "visao.sapo.pt"},
        {"name": "sol", "url": "sol.sapo.pt"},
        
    ]
    
    # Time periods to download (by year - sample a few years for testing)
    years = list(np.arange(2010, 2025))  # Recent years more likely to have data
    
    total_articles = 0
    
    for site in news_sites:
        site_name = site["name"]
        site_url = site["url"]
        
        print(f"\n{'='*60}")
        print(f"Downloading from {site_name} ({site_url})")
        print(f"{'='*60}")
        
        site_articles = []
        
        for year in tqdm(years, desc=f"Processing {site_name}"):
            # Define time range for the year
            from_date = f"{year}0101000000"
            to_date = f"{year}1231235959"
            
            # Search parameters
            params = {
                "q": "",  # Empty query to get all pages
                "siteSearch": site_url,
                "from": from_date,
                "to": to_date,
                "maxItems": 250,  # Maximum allowed per request
                "type": "html",
                "offset": 0,
                "prettyPrint": "false"
            }
            
            year_articles = []
            
            try:
                print(f"\n    Querying API for {year}...", flush=True)
                while True:
                    # Make API request
                    print(f"      Making API request (timeout=30s)...", end=' ', flush=True)
                    response = requests.get(api_endpoint, params=params, timeout=30)
                    print(f"Got response (status={response.status_code})", flush=True)
                    
                    if response.status_code == 429:
                        print(f"      Rate limit reached, waiting 60 seconds...")
                        time.sleep(60)
                        continue
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract articles
                    items = data.get("response_items", [])
                    print(f"      Found {len(items)} items in API response", flush=True)
                    
                    if not items:
                        print(f"      No items found, moving to next year", flush=True)
                        break
                    
                    # Extract text from each article
                    for idx, item in enumerate(items):
                        try:
                            # Get extracted text link
                            extracted_text_url = item.get("linkToExtractedText", "")
                            
                            if extracted_text_url:
                                print(f"      Article {idx+1}/{len(items)}: Fetching text...", end=' ', flush=True)
                                # Wait to respect rate limit
                                time.sleep(delay_between_requests)
                                
                                # Download extracted text
                                text_response = requests.get(extracted_text_url, timeout=10)
                                text_response.raise_for_status()
                                
                                article_text = text_response.text.strip()
                                print(f"Got {len(article_text)} chars", flush=True)
                                
                                # Only keep articles with substantial text
                                if len(article_text) > 250:
                                    article_info = {
                                        "text": article_text,
                                        "title": item.get("title", ""),
                                        "url": item.get("originalURL", ""),
                                        "timestamp": item.get("tstamp", ""),
                                        "year": year
                                    }
                                    year_articles.append(article_info)
                                    print(f"      ✓ Saved article (total: {len(year_articles)})", flush=True)
                                else:
                                    print(f"      ✗ Too short, skipped", flush=True)
                            else:
                                print(f"      Article {idx+1}/{len(items)}: No extracted text URL, skipping", flush=True)
                        
                        except Exception as e:
                            print(f"      ✗ Error on article {idx+1}: {str(e)[:50]}", flush=True)
                            continue
                    
                    # Check if there are more results
                    # if "next_page" in data and len(items) == params["maxItems"]:
                    #     params["offset"] += params["maxItems"]
                    # else:
                    #     break
                    
                    print(f"      Breaking after processing {len(items)} items (single batch mode)", flush=True)
                    break
                
                print(f"    ✓ Year {year}: Downloaded {len(year_articles)} articles total", flush=True)
                site_articles.extend(year_articles)
            
            except Exception as e:
                print(f"  Error downloading {year}: {e}")
                continue
        
        # Save articles for this site with year metadata
        if site_articles:
            output_file = news_dir / f"{site_name}_articles.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for article in site_articles:
                    # Clean and write article text with year prefix
                    cleaned_text = article["text"].replace('\n', ' ').replace('\r', ' ')
                    year = article.get("year", "unknown")
                    # Split into sentences (simple approach)
                    sentences = re.split(r'[.!?]+', cleaned_text)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 20:  # Only keep substantial sentences
                            # Write with year prefix: YEAR|sentence
                            f.write(f"{year}|{sentence}\n")
            
            print(f"\n  Saved {len(site_articles)} articles to {output_file}")
            total_articles += len(site_articles)
    
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"Total articles downloaded: {total_articles}")
    print(f"Saved to: {news_dir}")
    print("="*60)
    
    return news_dir

def analyze_news_timeseries():
    """
    Analyze Portuguese dialect in news articles from Arquivo.pt as a time series.
    Parses year information from sentences and shows dialect distribution trends over time.
    
    Expected format: YEAR|sentence (e.g., "2018|Esta é uma frase em português")
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import re
    from collections import defaultdict
    
    print("="*60)
    print("Analyzing Portuguese News Timeseries from Arquivo.pt")
    print("="*60)
    
    # Load the trained models
    model_path = CACHE_DIR + "/pt-dialect-model"
    catboost_path = CACHE_DIR + "/pt-dialect-catboost.cbm"
    
    print(f"\nLoading fine-tuned embedding model from: {model_path}")
    embedding_model = SentenceTransformer(model_path)
    
    print(f"Loading CatBoost classifier from: {catboost_path}")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(catboost_path)
    
    # Find Arquivo.pt articles
    news_dir = Path(CACHE_DIR) / "arquivo_pt_news"
    
    # Look for article files with year metadata
    article_files = sorted(list(news_dir.glob("*_articles.txt")))
    
    print(f"\nFound {len(article_files)} article files in {news_dir}")
    
    if not article_files:
        print(f"ERROR: No article files found in {news_dir}")
        print("Please run download_news_timeseries() first.")
        return
    
    # Load all sentences with year and source information
    print("\nLoading sentences with year metadata...")
    all_sentences = []
    all_years = []
    all_sources = []
    sentences_by_year = defaultdict(list)
    sentences_by_source = defaultdict(list)
    
    for file_path in article_files:
        site_name = file_path.stem.replace('_articles', '')
        print(f"\n  Loading from {site_name}...")
        
        site_count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: YEAR|sentence
                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        year_str, sentence = parts
                        try:
                            year = int(year_str)
                            if len(sentence) > 20:  # Keep substantial sentences
                                all_sentences.append(sentence)
                                all_years.append(year)
                                all_sources.append(site_name)
                                sentences_by_year[year].append(sentence)
                                site_count += 1
                        except ValueError:
                            # Skip malformed lines
                            pass
        
        print(f"    Loaded {site_count} sentences from {site_name}")
    
    if len(all_sentences) == 0:
        print("\nERROR: No sentences with year information found!")
        print("The file format should be: YEAR|sentence")
        print("Please ensure download_news_timeseries() saved the data correctly.")
        return
    
    print(f"\nTotal loaded: {len(all_sentences)} sentences across {len(sentences_by_year)} years")
    for year in sorted(sentences_by_year.keys()):
        print(f"  {year}: {len(sentences_by_year[year])} sentences")
    
    # Generate embeddings for all sentences
    print("\nGenerating embeddings for all sentences...")
    embeddings = embedding_model.encode(
        all_sentences, 
        show_progress_bar=True, 
        batch_size=32
    )
    
    # Predict probabilities for all sentences (get probability of pt-BR class)
    print("\nClassifying all sentences and extracting probabilities...")
    # predict_proba returns [prob_pt_pt, prob_pt_br] for each sentence
    probabilities = catboost_model.predict_proba(embeddings)
    # Extract pt-BR probability (class 1)
    pt_br_probs = probabilities[:, 1]
    
    # Also get binary predictions for overall statistics
    predictions = catboost_model.predict(embeddings)
    
    # Organize results by source and year - store probabilities
    source_year_results = defaultdict(lambda: defaultdict(lambda: {'probs': [], 'count': 0}))
    year_results = defaultdict(lambda: {'probs': [], 'count': 0})
    
    for source, year, prob in zip(all_sources, all_years, pt_br_probs):
        source_year_results[source][year]['probs'].append(prob)
        source_year_results[source][year]['count'] += 1
        year_results[year]['probs'].append(prob)
        year_results[year]['count'] += 1
    
    # Calculate statistics by year (filter out years with < 50 sentences)
    min_sentences_threshold = 50
    years_sorted = sorted(year_results.keys())
    pt_br_mean_probs = []
    
    print(f"\n{'='*60}")
    print("Results by Year (Mean PT-BR Probability)")
    print(f"{'='*60}")
    
    for year in years_sorted:
        total = year_results[year]['count']
        
        if total >= min_sentences_threshold:
            mean_prob = np.mean(year_results[year]['probs'])
            pt_br_mean_probs.append(mean_prob * 100)  # Convert to percentage for plotting
            
            print(f"{year}: Mean PT-BR Probability={mean_prob:.4f} ({mean_prob*100:.2f}%), "
                  f"N={total} sentences")
        else:
            # Not enough data for this year
            pt_br_mean_probs.append(np.nan)
            
            print(f"{year}: SKIPPED (only {total} sentences, need >={min_sentences_threshold})")
    
    # Overall statistics
    total_pt_pt = sum(predictions == 0)
    total_pt_br = sum(predictions == 1)
    
    print(f"\n{'='*60}")
    print("Overall Results")
    print(f"{'='*60}")
    print(f"Total sentences: {len(all_sentences)}")
    print(f"PT-PT (European): {total_pt_pt} ({total_pt_pt/len(all_sentences)*100:.2f}%)")
    print(f"PT-BR (Brazilian): {total_pt_br} ({total_pt_br/len(all_sentences)*100:.2f}%)")
    
    # Create timeseries visualization by source
    print("\n" + "="*60)
    print("Creating Timeseries Visualization (by source)")
    print("="*60)
    
    # Calculate percentages for each source
    sources = sorted(source_year_results.keys())
    colors = ['#A23B72', '#F18F01', '#2E86AB', '#06A77D', '#D65780', '#8B7AAF']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # Create figure with single time series plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # First plot the combined/aggregate line (all sources together) using mean probability
    # Also calculate 95% confidence intervals
    combined_br_mean_probs = []
    combined_br_ci_lower = []
    combined_br_ci_upper = []
    
    for year in years_sorted:
        total = year_results[year]['count']
        
        if total >= min_sentences_threshold:
            probs_array = np.array(year_results[year]['probs']) * 100  # Convert to percentage
            mean_prob = np.mean(probs_array)
            
            # Calculate 95% confidence interval using t-distribution
            # This accounts for sample size uncertainty
            sem = stats.sem(probs_array)  # Standard error of the mean
            ci = stats.t.interval(0.95, len(probs_array)-1, loc=mean_prob, scale=sem)
            
            combined_br_mean_probs.append(mean_prob)
            combined_br_ci_lower.append(ci[0])
            combined_br_ci_upper.append(ci[1])
        else:
            combined_br_mean_probs.append(np.nan)
            combined_br_ci_lower.append(np.nan)
            combined_br_ci_upper.append(np.nan)
    
    # Plot confidence interval as shaded region
    ax.fill_between(years_sorted, combined_br_ci_lower, combined_br_ci_upper, 
                     color='#000000', alpha=0.15, zorder=5, 
                     label='95% Confidence Interval')
    
    # Plot combined line with bold styling
    ax.plot(years_sorted, combined_br_mean_probs, 
            marker='*', linewidth=4, markersize=15,
            color='#000000', label='Combined (All Sources)', 
            alpha=0.9, zorder=10)  # zorder=10 makes it draw on top
    
    # Plot a line for each individual source
    for idx, source in enumerate(sources):
        source_br_mean_probs = []
        
        for year in years_sorted:
            total = source_year_results[source][year]['count']
            
            if total >= min_sentences_threshold:
                mean_prob = np.mean(source_year_results[source][year]['probs']) * 100  # Convert to percentage
                source_br_mean_probs.append(mean_prob)
            else:
                source_br_mean_probs.append(np.nan)
        
        # Plot line for this source
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(years_sorted, source_br_mean_probs, 
                marker=marker, linewidth=2, markersize=8,
                color=color, label=source, alpha=0.7, linestyle='--')
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Brazilian Portuguese Probability (%)', fontsize=14, fontweight='bold')
    ax.set_title('Arquivo.pt News: Mean PT-BR Probability Over Time\n(Bold = Combined, Dashed = Individual Sources)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', frameon=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    ax.set_xticks(years_sorted)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "figures/arquivo_pt_dialect_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTimeseries plot saved to: {output_path}")
    
    # Show the plot (optional, may not work in headless environment)
    # plt.show()
    
    # Create summary tables by source
    print("\n" + "="*80)
    print("Summary Table by Source")
    print("="*80)
    
    for source in sources:
        print(f"\n{source.upper()}:")
        print(f"{'Year':<8} {'Total':<8} {'Mean PT-BR Prob':<18} {'Status':<15}")
        print("-" * 55)
        
        for year in years_sorted:
            total = source_year_results[source][year]['count']
            
            if total >= min_sentences_threshold:
                mean_prob = np.mean(source_year_results[source][year]['probs'])
                status = "OK"
                print(f"{year:<8} {total:<8} {mean_prob:<18.4f} {status:<15}")
            elif total > 0:
                status = f"SKIP(<{min_sentences_threshold})"
                print(f"{year:<8} {total:<8} {'N/A':<18} {status:<15}")
    
    print("\n" + "="*80)
    print("Overall Summary (All Sources Combined)")
    print("="*80)
    print(f"{'Year':<8} {'Total':<10} {'Mean PT-BR Prob':<18} {'Status':<10}")
    print("-" * 50)
    for i, year in enumerate(years_sorted):
        total = year_results[year]['count']
        
        if total >= min_sentences_threshold:
            mean_prob = np.mean(year_results[year]['probs'])
            status = "OK"
            print(f"{year:<8} {total:<10} {mean_prob:<18.4f} {status:<10}")
        else:
            status = f"SKIP(<{min_sentences_threshold})"
            print(f"{year:<8} {total:<10} {'N/A':<18} {status:<10}")
    print("="*50)
    
    # Save yearly results to CSV
    yearly_df = pl.DataFrame({
        'year': years_sorted,
        'total_sentences': [year_results[y]['count'] for y in years_sorted],
        'mean_pt_br_probability': [np.mean(year_results[y]['probs']) if year_results[y]['count'] >= min_sentences_threshold else np.nan for y in years_sorted]
    })
    
    csv_path = "figures/arquivo_pt_dialect_timeseries.csv"
    yearly_df.write_csv(csv_path)
    print(f"\nYearly results saved to: {csv_path}")
    
    # Save sentence-level results with probabilities
    sentence_df = pl.DataFrame({
        'year': all_years,
        'source': all_sources,
        'sentence': all_sentences,
        'pt_br_probability': pt_br_probs,
        'predicted_dialect': ['pt-PT' if p == 0 else 'pt-BR' for p in predictions]
    })
    
    sentence_csv = "figures/arquivo_pt_sentences_classified.csv"
    sentence_df.write_csv(sentence_csv)
    print(f"Sentence-level results saved to: {sentence_csv}")
    
    print("\n" + "="*60)
    print("Time Series Analysis Complete!")
    print("="*60)
    
    return {
        'yearly_results': year_results,
        'years': years_sorted,
        'pt_br_mean_probs': pt_br_mean_probs,
        'total_sentences': len(all_sentences),
        'total_pt_pt': int(total_pt_pt),
        'total_pt_br': int(total_pt_br)
    }


def main():
    # Stage 1: Data preparation (uncomment if needed)
    # process_save_data()
    
    # Display data info
    data = pl.read_parquet(CACHE_DIR + "/pt_dialect_data.parquet")
    print("Dataset preview:")
    print(data.head())
    print(f"\nTotal translation pairs: {len(data)}")
    
    # Stage 2: Train the embedding model
    print("\n" + "="*60)
    print("Starting embedding model training...")
    print("="*60)
    trained_model = train_embedding_model()
    print("\n" + "="*60)
    print("Embedding model training complete!")
    print("="*60)
    
    # Stage 3: Train CatBoost classifier
    print("\n" + "="*60)
    print("Starting CatBoost classifier training...")
    print("="*60)
    results = train_catboost_model()
    
    # print("\n" + "="*60)
    print("All training complete!")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")
    print("="*60)
    
    # Stage 4: Test on out-of-sample news sentences from teh 2011 news-crawl corpus
    print("="*60)
    print("FINAL EVALUATION: Testing on News Corpus")
    print("="*60)
    news_results = classify_news()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"News corpus accuracy: {news_results['accuracy']:.4f} ({news_results['accuracy']*100:.2f}%)")
    print("="*60)
    
    # Stage 5: Download news from Arquivo.pt
    download_news_timeseries()
    
    # Stage 6: Analyze news timeseries - apply model to each year and visualize trends
    analyze_news_timeseries()


if __name__ == "__main__":
    main()