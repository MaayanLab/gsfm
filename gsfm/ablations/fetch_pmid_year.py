# Credit to ChatGPT

from Bio import Entrez
import csv
import time
import os

# REQUIRED: replace with your email
Entrez.email = os.environ.get('ENTREZ_EMAIL')
# Optional: speeds up requests (get it from your NCBI account)
Entrez.api_key = os.environ.get('ENTREZ_API_KEY')

OUTPUT_FILE = "data/pmid_year.csv"

def fetch_pubmed_years(pmids):
    """Fetch publication years for a batch of PMIDs."""
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml")
    records = Entrez.read(handle)
    handle.close()

    result = {}
    for article in records["PubmedArticle"]:
        pmid = str(article["MedlineCitation"]["PMID"])
        pubdate = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]
        year = pubdate.get("Year")

        # fallback if only MedlineDate exists (e.g. "1998 Jan-Feb")
        if not year and "MedlineDate" in pubdate:
            year = pubdate["MedlineDate"].split(" ")[0]

        if year:
            result[pmid] = year
    return result

def bulk_fetch(pmids, batch_size=200, sleep_time=0.12):
    """Fetch in batches with resume support."""
    # Resume: load already processed PMIDs from CSV if exists
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            next(f)  # skip header
            for line in f:
                pmid, _ = line.strip().split(",")
                processed.add(pmid)
        print(f"Resuming: {len(processed)} PMIDs already processed")

    # Open CSV in append mode if resuming, else write header
    mode = "a" if processed else "w"
    with open(OUTPUT_FILE, mode, newline="") as f:
        writer = csv.writer(f)
        if not processed:
            writer.writerow(["PMID", "Year"])

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            # Skip if this batch is already done
            if all(p in processed for p in batch):
                continue

            print(f"Fetching {i+1}–{i+len(batch)} / {len(pmids)}")
            try:
                data = fetch_pubmed_years(batch)
                for pmid, year in data.items():
                    writer.writerow([pmid, year])
                    processed.add(pmid)
                f.flush()  # write to disk immediately (safety)
            except Exception as e:
                print(f"⚠️ Error on batch {i//batch_size}: {e}")

            time.sleep(sleep_time)  # respect NCBI limits

if __name__ == "__main__":
    # Load your PMID list from file
    with open("data/pmids.txt") as f:
        pmids = [line.strip() for line in f if line.strip()]

    bulk_fetch(pmids)
    print(f"✅ Done! Results saved to {OUTPUT_FILE}")
