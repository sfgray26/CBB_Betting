# CBB Rating Sources: Evaluation for Third-Source Integration (G-R7)

## 1. Executive Summary
To replace the problematic EvanMiya rating source (due to Cloudflare blocking and fragile scraping), we evaluated four candidates: **Haslametrics**, **Massey Ratings**, **ESPN BPI**, and **Sagarin**. 

**Recommendation:** **Haslametrics** is the ideal replacement for an efficiency-based model. **Massey Ratings** is the best alternative if a "Consensus" anchor is preferred.

---

## 2. Comparative Analysis

| Source | Predictive Type | Scrapeability | Format | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Haslametrics** | Efficiency (Play-by-play) | **High** | HTML Table | **PRIMARY REPLACEMENT** |
| **Massey Ratings** | Consensus (Composite) | **High** | CSV / Tab / JSON | **SECONDARY / ANCHOR** |
| **ESPN BPI** | Power Index | **Low** | HTML Only | Avoid (Fragile) |
| **Sagarin** | Predictor / Elo | **Medium** | Fixed-Width Text | Avoid (Legacy format) |

---

## 3. Deep Dive: Top Candidates

### **A. Haslametrics (Erik Haslam)**
Haslametrics is widely considered part of the "Big Three" of modern CBB analytics alongside KenPom and BartTorvik.

*   **Predictive Performance:** Consistently top-tier. It is unique in that it uses **play-by-play data** to filter out "garbage time" (Analytically Final), leading to cleaner efficiency numbers.
*   **Accessibility:** Very high. The site hosts clean HTML tables at `haslametrics.com/ratings.php`. These can be parsed easily using `pandas.read_html()` or `BeautifulSoup`.
*   **Key Metrics:**
    *   **All-Play Percentage:** Expected win rate against all of D1 on a neutral court.
    *   **Efficiency Margin:** Similar to AdjEM, making it a "drop-in" replacement for EvanMiya in the current weighting scheme.
*   **Pros:** Modern, play-by-play based, very stable URL.
*   **Cons:** No official API; requires HTML table parsing.

### **B. Massey Ratings (Kenneth Massey)**
Massey is a "meta-rating" that aggregates multiple systems into a consensus rank.

*   **Predictive Performance:** Excellent. Aggregated models historically outperform individual models by smoothing out variance (Wisdom of the Crowds).
*   **Accessibility:** High. Provides an "Export" option that generates a clean CSV/Tab-delimited file. It also uses an internal JSON endpoint: `masseyratings.com/json/rate.php?argv={token}&task=json`.
*   **Key Metrics:**
    *   **Power Rating:** An ordinal or points-based ranking.
*   **Pros:** Extremely robust, CSV export native to the site, represents "market consensus."
*   **Cons:** Scale is points-based (e.g., 4000+), requiring a normalization step to convert to an AdjEM-like margin (-30 to +30).

---

## 4. Implementation Strategy (Roadmap)

### **Phase 1: Haslametrics Integration (Recommended)**
1.  **Scraper Development:** Use `requests` + `BeautifulSoup` to target `https://haslametrics.com/ratings.php`.
2.  **Parsing:** Extract the `All-Play %` or `Net Efficiency` columns.
3.  **Normalization:** Use the existing `normalize_team_name` service to map Haslam names (e.g., "NC State") to the DB standard.
4.  **Weighting:** Replace EvanMiya's 32.5% weight with Haslametrics.

### **Phase 2: Massey Ratings (Optional Anchor)**
If the system requires a "Truth" anchor to prevent model drift, Massey Ratings should be integrated using the CSV export feature to provide a consensus baseline.

---

## 5. Decision Log
*   **T-Rank (BartTorvik):** Noted as already integrated (Mission G-R7 correctly identified it, but it is currently Source #2).
*   **EvanMiya:** Confirmed as "Auto-Dropped" in `backend/services/ratings.py` due to persistent Cloudflare challenges.
