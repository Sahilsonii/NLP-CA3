# LaTeX Report Instructions - Two Versions Available

## 📄 Two Report Formats

You now have **TWO LaTeX report versions** to choose from:

### 1. **Standard Academic Format** (Recommended)
- **File:** `NLP_CA3_Report.tex`
- **Format:** Traditional academic paper (single column, 12pt)
- **Pages:** ~6 pages
- **For:** General academic submissions, thesis chapters
- **✅ Use this unless specifically asked for JACoW format**

### 2. **JACoW Conference Format**
- **File:** `NLP_CA3_Report_JACoW.tex`
- **Format:** JACoW conference proceedings (two-column, 10pt)
- **Pages:** ~4 pages (more compact)
- **For:** Conference submissions (IPAC, FEL, etc.)
- **⚠️ Only use if submitting to JACoW conference**

---

## Quick Start (Standard Format)

### 1. Upload to Overleaf:
   - Go to [Overleaf](https://www.overleaf.com/)
   - Click "New Project" → "Upload Project"
   - Upload `NLP_CA3_Report.tex` file (standard format)
   - Upload all images from `../results/` folders

### 2. Required Images:
   Upload the entire **results folder** to your Overleaf project:
   - Upload the complete `results/` folder (it will maintain the structure)
   - Or create folders manually and upload images:
     ```
     results/plots/dataset_samples.png
     results/plots/dataset_preview.png
     results/plots/metrics_comparison.png
     results/plots/roc_curves.png
     results/confusion_matrices/lstm_cm.png
     ```

### 3. Customize Before Submission:
   - Replace `[Your Name]` with your actual name (line 38)
   - Replace `[Your Roll Number]` with your roll number (line 39)
   - Replace `[Your University]` with your institution (line 40)
   - Replace `[your-username]` with your GitHub username in URLs

### 4. Compile:
   - Click "Recompile" in Overleaf
   - Download PDF when ready

**Note:** Image paths are configured for the results folder to be at the same level as the .tex file (i.e., `results/plots/...`). If you organize differently, adjust paths in the document.

---

## JACoW Format Instructions

If you need the JACoW format:

### 1. Upload to Overleaf:
   - Use `NLP_CA3_Report_JACoW.tex` instead
   - Upload same images as above

### 2. Customize:
   - Line 26: Replace `[Your Name]` with your name
   - Line 26: Replace `[Your Roll Number]` with roll number
   - Line 26: Replace `[Your University], [City], [Country]`
   - Line 26: Replace email address
   - Bottom: Replace `[your-username]` with GitHub username

### 3. Compile:
   - Compiler: **LuaLaTeX** (select in Overleaf menu)
   - Download PDF

---

## Comparison Table

| Feature | Standard Format | JACoW Format |
|---------|----------------|--------------|
| Columns | Single | Two |
| Font Size | 12pt body | 10pt body |
| Pages | ~6 pages | ~4 pages |
| Title | 14pt normal | 14pt uppercase |
| References | 10 citations | 7 citations |
| Style | Academic paper | Conference paper |
| Compiler | pdfLaTeX | LuaLaTeX |

---

## Structure (Both Versions)

✓ Abstract
✓ Introduction (with dataset preview)
✓ Literature Review
✓ Methodology
✓ Results
✓ Conclusion
✓ GitHub Link
✓ References

## Included Visualizations

✓ Dataset Sample Messages
✓ Dataset Statistics & Overview
✓ Confusion Matrix (LSTM)
✓ Metrics Comparison Chart
✓ ROC Curves
✓ Performance Tables
✓ Benchmark Comparison

---

## Which Format Should I Use?

### Use **Standard Format** if:
- ✅ Submitting to your instructor/professor
- ✅ General academic assignment
- ✅ Thesis/dissertation chapter
- ✅ No specific format requirement mentioned

### Use **JACoW Format** if:
- Conference paper submission
- Specifically asked for JACoW format
- Submitting to IPAC, FEL, or other JACoW conferences

**When in doubt, use Standard Format!**

---

## Customization Tips

### To Add More Images:
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{path/to/image.png}
    \caption{Your caption here}
    \label{fig:label}
\end{figure}
```

### To Add Tables:
```latex
\begin{table}[H]
\centering
\caption{Table caption}
\begin{tabular}{lcc}
\toprule
Column 1 & Column 2 & Column 3 \\
\midrule
Data 1 & Data 2 & Data 3 \\
\bottomrule
\end{tabular}
\end{table}
```

### To Adjust Page Count:
If document exceeds required pages:
- Reduce figure sizes: `[width=0.8\textwidth]`
- Make tables more compact
- Condense sections
- Remove extra whitespace

---

## Export to Word (if needed)

If you need .docx instead of PDF:
1. Download PDF from Overleaf
2. Use online converter: [pdf2doc.com](https://pdf2doc.com)
3. Or use Pandoc: `pandoc report.tex -o report.docx`

---

## Troubleshooting

### Images Not Showing?
- Check image paths start with `../results/`
- Ensure images are uploaded to Overleaf
- Use PNG format (already done)

### Compilation Errors?
- **Standard format:** Use pdfLaTeX compiler
- **JACoW format:** Use LuaLaTeX compiler
- Check all curly braces { } are balanced

### Page Count Too High?
- Reduce image widths (0.9 instead of 1.0)
- Shorten Literature Review section
- Remove extra examples

---

## Contact Information

Add your contact info to the title page if required by your institution.

## Notes

- **Standard format:** ~6 pages, traditional academic style
- **JACoW format:** ~4 pages, conference proceedings style
- Both versions have all mandatory sections
- Professional academic formatting
- Ready for submission
- All visualizations included

Good luck with your submission! 🚀
