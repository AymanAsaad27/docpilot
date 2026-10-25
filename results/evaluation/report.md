# DocPilot Evaluation Report

Generated: 2026-10-23T10:47:58.203916+00:00

## Overall results

| Metric | Result |
|---|---:|
| Total Questions | 30 |
| Successful Runs | 30 |
| Failed Runs | 0 |
| Source Recall | 0.9464 |
| Expected Source Hit Rate | 1.0 |
| Citation Validity Rate | 1.0 |
| Answerable Questions With Citations | 0.8571 |
| Answerable Questions With Valid Citations | 0.8571 |
| Unanswerable Abstention Rate | 1.0 |
| Answerability Match Rate | 1.0 |
| Mean Faithfulness | 0.9788 |
| Mean Relevance | 0.9467 |
| Mean Correctness | 0.9733 |
| Mean Latency Seconds | 102.9422 |
| Mean Retries | 0.1333 |
| Mean Regenerations | 0.0333 |

## Per-question results

| ID | Category | Source recall | Citation valid | Abstention match | Faithfulness | Correctness | Latency (s) | Error |
|---|---|---:|---|---|---:|---:|---:|---|
| F001 | basics | 1.0 | True | True | 0.9230769230769231 | 1.0 | 106.047 |  |
| F002 | request_parameters | 0.5 | True | True | 1.0 | 1.0 | 101.447 |  |
| F003 | request_parameters | 1.0 | True | True | 1.0 | 1.0 | 170.808 |  |
| F004 | request_parameters | 1.0 | True | True | 1.0 | 1.0 | 73.514 |  |
| F005 | request_body | 1.0 | True | True | 1.0 | 1.0 | 115.583 |  |
| F006 | request_body | 1.0 | True | True | 1.0 | 1.0 | 101.935 |  |
| F007 | request_body | 1.0 | True | True | 0.9166666666666666 | 1.0 | 105.11 |  |
| F008 | request_body | 1.0 | True | True | 1.0 | 1.0 | 113.196 |  |
| F009 | responses | 1.0 | True | True | 1.0 | 1.0 | 115.659 |  |
| F010 | responses | 0.5 | True | True | 1.0 | 1.0 | 101.424 |  |
| F011 | errors | 1.0 | True | True | 1.0 | 1.0 | 111.115 |  |
| F012 | dependencies | 1.0 | True | True | 0.8333333333333334 | 1.0 | 98.832 |  |
| F013 | dependencies | 1.0 | True | True | 0.8461538461538461 | 1.0 | 83.128 |  |
| F014 | dependencies | 1.0 | True | True | 1.0 | 1.0 | 108.341 |  |
| F015 | dependencies | 1.0 | True | True | 1.0 | 1.0 | 109.438 |  |
| F016 | security | 1.0 | True | True | 1.0 | 1.0 | 122.434 |  |
| F017 | security | 1.0 | True | True | 1.0 | 1.0 | 135.471 |  |
| F018 | security | 1.0 | True | True | 1.0 | 1.0 | 105.296 |  |
| F019 | testing | 1.0 | True | True | 1.0 | 1.0 | 93.595 |  |
| F020 | testing | 1.0 | True | True | 1.0 | 1.0 | 100.277 |  |
| F021 | middleware | 1.0 | True | True | 1.0 | 1.0 | 107.142 |  |
| F022 | middleware | 1.0 | True | True | 1.0 | 1.0 | 98.994 |  |
| F023 | background_tasks | 1.0 | True | True | 0.8461538461538461 | 1.0 | 108.166 |  |
| F024 | applications | 1.0 | True | True | 1.0 | 1.0 | 89.323 |  |
| F025 | files | 1.0 | True | True | 1.0 | 1.0 | 87.931 |  |
| F026 | database | 1.0 | True | True | 1.0 | 1.0 | 103.048 |  |
| F027 | openapi | 0.5 | True | True | 1.0 | 1.0 | 83.7 |  |
| F028 | static_files | 1.0 | True | True | 1.0 | 1.0 | 94.848 |  |
| F029 | unanswerable | None | True | True | 1.0 | 0.2 | 69.883 |  |
| F030 | unanswerable | None | True | True | 1.0 | 1.0 | 72.58 |  |

## Interpretation notes

- Source recall measures expected source-file overlap among returned passages.
- Citation validity checks whether numeric citation markers point to available passages; it does not prove the cited passage supports the claim.
- Abstention is detected using a phrase-based heuristic and requires manual review.
- Faithfulness, relevance, and correctness are model-judged scores, not independently verified ground truth.
- Review individual answers and errors before treating these results as a quality claim.
