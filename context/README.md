# context/  – your private knowledge files (not committed)

Drop your files here. Supported formats: `.docx`, `.pdf`, `.txt`, `.md`.

| What                          | Name it like…                                   | Category |
|-------------------------------|-------------------------------------------------|----------|
| Interview questions & answers | `QA_*.docx`, `Interview Questions.docx`, `Q&A…` | **QA**   |
| HEAT / training notes         | `HEAT_*.pdf`, `Hostile Environment Training…`   | **HEAT** |
| Your CV                       | `CV_*.pdf`, `Abujami_CV.docx`, `resume…`        | CV       |
| Job description               | `JD_*.docx`, `Job description…`                 | JD       |
| Anything else                 | `extra_*.txt`, other names                      | EXTRA    |

Matching is case-insensitive and looks at any part of the file name.
Answers are searched **QA first, then HEAT/EXTRA, then Gemini**.

Check how your files were parsed:

```bash
python knowledge_base.py context
```
