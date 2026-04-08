# Email to ClassMind Authors

**Recipient:** Ao Qu (lead author) — find via arxiv 2509.18020 "view email" link, or MIT DUSP directory
**Cc (suggested):** Jinhua Zhao, Paul Pu Liang
**Subject:** Inquiry about ClassMind code release — related work in multimodal classroom transcription

---

Dear Ao,

I'm Jennifer Kleiman, a researcher at the University of Georgia (College of Education / COMS group). I read your ClassMind paper (arxiv 2509.18020) with great interest — the AVA-Align framework and your approach to applying multimodal LLMs to classroom video are tackling problems I've been working on independently from a complementary angle.

I'm writing for two reasons.

**First**, my colleagues and I have been developing a multimodal video transcription pipeline for qualitative classroom research. Our system uses Gemini in a single-pass multimodal call (rather than splitting audio and visual processing) to produce interleaved speech and visual descriptions, with speaker diarization based on visual features. The goal differs from yours — we are aiming at producing research-grade transcripts to support qualitative analysis of student learning, rather than rubric-aligned teacher feedback — but the underlying technical problems (long-video chunking, hallucination control, speaker consistency) overlap substantially with what your paper addresses. We're preparing a methods paper and ClassMind is the closest related work we've identified.

**Second**, your paper describes ClassMind as an "open-source platform" but I wasn't able to locate a code repository or project page in the manuscript or via searching. Could you share whether the AVA-Align implementation is publicly available, or when you plan to release it? We would like to include ClassMind as a baseline in our evaluation, both to give your work proper credit and to enable a fair architectural comparison (single-pass multimodal vs. split-pipeline-with-merge).

If a public release isn't yet possible, we would still be very grateful if you could share enough implementation detail that we could reproduce a faithful baseline — even at the level of the prompts you use for the captioning step, or your sentence-segmentation procedure for merging Whisper transcripts with Gemini-2.5-Flash captions.

Either way, I'd be glad to send you a draft of our paper before submission so you can confirm we are characterizing your work accurately, and so we can cite any updates or follow-up work you have in progress.

Thank you for the careful work you and your co-authors did on this. I think the field needs both of our angles, and I would welcome any chance to compare notes.

Best regards,

Jennifer Kleiman
University of Georgia, COMS
[email]
[institutional page]

---

## Notes for Jennifer
- Tone: collegial, specific, gives them an out (no public release), offers a courtesy review of how we describe their work
- Asks for two concrete things: (1) code or (2) implementation detail sufficient to reproduce
- Doesn't promise anything beyond proper citation
- Mentioning Paul Liang (MIT MAS) is optional — he's the most prominent name on the paper and may help the email get attention
- If no response in ~10 days, follow up once. After that, proceed with self-built baseline.
