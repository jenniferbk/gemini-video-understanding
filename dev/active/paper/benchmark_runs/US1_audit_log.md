# TIMSS gold correction log

- Source gold: `Math US1 transcript.txt`
- Audit: `review_US1_1775590429244.json`
- Corrections config: `gold_corrections_US1.yaml`
- Original gold turns: **939**
- Reviewed turns: **19** (2%)
- Explicit corrections applied: **8**
- Corrected gold turns: **936** (removed 3)

## Verdict counts

- **gold**: 2
- **inaud**: 3
- **pred**: 14

## Changes applied

- **KEPT (metric artifact)** `00:00:00 T` — '(inaudible) four. One for each group of four. Okay? Or three. Okay?'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: "(inaudible) four. " was truly inaudible.  fine to skip. the discrepancy at the end was just due to arbitrary turn division but should't be counted as error*
- **KEPT (metric artifact)** `00:00:06 T` — "Okay guys, let's get started here. Okay, listen closely."  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: due to arbitrary turn division but should't be counted as error*
- **KEPT (metric artifact)** `00:00:11 T` — "To save a little time I won't bring out my overhead like you're usually accustomed to."  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: due to arbitrary turn division but should't be counted as error*
- **REMOVED (correction)** `00:00:14 SN` — 'Dear...'  
  *reason: Inaudible — no student speech at this moment (JK spot-check)*
- **REMOVED (correction)** `00:00:15 SN` — 'Thank you.'  
  *reason: Inaudible — no student speech (JK spot-check)*
- **KEPT (metric artifact)** `00:00:16 T` — 'Okay? Instead, let me just tell you.'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: due to arbitrary turn division but should't be counted as error*
- **KEPT (metric artifact)** `00:00:19 T` — "What I'm going to give you for each group is a little three-page packet. Okay?"  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: due to arbitrary turn division but should't be counted as error*
- **KEPT (metric artifact)** `00:00:26 T` — 'What I need for you to do right now is, every one of you get out a blank sheet of paper. Every one of you needs a blank sheet of paper.'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: due to arbitrary turn division but should't be counted as error*
- **KEPT (metric artifact)** `00:00:35 SN` — 'Thank you.'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: same but only 1 second off, shouldn't count as error*
- **EDITED (correction)** `00:00:38 S`  
  from: 'Ashley, can I have a piece of paper?'  
  to:   'Can I have a piece of paper?'  
  *reason: "Ashley" is inaudible in source audio; "can I have a piece of paper" is barely audible but present (JK spot-check)*
- **KEPT (confirmed correct)** `00:00:41 T` — 'Just put it right on the table back there, (inaudible).'  
  *verdict: gold-correct; pred had a genuine error*  
  *note: "man" is probably not what was said but its hard to make out*
- **KEPT (metric artifact)** `00:00:43 S` — 'Thank you.'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: both good? no difference?*
- **KEPT (confirmed correct)** `00:00:58 T` — "Thank you Sarah. Don't leave without it, all right."  
  *verdict: gold-correct; pred had a genuine error*  
  *note: "sarah" was correct.  but 'all right' and 'alright?' should be counted as the same, no difference*
- **REMOVED (correction)** `00:01:01 SN` — 'Oh, (inaudible).'  
  *reason: Inaudible — not discernible in audio (JK spot-check)*
- **KEPT (metric artifact)** `00:01:02 T` — 'Okay. Now- Here you go.'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: I did hear "Now here you go"*
- **KEPT (metric artifact)** `00:01:11 T` — 'All right.'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: all right' and 'alright?' should be counted as the same, no difference*
- **KEPT (metric artifact)** `00:01:13 T` — 'Okay, so... listen closely now. Everybody ready?'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: 1:13 in gold vs 1:15 in v10, different utterance cutoff but should be counted the same since it adds up to the same*
- **KEPT (metric artifact)** `00:01:18 S` — 'Yep.'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: 1:18 yep should match with v10 1:20 yep and girl-whitejacket shoudl be student, then no error*
- **KEPT (metric artifact)** `00:01:20 T` — 'Okay, everybody within your group of four or three, you need to work together. I know normally we work in pairs.'  
  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*  
  *note: again just slight timing difference but 1:20 gold = 1:21 plus 1:27 v10 so should be counted the same*
- **EDITED (correction)** `00:02:07 SN`  
  from: 'Y intercept and X intercept.'  
  to:   'Y intercept.'  
  *reason: "X intercept" was not actually said — gold over-claim (JK spot-check)*
- **EDITED (correction)** `00:03:48 SN`  
  from: 'Mr. Ormsby, you know, when you put, like, a one in here, would you go like, times-'  
  to:   'When you put, like, a one in here, would you go like, times-'  
  *reason: "Mr. Ormsby, you know" is not clearly audible; remainder is audible (JK spot-check)*
- **EDITED (correction)** `00:03:53 T`  
  from: "Ah, see, so now you've already forgotten, right? First of all zeros are our favorite number, right?"  
  to:   "Ah, see, so now you've already forgotten, right? First of all zero's our favorite number, right?"  
  *reason: Gold transcribed the contraction wrong: "zero's" not "zeros are" (JK spot-check)*
- **EDITED (correction)** `00:04:52 S`  
  from: 'You want to pick three as like-'  
  to:   'You want to pick threes? ... like-'  
  *reason: Gold mis-heard: student says "threes" not "three as"; "like" trailing off (JK spot-check)*
