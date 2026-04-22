# FAQ (Visual Validation Rating)

## 1. The event is described but I can't find it in the video.

Rate **0 on factual accuracy** and **0 on temporal precision**. Leave a note describing what you looked for and where you looked. This is a hallucination and it's important to catch and document.

## 2. The description mentions a student I can't identify.

Rate on whether the action happened, not on whether the label is right. Speaker-ID accuracy is handled elsewhere in the paper; your job here is the visual event itself. Example: if the transcript says "S-BoyRed points at the worksheet" and *a* student pointed at the worksheet near that timestamp, that's factually accurate even if you can't confirm which specific student.

## 3. The description reads text from the board but I can't tell if the text is really there.

Pause and zoom if your player allows. If the text is genuinely unreadable from video quality, rate **1 on factual accuracy** (partially accurate, since we can't verify) and note "unverifiable from video."

## 4. Two descriptions seem to describe the same physical event.

The sampler should have filtered near-duplicates within 5 seconds, but if one slipped through, rate both independently and flag in the notes column. Don't try to pick the better one; that's not your call, and the data point matters either way.

## 5. I want to give a 1.5.

Round down. Note the ambivalence in the notes column. The 3-point scale is intentional (Gwet's argument about reliability at small N), but your narrative comments are where the nuance lives.

## 6. Video playback is laggy, or timestamps drift by a second or two.

Expected. The rubric's windows (5 seconds for "precise," 15 seconds for "proximate") are designed for exactly this kind of drift. Don't penalize small offsets that fall inside those windows.

## 7. The event is tagged as teacher-attributed but the actor looks like a student to me.

Rate **factual accuracy 1** (action right, agent wrong). Note it. This is the "agent confusion" failure mode from the protocol, and it's something the paper specifically reports on.

## 8. How long should this really take?

Plan 3–4 hours for the 45 events. Some take 30 seconds to rate. A few will take 5 minutes because you'll want to rewatch. That's normal and fine.

If you find yourself consistently at 10+ minutes per event, stop and ping me. Something about the protocol isn't working.
