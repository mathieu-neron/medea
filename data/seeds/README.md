# Seed channels

Two files seed the training data:

- `offenders.txt` — channels you've identified as AI-slop (label = 1)
- `controls.txt` — clean, human-made channels for comparison (label = 0)

## Format

One channel URL per line. Comments allowed with `#`. Blank lines OK.

Accepted forms:
```
https://www.youtube.com/@SomeChannel
https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxx
@SomeChannel
```

## Curation tips

**Offenders — pick channels with clear AI signals.** Diversify across:
- Fully AI-generated visuals (Sora/Runway/Veo style)
- AI voice over stock or stolen footage (the "history slop" pattern)
- AI image animation (Ken-Burns over Midjourney stills)
- Pure LLM-written scripts on real footage

5–10 to start; you can grow later.

**Controls — match domain, not style.**
Pick channels that talk about *the same topics* as your offenders but are clearly human (long-time creators, on-camera presence, recognizable voices). If your offenders are "history shorts" then your controls should also be history channels — otherwise the model learns "history" not "AI-ness."

5–10 to start, ideally roughly matched to offender domains.

## Channel-level splits

The training split is by channel, not by video. Don't put the same channel in both offenders and controls. Don't expect the model to generalize beyond the kinds of channels in this list — only as well as the diversity here.
