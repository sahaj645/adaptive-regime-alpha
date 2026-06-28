# Interview Pitch — Regime-Switching SPY Overlay

## Mindset (read this first)

"You don't know what you're doing" in quant almost always means one of four things:
hidden look-ahead, survivorship bias, overfitting mistaken for skill, or not knowing
whether a result is significant. Those are the four ways alpha research lies to you.
This project is built to catch all four. So: **lead with the rigor, own the modest
result, and never get defensive about the Sharpe.** A number you can defend beats a
better number you can't.

## The ~2-minute pitch (say this, in your own words)

> The brief was to combine the Absorption Ratio and a statistical jump model into a
> regime signal for SPY. I'll be upfront that I didn't treat this as "produce a winning
> backtest" — I treated it as "find out whether this signal carries real, tradeable
> information out-of-sample, and build the machinery to know whether I'm fooling
> myself." The easy thing here is a pretty equity curve; the hard thing is a number you
> can actually defend.
>
> Three things I was disciplined about. **Look-ahead**: it's a walk-forward with
> train-only fitting and next-day execution, and I don't just claim it's causal — I
> wrote a test that corrupts all future data and verifies every past signal is
> bit-for-bit unchanged. I also built a deliberately leaked version to measure what
> look-ahead is worth; it nearly doubled the Sharpe, and that's the number I refuse to
> report. **Data**: I rebuilt the cross-section point-in-time from the S&P 1500,
> survivorship-free, membership lagged a month, splits neutralized, validated to 0.997
> correlation with the market. **Evaluation**: every claim has a bootstrap confidence
> interval, and I ablated each component to see what actually carries the signal.
>
> What I found, honestly: the jump model carries the signal; the absorption ratio adds
> essentially nothing over it, across windows, weightings and granularities. More
> striking — the whole regime model is matched by a 200-day moving average and beaten by
> simple volatility targeting. I benchmarked my own model against trivial overlays and
> they win. And the edge over buy-and-hold isn't statistically significant; the interval
> straddles zero, because six years is effectively two bear markets. The one robust
> property is risk — drawdown and volatility roughly halved.
>
> So the number I'll defend is a Sharpe around 0.6 for the pure model, below
> buy-and-hold, with its interval — and about 0.9 if you add standard vol targeting,
> though I'd be the first to say that gain is the vol targeting, not my signal. What I'm
> really demonstrating isn't a strategy. It's that I know the four ways alpha research
> lies to you, I built the tools to catch each one, and I used them to talk myself out
> of my own backtest. That's the discipline I'd bring to your book.

## Your five competence signals (the evidence)

1. **No look-ahead, machine-proven** — perturb the future, past signals unchanged; plus a
   leak test quantifying the premium (Sharpe 0.76 → 1.45).
2. **Point-in-time, survivorship-free data** — monthly membership lagged a month, split
   neutralization, winsorization, validated 0.997 vs SPY.
3. **Honest ablation** — tested my own components; the absorption ratio adds no value
   (held in only 6/7 robustness configs); the model is matched by a 200-day MA.
4. **Trivial-baseline benchmarking** — vol-targeted buy-hold (0.93) and a 200-day trend
   (0.93) match or beat the regime model; I report it.
5. **Statistical honesty** — bootstrap CIs on every claim; ~2 regime cycles in the sample;
   a robust CV that did *not* rescue the penalty instability (proving it's a data limit).

## Numbers you defend vs numbers you won't

| Defend (with CI) | Won't claim (and why) |
|------------------|------------------------|
| ~0.62 Sharpe, pure model, below buy-hold | 0.99 — lucky single-fold CV |
| ~0.93 with vol targeting (credit the vol targeting) | 1.01 — oracle / test-set-selected penalty |
| Drawdown ~halved (robust, repeatable) | 1.45 — in-sample leak (look-ahead) |

## Tough questions, crisp answers

- **"Your Sharpe is below buy-and-hold — why care?"** The honesty is the point. I can tell
  you how much of any number is real, luck, or leakage. The repeatable result is a halved
  drawdown; the Sharpe edge isn't significant and I won't pretend it is.
- **"Did you have look-ahead?"** No — and I don't assert it, I test it: corrupt the future,
  the past is bit-identical.
- **"Isn't this just a moving average?"** Yes — a 200-day MA matches it, and I'm the one
  telling you, because I benchmarked against it.
- **"How would you improve it?"** Not by tuning — I proved that overfits (I walked the
  Sharpe 0.72 → 1.01 by selection alone). More regime cycles, or relaxing the binary
  mandate; vol targeting reaches ~0.93.
- **"What would you fix with more time?"** The sample. Six years can't support a tunable
  edge; the rest is engineering.

## One-liners that signal seniority

- "I built the tools to talk myself out of my own backtest."
- "My model is matched by a 200-day moving average — and I'm the one telling you that."
- "I can tell you how much of any number is real, luck, or leakage."
- "The effective sample is two bear markets; no amount of tuning fixes that."
- "I'd rather hand you a modest result I've stress-tested to its limits than a backtest I can't defend."
