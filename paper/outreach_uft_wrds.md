# WRDS access request — UofT librarian outreach

**Status**: drop-in-and-send. Mahimn copies into Gmail, fills the recipient line and any name details he learns at orientation, sends.

---

## Recipient

The two most likely correct addresses (pick one — Mahimn confirms via the
[UofT library research-help form](https://library.utoronto.ca/use/research-and-publish)
or by phone, **416-978-5589**, Monday–Friday 11 am–5 pm):

- **Map & Data Library (Robarts, 5th floor)** — handles dataset licensing
  and most non-business databases. Form intake: <https://mdl.library.utoronto.ca>.
- **Rotman Business Information Centre** — administers most WRDS subscriptions
  at UofT through the Rotman School of Management. Search the Rotman Library
  staff directory for the data/finance librarian (often listed under
  "Business Information" or "Research Services").

If unsure, send to both with a single body and a short header noting "I'm
unsure which office handles WRDS — please forward as appropriate."

## Subject line

> WRDS / OptionMetrics access request — incoming Economics student, pre-registered options-market research

## Email body

> Dear Map & Data Library team / Business Information Centre,
>
> I'm Mahimn Patel, an incoming first-year undergraduate accepting my UofT
> Economics offer for September 2026. I'm writing to ask about the workflow
> for requesting access to **WRDS (Wharton Research Data Services)**,
> specifically the **OptionMetrics IvyDB US** product, ahead of orientation.
>
> **Project context.** Over the past several months I've built an open
> research codebase, *Reflexivity in Options Markets* (GitHub:
> <https://github.com/mahimn01/reflexive-options>), studying how dealer-gamma
> feedback in the options market can drive endogenous volatility cycles
> in the underlying. The current v0.3.3 release contains a 30-page
> manuscript with four theorems — including a Hopf bifurcation calculus
> with closed-form first Lyapunov coefficient and a formal Hawkes-SV
> equivalence at the criticality boundary — together with a
> **pre-registered evaluation framework** that is anchored to a specific
> git commit hash via an OpenTimestamps Bitcoin-anchored proof. The paper
> is targeted at the NeurIPS 2026 GenAI Finance Workshop and ICAIF 2026
> conference. I would be happy to share the manuscript PDF and the
> pre-registration document for context.
>
> **Why I need WRDS.** The pre-registered Phase-4 empirical evaluation
> requires SPX index-options surface data over three pre-specified event
> windows: Volmageddon (around 5 February 2018), the COVID volatility
> shock (around 12 March 2020), and the Yen carry-trade unwind (around
> 5 August 2024). For each event I need ±60 trading days of:
>
> - implied-volatility surface across the locked 11 strikes × 7
>   maturities pre-registered grid;
> - per-strike-and-maturity open interest and trade volume;
> - end-of-day mid-quote prices.
>
> The total volume is roughly 360 trading days × 3 events × ~77 contracts
> per surface, well within the per-user OptionMetrics quota at typical
> WRDS subscriptions. I have read the data-use citation requirement and
> will cite OptionMetrics + WRDS in the published paper as required.
>
> **My ask.**
>
> 1. Is OptionMetrics IvyDB US included in UofT's current WRDS
>    subscription?
> 2. Are incoming undergraduates eligible to request a WRDS account, or
>    do I need to wait until I am formally registered in September? If
>    the latter, can I begin the application paperwork in advance so I
>    can start data collection promptly?
> 3. What is the standard onboarding workflow — required training, terms
>    of use, faculty sponsor, anything else I should prepare?
> 4. Are there usage caps (queries per day, GB per month) I should know
>    about so I can plan the data extraction within them?
>
> I would be glad to provide my admission letter, a brief CV, the
> manuscript PDF, or anything else useful. If a short call or in-person
> meeting at the Map & Data Library or the Rotman BIC would be easier
> than email, I am very happy to come in.
>
> Thank you for your time and for helping me get started on this work.
>
> Best regards,
> Mahimn Patel
> Incoming, Department of Economics, University of Toronto
> mahimn.patel.k@gmail.com
> +1 [phone, optional]

## Attachments to mention (don't pre-attach — share on response)

- `paper/main.pdf` (30-page manuscript at v0.3.3)
- `paper/pre_registration.md` and `paper/pre_registration.md.ots` (the
  pre-registration document and its OpenTimestamps proof)
- GitHub repository: <https://github.com/mahimn01/reflexive-options>
- (optional) UofT admission letter PDF
- (optional) one-page CV

## Send-readiness checklist

- [ ] Resolve the recipient: phoned 416-978-5589 OR confirmed via the
  research-help form OR sent to both MDL and Rotman BIC with the
  forwarding note.
- [ ] Confirmed the subject line is intact (Gmail sometimes clips long
  subject lines on mobile composers).
- [ ] Spell-checked the body, especially the institution / personal-name
  fields if a specific librarian name is added.
- [ ] Verified all three event-date wordings against `paper/event_windows.txt`
  (Volmageddon `2018-02-05`, COVID `2020-03-12`, Yen carry `2024-08-05`).
- [ ] Verified `paper/main.pdf` is the latest v0.3.3 build (not a stale
  copy) before offering to share it on a follow-up reply.
- [ ] Decided whether to include a phone number in the signature.
- [ ] CC: optional — your future academic advisor or the Economics
  Department's undergraduate office, if you have a contact there. A CC
  to a faculty member often unlocks WRDS faster, especially for new
  undergrads who don't yet have a formal "research project" supervisor.
- [ ] Consider scheduling the send for a weekday morning (UofT librarians
  triage email more responsively at the start of the workday).
- [ ] Save a copy of the sent email to `~/Documents/reflexivity-research/`
  for chain-of-custody.
- [ ] Add a follow-up reminder (~2 weeks) so the request doesn't go stale
  if there is no response.

## After access is granted

- Pull the OptionMetrics IvyDB US data into `data/optionmetrics/` (gitignored).
- Run the locked Phase-4 evaluation pipeline; the analysis code is already
  written and the pre-registration document at `paper/pre_registration.md`
  is anchored to commit `268c061` via OpenTimestamps.
- Any deviation from the locked spec must be disclosed in the paper as an
  exploratory analysis under the §9 deviations clause; the amendments file
  is closed at commit `63078f5`.
