# WRDS access request — UofT librarian outreach

**Status**: draft only. Verify the current recipient, subscription, and eligibility rules
before sending; do not imply current enrolment, departmental membership, or faculty sponsorship.

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

> WRDS / OptionMetrics access question — incoming UofT undergraduate researcher

## Email body

> Dear Map & Data Library team / Business Information Centre,
>
> I'm Mahimn Patel, an incoming University of Toronto undergraduate for
> September 2026. I'm writing to ask about the workflow
> for requesting access to **WRDS (Wharton Research Data Services)**,
> specifically the **OptionMetrics IvyDB US** product, ahead of orientation.
>
> **Project context.** I have built an open research codebase (GitHub:
> <https://github.com/mahimn01/reflexive-options>), studying how dealer-gamma
> feedback could generate local volatility cycles in a reduced-form model.
> The current sole-authored manuscript presents a local mathematical
> possibility result, not an SPX calibration or a claim that public open
> interest reveals dealer positions. The repository also contains a
> timestamped, pre-extraction protocol for a separate sign-agnostic proxy
> analysis. No registered market dataset has yet been analyzed. I would be
> happy to share the manuscript and protocol for context.
>
> **Why I need WRDS.** The registered-horizon proxy analysis requires
> historical SPX option records from OptionMetrics, including:
>
> - option identifiers and contract terms;
> - end-of-day bids, asks, implied volatility, volume, and open interest;
> - the accompanying index, rate, and distribution inputs needed to apply
>   the protocol's filters and construct its book summaries.
>
> I will comply with the applicable licence, access, storage, citation, and
> non-redistribution requirements. Proprietary observations will not be
> committed to the public repository.
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
> Incoming undergraduate student, University of Toronto
> mahimn.patel.k@gmail.com
> +1 [phone, optional]

## Attachments to mention (don't pre-attach — share on response)

- `paper/main.pdf` (current 24-page v0.4.1 manuscript)
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
- [ ] Verified `paper/main.pdf` is the latest v0.4.1 build (not a stale
  copy) before offering to share it on a follow-up reply.
- [ ] Decided whether to include a phone number in the signature.
- [ ] Do not CC or name a faculty member unless that person has agreed to
  sponsor or advise the project.
- [ ] Consider scheduling the send for a weekday morning (UofT librarians
  triage email more responsively at the start of the workday).
- [ ] Save a copy of the sent email to `~/Documents/reflexivity-research/`
  for chain-of-custody.
- [ ] Add a follow-up reminder (~2 weeks) so the request doesn't go stale
  if there is no response.

## After access is granted

- Pull the OptionMetrics IvyDB US data into `data/optionmetrics/` (gitignored).
- Freeze and hash the complete option extraction before constructing outcomes,
  as required by the current protocol and Amendments A13--A16.
- Record any deviation from the registered specification before inspecting
  affected results, and label non-registered analyses exploratory.
