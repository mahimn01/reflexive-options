# Amendment A15 — market-time alignment and registration-language correction

**Date:** 2026-07-10  
**Status:** Made before WRDS/OptionMetrics access and before any project extraction of the
registered option-chain, CRSP-return, or VIX series. This file supplements the preserved and
timestamped A13--A14 amendment record; it does not overwrite that record.

## Reason for the amendment

A clean-room pre-publication audit found two implementation ambiguities and one disclosure
problem. First, "next common day" could be implemented by compressing the data to dates on
which both OptionMetrics and CRSP are present. If an option-chain date were missing, that would
make the outcome an irregularly spaced return and would also compress the five- and 22-session
controls. Second, A14 attached weekday indicators to the regressor date even though their role is
to control calendar seasonality in the next-session outcome. Third, "full-sample" and "pre-data"
were too broad: the fixed 2017-01-03--2024-10-29 period is the registered historical horizon, not
all data that may be available in September 2026, and the historical market outcomes were public
before this analysis plan was written. The proprietary option-position proxies have not been
extracted, but the design is not blinded to the existence of the named historical episodes.

## A15.1 Complete market-calendar alignment

1. Build the ordered CRSP trading-session calendar first. Do not compress it to dates with a valid
   option chain, VIX observation, or proxy.
2. Construct CRSP log returns and the one-, five-, and 22-session outcome controls on that complete
   calendar. A missing option proxy is represented as missing on its actual session; it does not
   remove the session before leads or lags are constructed.
3. For a regressor session `t`, the primary outcome is the CRSP return for the immediately following
   CRSP trading session `t+1`. This remains true when the option proxy is missing on `t+1`.
4. A regression row is retained only when the proxy and VIX are observed on `t`, all registered
   return controls are available on their actual preceding CRSP sessions, the monthly-expiration
   indicator is defined, and the return is observed on the immediately following CRSP session.
   Every exclusion and reason is counted.
5. Automated tests must demonstrate that inserting a missing option date does not change any
   adjacent return horizon or compress the return-control calendar.

## A15.2 Weekday control

Replace the Tuesday--Friday indicators for the regressor session with Tuesday--Friday indicators
for the outcome session `t+1`, with Monday omitted. The next session's calendar identity is known
at the close of `t`, so this creates no look-ahead. It correctly controls seasonality in the dependent
variable around exchange holidays. The monthly-expiration indicator remains attached to the
regressor session `t` because it describes the option-book formation date.

## A15.3 Terminology and interpretation

- Replace "full-sample" with **registered-horizon** when referring to 2017-01-03 through
  2024-10-29.
- Replace unqualified "pre-data" with **pre-extraction** or **before proprietary-data access**.
- Describe the empirical plan as a retrospective pre-analysis protocol for a historically known
  period. Its timestamp prevents adaptation to the unobserved OptionMetrics proxy results; it does
  not make the historical market outcomes prospectively unknown.
- The end date is inherited from the original registered historical horizon and is not represented
  as the last date available from the vendor. Any later-period analysis is a separately labelled
  temporal extension and cannot replace the registered-horizon result.

## Effect

A15 changes no option-summary formula, primary coefficient, outcome transformation, HAC lag,
bootstrap setting, multiplicity rule, or interpretation threshold. It prevents irregular-horizon
lead construction, aligns the calendar control with the outcome it is meant to condition on, and
makes the registration claim accurately describe what was and was not unknown at the time of the
plan.
